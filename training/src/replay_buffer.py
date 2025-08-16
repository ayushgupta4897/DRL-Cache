"""
Prioritized Experience Replay Buffer for DRL Cache Training

Implements prioritized experience replay (PER) as described in 
"Prioritized Experience Replay" by Schaul et al., 2016.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
import random
from collections import deque
from dataclasses import dataclass

from .config import ReplayBufferConfig


@dataclass
class Transition:
    """Single transition in the replay buffer."""
    state: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor
    priority: float = 1.0


class SumTree:
    """
    Sum tree data structure for efficient prioritized sampling.
    
    This is a complete binary tree where each leaf stores a priority value
    and each internal node stores the sum of priorities in its subtree.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Internal nodes + leaves
        self.data = [None] * capacity  # Store transitions
        self.data_pointer = 0
        self.size = 0
    
    def add(self, priority: float, data) -> None:
        """Add new data with given priority."""
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
    
    def update(self, tree_idx: int, priority: float) -> None:
        """Update priority of a tree node and propagate changes."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get_leaf(self, s: float) -> Tuple[int, float, any]:
        """
        Get leaf node for sampling.
        
        Args:
            s: Sample value in [0, total_priority]
        
        Returns:
            (tree_index, priority, data)
        """
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we reach a leaf
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            # Navigate based on cumulative sum
            if s <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                s -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    @property
    def total_priority(self) -> float:
        """Get total priority (root of the tree)."""
        return self.tree[0]
    
    @property
    def min_priority(self) -> float:
        """Get minimum priority among stored transitions."""
        if self.size == 0:
            return 1.0
        return np.min(self.tree[-self.capacity:self.capacity - self.size])
    
    @property
    def max_priority(self) -> float:
        """Get maximum priority among stored transitions."""
        if self.size == 0:
            return 1.0
        return np.max(self.tree[-self.capacity:self.capacity - self.size])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Stores transitions with associated priorities and samples them
    proportionally to their TD-error based priorities.
    """
    
    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self.alpha = config.alpha
        self.beta = config.beta_start
        self.beta_increment = (config.beta_end - config.beta_start) / 100000  # Over 100k steps
        self.epsilon = config.epsilon
        
        if config.prioritized:
            self.tree = SumTree(config.capacity)
        else:
            self.buffer = deque(maxlen=config.capacity)
        
        self.max_priority = 1.0
        self.global_step = 0
    
    def add(self, state: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
            next_state: torch.Tensor, done: torch.Tensor, priority: Optional[float] = None) -> None:
        """Add a transition to the buffer."""
        transition = Transition(
            state=state.clone(),
            actions=actions.clone(),
            rewards=rewards.clone(),
            next_state=next_state.clone(),
            done=done.clone()
        )
        
        if self.config.prioritized:
            # Use max priority for new transitions
            priority = priority if priority is not None else self.max_priority
            self.tree.add(priority ** self.alpha, transition)
        else:
            self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Sample a batch of transitions.
        
        Returns:
            If prioritized: (states, actions, rewards, next_states, dones, weights, indices)
            If uniform: (states, actions, rewards, next_states, dones)
        """
        if len(self) < self.config.min_size_to_sample:
            return None
        
        if self.config.prioritized:
            return self._sample_prioritized(batch_size)
        else:
            return self._sample_uniform(batch_size)
    
    def _sample_prioritized(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch using prioritized sampling."""
        transitions = []
        indices = []
        priorities = []
        
        # Calculate sampling probability range
        segment = self.tree.total_priority / batch_size
        
        for i in range(batch_size):
            # Sample from segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get_leaf(s)
            transitions.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        weights = (len(self) * sampling_probabilities) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Convert to tensors
        states = torch.stack([t.state for t in transitions])
        actions = torch.stack([t.actions for t in transitions])
        rewards = torch.stack([t.rewards for t in transitions])
        next_states = torch.stack([t.next_state for t in transitions])
        dones = torch.stack([t.done for t in transitions])
        weights = torch.FloatTensor(weights)
        
        # Update beta
        self.beta = min(self.config.beta_end, 
                       self.beta + self.beta_increment * self.global_step)
        self.global_step += 1
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def _sample_uniform(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch uniformly."""
        transitions = random.sample(list(self.buffer), batch_size)
        
        states = torch.stack([t.state for t in transitions])
        actions = torch.stack([t.actions for t in transitions])
        rewards = torch.stack([t.rewards for t in transitions])
        next_states = torch.stack([t.next_state for t in transitions])
        dones = torch.stack([t.done for t in transitions])
        
        return states, actions, rewards, next_states, dones
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """Update priorities for given transitions."""
        if not self.config.prioritized:
            return
        
        for idx, priority in zip(indices, priorities):
            # Add small epsilon to avoid zero priorities
            priority = abs(priority) + self.epsilon
            self.tree.update(idx, priority ** self.alpha)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return current size of the buffer."""
        if self.config.prioritized:
            return self.tree.size
        else:
            return len(self.buffer)
    
    def get_statistics(self) -> dict:
        """Get buffer statistics."""
        if self.config.prioritized:
            return {
                'size': self.tree.size,
                'capacity': self.tree.capacity,
                'total_priority': self.tree.total_priority,
                'min_priority': self.tree.min_priority,
                'max_priority': self.tree.max_priority,
                'beta': self.beta,
                'alpha': self.alpha
            }
        else:
            return {
                'size': len(self.buffer),
                'capacity': self.buffer.maxlen
            }


class UniformReplayBuffer:
    """
    Simple uniform replay buffer for comparison.
    
    This is a baseline implementation that samples transitions uniformly
    without prioritization.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
            next_state: torch.Tensor, done: torch.Tensor) -> None:
        """Add a transition to the buffer."""
        transition = Transition(
            state=state.clone(),
            actions=actions.clone(),
            rewards=rewards.clone(),
            next_state=next_state.clone(),
            done=done.clone()
        )
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """Sample a batch of transitions uniformly."""
        if len(self.buffer) < batch_size:
            return None
        
        transitions = random.sample(list(self.buffer), batch_size)
        
        states = torch.stack([t.state for t in transitions])
        actions = torch.stack([t.actions for t in transitions])
        rewards = torch.stack([t.rewards for t in transitions])
        next_states = torch.stack([t.next_state for t in transitions])
        dones = torch.stack([t.done for t in transitions])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


def test_replay_buffer():
    """Test the replay buffer implementation."""
    from .config import ReplayBufferConfig
    
    # Test configuration
    config = ReplayBufferConfig(
        capacity=1000,
        prioritized=True,
        alpha=0.6,
        beta_start=0.4
    )
    
    buffer = PrioritizedReplayBuffer(config)
    
    # Add some test transitions
    for i in range(100):
        state = torch.randn(16 * 6)  # 16 candidates, 6 features each
        actions = torch.randint(0, 2, (16,)).float()
        rewards = torch.randn(16)
        next_state = torch.randn(16 * 6)
        done = torch.zeros(16)
        
        buffer.add(state, actions, rewards, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Buffer stats: {buffer.get_statistics()}")
    
    # Test sampling
    batch = buffer.sample(32)
    if batch is not None:
        if config.prioritized:
            states, actions, rewards, next_states, dones, weights, indices = batch
            print(f"Sampled batch shapes:")
            print(f"  States: {states.shape}")
            print(f"  Actions: {actions.shape}")
            print(f"  Rewards: {rewards.shape}")
            print(f"  Weights: {weights.shape}")
            print(f"  Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
        else:
            states, actions, rewards, next_states, dones = batch
            print(f"Sampled batch shapes:")
            print(f"  States: {states.shape}")
            print(f"  Actions: {actions.shape}")
            print(f"  Rewards: {rewards.shape}")
    
    print("Replay buffer test completed successfully!")


if __name__ == "__main__":
    test_replay_buffer()
