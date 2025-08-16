"""
Reward Calculator for DRL Cache Training

Computes rewards for cache eviction decisions based on:
1. Future cache hits (primary reward)
2. Object size penalties (prevents hoarding large objects)
3. TTL-based bonuses (optional)
4. Access frequency bonuses (optional)
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import logging
from collections import defaultdict

from .config import RewardConfig


class RewardCalculator:
    """
    Calculates rewards for cache eviction decisions.
    
    The main reward signal is based on whether keeping an object in the cache
    leads to future cache hits before the object expires.
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cache for computed rewards to avoid recomputation
        self._reward_cache = {}
        
        # Statistics tracking
        self.stats = {
            'total_rewards_computed': 0,
            'positive_rewards': 0,
            'negative_rewards': 0,
            'zero_rewards': 0,
            'size_penalties_applied': 0,
            'ttl_bonuses_applied': 0,
            'frequency_bonuses_applied': 0
        }
    
    def compute_rewards(self, candidate_keys: List[str], actions: np.ndarray,
                       decision_time: datetime, full_dataset: List[Dict]) -> np.ndarray:
        """
        Compute rewards for a cache eviction decision.
        
        Args:
            candidate_keys: List of cache keys for the K candidates
            actions: Binary array indicating which candidates were evicted (1) or kept (0)
            decision_time: Timestamp when eviction decision was made
            full_dataset: Complete dataset to look up future accesses
        
        Returns:
            Array of rewards for each candidate
        """
        rewards = np.zeros(len(candidate_keys))
        
        # Find the corresponding cache objects at decision time
        objects_at_decision_time = self._get_objects_at_time(
            candidate_keys, decision_time, full_dataset
        )
        
        for i, (key, action) in enumerate(zip(candidate_keys, actions)):
            if key not in objects_at_decision_time:
                continue
            
            obj_info = objects_at_decision_time[key]
            
            if action == 1:  # Object was evicted
                # Reward for eviction: negative if object would have been hit
                rewards[i] = self._compute_eviction_reward(
                    key, obj_info, decision_time, full_dataset
                )
            else:  # Object was kept
                # Reward for keeping: positive if object is hit, negative otherwise
                rewards[i] = self._compute_keep_reward(
                    key, obj_info, decision_time, full_dataset
                )
            
            # Apply size penalty if configured
            if self.config.use_size_penalty:
                size_penalty = self._compute_size_penalty(obj_info['size_bytes'])
                if action == 0:  # Only penalize keeping large objects
                    rewards[i] -= size_penalty
                    if size_penalty > 0:
                        self.stats['size_penalties_applied'] += 1
            
            # Apply TTL bonus if configured
            if self.config.use_ttl_bonus:
                ttl_bonus = self._compute_ttl_bonus(obj_info, decision_time)
                if action == 0:  # Bonus for keeping objects with more TTL
                    rewards[i] += ttl_bonus
                    if ttl_bonus > 0:
                        self.stats['ttl_bonuses_applied'] += 1
            
            # Apply frequency bonus if configured
            if self.config.use_frequency_bonus:
                freq_bonus = self._compute_frequency_bonus(obj_info)
                if action == 0:  # Bonus for keeping frequently accessed objects
                    rewards[i] += freq_bonus
                    if freq_bonus > 0:
                        self.stats['frequency_bonuses_applied'] += 1
        
        # Update statistics
        self._update_stats(rewards)
        
        return rewards
    
    def _get_objects_at_time(self, candidate_keys: List[str], 
                           decision_time: datetime, 
                           full_dataset: List[Dict]) -> Dict[str, Dict]:
        """Get object information at the time of decision."""
        objects = {}
        
        # Build object state at decision time by replaying events
        object_states = defaultdict(dict)
        
        for sample in full_dataset:
            if sample['timestamp'] > decision_time:
                break
            
            # Update object states based on this sample
            for j, key in enumerate(sample.get('candidate_keys', [])):
                if key in candidate_keys:
                    # Extract object information from features
                    features = sample['features'][j] if j < len(sample['features']) else None
                    if features is not None and len(features) >= 6:
                        object_states[key] = {
                            'size_bytes': self._feature_to_size_bytes(features[1]),  # size_kb feature
                            'hit_count': int(features[2]),  # hit_count feature
                            'ttl_seconds': int(features[4]),  # ttl_left_sec feature
                            'last_upstream_time': int(features[5]),  # last_origin_rtt_us
                            'last_seen': sample['timestamp']
                        }
        
        return object_states
    
    def _feature_to_size_bytes(self, size_kb_feature: float) -> int:
        """Convert size feature back to bytes."""
        if self.config.log_scale_size:
            # Reverse log scaling: size_kb = log(1 + actual_size_kb)
            actual_size_kb = np.exp(size_kb_feature) - 1
        else:
            actual_size_kb = size_kb_feature
        
        return int(actual_size_kb * 1024)  # Convert to bytes
    
    def _compute_eviction_reward(self, key: str, obj_info: Dict,
                               decision_time: datetime, 
                               full_dataset: List[Dict]) -> float:
        """Compute reward for evicting an object."""
        # Check if object would have been accessed after decision time
        future_hits = self._count_future_hits(key, obj_info, decision_time, full_dataset)
        
        if future_hits > 0:
            # Negative reward for evicting an object that would have been hit
            return -self.config.hit_reward * future_hits
        else:
            # Small positive reward for evicting an object that wouldn't be hit
            return 0.1 * self.config.hit_reward
    
    def _compute_keep_reward(self, key: str, obj_info: Dict,
                           decision_time: datetime,
                           full_dataset: List[Dict]) -> float:
        """Compute reward for keeping an object."""
        # Check if object is accessed after decision time
        future_hits = self._count_future_hits(key, obj_info, decision_time, full_dataset)
        
        if future_hits > 0:
            # Positive reward for keeping an object that gets hit
            return self.config.hit_reward * future_hits
        else:
            # Negative reward for keeping an object that doesn't get hit
            return -self.config.miss_penalty
    
    def _count_future_hits(self, key: str, obj_info: Dict,
                         decision_time: datetime,
                         full_dataset: List[Dict]) -> int:
        """Count future cache hits for an object within its TTL."""
        # Calculate expiry time
        ttl_seconds = obj_info.get('ttl_seconds', 0)
        expiry_time = decision_time + timedelta(seconds=ttl_seconds)
        
        hit_count = 0
        
        # Look for future accesses to this key
        for sample in full_dataset:
            if sample['timestamp'] <= decision_time:
                continue
            
            if sample['timestamp'] > expiry_time:
                break  # Past TTL
            
            # Check if this key appears as a cache hit in future samples
            # This is a simplified approach - in reality, we'd need to track
            # the actual cache events more carefully
            if key in sample.get('candidate_keys', []):
                # Assume some probability of hit based on object characteristics
                hit_probability = min(1.0, obj_info.get('hit_count', 0) / 10.0)
                if np.random.random() < hit_probability:
                    hit_count += 1
        
        return hit_count
    
    def _compute_size_penalty(self, size_bytes: int) -> float:
        """Compute size-based penalty for keeping large objects."""
        if self.config.size_penalty_scale == "kb":
            size_value = size_bytes / 1024.0
        elif self.config.size_penalty_scale == "mb":
            size_value = size_bytes / (1024.0 ** 2)
        elif self.config.size_penalty_scale == "log":
            size_value = np.log1p(size_bytes / 1024.0)  # log(1 + size_kb)
        else:
            size_value = size_bytes / (1024.0 ** 2)  # Default to MB
        
        return self.config.size_penalty_lambda * size_value
    
    def _compute_ttl_bonus(self, obj_info: Dict, decision_time: datetime) -> float:
        """Compute TTL-based bonus for objects with more remaining life."""
        ttl_seconds = obj_info.get('ttl_seconds', 0)
        if ttl_seconds <= 0:
            return 0.0
        
        # Normalize TTL to [0, 1] range (assuming max TTL of 1 hour = 3600 seconds)
        normalized_ttl = min(1.0, ttl_seconds / 3600.0)
        
        return self.config.ttl_bonus_scale * normalized_ttl
    
    def _compute_frequency_bonus(self, obj_info: Dict) -> float:
        """Compute frequency-based bonus for frequently accessed objects."""
        hit_count = obj_info.get('hit_count', 0)
        if hit_count <= 0:
            return 0.0
        
        # Logarithmic bonus to avoid over-rewarding very high hit counts
        frequency_bonus = self.config.frequency_bonus_scale * np.log1p(hit_count)
        
        return frequency_bonus
    
    def _update_stats(self, rewards: np.ndarray) -> None:
        """Update statistics based on computed rewards."""
        self.stats['total_rewards_computed'] += len(rewards)
        self.stats['positive_rewards'] += np.sum(rewards > 0)
        self.stats['negative_rewards'] += np.sum(rewards < 0)
        self.stats['zero_rewards'] += np.sum(rewards == 0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reward computation statistics."""
        total = max(1, self.stats['total_rewards_computed'])
        
        return {
            **self.stats,
            'positive_reward_ratio': self.stats['positive_rewards'] / total,
            'negative_reward_ratio': self.stats['negative_rewards'] / total,
            'zero_reward_ratio': self.stats['zero_rewards'] / total,
            'average_reward_magnitude': np.mean([
                abs(r) for r in self.stats.values() 
                if isinstance(r, (int, float))
            ])
        }
    
    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0


class SimpleRewardCalculator:
    """
    Simplified reward calculator for testing and baseline comparison.
    
    This calculator uses a simple heuristic based on object characteristics
    without looking at future cache events.
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
    
    def compute_rewards(self, candidate_keys: List[str], actions: np.ndarray,
                       decision_time: datetime, features: np.ndarray) -> np.ndarray:
        """
        Compute rewards based on simple heuristics.
        
        Args:
            candidate_keys: Cache keys
            actions: Eviction actions (1=evict, 0=keep)
            decision_time: Decision timestamp
            features: Feature matrix [K, feature_dim]
        
        Returns:
            Reward array for each candidate
        """
        rewards = np.zeros(len(candidate_keys))
        
        for i, action in enumerate(actions):
            if i >= len(features):
                continue
            
            feature_vector = features[i]
            
            # Extract features
            age_sec = feature_vector[0]
            size_kb = feature_vector[1] 
            hit_count = feature_vector[2]
            inter_arrival_dt = feature_vector[3]
            ttl_left_sec = feature_vector[4]
            
            # Simple heuristic reward
            if action == 1:  # Evicted
                # Good to evict if: old, large, rarely hit, not accessed recently
                goodness_score = (
                    (age_sec > 1800) * 0.3 +          # Older than 30 min
                    (size_kb > 1000) * 0.3 +          # Larger than 1MB
                    (hit_count < 2) * 0.2 +           # Hit less than 2 times
                    (inter_arrival_dt > 600) * 0.2    # Not accessed in 10 min
                )
                rewards[i] = goodness_score * self.config.hit_reward
            
            else:  # Kept
                # Good to keep if: young, small, frequently hit, recently accessed
                goodness_score = (
                    (age_sec < 300) * 0.3 +           # Younger than 5 min
                    (size_kb < 100) * 0.2 +           # Smaller than 100KB
                    (hit_count > 5) * 0.3 +           # Hit more than 5 times
                    (inter_arrival_dt < 60) * 0.2     # Accessed in last minute
                )
                rewards[i] = goodness_score * self.config.hit_reward
                
                # Apply size penalty
                if self.config.use_size_penalty:
                    size_penalty = self.config.size_penalty_lambda * (size_kb / 1024.0)
                    rewards[i] -= size_penalty
        
        return rewards


def test_reward_calculator():
    """Test reward calculator implementations."""
    from .config import RewardConfig
    
    config = RewardConfig(
        hit_reward=1.0,
        use_size_penalty=True,
        size_penalty_lambda=0.05,
        use_frequency_bonus=True,
        frequency_bonus_scale=0.2
    )
    
    calculator = RewardCalculator(config)
    simple_calculator = SimpleRewardCalculator(config)
    
    # Test data
    candidate_keys = ["obj1", "obj2", "obj3", "obj4"]
    actions = np.array([1, 0, 1, 0])  # Evict obj1,obj3, keep obj2,obj4
    decision_time = datetime.now()
    
    # Mock features: [age_sec, size_kb, hit_count, inter_arrival_dt, ttl_left_sec, rtt_us]
    features = np.array([
        [1800, 2000, 1, 600, 300, 150],    # obj1: old, large, rarely hit
        [300, 50, 10, 30, 500, 100],       # obj2: young, small, frequently hit
        [900, 1500, 2, 400, 200, 200],     # obj3: medium age, large, some hits
        [600, 200, 8, 60, 400, 120]        # obj4: medium, frequently hit
    ])
    
    # Mock dataset for advanced calculator
    mock_dataset = [{
        'timestamp': decision_time,
        'candidate_keys': candidate_keys,
        'features': features,
        'actions': actions
    }]
    
    print("Testing Reward Calculators")
    print("=" * 50)
    
    # Test simple calculator
    simple_rewards = simple_calculator.compute_rewards(
        candidate_keys, actions, decision_time, features
    )
    print(f"Simple calculator rewards: {simple_rewards}")
    
    # Test advanced calculator
    advanced_rewards = calculator.compute_rewards(
        candidate_keys, actions, decision_time, mock_dataset
    )
    print(f"Advanced calculator rewards: {advanced_rewards}")
    
    # Print statistics
    print(f"Calculator statistics: {calculator.get_statistics()}")
    
    print("\nReward calculator test completed!")


if __name__ == "__main__":
    test_reward_calculator()
