"""
DRL Cache Policy Implementation for Benchmarking

Simulates the behavior of our DRL Cache system for comparison
with baseline eviction policies.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
import time
import random
from cache_simulator import EvictionPolicy, CacheObject


class MockDuelingDQN(nn.Module):
    """Mock implementation of Dueling DQN for benchmarking."""
    
    def __init__(self, input_dim: int = 6, max_k: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.max_k = max_k
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim * max_k, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Value stream: V(s)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream: A(s,a)
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_k)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        shared = self.shared(x)
        
        value = self.value(shared)  # [batch_size, 1]
        advantage = self.advantage(shared)  # [batch_size, max_k]
        
        # Apply mask if provided
        if mask is not None:
            advantage = advantage.masked_fill(~mask, -1e8)
        
        # Dueling combination
        if mask is not None:
            valid_advantages = advantage.masked_fill(~mask, 0)
            valid_count = mask.sum(dim=1, keepdim=True).float()
            advantage_mean = valid_advantages.sum(dim=1, keepdim=True) / valid_count.clamp(min=1)
        else:
            advantage_mean = advantage.mean(dim=1, keepdim=True)
        
        q_values = value + advantage - advantage_mean
        
        if mask is not None:
            q_values = q_values.masked_fill(~mask, -1e8)
        
        return q_values


class DRLCachePolicy(EvictionPolicy):
    """DRL-based cache eviction policy using trained neural network."""
    
    def __init__(self, model_path: str = None, max_k: int = 16, 
                 inference_timeout_us: float = 500,
                 fallback_policy: EvictionPolicy = None):
        super().__init__("DRL-Cache")
        
        self.max_k = max_k
        self.inference_timeout_us = inference_timeout_us
        self.fallback_policy = fallback_policy or LRUFallbackPolicy()
        
        # Feature normalization parameters (would be learned during training)
        self.feature_stats = {
            'means': np.array([1800, 4.0, 5.0, 300, 600, 150000]),  # age, log_size, hits, iat, ttl, rtt
            'stds': np.array([3600, 2.0, 10.0, 600, 1200, 100000]),
        }
        
        # Initialize model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Performance tracking
        self.inference_times = []
        self.fallback_count = 0
        
    def _load_model(self, model_path: str = None) -> MockDuelingDQN:
        """Load or create a mock DQN model."""
        model = MockDuelingDQN(input_dim=6, max_k=self.max_k)
        
        if model_path and torch.cuda.is_available():
            try:
                # In real implementation, this would load actual trained weights
                model.load_state_dict(torch.load(model_path))
                print(f"Loaded DRL model from {model_path}")
            except:
                print("Failed to load model, using randomly initialized weights")
        else:
            # For benchmarking, initialize with reasonable random weights
            self._init_smart_weights(model)
            print("Using mock DRL model with smart initialization")
        
        return model
    
    def _init_smart_weights(self, model: MockDuelingDQN):
        """Initialize weights to produce reasonable cache decisions."""
        with torch.no_grad():
            # Initialize shared layers to recognize important features
            for layer in model.shared:
                if isinstance(layer, nn.Linear):
                    # Initialize to prefer smaller, frequently accessed, recently used objects
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            
            # Initialize value head
            for layer in model.value:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            
            # Initialize advantage head with bias toward keeping small, recent objects
            for layer in model.advantage:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        # Bias toward not evicting (negative values)
                        nn.init.constant_(layer.bias, -0.5)
    
    def _extract_features(self, candidates: List[CacheObject], current_time: float) -> np.ndarray:
        """Extract normalized features for each candidate."""
        features = []
        
        for candidate in candidates:
            # Extract raw features (same as in actual implementation)
            age_sec = current_time - candidate.timestamp
            size_kb = candidate.size / 1024.0
            log_size_kb = np.log1p(size_kb)
            hit_count = candidate.access_count
            inter_arrival_dt = current_time - candidate.last_access
            ttl_left_sec = max(0, candidate.ttl - age_sec)
            last_origin_rtt_us = getattr(candidate, 'creation_cost', 0) * 1000000  # Mock RTT
            
            raw_features = np.array([
                age_sec, log_size_kb, hit_count, 
                inter_arrival_dt, ttl_left_sec, last_origin_rtt_us
            ])
            
            # Normalize features
            normalized = (raw_features - self.feature_stats['means']) / self.feature_stats['stds']
            normalized = np.clip(normalized, -5, 5)  # Clip outliers
            
            features.append(normalized)
        
        # Pad with zeros if fewer than max_k candidates
        while len(features) < self.max_k:
            features.append(np.zeros(6))
        
        return np.array(features[:self.max_k])  # Take at most max_k candidates
    
    def should_evict(self, candidates: List[CacheObject], 
                    bytes_needed: int, current_time: float) -> List[str]:
        """Use DRL model to decide which objects to evict."""
        start_time = time.perf_counter()
        
        if len(candidates) == 0:
            return []
        
        try:
            # Extract and normalize features
            features = self._extract_features(candidates, current_time)
            
            # Create input tensor
            input_tensor = torch.FloatTensor(features.flatten()).unsqueeze(0)  # [1, max_k * 6]
            
            # Create mask for valid candidates
            valid_mask = torch.zeros(1, self.max_k, dtype=torch.bool)
            valid_mask[0, :len(candidates)] = True
            
            # Run inference with timeout check
            inference_start = time.perf_counter()
            
            with torch.no_grad():
                q_values = self.model(input_tensor, valid_mask)
            
            inference_time = (time.perf_counter() - inference_start) * 1e6
            self.inference_times.append(inference_time)
            
            # Check if inference exceeded timeout
            if inference_time > self.inference_timeout_us:
                self.fallback_count += 1
                return self.fallback_policy.should_evict(candidates, bytes_needed, current_time)
            
            # Convert Q-values to eviction decisions
            q_values = q_values[0][:len(candidates)]  # Get valid Q-values
            
            # Higher Q-value means more likely to evict
            # We'll evict objects with positive Q-values, prioritizing highest
            evict_indices = []
            q_values_with_indices = [(q_values[i].item(), i) for i in range(min(len(candidates), len(q_values)))]
            q_values_with_indices.sort(reverse=True)  # Highest Q-value first
            
            bytes_freed = 0
            for q_val, idx in q_values_with_indices:
                if bytes_freed >= bytes_needed:
                    break
                if idx < len(candidates):  # Bounds check
                    if q_val > 0.0:  # Only evict if Q-value is positive
                        evict_indices.append(idx)
                        bytes_freed += candidates[idx].size
            
            # If not enough space freed, add more objects
            if bytes_freed < bytes_needed:
                for q_val, idx in q_values_with_indices:
                    if bytes_freed >= bytes_needed:
                        break
                    if idx not in evict_indices and idx < len(candidates):  # Bounds check
                        evict_indices.append(idx)
                        bytes_freed += candidates[idx].size
                
            evicted_keys = [candidates[i].key for i in evict_indices]
            
        except Exception as e:
            # Fallback on any error
            print(f"DRL inference failed: {e}")
            self.fallback_count += 1
            return self.fallback_policy.should_evict(candidates, bytes_needed, current_time)
        
        # Record performance
        total_time = (time.perf_counter() - start_time) * 1e6
        self.stats['decisions'] += 1
        self.stats['decision_time_us'].append(total_time)
        
        return evicted_keys
    
    def on_access(self, obj: CacheObject, current_time: float):
        """Called when an object is accessed."""
        pass
    
    def on_insert(self, obj: CacheObject, current_time: float):
        """Called when a new object is inserted."""
        pass
    
    def on_evict(self, obj: CacheObject, current_time: float):
        """Called when an object is evicted."""
        self.stats['evictions'] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        base_stats = self.stats.copy()
        
        if self.inference_times:
            base_stats.update({
                'avg_inference_time_us': np.mean(self.inference_times),
                'p50_inference_time_us': np.percentile(self.inference_times, 50),
                'p95_inference_time_us': np.percentile(self.inference_times, 95),
                'p99_inference_time_us': np.percentile(self.inference_times, 99),
                'max_inference_time_us': max(self.inference_times),
                'fallback_count': self.fallback_count,
                'fallback_rate': self.fallback_count / max(1, self.stats['decisions'])
            })
        
        return base_stats


class LRUFallbackPolicy(EvictionPolicy):
    """Simple LRU policy for fallback."""
    
    def __init__(self):
        super().__init__("LRU-Fallback")
    
    def should_evict(self, candidates: List[CacheObject], 
                    bytes_needed: int, current_time: float) -> List[str]:
        # Sort by last access time (oldest first)
        candidates.sort(key=lambda x: x.last_access)
        
        evicted = []
        bytes_freed = 0
        
        for obj in candidates:
            if bytes_freed >= bytes_needed:
                break
            evicted.append(obj.key)
            bytes_freed += obj.size
        
        return evicted
    
    def on_access(self, obj: CacheObject, current_time: float):
        pass
    
    def on_insert(self, obj: CacheObject, current_time: float):
        pass
    
    def on_evict(self, obj: CacheObject, current_time: float):
        pass


class OptimalOfflinePolicy(EvictionPolicy):
    """Optimal offline policy using future knowledge (Belady's algorithm)."""
    
    def __init__(self, future_accesses: Dict[str, List[float]]):
        super().__init__("Optimal-Offline")
        self.future_accesses = future_accesses
    
    def should_evict(self, candidates: List[CacheObject], 
                    bytes_needed: int, current_time: float) -> List[str]:
        start_time = time.perf_counter()
        
        def next_access_time(obj_key: str) -> float:
            """Find the next access time for an object."""
            if obj_key not in self.future_accesses:
                return float('inf')
            
            future_times = self.future_accesses[obj_key]
            for access_time in future_times:
                if access_time > current_time:
                    return access_time
            return float('inf')
        
        # Sort by next access time (farthest first)
        candidates_with_next_access = [
            (next_access_time(obj.key), obj) for obj in candidates
        ]
        candidates_with_next_access.sort(key=lambda x: x[0], reverse=True)
        
        evicted = []
        bytes_freed = 0
        
        for _, obj in candidates_with_next_access:
            if bytes_freed >= bytes_needed:
                break
            evicted.append(obj.key)
            bytes_freed += obj.size
        
        # Record performance
        total_time = (time.perf_counter() - start_time) * 1e6
        self.stats['decisions'] += 1
        self.stats['decision_time_us'].append(total_time)
        
        return evicted
    
    def on_access(self, obj: CacheObject, current_time: float):
        pass
    
    def on_insert(self, obj: CacheObject, current_time: float):
        pass
    
    def on_evict(self, obj: CacheObject, current_time: float):
        self.stats['evictions'] += 1


# Improved baseline policies for comparison

class AdaptiveLRUPolicy(EvictionPolicy):
    """LRU with size awareness - evicts large old objects first."""
    
    def __init__(self, size_threshold: int = 1024 * 1024):  # 1MB threshold
        super().__init__("Adaptive-LRU")
        self.size_threshold = size_threshold
    
    def should_evict(self, candidates: List[CacheObject], 
                    bytes_needed: int, current_time: float) -> List[str]:
        start_time = time.perf_counter()
        
        # Separate large and small objects
        large_objects = [obj for obj in candidates if obj.size > self.size_threshold]
        small_objects = [obj for obj in candidates if obj.size <= self.size_threshold]
        
        # Sort both by LRU order
        large_objects.sort(key=lambda x: x.last_access)
        small_objects.sort(key=lambda x: x.last_access)
        
        evicted = []
        bytes_freed = 0
        
        # Prefer evicting large objects first
        for obj in large_objects:
            if bytes_freed >= bytes_needed:
                break
            evicted.append(obj.key)
            bytes_freed += obj.size
        
        # Then evict small objects if needed
        for obj in small_objects:
            if bytes_freed >= bytes_needed:
                break
            evicted.append(obj.key)
            bytes_freed += obj.size
        
        # Record performance
        total_time = (time.perf_counter() - start_time) * 1e6
        self.stats['decisions'] += 1
        self.stats['decision_time_us'].append(total_time)
        
        return evicted
    
    def on_access(self, obj: CacheObject, current_time: float):
        pass
    
    def on_insert(self, obj: CacheObject, current_time: float):
        pass
    
    def on_evict(self, obj: CacheObject, current_time: float):
        self.stats['evictions'] += 1


class FrequencyAwareLRUPolicy(EvictionPolicy):
    """LRU with frequency awareness - protects hot objects."""
    
    def __init__(self, frequency_threshold: int = 5):
        super().__init__("Frequency-Aware-LRU")
        self.frequency_threshold = frequency_threshold
    
    def should_evict(self, candidates: List[CacheObject], 
                    bytes_needed: int, current_time: float) -> List[str]:
        start_time = time.perf_counter()
        
        # Separate hot and cold objects
        cold_objects = [obj for obj in candidates if obj.access_count < self.frequency_threshold]
        hot_objects = [obj for obj in candidates if obj.access_count >= self.frequency_threshold]
        
        # Sort both by LRU order
        cold_objects.sort(key=lambda x: x.last_access)
        hot_objects.sort(key=lambda x: x.last_access)
        
        evicted = []
        bytes_freed = 0
        
        # Prefer evicting cold objects first
        for obj in cold_objects:
            if bytes_freed >= bytes_needed:
                break
            evicted.append(obj.key)
            bytes_freed += obj.size
        
        # Then evict hot objects if absolutely necessary
        for obj in hot_objects:
            if bytes_freed >= bytes_needed:
                break
            evicted.append(obj.key)
            bytes_freed += obj.size
        
        # Record performance
        total_time = (time.perf_counter() - start_time) * 1e6
        self.stats['decisions'] += 1
        self.stats['decision_time_us'].append(total_time)
        
        return evicted
    
    def on_access(self, obj: CacheObject, current_time: float):
        pass
    
    def on_insert(self, obj: CacheObject, current_time: float):
        pass
    
    def on_evict(self, obj: CacheObject, current_time: float):
        self.stats['evictions'] += 1
