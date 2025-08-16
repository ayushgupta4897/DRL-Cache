"""
Clean DRL Policy Implementation for Research Benchmark

Contains only the essential policies needed for the trap scenario benchmark
without heavy dependencies like PyTorch.
"""

import numpy as np
import math
from typing import List, Dict, Any
from collections import defaultdict, deque
from cache_simulator import EvictionPolicy, CacheObject


class AdaptiveLRUPolicy(EvictionPolicy):
    """LRU policy that adapts based on object size."""
    
    def __init__(self, size_threshold: int = 100 * 1024):
        self.name = "AdaptiveLRU"
        self.size_threshold = size_threshold
        self.access_times = {}
    
    def should_evict(self, candidates: List[CacheObject], bytes_needed: int, current_time: float) -> List[str]:
        if not candidates:
            return []
        
        # Sort by access time, but prioritize large objects for eviction
        def adaptive_key(obj):
            access_time = self.access_times.get(obj.key, 0)
            size_penalty = 1.0 if obj.size > self.size_threshold else 0.0
            return (access_time - size_penalty * 1000)  # Earlier access time + size penalty = evict first
        
        candidates.sort(key=adaptive_key)
        
        evict_keys = []
        space_freed = 0
        
        for candidate in candidates:
            if space_freed >= bytes_needed:
                break
            evict_keys.append(candidate.key)
            space_freed += candidate.size
        
        return evict_keys
    
    def on_insert(self, cache_obj: CacheObject, current_time: float):
        self.access_times[cache_obj.key] = current_time
    
    def on_access(self, cache_obj: CacheObject, current_time: float):
        self.access_times[cache_obj.key] = current_time
    
    def on_evict(self, cache_obj: CacheObject, current_time: float):
        self.access_times.pop(cache_obj.key, None)


class FrequencyAwareLRUPolicy(EvictionPolicy):
    """LRU policy that considers access frequency."""
    
    def __init__(self, frequency_weight: float = 0.5):
        self.name = "FrequencyAwareLRU"
        self.frequency_weight = frequency_weight
        self.access_times = {}
        self.access_counts = defaultdict(int)
    
    def should_evict(self, candidates: List[CacheObject], bytes_needed: int, current_time: float) -> List[str]:
        if not candidates:
            return []
        
        # Score based on recency and frequency
        def frequency_aware_score(obj):
            last_access = self.access_times.get(obj.key, 0)
            frequency = self.access_counts[obj.key]
            
            recency_score = current_time - last_access
            frequency_score = 1.0 / (frequency + 1)  # Lower is better (less frequent)
            
            return self.frequency_weight * frequency_score + (1 - self.frequency_weight) * recency_score
        
        candidates.sort(key=frequency_aware_score, reverse=True)  # Highest score (worst) first
        
        evict_keys = []
        space_freed = 0
        
        for candidate in candidates:
            if space_freed >= bytes_needed:
                break
            evict_keys.append(candidate.key)
            space_freed += candidate.size
        
        return evict_keys
    
    def on_insert(self, cache_obj: CacheObject, current_time: float):
        self.access_times[cache_obj.key] = current_time
        self.access_counts[cache_obj.key] += 1
    
    def on_access(self, cache_obj: CacheObject, current_time: float):
        self.access_times[cache_obj.key] = current_time
        self.access_counts[cache_obj.key] += 1
    
    def on_evict(self, cache_obj: CacheObject, current_time: float):
        self.access_times.pop(cache_obj.key, None)
        self.access_counts.pop(cache_obj.key, None)


class MockDRLPolicy(EvictionPolicy):
    """Simple mock DRL policy for comparison (not the real breakthrough)."""
    
    def __init__(self):
        self.name = "MockDRL"
        self.access_history = defaultdict(list)
        self.size_bias = 0.3
    
    def should_evict(self, candidates: List[CacheObject], bytes_needed: int, current_time: float) -> List[str]:
        if not candidates:
            return []
        
        # Simple heuristic that mimics some DRL-like behavior
        def mock_drl_score(obj):
            # Frequency component
            frequency = len(self.access_history[obj.key])
            frequency_score = min(frequency / 10.0, 1.0)
            
            # Size component (smaller is better)
            size_score = 1.0 - min(obj.size / (2 * 1024 * 1024), 1.0)
            
            # Recency component
            if self.access_history[obj.key]:
                time_since = current_time - self.access_history[obj.key][-1]
                recency_score = math.exp(-time_since / 3600)  # 1 hour decay
            else:
                recency_score = 0.0
            
            # Combined score (higher is better, so we'll evict lowest)
            return 0.4 * frequency_score + 0.3 * size_score + 0.3 * recency_score
        
        scored_candidates = [(obj, mock_drl_score(obj)) for obj in candidates]
        scored_candidates.sort(key=lambda x: x[1])  # Lowest score first (evict first)
        
        evict_keys = []
        space_freed = 0
        
        for candidate, score in scored_candidates:
            if space_freed >= bytes_needed:
                break
            evict_keys.append(candidate.key)
            space_freed += candidate.size
        
        return evict_keys
    
    def on_insert(self, cache_obj: CacheObject, current_time: float):
        self.access_history[cache_obj.key].append(current_time)
    
    def on_access(self, cache_obj: CacheObject, current_time: float):
        self.access_history[cache_obj.key].append(current_time)
        # Keep history manageable
        if len(self.access_history[cache_obj.key]) > 20:
            self.access_history[cache_obj.key] = self.access_history[cache_obj.key][-10:]
    
    def on_evict(self, cache_obj: CacheObject, current_time: float):
        pass  # Keep history for learning


# Note: The actual TrapAwareDRL policy is implemented directly in trap_scenario_drl.py
# This file provides the baseline policies for comparison
