"""
Comprehensive Cache Simulator for Benchmarking

This module provides a high-fidelity cache simulator that can evaluate
different eviction policies on real-world datasets with detailed metrics.
"""

import time
import heapq
import random
from abc import ABC, abstractmethod
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


@dataclass
class CacheRequest:
    """Represents a single cache request."""
    timestamp: float
    key: str
    size: int
    ttl: Optional[int] = None  # Time to live in seconds
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_type: str = "GET"
    response_time: Optional[float] = None
    
    def __post_init__(self):
        if self.ttl is None:
            self.ttl = 3600  # Default 1 hour TTL


@dataclass
class CacheObject:
    """Represents an object stored in cache."""
    key: str
    size: int
    timestamp: float
    last_access: float
    access_count: int = 0
    ttl: int = 3600
    creation_cost: float = 0.0  # Cost to fetch from origin
    
    @property
    def age(self) -> float:
        """Age of object in seconds."""
        return time.time() - self.timestamp
    
    @property
    def idle_time(self) -> float:
        """Time since last access."""
        return time.time() - self.last_access
    
    def is_expired(self, current_time: float) -> bool:
        """Check if object has expired."""
        return current_time > (self.timestamp + self.ttl)
    
    def access(self, current_time: float):
        """Record an access to this object."""
        self.last_access = current_time
        self.access_count += 1


class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""
    
    def __init__(self, name: str):
        self.name = name
        self.stats = {
            'evictions': 0,
            'decisions': 0,
            'decision_time_us': []
        }
    
    @abstractmethod
    def should_evict(self, candidates: List[CacheObject], 
                    bytes_needed: int, current_time: float) -> List[str]:
        """Return list of cache keys to evict."""
        pass
    
    @abstractmethod
    def on_access(self, obj: CacheObject, current_time: float):
        """Called when an object is accessed."""
        pass
    
    @abstractmethod
    def on_insert(self, obj: CacheObject, current_time: float):
        """Called when a new object is inserted."""
        pass
    
    @abstractmethod
    def on_evict(self, obj: CacheObject, current_time: float):
        """Called when an object is evicted."""
        pass
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'evictions': 0,
            'decisions': 0,
            'decision_time_us': []
        }


class LRUPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def __init__(self):
        super().__init__("LRU")
        self.access_order = OrderedDict()  # key -> timestamp
    
    def should_evict(self, candidates: List[CacheObject], 
                    bytes_needed: int, current_time: float) -> List[str]:
        start_time = time.perf_counter()
        
        # Sort by last access time (oldest first)
        candidates.sort(key=lambda x: x.last_access)
        
        evicted = []
        bytes_freed = 0
        
        for obj in candidates:
            if bytes_freed >= bytes_needed:
                break
            evicted.append(obj.key)
            bytes_freed += obj.size
        
        self.stats['decisions'] += 1
        self.stats['decision_time_us'].append((time.perf_counter() - start_time) * 1e6)
        
        return evicted
    
    def on_access(self, obj: CacheObject, current_time: float):
        self.access_order[obj.key] = current_time
    
    def on_insert(self, obj: CacheObject, current_time: float):
        self.access_order[obj.key] = current_time
    
    def on_evict(self, obj: CacheObject, current_time: float):
        self.access_order.pop(obj.key, None)
        self.stats['evictions'] += 1


class LFUPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def __init__(self):
        super().__init__("LFU")
        self.frequencies = defaultdict(int)
        self.min_heap = []  # (frequency, timestamp, key)
    
    def should_evict(self, candidates: List[CacheObject], 
                    bytes_needed: int, current_time: float) -> List[str]:
        start_time = time.perf_counter()
        
        # Sort by access count (least frequent first), then by age
        candidates.sort(key=lambda x: (x.access_count, x.timestamp))
        
        evicted = []
        bytes_freed = 0
        
        for obj in candidates:
            if bytes_freed >= bytes_needed:
                break
            evicted.append(obj.key)
            bytes_freed += obj.size
        
        self.stats['decisions'] += 1
        self.stats['decision_time_us'].append((time.perf_counter() - start_time) * 1e6)
        
        return evicted
    
    def on_access(self, obj: CacheObject, current_time: float):
        self.frequencies[obj.key] = obj.access_count
    
    def on_insert(self, obj: CacheObject, current_time: float):
        self.frequencies[obj.key] = 1
    
    def on_evict(self, obj: CacheObject, current_time: float):
        self.frequencies.pop(obj.key, None)
        self.stats['evictions'] += 1


class FIFOPolicy(EvictionPolicy):
    """First In, First Out eviction policy."""
    
    def __init__(self):
        super().__init__("FIFO")
        self.insertion_order = {}  # key -> timestamp
    
    def should_evict(self, candidates: List[CacheObject], 
                    bytes_needed: int, current_time: float) -> List[str]:
        start_time = time.perf_counter()
        
        # Sort by insertion time (oldest first)
        candidates.sort(key=lambda x: x.timestamp)
        
        evicted = []
        bytes_freed = 0
        
        for obj in candidates:
            if bytes_freed >= bytes_needed:
                break
            evicted.append(obj.key)
            bytes_freed += obj.size
        
        self.stats['decisions'] += 1
        self.stats['decision_time_us'].append((time.perf_counter() - start_time) * 1e6)
        
        return evicted
    
    def on_access(self, obj: CacheObject, current_time: float):
        pass  # FIFO doesn't care about access
    
    def on_insert(self, obj: CacheObject, current_time: float):
        self.insertion_order[obj.key] = current_time
    
    def on_evict(self, obj: CacheObject, current_time: float):
        self.insertion_order.pop(obj.key, None)
        self.stats['evictions'] += 1


class RandomPolicy(EvictionPolicy):
    """Random eviction policy."""
    
    def __init__(self, seed: int = 42):
        super().__init__("Random")
        self.rng = random.Random(seed)
    
    def should_evict(self, candidates: List[CacheObject], 
                    bytes_needed: int, current_time: float) -> List[str]:
        start_time = time.perf_counter()
        
        # Randomly shuffle candidates
        shuffled = candidates.copy()
        self.rng.shuffle(shuffled)
        
        evicted = []
        bytes_freed = 0
        
        for obj in shuffled:
            if bytes_freed >= bytes_needed:
                break
            evicted.append(obj.key)
            bytes_freed += obj.size
        
        self.stats['decisions'] += 1
        self.stats['decision_time_us'].append((time.perf_counter() - start_time) * 1e6)
        
        return evicted
    
    def on_access(self, obj: CacheObject, current_time: float):
        pass
    
    def on_insert(self, obj: CacheObject, current_time: float):
        pass
    
    def on_evict(self, obj: CacheObject, current_time: float):
        self.stats['evictions'] += 1


class SizeBasedPolicy(EvictionPolicy):
    """Size-based eviction (evict largest objects first)."""
    
    def __init__(self):
        super().__init__("SizeBased")
    
    def should_evict(self, candidates: List[CacheObject], 
                    bytes_needed: int, current_time: float) -> List[str]:
        start_time = time.perf_counter()
        
        # Sort by size (largest first)
        candidates.sort(key=lambda x: x.size, reverse=True)
        
        evicted = []
        bytes_freed = 0
        
        for obj in candidates:
            if bytes_freed >= bytes_needed:
                break
            evicted.append(obj.key)
            bytes_freed += obj.size
        
        self.stats['decisions'] += 1
        self.stats['decision_time_us'].append((time.perf_counter() - start_time) * 1e6)
        
        return evicted
    
    def on_access(self, obj: CacheObject, current_time: float):
        pass
    
    def on_insert(self, obj: CacheObject, current_time: float):
        pass
    
    def on_evict(self, obj: CacheObject, current_time: float):
        self.stats['evictions'] += 1


class HybridLRUSizePolicy(EvictionPolicy):
    """Hybrid policy that considers both recency and size."""
    
    def __init__(self, size_weight: float = 0.3):
        super().__init__("HybridLRUSize")
        self.size_weight = size_weight
    
    def should_evict(self, candidates: List[CacheObject], 
                    bytes_needed: int, current_time: float) -> List[str]:
        start_time = time.perf_counter()
        
        # Compute hybrid score: (1-w) * recency_score + w * size_score
        max_size = max(obj.size for obj in candidates) if candidates else 1
        
        def hybrid_score(obj):
            recency_score = current_time - obj.last_access  # Higher = older
            size_score = obj.size / max_size  # Normalized size
            return (1 - self.size_weight) * recency_score + self.size_weight * size_score
        
        # Sort by hybrid score (higher score = more likely to evict)
        candidates.sort(key=hybrid_score, reverse=True)
        
        evicted = []
        bytes_freed = 0
        
        for obj in candidates:
            if bytes_freed >= bytes_needed:
                break
            evicted.append(obj.key)
            bytes_freed += obj.size
        
        self.stats['decisions'] += 1
        self.stats['decision_time_us'].append((time.perf_counter() - start_time) * 1e6)
        
        return evicted
    
    def on_access(self, obj: CacheObject, current_time: float):
        pass
    
    def on_insert(self, obj: CacheObject, current_time: float):
        pass
    
    def on_evict(self, obj: CacheObject, current_time: float):
        self.stats['evictions'] += 1


@dataclass
class CacheStats:
    """Comprehensive cache performance statistics."""
    policy_name: str
    
    # Basic metrics
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Byte-level metrics
    total_bytes_requested: int = 0
    bytes_served_from_cache: int = 0
    bytes_served_from_origin: int = 0
    
    # Latency metrics
    total_response_time: float = 0.0
    cache_response_times: List[float] = field(default_factory=list)
    origin_response_times: List[float] = field(default_factory=list)
    
    # Eviction metrics
    total_evictions: int = 0
    bytes_evicted: int = 0
    eviction_decisions: int = 0
    
    # Cost metrics
    total_cost: float = 0.0  # Cost of serving from origin
    
    @property
    def hit_ratio(self) -> float:
        """Request-level hit ratio."""
        return self.cache_hits / max(1, self.total_requests)
    
    @property
    def byte_hit_ratio(self) -> float:
        """Byte-level hit ratio."""
        return self.bytes_served_from_cache / max(1, self.total_bytes_requested)
    
    @property
    def miss_ratio(self) -> float:
        """Request-level miss ratio."""
        return 1.0 - self.hit_ratio
    
    @property
    def avg_response_time(self) -> float:
        """Average response time."""
        return self.total_response_time / max(1, self.total_requests)
    
    @property
    def avg_cache_response_time(self) -> float:
        """Average cache response time."""
        return np.mean(self.cache_response_times) if self.cache_response_times else 0.0
    
    @property
    def avg_origin_response_time(self) -> float:
        """Average origin response time."""
        return np.mean(self.origin_response_times) if self.origin_response_times else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'policy_name': self.policy_name,
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_ratio': self.hit_ratio,
            'byte_hit_ratio': self.byte_hit_ratio,
            'total_bytes_requested': self.total_bytes_requested,
            'bytes_served_from_cache': self.bytes_served_from_cache,
            'bytes_served_from_origin': self.bytes_served_from_origin,
            'total_evictions': self.total_evictions,
            'bytes_evicted': self.bytes_evicted,
            'avg_response_time': self.avg_response_time,
            'avg_cache_response_time': self.avg_cache_response_time,
            'avg_origin_response_time': self.avg_origin_response_time,
            'total_cost': self.total_cost,
        }


class CacheSimulator:
    """High-fidelity cache simulator with comprehensive metrics."""
    
    def __init__(self, 
                 cache_size: int,
                 eviction_policy: EvictionPolicy,
                 cache_response_time: float = 0.001,  # 1ms
                 origin_response_time: float = 0.100,  # 100ms
                 origin_cost_per_byte: float = 1e-6):  # $1/MB
        
        self.cache_size = cache_size
        self.policy = eviction_policy
        self.cache_response_time = cache_response_time
        self.origin_response_time = origin_response_time
        self.origin_cost_per_byte = origin_cost_per_byte
        
        # Cache storage
        self.cache: Dict[str, CacheObject] = {}
        self.current_size = 0
        
        # Statistics
        self.stats = CacheStats(policy_name=eviction_policy.name)
        
        # Time tracking
        self.current_time = 0.0
    
    def _evict_expired(self, current_time: float) -> int:
        """Remove expired objects and return bytes freed."""
        expired_keys = []
        for key, obj in self.cache.items():
            if obj.is_expired(current_time):
                expired_keys.append(key)
        
        bytes_freed = 0
        for key in expired_keys:
            obj = self.cache.pop(key)
            bytes_freed += obj.size
            self.current_size -= obj.size
            self.policy.on_evict(obj, current_time)
        
        return bytes_freed
    
    def _make_space(self, bytes_needed: int, current_time: float) -> bool:
        """Make space in cache by evicting objects."""
        # First, remove expired objects
        self._evict_expired(current_time)
        
        if self.current_size + bytes_needed <= self.cache_size:
            return True
        
        # Need to evict more objects
        space_needed = self.current_size + bytes_needed - self.cache_size
        candidates = list(self.cache.values())
        
        if not candidates:
            return bytes_needed <= self.cache_size
        
        evict_keys = self.policy.should_evict(candidates, space_needed, current_time)
        
        bytes_freed = 0
        for key in evict_keys:
            if key in self.cache:
                obj = self.cache.pop(key)
                bytes_freed += obj.size
                self.current_size -= obj.size
                self.policy.on_evict(obj, current_time)
                
                self.stats.total_evictions += 1
                self.stats.bytes_evicted += obj.size
        
        return self.current_size + bytes_needed <= self.cache_size
    
    def process_request(self, request: CacheRequest) -> Tuple[bool, float, float]:
        """
        Process a cache request.
        
        Returns:
            (hit, response_time, cost)
        """
        self.current_time = request.timestamp
        self.stats.total_requests += 1
        self.stats.total_bytes_requested += request.size
        
        # Check if object is in cache
        if request.key in self.cache:
            obj = self.cache[request.key]
            
            # Check if expired
            if obj.is_expired(self.current_time):
                # Remove expired object
                self.cache.pop(request.key)
                self.current_size -= obj.size
                self.policy.on_evict(obj, self.current_time)
            else:
                # Cache hit
                obj.access(self.current_time)
                self.policy.on_access(obj, self.current_time)
                
                self.stats.cache_hits += 1
                self.stats.bytes_served_from_cache += request.size
                self.stats.total_response_time += self.cache_response_time
                self.stats.cache_response_times.append(self.cache_response_time)
                
                return True, self.cache_response_time, 0.0
        
        # Cache miss - need to fetch from origin
        response_time = request.response_time or self.origin_response_time
        cost = request.size * self.origin_cost_per_byte
        
        self.stats.cache_misses += 1
        self.stats.bytes_served_from_origin += request.size
        self.stats.total_response_time += response_time
        self.stats.origin_response_times.append(response_time)
        self.stats.total_cost += cost
        
        # Try to cache the object
        if self._make_space(request.size, self.current_time):
            new_obj = CacheObject(
                key=request.key,
                size=request.size,
                timestamp=self.current_time,
                last_access=self.current_time,
                access_count=1,
                ttl=request.ttl,
                creation_cost=cost
            )
            
            self.cache[request.key] = new_obj
            self.current_size += request.size
            self.policy.on_insert(new_obj, self.current_time)
        
        return False, response_time, cost
    
    def get_cache_utilization(self) -> float:
        """Get current cache utilization as a fraction."""
        return self.current_size / self.cache_size
    
    def get_object_count(self) -> int:
        """Get number of objects in cache."""
        return len(self.cache)
    
    def reset(self):
        """Reset simulator state."""
        self.cache.clear()
        self.current_size = 0
        self.stats = CacheStats(policy_name=self.policy.name)
        self.policy.reset_stats()
        self.current_time = 0.0


def run_simulation(requests: List[CacheRequest], 
                  cache_size: int,
                  eviction_policy: EvictionPolicy,
                  progress_callback=None) -> CacheStats:
    """Run a complete cache simulation and return statistics."""
    
    simulator = CacheSimulator(cache_size, eviction_policy)
    
    for i, request in enumerate(requests):
        simulator.process_request(request)
        
        if progress_callback and i % 10000 == 0:
            progress_callback(i, len(requests))
    
    return simulator.stats


if __name__ == "__main__":
    # Simple test
    requests = [
        CacheRequest(1.0, "obj1", 1000),
        CacheRequest(2.0, "obj2", 2000),
        CacheRequest(3.0, "obj1", 1000),  # Hit
        CacheRequest(4.0, "obj3", 3000),
    ]
    
    policies = [LRUPolicy(), LFUPolicy(), FIFOPolicy(), RandomPolicy()]
    
    for policy in policies:
        stats = run_simulation(requests, cache_size=5000, eviction_policy=policy)
        print(f"{policy.name}: Hit ratio = {stats.hit_ratio:.3f}")
