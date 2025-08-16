"""
Data Pipeline for DRL Cache Training

This module handles parsing NGINX access logs, extracting cache events,
and preparing training data for the DRL cache eviction model.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional, Iterator, Tuple, NamedTuple
from pathlib import Path
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import pickle
from tqdm import tqdm

from .config import DataConfig, FeatureConfig


class CacheEvent(NamedTuple):
    """Single cache event from access logs."""
    timestamp: datetime
    cache_key: str
    cache_status: str  # HIT, MISS, BYPASS, EXPIRED, etc.
    size_bytes: int
    ttl_seconds: int
    upstream_time_us: int
    request_uri: str
    user_agent: Optional[str] = None
    referer: Optional[str] = None


@dataclass
class CacheObject:
    """Represents a cached object with its history."""
    key: str
    size_bytes: int
    created_at: datetime
    last_access: datetime
    hit_count: int
    ttl_seconds: int
    access_history: List[datetime]
    upstream_times: List[int]
    
    @property
    def age_seconds(self) -> float:
        """Age since creation."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()
    
    @property
    def idle_seconds(self) -> float:
        """Time since last access."""
        return (datetime.now(timezone.utc) - self.last_access).total_seconds()
    
    @property
    def avg_upstream_time_us(self) -> float:
        """Average upstream response time."""
        return np.mean(self.upstream_times) if self.upstream_times else 0.0


class LogParser:
    """Parser for NGINX access logs with cache information."""
    
    # NGINX combined log format with cache status
    NGINX_COMBINED_PATTERN = re.compile(
        r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] '
        r'"(?P<method>\S+) (?P<uri>\S+) (?P<protocol>[^"]+)" '
        r'(?P<status>\d+) (?P<size>\d+) '
        r'"(?P<referer>[^"]*)" "(?P<user_agent>[^"]*)" '
        r'"(?P<cache_status>\S+)" "(?P<cache_key>[^"]*)" '
        r'"(?P<upstream_time>\d+)" "(?P<ttl>\d+)"'
    )
    
    # Custom log format patterns can be added here
    CUSTOM_PATTERNS = {
        "nginx_cache": re.compile(
            r'(?P<timestamp>\S+ \S+) \S+ (?P<cache_status>\S+) '
            r'(?P<cache_key>\S+) (?P<size>\d+) (?P<ttl>\d+) '
            r'(?P<upstream_time>\d+) "(?P<uri>[^"]*)"'
        )
    }
    
    def __init__(self, log_format: str = "nginx_combined"):
        self.log_format = log_format
        self.pattern = self._get_pattern(log_format)
        self.logger = logging.getLogger(__name__)
    
    def _get_pattern(self, log_format: str) -> re.Pattern:
        """Get regex pattern for specified log format."""
        if log_format == "nginx_combined":
            return self.NGINX_COMBINED_PATTERN
        elif log_format in self.CUSTOM_PATTERNS:
            return self.CUSTOM_PATTERNS[log_format]
        else:
            raise ValueError(f"Unsupported log format: {log_format}")
    
    def parse_line(self, line: str) -> Optional[CacheEvent]:
        """Parse a single log line into a CacheEvent."""
        match = self.pattern.match(line.strip())
        if not match:
            return None
        
        try:
            data = match.groupdict()
            
            # Parse timestamp
            timestamp_str = data.get('timestamp', '')
            if self.log_format == "nginx_combined":
                # Format: 23/Dec/2023:10:30:45 +0000
                timestamp = datetime.strptime(timestamp_str, '%d/%b/%Y:%H:%M:%S %z')
            else:
                # Custom format: 2023-12-23 10:30:45
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            
            # Extract cache information
            cache_status = data.get('cache_status', 'UNKNOWN')
            cache_key = data.get('cache_key', '')
            size_bytes = int(data.get('size', '0'))
            ttl_seconds = int(data.get('ttl', '0'))
            upstream_time = int(data.get('upstream_time', '0'))  # microseconds
            request_uri = data.get('uri', '')
            
            # Optional fields
            user_agent = data.get('user_agent')
            referer = data.get('referer')
            
            return CacheEvent(
                timestamp=timestamp,
                cache_key=cache_key,
                cache_status=cache_status,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds,
                upstream_time_us=upstream_time,
                request_uri=request_uri,
                user_agent=user_agent,
                referer=referer
            )
        
        except (ValueError, KeyError) as e:
            self.logger.debug(f"Failed to parse line: {e}")
            return None
    
    def parse_file(self, file_path: str, max_lines: Optional[int] = None) -> Iterator[CacheEvent]:
        """Parse an entire log file."""
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                if max_lines and line_num >= max_lines:
                    break
                
                event = self.parse_line(line)
                if event:
                    yield event


class CacheSimulator:
    """Simulates cache behavior for generating training data."""
    
    def __init__(self, config: DataConfig, feature_config: FeatureConfig):
        self.config = config
        self.feature_config = feature_config
        self.cache: Dict[str, CacheObject] = {}
        self.current_size = 0
        self.max_size = int(config.simulation.max_size_gb * 1024**3)  # Convert to bytes
        self.lru_order: List[str] = []  # Keys in LRU order (least recent first)
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'bytes_served_from_cache': 0,
            'bytes_served_from_origin': 0
        }
        self.logger = logging.getLogger(__name__)
    
    def _update_lru(self, key: str) -> None:
        """Update LRU order for a key."""
        if key in self.lru_order:
            self.lru_order.remove(key)
        self.lru_order.append(key)
    
    def _remove_from_lru(self, key: str) -> None:
        """Remove key from LRU order."""
        if key in self.lru_order:
            self.lru_order.remove(key)
    
    def _get_lru_candidates(self, k: int) -> List[str]:
        """Get K least recently used candidates."""
        return self.lru_order[:min(k, len(self.lru_order))]
    
    def _needs_eviction(self, new_object_size: int) -> bool:
        """Check if eviction is needed to fit new object."""
        return self.current_size + new_object_size > self.max_size
    
    def _evict_lru(self, bytes_needed: int) -> List[str]:
        """Evict objects using LRU policy until enough space is freed."""
        evicted_keys = []
        bytes_freed = 0
        
        while bytes_freed < bytes_needed and self.lru_order:
            key = self.lru_order[0]  # Least recently used
            if key in self.cache:
                obj = self.cache[key]
                bytes_freed += obj.size_bytes
                self.current_size -= obj.size_bytes
                del self.cache[key]
                evicted_keys.append(key)
                self._remove_from_lru(key)
                self.stats['evictions'] += 1
        
        return evicted_keys
    
    def _extract_features(self, keys: List[str], current_time: datetime) -> np.ndarray:
        """Extract features for candidate objects."""
        features = []
        
        for key in keys:
            if key not in self.cache:
                continue
            
            obj = self.cache[key]
            
            # Calculate features
            age_sec = (current_time - obj.created_at).total_seconds()
            size_kb = obj.size_bytes / 1024.0
            hit_count = obj.hit_count
            inter_arrival_dt = (current_time - obj.last_access).total_seconds()
            ttl_left_sec = max(0, obj.ttl_seconds - age_sec)
            last_origin_rtt_us = obj.avg_upstream_time_us
            
            # Apply transformations
            if self.feature_config.log_scale_size:
                size_kb = np.log1p(size_kb)
            
            if self.feature_config.sqrt_transform_hits:
                hit_count = np.sqrt(hit_count)
            
            feature_vector = [
                age_sec,
                size_kb,
                hit_count,
                inter_arrival_dt,
                ttl_left_sec,
                last_origin_rtt_us
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def process_event(self, event: CacheEvent) -> Optional[Dict]:
        """Process a cache event and return training data if eviction occurred."""
        self.stats['total_requests'] += 1
        
        if event.cache_status == 'HIT':
            # Cache hit - update object access info
            if event.cache_key in self.cache:
                obj = self.cache[event.cache_key]
                obj.hit_count += 1
                obj.last_access = event.timestamp
                obj.access_history.append(event.timestamp)
                if event.upstream_time_us > 0:
                    obj.upstream_times.append(event.upstream_time_us)
                
                self._update_lru(event.cache_key)
                self.stats['cache_hits'] += 1
                self.stats['bytes_served_from_cache'] += obj.size_bytes
            
            return None
        
        elif event.cache_status in ['MISS', 'EXPIRED']:
            # Cache miss - potentially add new object
            self.stats['cache_misses'] += 1
            self.stats['bytes_served_from_origin'] += event.size_bytes
            
            # Skip if object is too small/large or TTL too short
            if (event.size_bytes < self.config.min_object_size or
                event.size_bytes > self.config.max_object_size or
                event.ttl_seconds < self.config.min_cache_duration):
                return None
            
            # Check if eviction is needed
            if not self._needs_eviction(event.size_bytes):
                # No eviction needed - just add object
                self._add_object(event)
                return None
            
            # Eviction needed - generate training data
            k = self.config.simulation.k_candidates
            candidate_keys = self._get_lru_candidates(k)
            
            if len(candidate_keys) < 2:  # Need at least 2 candidates
                return None
            
            # Extract features
            features = self._extract_features(candidate_keys, event.timestamp)
            
            if len(features) == 0:
                return None
            
            # Simulate eviction (using LRU for ground truth)
            bytes_needed = event.size_bytes
            evicted_keys = self._evict_lru(bytes_needed)
            
            # Create action labels (1 = evicted, 0 = kept)
            actions = np.zeros(len(candidate_keys))
            for i, key in enumerate(candidate_keys):
                if key in evicted_keys:
                    actions[i] = 1.0
            
            # Add new object
            self._add_object(event)
            
            # Calculate rewards (will be computed during replay)
            training_data = {
                'timestamp': event.timestamp,
                'features': features,
                'candidate_keys': candidate_keys,
                'actions': actions,
                'evicted_keys': evicted_keys,
                'new_object_size': event.size_bytes,
                'cache_size_before': self.current_size - event.size_bytes,
                'cache_size_after': self.current_size
            }
            
            return training_data
        
        return None
    
    def _add_object(self, event: CacheEvent) -> None:
        """Add new object to cache."""
        obj = CacheObject(
            key=event.cache_key,
            size_bytes=event.size_bytes,
            created_at=event.timestamp,
            last_access=event.timestamp,
            hit_count=0,
            ttl_seconds=event.ttl_seconds,
            access_history=[event.timestamp],
            upstream_times=[event.upstream_time_us] if event.upstream_time_us > 0 else []
        )
        
        self.cache[event.cache_key] = obj
        self.current_size += event.size_bytes
        self._update_lru(event.cache_key)
    
    def get_statistics(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.stats['total_requests']
        if total_requests == 0:
            return self.stats
        
        hit_ratio = self.stats['cache_hits'] / total_requests
        total_bytes = self.stats['bytes_served_from_cache'] + self.stats['bytes_served_from_origin']
        byte_hit_ratio = self.stats['bytes_served_from_cache'] / total_bytes if total_bytes > 0 else 0
        
        return {
            **self.stats,
            'hit_ratio': hit_ratio,
            'byte_hit_ratio': byte_hit_ratio,
            'current_cache_size': self.current_size,
            'cache_utilization': self.current_size / self.max_size,
            'num_cached_objects': len(self.cache)
        }


class DataPipeline:
    """Main data processing pipeline."""
    
    def __init__(self, config: DataConfig, feature_config: FeatureConfig):
        self.config = config
        self.feature_config = feature_config
        self.parser = LogParser(config.log_format)
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory
        Path(config.data_cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, log_path: str) -> str:
        """Get cache file path for processed data."""
        # Create hash of log file path and modification time
        log_file = Path(log_path)
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        
        file_hash = hashlib.md5(
            f"{log_path}_{log_file.stat().st_mtime}".encode()
        ).hexdigest()
        
        return str(Path(self.config.data_cache_dir) / f"processed_{file_hash}.pkl")
    
    def process_log_chunk(self, events: List[CacheEvent]) -> List[Dict]:
        """Process a chunk of log events."""
        simulator = CacheSimulator(self.config, self.feature_config)
        training_data = []
        
        # Warmup phase
        warmup_size = int(len(events) * self.config.simulation.warmup_ratio)
        
        for i, event in enumerate(tqdm(events, desc="Processing events")):
            result = simulator.process_event(event)
            
            # Only collect training data after warmup
            if i >= warmup_size and result is not None:
                training_data.append(result)
        
        stats = simulator.get_statistics()
        self.logger.info(f"Chunk statistics: {stats}")
        
        return training_data
    
    def process_log_file(self, log_path: str, use_cache: bool = True) -> List[Dict]:
        """Process entire log file and return training data."""
        cache_path = self._get_cache_path(log_path)
        
        # Check cache
        if use_cache and Path(cache_path).exists():
            self.logger.info(f"Loading cached data from: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        self.logger.info(f"Processing log file: {log_path}")
        
        # Parse events
        events = list(self.parser.parse_file(log_path))
        self.logger.info(f"Parsed {len(events)} events")
        
        if len(events) == 0:
            return []
        
        # Process in chunks if file is large
        chunk_size = self.config.chunk_size
        all_training_data = []
        
        if len(events) <= chunk_size:
            training_data = self.process_log_chunk(events)
            all_training_data.extend(training_data)
        else:
            # Process in parallel chunks
            chunks = [events[i:i + chunk_size] 
                     for i in range(0, len(events), chunk_size)]
            
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(self.process_log_chunk, chunk) 
                          for chunk in chunks]
                
                for future in as_completed(futures):
                    try:
                        chunk_data = future.result()
                        all_training_data.extend(chunk_data)
                    except Exception as e:
                        self.logger.error(f"Chunk processing failed: {e}")
        
        self.logger.info(f"Generated {len(all_training_data)} training samples")
        
        # Cache results
        if use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(all_training_data, f)
            self.logger.info(f"Cached processed data to: {cache_path}")
        
        return all_training_data
    
    def create_datasets(self, training_data: List[Dict]) -> Tuple[List, List, List]:
        """Split training data into train/val/test sets."""
        if len(training_data) == 0:
            return [], [], []
        
        # Sort by timestamp for temporal splitting
        training_data.sort(key=lambda x: x['timestamp'])
        
        # Calculate split indices
        total_size = len(training_data)
        train_size = int(total_size * self.config.train_ratio)
        val_size = int(total_size * self.config.val_ratio)
        
        # Split data
        train_data = training_data[:train_size]
        val_data = training_data[train_size:train_size + val_size]
        test_data = training_data[train_size + val_size:]
        
        self.logger.info(f"Dataset splits - Train: {len(train_data)}, "
                        f"Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data


def test_data_pipeline():
    """Test the data pipeline with sample data."""
    from .config import DataConfig, FeatureConfig
    
    # Create sample log data
    sample_logs = [
        '192.168.1.1 - - [23/Dec/2023:10:30:45 +0000] "GET /api/data.json HTTP/1.1" 200 1024 "-" "Mozilla/5.0" "MISS" "/api/data.json" "150000" "300"',
        '192.168.1.2 - - [23/Dec/2023:10:30:46 +0000] "GET /api/data.json HTTP/1.1" 200 1024 "-" "Mozilla/5.0" "HIT" "/api/data.json" "0" "300"',
        '192.168.1.3 - - [23/Dec/2023:10:30:47 +0000] "GET /large-file.zip HTTP/1.1" 200 5242880 "-" "Mozilla/5.0" "MISS" "/large-file.zip" "500000" "3600"',
    ]
    
    # Test parser
    parser = LogParser()
    events = [parser.parse_line(log) for log in sample_logs]
    events = [e for e in events if e is not None]
    
    print(f"Parsed {len(events)} events:")
    for event in events:
        print(f"  {event.cache_status}: {event.cache_key} ({event.size_bytes} bytes)")
    
    # Test simulator
    config = DataConfig()
    feature_config = FeatureConfig()
    pipeline = DataPipeline(config, feature_config)
    
    training_data = pipeline.process_log_chunk(events)
    print(f"\nGenerated {len(training_data)} training samples")


if __name__ == "__main__":
    test_data_pipeline()
