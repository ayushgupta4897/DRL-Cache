"""
Generate realistic CDN workload with challenging cache characteristics.

This creates CloudFlare-style traffic patterns that differentiate between
eviction policies and provide learning opportunities for DRL models.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from cache_simulator import CacheRequest
import random
from pathlib import Path

class RealisticCDNGenerator:
    """Generate realistic CDN traffic patterns."""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_realistic_cloudflare_traffic(self, 
                                            num_requests=200000,
                                            num_unique_objects=50000,
                                            cache_working_set_ratio=0.8) -> list:
        """
        Generate realistic CDN traffic with challenging characteristics:
        - Flash crowd events (viral content)
        - Diurnal patterns 
        - Long-tail popularity distribution
        - Geographic clustering
        - Content type diversity
        - Cache-unfriendly patterns
        """
        
        print(f"Generating {num_requests} requests for {num_unique_objects} objects...")
        
        # Create diverse content catalog
        objects = self._create_content_catalog(num_unique_objects)
        
        # Generate time-based request pattern
        base_time = datetime.now().timestamp()
        requests = []
        
        # Simulate 24 hours of traffic with realistic patterns
        time_span = 24 * 3600  # 24 hours in seconds
        
        # Create flash crowd events (viral content)
        flash_events = self._generate_flash_crowd_events(objects, time_span)
        
        # Generate requests with complex patterns
        for i in range(num_requests):
            current_time = base_time + (i / num_requests) * time_span
            hour_of_day = ((current_time - base_time) % (24 * 3600)) / 3600
            
            # Choose content based on time and events
            obj = self._select_object_realistic(objects, current_time, base_time, flash_events, hour_of_day)
            
            # Add geographic and temporal clustering
            obj = self._apply_geographic_clustering(obj, current_time)
            
            request = CacheRequest(
                timestamp=current_time,
                key=obj['key'],
                size=obj['size'],
                ttl=obj['ttl'],
                client_ip=obj['client_ip'],
                response_time=obj['response_time']
            )
            
            requests.append(request)
            
            if i % 20000 == 0:
                print(f"  Generated {i:,} requests...")
        
        print(f"Generated {len(requests)} realistic CDN requests")
        return requests
    
    def _create_content_catalog(self, num_objects):
        """Create diverse content catalog with realistic characteristics."""
        objects = []
        
        # Content types with different characteristics
        content_types = {
            'image_thumbnail': {
                'count_ratio': 0.4,
                'size_range': (5000, 50000),    # 5-50KB
                'ttl_range': (86400, 604800),   # 1-7 days
                'popularity_exp': 1.2,          # Moderate skew
                'access_pattern': 'bursty'
            },
            'image_full': {
                'count_ratio': 0.2, 
                'size_range': (100000, 2000000), # 100KB-2MB
                'ttl_range': (86400, 2592000),   # 1-30 days
                'popularity_exp': 1.5,           # Higher skew
                'access_pattern': 'long_tail'
            },
            'video_segment': {
                'count_ratio': 0.15,
                'size_range': (500000, 10000000), # 500KB-10MB  
                'ttl_range': (3600, 86400),       # 1-24 hours
                'popularity_exp': 2.0,            # Very skewed
                'access_pattern': 'sequential'
            },
            'api_response': {
                'count_ratio': 0.15,
                'size_range': (500, 10000),       # 500B-10KB
                'ttl_range': (60, 3600),          # 1min-1hour  
                'popularity_exp': 0.8,            # Less skewed
                'access_pattern': 'uniform'
            },
            'static_asset': {
                'count_ratio': 0.1,
                'size_range': (10000, 500000),   # 10KB-500KB
                'ttl_range': (604800, 2592000),  # 7-30 days
                'popularity_exp': 0.5,           # Uniform-ish
                'access_pattern': 'steady'
            }
        }
        
        obj_id = 0
        for content_type, config in content_types.items():
            type_count = int(num_objects * config['count_ratio'])
            
            for i in range(type_count):
                size = np.random.randint(config['size_range'][0], config['size_range'][1])
                ttl = np.random.randint(config['ttl_range'][0], config['ttl_range'][1])
                
                # Assign popularity rank for this content type
                popularity_rank = i + 1
                
                obj = {
                    'key': f'/{content_type}/obj_{obj_id:06d}',
                    'size': size,
                    'ttl': ttl,
                    'content_type': content_type,
                    'popularity_rank': popularity_rank,
                    'popularity_exp': config['popularity_exp'],
                    'access_pattern': config['access_pattern'],
                    'base_popularity': 1.0 / (popularity_rank ** config['popularity_exp'])
                }
                
                objects.append(obj)
                obj_id += 1
        
        # Sort by base popularity for easier selection
        objects.sort(key=lambda x: x['base_popularity'], reverse=True)
        
        print(f"Created {len(objects)} diverse content objects")
        return objects
    
    def _generate_flash_crowd_events(self, objects, time_span):
        """Generate flash crowd events for viral content."""
        events = []
        
        # 3-5 flash events over 24 hours
        num_events = np.random.randint(3, 6)
        
        for i in range(num_events):
            # Pick random object (prefer images/videos for viral content)
            viral_objects = [obj for obj in objects 
                           if obj['content_type'] in ['image_full', 'video_segment', 'image_thumbnail']]
            viral_obj = np.random.choice(viral_objects)
            
            # Event timing
            start_time = np.random.uniform(0, time_span * 0.8)
            duration = np.random.exponential(3600)  # Average 1 hour duration
            peak_intensity = np.random.uniform(10, 100)  # 10-100x normal traffic
            
            events.append({
                'object': viral_obj,
                'start_time': start_time,
                'duration': duration,
                'peak_intensity': peak_intensity
            })
        
        return events
    
    def _select_object_realistic(self, objects, current_time, base_time, flash_events, hour_of_day):
        """Select object based on realistic patterns."""
        elapsed_time = current_time - base_time
        
        # Check for flash crowd events
        for event in flash_events:
            if (elapsed_time >= event['start_time'] and 
                elapsed_time < event['start_time'] + event['duration']):
                
                # Higher probability of selecting viral content
                event_progress = (elapsed_time - event['start_time']) / event['duration']
                intensity = event['peak_intensity'] * np.exp(-2 * event_progress)  # Exponential decay
                
                if np.random.random() < intensity / (intensity + 10):
                    return event['object']
        
        # Diurnal patterns - different content types popular at different times
        time_weights = {
            'image_thumbnail': 1.0 + 0.5 * np.sin(2 * np.pi * (hour_of_day - 12) / 24),  # Peaks afternoon
            'image_full': 1.0 + 0.3 * np.sin(2 * np.pi * (hour_of_day - 20) / 24),      # Peaks evening  
            'video_segment': 1.0 + 0.8 * np.sin(2 * np.pi * (hour_of_day - 21) / 24),   # Peaks prime time
            'api_response': 1.0 + 0.4 * np.sin(2 * np.pi * (hour_of_day - 14) / 24),    # Peaks work hours
            'static_asset': 1.0  # Constant
        }
        
        # Weight objects by time-of-day preferences
        weighted_objects = []
        weights = []
        
        for obj in objects[:10000]:  # Consider top 10K objects for performance
            time_weight = time_weights.get(obj['content_type'], 1.0)
            final_weight = obj['base_popularity'] * time_weight
            weighted_objects.append(obj)
            weights.append(final_weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Select object
        selected_idx = np.random.choice(len(weighted_objects), p=weights)
        return weighted_objects[selected_idx]
    
    def _apply_geographic_clustering(self, obj, current_time):
        """Apply geographic and temporal clustering effects."""
        # Simulate different geographic regions with clustering
        regions = [
            {'name': 'US-East', 'ip_prefix': '10.1', 'timezone_offset': 0},
            {'name': 'US-West', 'ip_prefix': '10.2', 'timezone_offset': -3},
            {'name': 'Europe', 'ip_prefix': '10.3', 'timezone_offset': 6},
            {'name': 'Asia', 'ip_prefix': '10.4', 'timezone_offset': 12},
        ]
        
        # Select region with clustering (70% chance same as last request)
        if hasattr(self, '_last_region') and np.random.random() < 0.7:
            region = self._last_region
        else:
            region = np.random.choice(regions)
            self._last_region = region
        
        # Generate IP in region
        client_ip = f"{region['ip_prefix']}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        
        # Response time varies by content type and region
        base_response_time = {
            'image_thumbnail': 0.02,   # 20ms
            'image_full': 0.05,        # 50ms  
            'video_segment': 0.1,      # 100ms
            'api_response': 0.01,      # 10ms
            'static_asset': 0.03       # 30ms
        }.get(obj['content_type'], 0.05)
        
        # Add regional latency and jitter
        region_latency = abs(region['timezone_offset']) * 0.01  # 10ms per timezone
        jitter = np.random.exponential(0.02)  # Exponential jitter
        
        response_time = base_response_time + region_latency + jitter
        
        return {
            **obj,
            'client_ip': client_ip,
            'response_time': response_time,
            'region': region['name']
        }

def save_realistic_cloudflare_dataset():
    """Generate and save realistic CloudFlare dataset."""
    generator = RealisticCDNGenerator(seed=42)
    
    # Generate challenging dataset
    requests = generator.generate_realistic_cloudflare_traffic(
        num_requests=150000,       # 150K requests for thorough testing
        num_unique_objects=30000,  # 30K unique objects  
        cache_working_set_ratio=0.7  # Working set is 70% of cache
    )
    
    # Save to dataset directory
    dataset_dir = Path("datasets/cloudflare_realistic")
    dataset_dir.mkdir(exist_ok=True)
    
    # Convert to DataFrame and save
    df = pd.DataFrame([
        {
            'timestamp': r.timestamp,
            'key': r.key, 
            'size': r.size,
            'ttl': r.ttl,
            'client_ip': r.client_ip,
            'response_time': r.response_time
        }
        for r in requests
    ])
    
    output_file = dataset_dir / "processed_requests.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Saved realistic CloudFlare dataset: {output_file}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total requests: {len(requests):,}")
    print(f"Unique objects: {df['key'].nunique():,}")
    print(f"Size range: {df['size'].min():,} - {df['size'].max():,} bytes")
    print(f"Time span: {(df['timestamp'].max() - df['timestamp'].min()) / 3600:.1f} hours")
    
    return requests

if __name__ == "__main__":
    save_realistic_cloudflare_dataset()
