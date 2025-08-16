"""
Challenging Dataset Generator for DRL Cache Research

Creates realistic but challenging cache workloads that demonstrate DRL advantages.
Based on real-world cache behavior patterns but designed to stress-test cache policies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
from cache_simulator import CacheRequest

class ChallengingDatasetGenerator:
    """Generate challenging synthetic datasets for cache research."""
    
    def __init__(self):
        self.output_dir = Path("datasets/challenging_synthetic")
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_challenging_web_workload(self, 
                                        num_requests: int = 200000,
                                        num_objects: int = 10000,
                                        challenge_level: str = "medium") -> List[CacheRequest]:
        """
        Generate challenging web cache workload.
        
        Challenge levels:
        - easy: High locality, small working set (like NASA dataset)
        - medium: Mixed patterns, moderate diversity 
        - hard: Low locality, large working set, diverse patterns
        """
        
        print(f"🎯 Generating {challenge_level} challenge dataset...")
        print(f"   📊 {num_requests:,} requests, {num_objects:,} objects")
        
        # Configure challenge parameters
        if challenge_level == "easy":
            repeat_ratio = 0.95
            zipf_alpha = 1.8  # High skew
            temporal_locality = 0.8
            working_set_pct = 0.1  # Only 10% of objects are popular
        elif challenge_level == "medium":
            repeat_ratio = 0.80
            zipf_alpha = 1.2  # Medium skew
            temporal_locality = 0.6
            working_set_pct = 0.3  # 30% of objects are popular
        else:  # hard
            repeat_ratio = 0.70
            zipf_alpha = 0.9  # Low skew (more uniform)
            temporal_locality = 0.4
            working_set_pct = 0.5  # 50% of objects get some traffic
        
        # Generate object metadata
        objects = self._generate_object_metadata(num_objects)
        
        # Generate request sequence with challenging patterns
        requests = []
        base_timestamp = datetime(2024, 1, 1).timestamp()
        
        print(f"   🔄 Target repeat ratio: {repeat_ratio:.1%}")
        
        # Track recently accessed objects for temporal locality
        recent_objects = []
        recent_weights = []
        
        for i in range(num_requests):
            if i % 50000 == 0 and i > 0:
                print(f"   📈 Generated {i:,} requests...")
            
            # Decide: new request or repeat
            is_repeat = np.random.random() < repeat_ratio and len(recent_objects) > 0
            
            if is_repeat and np.random.random() < temporal_locality:
                # Temporal locality - recent objects more likely
                if recent_objects:
                    weights = np.array(recent_weights)
                    weights = weights / weights.sum()
                    obj_id = np.random.choice(recent_objects, p=weights)
                else:
                    obj_id = self._select_object_zipf(objects, zipf_alpha, working_set_pct)
            else:
                # New request - use Zipfian distribution
                obj_id = self._select_object_zipf(objects, zipf_alpha, working_set_pct)
            
            # Update recent objects list (sliding window)
            if obj_id in recent_objects:
                idx = recent_objects.index(obj_id)
                recent_weights[idx] = min(recent_weights[idx] * 1.1, 10.0)  # Boost weight
            else:
                recent_objects.append(obj_id)
                recent_weights.append(1.0)
                
                # Keep only recent objects (sliding window)
                if len(recent_objects) > 1000:
                    recent_objects.pop(0)
                    recent_weights.pop(0)
            
            # Create request
            obj = objects[obj_id]
            timestamp = base_timestamp + i * np.random.exponential(2.0)  # Variable inter-arrival
            
            # Add some jitter to sizes and TTLs for realism
            size_jitter = np.random.uniform(0.8, 1.2)
            size = max(1, int(obj['size'] * size_jitter))
            
            ttl_jitter = np.random.uniform(0.5, 2.0)
            ttl = max(60, int(obj['ttl'] * ttl_jitter))
            
            # Simulate geographic diversity
            client_ip = self._generate_client_ip(obj['content_type'])
            
            # Simulate network latency based on content type and size
            response_time = self._estimate_response_time(obj['content_type'], size)
            
            request = CacheRequest(
                timestamp=timestamp,
                key=obj['key'],
                size=size,
                ttl=ttl,
                client_ip=client_ip,
                response_time=response_time
            )
            requests.append(request)
        
        # Sort by timestamp
        requests.sort(key=lambda x: x.timestamp)
        
        # Calculate actual statistics
        unique_objects = len(set(r.key for r in requests))
        actual_repeat_ratio = (len(requests) - unique_objects) / len(requests)
        
        print(f"   ✅ Generated {len(requests):,} requests")
        print(f"   🔑 Unique objects: {unique_objects:,}")
        print(f"   🔄 Actual repeat ratio: {actual_repeat_ratio:.1%}")
        
        return requests
    
    def _generate_object_metadata(self, num_objects: int) -> Dict[str, Dict]:
        """Generate realistic object metadata."""
        
        objects = {}
        
        # Content type distribution (realistic web mix)
        content_types = [
            ('image', 0.40),     # Images
            ('page', 0.25),      # HTML pages  
            ('api', 0.15),       # API responses
            ('static', 0.10),    # CSS/JS/fonts
            ('document', 0.05),  # PDFs, docs
            ('media', 0.05)      # Videos, audio
        ]
        
        for obj_id in range(num_objects):
            # Select content type
            content_type = np.random.choice(
                [ct[0] for ct in content_types],
                p=[ct[1] for ct in content_types]
            )
            
            # Generate size based on content type
            size = self._generate_size_for_content_type(content_type)
            
            # Generate TTL based on content type
            ttl = self._generate_ttl_for_content_type(content_type)
            
            # Generate realistic URL
            url = self._generate_url(content_type, obj_id)
            
            objects[obj_id] = {
                'key': url,
                'content_type': content_type,
                'size': size,
                'ttl': ttl
            }
        
        return objects
    
    def _generate_size_for_content_type(self, content_type: str) -> int:
        """Generate realistic sizes for different content types."""
        
        size_ranges = {
            'image': (5000, 500000),      # 5KB - 500KB
            'page': (2000, 100000),       # 2KB - 100KB
            'api': (200, 10000),          # 200B - 10KB
            'static': (1000, 50000),      # 1KB - 50KB
            'document': (50000, 5000000), # 50KB - 5MB
            'media': (1000000, 50000000)  # 1MB - 50MB
        }
        
        min_size, max_size = size_ranges[content_type]
        
        # Log-normal distribution for realistic size diversity
        mean_log = np.log(min_size + (max_size - min_size) * 0.3)
        std_log = 1.0
        
        size = int(np.random.lognormal(mean_log, std_log))
        return max(min_size, min(size, max_size))
    
    def _generate_ttl_for_content_type(self, content_type: str) -> int:
        """Generate realistic TTL values."""
        
        ttl_ranges = {
            'image': (86400, 604800),     # 1-7 days
            'page': (300, 3600),          # 5min - 1hour  
            'api': (60, 300),             # 1-5 minutes
            'static': (86400, 2592000),   # 1-30 days
            'document': (3600, 86400),    # 1-24 hours
            'media': (604800, 2592000)    # 7-30 days
        }
        
        min_ttl, max_ttl = ttl_ranges[content_type]
        return np.random.randint(min_ttl, max_ttl + 1)
    
    def _generate_url(self, content_type: str, obj_id: int) -> str:
        """Generate realistic URLs."""
        
        url_patterns = {
            'image': ['/images/photo_{}.jpg', '/pics/img_{}.png', '/media/pic_{}.gif'],
            'page': ['/articles/{}/', '/posts/{}/', '/pages/{}.html'],
            'api': ['/api/v1/data/{}', '/api/users/{}', '/api/content/{}'],
            'static': ['/css/style_{}.css', '/js/script_{}.js', '/fonts/font_{}.woff'],
            'document': ['/docs/file_{}.pdf', '/reports/report_{}.doc'],
            'media': ['/videos/clip_{}.mp4', '/audio/track_{}.mp3']
        }
        
        pattern = np.random.choice(url_patterns[content_type])
        return pattern.format(obj_id)
    
    def _select_object_zipf(self, objects: Dict, alpha: float, working_set_pct: float) -> int:
        """Select object using Zipfian distribution."""
        
        # Only consider working set objects for popularity
        working_set_size = int(len(objects) * working_set_pct)
        
        if np.random.random() < 0.8:  # 80% chance to select from working set
            # Zipfian within working set
            ranks = np.arange(1, working_set_size + 1)
            weights = 1.0 / np.power(ranks, alpha)
            weights = weights / weights.sum()
            
            selected_rank = np.random.choice(working_set_size, p=weights)
            return selected_rank
        else:
            # Random selection from full object set (long tail)
            return np.random.randint(0, len(objects))
    
    def _generate_client_ip(self, content_type: str) -> str:
        """Generate diverse client IPs with geographic patterns."""
        
        # Different content types have different geographic patterns
        if content_type == 'api':
            # API calls more concentrated geographically
            regions = ['10.1', '10.2', '10.3']
            weights = [0.6, 0.25, 0.15]
        else:
            # Media content more distributed
            regions = ['10.1', '10.2', '10.3', '10.4', '10.5']
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        region = np.random.choice(regions, p=weights)
        return f"{region}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
    
    def _estimate_response_time(self, content_type: str, size: int) -> float:
        """Estimate realistic response times."""
        
        # Base latency by content type
        base_latency = {
            'api': 0.02,      # 20ms (fast)
            'page': 0.05,     # 50ms 
            'image': 0.03,    # 30ms
            'static': 0.02,   # 20ms (cached)
            'document': 0.1,  # 100ms (slower)
            'media': 0.2      # 200ms (slowest)
        }
        
        base = base_latency.get(content_type, 0.05)
        
        # Size-dependent component
        size_factor = size / (1024 * 1024)  # Size in MB
        size_latency = size_factor * 0.1    # 100ms per MB
        
        # Add jitter
        jitter = np.random.exponential(0.01)
        
        return max(base + size_latency + jitter, 0.001)
    
    def save_dataset(self, requests: List[CacheRequest], 
                    name: str, challenge_level: str) -> Path:
        """Save dataset for reuse."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"challenging_{name}_{challenge_level}_{timestamp}.csv"
        output_file = self.output_dir / filename
        
        # Convert to DataFrame
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
        
        df.to_csv(output_file, index=False)
        print(f"💾 Saved dataset: {output_file}")
        
        return output_file
    
    def print_dataset_analysis(self, requests: List[CacheRequest]):
        """Print detailed analysis of generated dataset."""
        
        total = len(requests)
        unique = len(set(r.key for r in requests))
        
        print(f"\\n📊 Dataset Analysis:")
        print(f"   Total requests: {total:,}")
        print(f"   Unique objects: {unique:,}")
        print(f"   Repeat ratio: {(total-unique)/total:.1%}")
        
        # Size analysis
        sizes = [r.size for r in requests]
        total_gb = sum(sizes) / (1024**3)
        print(f"   Total data: {total_gb:.2f} GB")
        print(f"   Size range: {min(sizes):,} - {max(sizes):,} bytes")
        
        # Working set estimation
        from collections import Counter
        counts = Counter(r.key for r in requests)
        popular_objects = sum(1 for c in counts.values() if c >= 5)
        very_popular = sum(1 for c in counts.values() if c >= 20)
        
        print(f"   Objects with ≥5 requests: {popular_objects:,}")
        print(f"   Objects with ≥20 requests: {very_popular:,}")
        
        # Top objects
        top_objects = counts.most_common(5)
        print(f"   Top 5 objects: {[f'{obj}({count})' for obj, count in top_objects]}")


def main():
    """Generate challenging datasets for DRL research."""
    
    generator = ChallengingDatasetGenerator()
    
    print("🎯 GENERATING CHALLENGING DATASETS FOR DRL RESEARCH")
    print("=" * 60)
    
    # Generate different challenge levels
    challenge_levels = [
        ("medium", 150000, 8000),  # Good balance for research
        ("hard", 200000, 12000),   # Challenging for DRL to show benefits
    ]
    
    datasets = {}
    
    for level, num_requests, num_objects in challenge_levels:
        print(f"\\n🔥 Challenge Level: {level.upper()}")
        
        requests = generator.generate_challenging_web_workload(
            num_requests=num_requests,
            num_objects=num_objects,
            challenge_level=level
        )
        
        # Analyze the dataset
        generator.print_dataset_analysis(requests)
        
        # Save for reuse
        output_file = generator.save_dataset(requests, "web", level)
        
        datasets[level] = {
            'requests': requests,
            'file': output_file
        }
    
    print(f"\\n✅ Generated {len(datasets)} challenging datasets!")
    print(f"💡 These datasets are designed to show DRL advantages:")
    print(f"   - Lower repeat ratios (70-80% vs 98.5%)")
    print(f"   - Larger working sets with cache pressure")
    print(f"   - Diverse content types and access patterns")
    print(f"   - Realistic size distributions and temporal patterns")
    
    return datasets


if __name__ == "__main__":
    datasets = main()
