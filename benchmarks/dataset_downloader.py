"""
Dataset Downloader for Cache Benchmarking

Downloads and prepares real-world cache traces from various sources
for benchmarking cache eviction algorithms.
"""

import os
import requests
import gzip
import zipfile
import tarfile
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple
import hashlib
import json
from tqdm import tqdm
from datetime import datetime, timezone
import logging

from cache_simulator import CacheRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Downloads and preprocesses cache datasets."""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Registry of available datasets
        self.datasets = {
            # Academic datasets
            "wikipedia_trace": {
                "url": "https://dumps.wikimedia.org/other/pagecounts-raw/2023/2023-01/",
                "description": "Wikipedia page access traces",
                "format": "custom",
                "size_gb": 15.0,
                "processor": self._process_wikipedia_trace
            },
            
            # CDN traces (CloudFlare research data)
            "cloudflare_sample": {
                "url": "https://github.com/cloudflare/cf-ui/raw/main/packages/cloudflare-analytics/test/fixtures/",
                "description": "Sample CloudFlare analytics data",
                "format": "json",
                "size_gb": 0.001,
                "processor": self._process_cloudflare_sample
            },
            
            # Web server logs (sample datasets)
            "nasa_web_logs": {
                "url": "http://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz",
                "description": "NASA web server logs from 1995",
                "format": "common_log",
                "size_gb": 0.2,
                "processor": self._process_nasa_logs
            },
            
            # CDN research datasets
            "akamai_trace": {
                "url": "https://www.akamai.com/us/en/multimedia/documents/technical-publication/",
                "description": "Akamai CDN traces (requires manual download)",
                "format": "custom",
                "size_gb": 5.0,
                "processor": self._process_akamai_trace
            },
            
            # Synthetic workloads for controlled experiments
            "synthetic_zipf": {
                "description": "Synthetic Zipfian workload",
                "format": "synthetic",
                "size_gb": 0.1,
                "processor": self._generate_synthetic_zipf
            },
            
            "synthetic_temporal": {
                "description": "Synthetic workload with temporal patterns",
                "format": "synthetic", 
                "size_gb": 0.1,
                "processor": self._generate_synthetic_temporal
            }
        }
    
    def list_datasets(self) -> Dict[str, Dict]:
        """List all available datasets."""
        return self.datasets
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """Download and preprocess a dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_info = self.datasets[dataset_name]
        dataset_dir = self.data_dir / dataset_name
        
        # Check if already processed
        if not force and (dataset_dir / "processed_requests.csv").exists():
            logger.info(f"Dataset {dataset_name} already downloaded and processed")
            return True
        
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            logger.info(f"Processing dataset: {dataset_name}")
            success = dataset_info["processor"](dataset_info, dataset_dir)
            
            if success:
                logger.info(f"Successfully processed dataset: {dataset_name}")
                return True
            else:
                logger.error(f"Failed to process dataset: {dataset_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {e}")
            return False
    
    def load_dataset(self, dataset_name: str) -> List[CacheRequest]:
        """Load preprocessed dataset as CacheRequest objects."""
        dataset_dir = self.data_dir / dataset_name
        processed_file = dataset_dir / "processed_requests.csv"
        
        if not processed_file.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found. Run download_dataset() first.")
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        df = pd.read_csv(processed_file)
        requests = []
        
        for _, row in df.iterrows():
            request = CacheRequest(
                timestamp=row['timestamp'],
                key=row['key'],
                size=int(row['size']),
                ttl=int(row.get('ttl', 3600)),
                client_ip=row.get('client_ip'),
                response_time=row.get('response_time')
            )
            requests.append(request)
        
        logger.info(f"Loaded {len(requests)} requests from {dataset_name}")
        return requests
    
    def _download_file(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """Download a file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def _process_nasa_logs(self, dataset_info: Dict, dataset_dir: Path) -> bool:
        """Process NASA web server logs."""
        url = dataset_info["url"]
        compressed_file = dataset_dir / "nasa_logs.gz"
        
        # Download compressed file
        if not compressed_file.exists():
            logger.info(f"Downloading NASA logs from {url}")
            if not self._download_file(url, compressed_file):
                return False
        
        # Parse log file
        requests = []
        
        try:
            with gzip.open(compressed_file, 'rt') as f:
                for line_num, line in enumerate(tqdm(f, desc="Parsing logs")):
                    if line_num > 500000:  # Limit for testing
                        break
                    
                    request = self._parse_common_log_line(line, line_num)
                    if request:
                        requests.append(request)
            
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
            
            logger.info(f"Processed {len(requests)} requests from NASA logs")
            return True
            
        except Exception as e:
            logger.error(f"Error processing NASA logs: {e}")
            return False
    
    def _parse_common_log_line(self, line: str, line_num: int) -> Optional[CacheRequest]:
        """Parse a line in Common Log Format."""
        try:
            # NASA log format: host logname user [timestamp] "request" status size
            parts = line.strip().split(' ')
            if len(parts) < 7:
                return None
            
            host = parts[0]
            timestamp_str = ' '.join(parts[3:5]).strip('[]')
            request_str = ' '.join(parts[5:8]).strip('"')
            status = parts[8] if len(parts) > 8 else "200"
            size_str = parts[9] if len(parts) > 9 else "1024"
            
            # Parse timestamp
            try:
                # Format: [01/Jul/1995:00:00:01 -0400]
                dt = datetime.strptime(timestamp_str, '%d/%b/%Y:%H:%M:%S %z')
                timestamp = dt.timestamp()
            except:
                timestamp = float(line_num)  # Fallback to line number
            
            # Extract URL from request
            request_parts = request_str.split(' ')
            if len(request_parts) >= 2:
                url = request_parts[1]
            else:
                url = f"/resource_{line_num}"
            
            # Parse size
            try:
                size = int(size_str) if size_str != '-' else 1024
            except:
                size = 1024
            
            # Only include successful requests for certain file types
            if status.startswith('2') and any(url.endswith(ext) for ext in 
                                            ['.html', '.htm', '.gif', '.jpg', '.png', '.css', '.js', '/']):
                return CacheRequest(
                    timestamp=timestamp,
                    key=url,
                    size=max(size, 100),  # Minimum size
                    ttl=3600,  # 1 hour default
                    client_ip=host,
                    response_time=0.1  # 100ms default
                )
            
        except Exception as e:
            logger.debug(f"Failed to parse log line {line_num}: {e}")
        
        return None
    
    def _process_cloudflare_sample(self, dataset_info: Dict, dataset_dir: Path) -> bool:
        """Process sample CloudFlare data."""
        # Generate sample CloudFlare-like data since actual data requires API access
        requests = []
        base_timestamp = datetime.now().timestamp()
        
        # Simulate CDN traffic patterns
        popular_resources = [
            ('/api/v1/users', 2048, 300),
            ('/static/app.js', 51200, 86400),
            ('/static/style.css', 25600, 86400),
            ('/images/logo.png', 10240, 3600),
            ('/api/v1/posts', 4096, 600),
        ]
        
        np.random.seed(42)
        
        for i in range(100000):
            # Choose resource with Zipfian distribution
            weights = [1/((i+1)**0.8) for i in range(len(popular_resources))]
            weights = np.array(weights) / sum(weights)
            
            resource_idx = np.random.choice(len(popular_resources), p=weights)
            url, size, ttl = popular_resources[resource_idx]
            
            # Add some variability
            size = int(size * np.random.uniform(0.8, 1.2))
            timestamp = base_timestamp + i * np.random.exponential(0.1)
            
            request = CacheRequest(
                timestamp=timestamp,
                key=f"{url}?v={i // 1000}",  # Add versioning
                size=size,
                ttl=ttl,
                client_ip=f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                response_time=np.random.exponential(0.05)  # 50ms average
            )
            requests.append(request)
        
        # Save to CSV
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
        
        logger.info(f"Generated {len(requests)} CloudFlare-like requests")
        return True
    
    def _generate_synthetic_zipf(self, dataset_info: Dict, dataset_dir: Path) -> bool:
        """Generate synthetic workload with Zipfian popularity distribution."""
        requests = []
        base_timestamp = datetime.now().timestamp()
        
        # Parameters for Zipfian distribution
        num_objects = 10000
        num_requests = 200000
        zipf_param = 1.2  # Skewness parameter
        
        # Generate object catalog
        objects = []
        for i in range(num_objects):
            size = int(np.random.lognormal(8, 1.5))  # Log-normal size distribution
            size = max(1000, min(10000000, size))  # Clamp between 1KB and 10MB
            ttl = np.random.choice([300, 600, 1800, 3600, 7200], p=[0.1, 0.2, 0.3, 0.3, 0.1])
            
            objects.append({
                'key': f'/object_{i:06d}',
                'size': size,
                'ttl': ttl
            })
        
        # Generate requests with Zipfian popularity
        np.random.seed(42)
        
        # Zipfian weights
        ranks = np.arange(1, num_objects + 1)
        weights = 1.0 / (ranks ** zipf_param)
        weights = weights / weights.sum()
        
        for i in range(num_requests):
            # Select object based on Zipfian distribution
            obj_idx = np.random.choice(num_objects, p=weights)
            obj = objects[obj_idx]
            
            # Generate timestamp with some clustering
            if i == 0:
                timestamp = base_timestamp
            else:
                # Add temporal locality - sometimes request comes soon after previous
                if np.random.random() < 0.3:
                    timestamp = requests[-1].timestamp + np.random.exponential(0.5)
                else:
                    timestamp = requests[-1].timestamp + np.random.exponential(5.0)
            
            request = CacheRequest(
                timestamp=timestamp,
                key=obj['key'],
                size=obj['size'],
                ttl=obj['ttl'],
                client_ip=f"10.0.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}",
                response_time=np.random.exponential(0.1)
            )
            requests.append(request)
        
        # Save to CSV
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
        
        logger.info(f"Generated {len(requests)} requests with Zipfian distribution")
        return True
    
    def _generate_synthetic_temporal(self, dataset_info: Dict, dataset_dir: Path) -> bool:
        """Generate synthetic workload with temporal patterns."""
        requests = []
        base_timestamp = datetime.now().timestamp()
        
        # Create different content types with different access patterns
        content_types = {
            'news': {'count': 1000, 'size_range': (5000, 50000), 'ttl': 1800, 'burst_prob': 0.1},
            'images': {'count': 5000, 'size_range': (100000, 1000000), 'ttl': 86400, 'burst_prob': 0.05},
            'api': {'count': 100, 'size_range': (1000, 10000), 'ttl': 300, 'burst_prob': 0.2},
            'static': {'count': 500, 'size_range': (10000, 100000), 'ttl': 86400, 'burst_prob': 0.02}
        }
        
        # Generate object catalog
        objects = []
        obj_id = 0
        
        for content_type, config in content_types.items():
            for i in range(config['count']):
                size = np.random.randint(config['size_range'][0], config['size_range'][1])
                objects.append({
                    'key': f'/{content_type}/object_{obj_id:06d}',
                    'size': size,
                    'ttl': config['ttl'],
                    'type': content_type,
                    'burst_prob': config['burst_prob']
                })
                obj_id += 1
        
        # Generate requests with temporal patterns
        np.random.seed(42)
        current_time = base_timestamp
        
        # Simulate daily patterns
        for day in range(7):  # One week
            daily_requests = np.random.poisson(20000)  # Average requests per day
            
            for req_num in range(daily_requests):
                # Time of day affects request pattern
                hour_of_day = (req_num / daily_requests) * 24
                
                # More activity during business hours
                if 9 <= hour_of_day <= 17:
                    intensity = 3.0
                elif 19 <= hour_of_day <= 23:
                    intensity = 2.0
                else:
                    intensity = 1.0
                
                # Select object type based on time
                if 9 <= hour_of_day <= 17:
                    # Business hours - more API and news
                    type_probs = {'news': 0.4, 'api': 0.3, 'static': 0.2, 'images': 0.1}
                else:
                    # Off hours - more images and static content
                    type_probs = {'news': 0.2, 'api': 0.1, 'static': 0.3, 'images': 0.4}
                
                # Choose content type
                chosen_type = np.random.choice(list(type_probs.keys()), 
                                             p=list(type_probs.values()))
                
                # Choose specific object of that type
                type_objects = [obj for obj in objects if obj['type'] == chosen_type]
                
                # Zipfian selection within type
                ranks = np.arange(1, len(type_objects) + 1)
                weights = 1.0 / (ranks ** 0.8)
                weights = weights / weights.sum()
                
                obj_idx = np.random.choice(len(type_objects), p=weights)
                obj = type_objects[obj_idx]
                
                # Check for burst pattern
                if np.random.random() < obj['burst_prob']:
                    # Generate burst of requests for this object
                    burst_size = np.random.poisson(10)
                    for _ in range(burst_size):
                        request = CacheRequest(
                            timestamp=current_time,
                            key=obj['key'],
                            size=obj['size'],
                            ttl=obj['ttl'],
                            client_ip=f"172.16.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}",
                            response_time=np.random.exponential(0.08)
                        )
                        requests.append(request)
                        current_time += np.random.exponential(0.1)
                else:
                    # Single request
                    request = CacheRequest(
                        timestamp=current_time,
                        key=obj['key'],
                        size=obj['size'],
                        ttl=obj['ttl'],
                        client_ip=f"172.16.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}",
                        response_time=np.random.exponential(0.08)
                    )
                    requests.append(request)
                
                # Advance time
                current_time += intensity * np.random.exponential(60.0 / daily_requests)
        
        # Sort by timestamp
        requests.sort(key=lambda x: x.timestamp)
        
        # Save to CSV
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
        
        logger.info(f"Generated {len(requests)} requests with temporal patterns")
        return True
    
    def _process_wikipedia_trace(self, dataset_info: Dict, dataset_dir: Path) -> bool:
        """Process Wikipedia trace (placeholder - would need actual implementation)."""
        logger.info("Wikipedia trace processing not implemented (requires large download)")
        return False
    
    def _process_akamai_trace(self, dataset_info: Dict, dataset_dir: Path) -> bool:
        """Process Akamai trace (placeholder - requires manual download)."""
        logger.info("Akamai trace processing requires manual download")
        return False


def main():
    """Main function for testing dataset downloader."""
    downloader = DatasetDownloader()
    
    print("Available datasets:")
    for name, info in downloader.list_datasets().items():
        print(f"  {name}: {info['description']} ({info['size_gb']} GB)")
    
    # Download and test synthetic datasets
    test_datasets = ["synthetic_zipf", "synthetic_temporal", "nasa_web_logs", "cloudflare_sample"]
    
    for dataset in test_datasets:
        print(f"\nProcessing {dataset}...")
        if downloader.download_dataset(dataset):
            requests = downloader.load_dataset(dataset)
            print(f"Loaded {len(requests)} requests")
            
            # Print some basic statistics
            sizes = [r.size for r in requests[:1000]]  # Sample first 1000
            print(f"  Size range: {min(sizes)} - {max(sizes)} bytes")
            print(f"  Unique objects: {len(set(r.key for r in requests[:1000]))}")
        else:
            print(f"Failed to process {dataset}")


if __name__ == "__main__":
    main()
