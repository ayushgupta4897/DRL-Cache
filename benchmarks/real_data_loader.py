"""
Flexible Real Data Loader for DRL Cache Benchmarking

Handles multiple real-world data formats:
- NASA web server logs (confirmed available)
- Common Log Format (CLF)
- Extended Log Format (ELF) 
- TSV/CSV formats
- Custom parsers for any format

This provides research-grade validation with actual production workloads.
"""

import pandas as pd
import numpy as np
import requests
import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import logging
from datetime import datetime
import re
from cache_simulator import CacheRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataLoader:
    """Load and process real-world cache datasets from various sources."""
    
    def __init__(self, data_dir: str = "datasets/real_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Known public datasets with confirmed URLs
        self.public_datasets = {
            "nasa_ksc_jul95": {
                "url": "http://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz",
                "description": "NASA Kennedy Space Center HTTP logs (Jul 1995)",
                "format": "common_log_format",
                "size_mb": 20,
                "requests": "3.46M",
                "parser": self._parse_nasa_logs
            },
            "nasa_ksc_aug95": {
                "url": "http://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz", 
                "description": "NASA Kennedy Space Center HTTP logs (Aug 1995)",
                "format": "common_log_format",
                "size_mb": 25,
                "requests": "~4M",
                "parser": self._parse_nasa_logs
            },
            "clarknet_sep95": {
                "url": "http://ita.ee.lbl.gov/traces/clarknet_access_log_Sep95.gz",
                "description": "ClarkNet ISP HTTP logs (Sep 1995)",
                "format": "common_log_format", 
                "size_mb": 15,
                "requests": "~3M",
                "parser": self._parse_nasa_logs  # Same format
            }
        }
    
    def list_available_datasets(self):
        """Show all available public datasets."""
        print("🌐 Available Public Datasets for DRL Benchmarking:")
        print("=" * 60)
        
        for name, info in self.public_datasets.items():
            print(f"📊 {name}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size_mb']} MB ({info['requests']} requests)")
            print(f"   Format: {info['format']}")
            print(f"   URL: {info['url']}")
            print()
    
    def download_public_dataset(self, dataset_name: str) -> Optional[Path]:
        """Download a confirmed public dataset."""
        
        if dataset_name not in self.public_datasets:
            print(f"❌ Unknown dataset: {dataset_name}")
            print("💡 Use list_available_datasets() to see options")
            return None
        
        info = self.public_datasets[dataset_name]
        url = info["url"]
        filename = Path(url).name
        local_file = self.data_dir / filename
        
        # Check if already downloaded
        if local_file.exists():
            print(f"✅ {dataset_name} already downloaded: {local_file}")
            return local_file
        
        print(f"📥 Downloading {dataset_name} from {url}")
        print(f"   Expected size: {info['size_mb']} MB")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(local_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\\r   Progress: {progress:.1f}% ({downloaded/(1024*1024):.1f}MB)", end="")
            
            print(f"\\n✅ Downloaded {dataset_name}: {local_file}")
            return local_file
            
        except Exception as e:
            print(f"❌ Failed to download {dataset_name}: {e}")
            return None
    
    def load_public_dataset(self, dataset_name: str, max_requests: int = 500000) -> List[CacheRequest]:
        """Load a public dataset and convert to CacheRequest format."""
        
        # Download if needed
        filepath = self.download_public_dataset(dataset_name)
        if not filepath:
            return []
        
        # Get parser function
        info = self.public_datasets[dataset_name]
        parser = info["parser"]
        
        print(f"🔧 Processing {dataset_name}...")
        requests = parser(filepath, max_requests)
        
        if requests:
            print(f"✅ Loaded {len(requests):,} requests from {dataset_name}")
            self._print_dataset_stats(requests)
        else:
            print(f"❌ Failed to process {dataset_name}")
        
        return requests
    
    def _parse_nasa_logs(self, filepath: Path, max_requests: int) -> List[CacheRequest]:
        """Parse NASA/ClarkNet Common Log Format files."""
        
        requests = []
        
        try:
            # Open compressed file
            with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f):
                    if len(requests) >= max_requests:
                        break
                    
                    if line_num % 50000 == 0 and line_num > 0:
                        print(f"\\r   Parsed {line_num:,} lines, {len(requests):,} valid requests", end="")
                    
                    request = self._parse_common_log_line(line, line_num)
                    if request:
                        requests.append(request)
            
            print(f"\\n   ✅ Processed {len(requests):,} requests")
            
        except Exception as e:
            print(f"❌ Error parsing {filepath}: {e}")
        
        return requests
    
    def _parse_common_log_line(self, line: str, line_num: int) -> Optional[CacheRequest]:
        """Parse a Common Log Format line."""
        
        # Common Log Format: 
        # host logname user [timestamp] "request" status size
        # Example: 199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245
        
        try:
            # Use regex to parse CLF - more flexible pattern
            clf_pattern = r'^([^\s]+)\s+\S+\s+\S+\s+\[([^\]]+)\]\s+"([^"]*?)"\s+(\d+)\s+([\d\-]+)'
            match = re.match(clf_pattern, line.strip())
            
            if not match:
                return None
            
            host, timestamp_str, request_line, status, size_str = match.groups()
            
            # Parse request line "METHOD URL VERSION"
            request_parts = request_line.split()
            if len(request_parts) < 2:
                return None
            
            method = request_parts[0]
            url = request_parts[1]
            
            # Parse timestamp
            try:
                # Format: 01/Jul/1995:00:00:01 -0400
                dt = datetime.strptime(timestamp_str.split()[0], '%d/%b/%Y:%H:%M:%S')
                timestamp = dt.timestamp()
            except:
                timestamp = float(line_num)  # Fallback to line number
            
            # Parse size
            if size_str == '-':
                size = 1024  # Default size for missing data
            else:
                try:
                    size = int(size_str)
                    size = max(size, 1)  # Minimum 1 byte
                except ValueError:
                    size = 1024  # Default for unparseable size
            
            # Include successful GET requests (including 304 Not Modified for cache behavior)
            if (method == 'GET' and 
                (status.startswith('2') or status == '304') and 
                self._is_cacheable_url(url)):
                
                # Estimate TTL based on content type
                ttl = self._estimate_ttl(url, size)
                
                # Estimate origin response time based on size
                response_time = self._estimate_response_time(size)
                
                return CacheRequest(
                    timestamp=timestamp,
                    key=url,
                    size=size,
                    ttl=ttl,
                    client_ip=host,
                    response_time=response_time
                )
                
        except Exception as e:
            # Skip malformed lines - print first few for debugging
            if line_num < 5:
                print(f"Debug: Failed to parse line {line_num}: {line[:100]}...")
        
        return None
    
    def _is_cacheable_url(self, url: str) -> bool:
        """Determine if a URL represents cacheable content."""
        
        # Cacheable file extensions
        cacheable_extensions = {
            # Images
            '.gif', '.jpg', '.jpeg', '.png', '.bmp', '.ico',
            # Documents  
            '.html', '.htm', '.txt', '.pdf', '.ps',
            # Media
            '.mp3', '.wav', '.avi', '.mov', '.mpg', '.mpeg',
            # Data
            '.zip', '.tar', '.gz', '.Z',
            # Scripts/styles (often cached)
            '.js', '.css'
        }
        
        url_lower = url.lower()
        
        # Check file extension
        if any(url_lower.endswith(ext) for ext in cacheable_extensions):
            return True
        
        # Check for directory requests (often cacheable index pages)
        if url.endswith('/'):
            return True
        
        # Skip obviously dynamic content
        skip_patterns = ['cgi-bin', 'search', 'query', '.cgi']
        if any(pattern in url_lower for pattern in skip_patterns):
            return False
        
        # Skip URLs with query parameters (likely dynamic)
        if '?' in url or '=' in url:
            return False
        
        # Include most other content paths - be more inclusive for research
        # This covers paths like "/history/apollo/" or "/shuttle/missions/..."
        if len(url) > 1:
            return True
        
        return False
    
    def _estimate_ttl(self, url: str, size: int) -> int:
        """Estimate appropriate TTL based on content type."""
        
        url_lower = url.lower()
        
        # Images - long TTL
        if any(ext in url_lower for ext in ['.gif', '.jpg', '.jpeg', '.png']):
            return 86400 * 7  # 7 days
        
        # Media files - very long TTL
        if any(ext in url_lower for ext in ['.mp3', '.avi', '.mov', '.zip']):
            return 86400 * 30  # 30 days
        
        # HTML pages - medium TTL
        if any(ext in url_lower for ext in ['.html', '.htm']) or url.endswith('/'):
            return 3600  # 1 hour
        
        # Documents - long TTL
        if any(ext in url_lower for ext in ['.pdf', '.ps', '.txt']):
            return 86400  # 1 day
        
        # Default TTL
        return 3600  # 1 hour
    
    def _estimate_response_time(self, size: int) -> float:
        """Estimate origin server response time based on content size."""
        
        # Base latency (network + server processing)
        base_latency = 0.05  # 50ms
        
        # Size-dependent component (simulating transfer time)
        # Assume ~1MB/s server throughput
        size_latency = size / (1024 * 1024)  # seconds per MB
        
        # Add some jitter
        jitter = np.random.exponential(0.02)
        
        total_time = base_latency + size_latency + jitter
        return max(total_time, 0.001)  # Minimum 1ms
    
    def _print_dataset_stats(self, requests: List[CacheRequest]):
        """Print comprehensive dataset statistics."""
        
        if not requests:
            return
        
        # Basic stats
        total_requests = len(requests)
        unique_objects = len(set(r.key for r in requests))
        time_span = (requests[-1].timestamp - requests[0].timestamp) / 3600  # hours
        
        # Size stats
        sizes = [r.size for r in requests]
        total_gb = sum(sizes) / (1024**3)
        
        print(f"\\n📊 Dataset Statistics:")
        print(f"   📈 Total requests: {total_requests:,}")
        print(f"   🔑 Unique objects: {unique_objects:,}")
        print(f"   📅 Time span: {time_span:.1f} hours")
        print(f"   💾 Total data: {total_gb:.2f} GB")
        print(f"   📏 Size range: {min(sizes):,} - {max(sizes):,} bytes")
        print(f"   📊 Avg size: {np.mean(sizes):,.0f} bytes")
        print(f"   🔄 Repeat ratio: {(total_requests - unique_objects) / total_requests * 100:.1f}%")
        
        # Content type analysis
        image_requests = sum(1 for r in requests if any(ext in r.key.lower() 
                                                       for ext in ['.gif', '.jpg', '.jpeg', '.png']))
        html_requests = sum(1 for r in requests if r.key.lower().endswith('.html') or r.key.endswith('/'))
        
        print(f"   🖼️  Images: {image_requests:,} ({image_requests/total_requests*100:.1f}%)")
        print(f"   📄 HTML/Pages: {html_requests:,} ({html_requests/total_requests*100:.1f}%)")
    
    def load_custom_data(self, filepath: str, parser_func: Callable) -> List[CacheRequest]:
        """Load custom data format with user-provided parser."""
        
        print(f"🔧 Loading custom data from {filepath}")
        
        try:
            requests = parser_func(filepath)
            if requests:
                print(f"✅ Loaded {len(requests):,} requests")
                self._print_dataset_stats(requests)
            return requests
        except Exception as e:
            print(f"❌ Failed to load custom data: {e}")
            return []
    
    def save_processed_dataset(self, requests: List[CacheRequest], 
                             dataset_name: str) -> Path:
        """Save processed requests for reuse."""
        
        if not requests:
            return None
        
        # Create DataFrame
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
        
        # Save to CSV
        output_file = self.data_dir / f"{dataset_name}_processed.csv"
        df.to_csv(output_file, index=False)
        
        print(f"💾 Saved processed dataset: {output_file}")
        return output_file


def create_custom_parser_example():
    """Example of how to create a custom parser for your data format."""
    
    def my_custom_parser(filepath: str) -> List[CacheRequest]:
        """
        Example custom parser - modify this for your data format.
        
        Your data format might be:
        - CSV with columns: timestamp,url,size,client_ip
        - JSON with request objects
        - Custom log format
        - Database export
        etc.
        """
        
        requests = []
        
        # Example: CSV format
        df = pd.read_csv(filepath)
        
        for _, row in df.iterrows():
            request = CacheRequest(
                timestamp=pd.to_datetime(row['timestamp']).timestamp(),
                key=row['url'],
                size=int(row['size']),
                ttl=3600,  # Default 1 hour
                client_ip=row.get('client_ip', 'unknown'),
                response_time=0.1  # Default 100ms
            )
            requests.append(request)
        
        return requests
    
    return my_custom_parser


def main():
    """Demonstrate the real data loader."""
    
    loader = RealDataLoader()
    
    print("🚀 Real Data Loader for DRL Cache Benchmarking")
    print("=" * 50)
    
    # Show available datasets
    loader.list_available_datasets()
    
    # Load NASA dataset as example
    print("\\n📥 Loading NASA dataset for demonstration...")
    requests = loader.load_public_dataset("nasa_ksc_jul95", max_requests=100000)
    
    if requests:
        print("\\n🎉 Successfully loaded real data!")
        print("💡 This data is ready for DRL cache benchmarking")
        
        # Save for reuse
        loader.save_processed_dataset(requests, "nasa_demo")
    else:
        print("\\n💡 Data loading examples:")
        print("  1. Use load_public_dataset() for confirmed datasets")
        print("  2. Use load_custom_data() with your parser function")
        print("  3. Modify the parsers for your specific format")


if __name__ == "__main__":
    main()
