"""
Comprehensive Benchmark Runner for DRL Cache Evaluation

This module orchestrates comprehensive benchmarking experiments comparing
DRL Cache against various baseline cache eviction policies.
"""

import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from cache_simulator import (
    CacheSimulator, LRUPolicy, LFUPolicy, FIFOPolicy, 
    RandomPolicy, SizeBasedPolicy, HybridLRUSizePolicy,
    CacheRequest, CacheStats, run_simulation
)
from drl_policy import (
    DRLCachePolicy, OptimalOfflinePolicy, AdaptiveLRUPolicy,
    FrequencyAwareLRUPolicy
)
from dataset_downloader import DatasetDownloader


class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    
    def __init__(self):
        # Cache sizes to test (in bytes)
        self.cache_sizes = [
            100 * 1024 * 1024,   # 100MB
            500 * 1024 * 1024,   # 500MB
            1 * 1024 * 1024 * 1024,  # 1GB
            2 * 1024 * 1024 * 1024,  # 2GB
        ]
        
        # Datasets to evaluate
        self.datasets = [
            "synthetic_zipf",
            "synthetic_temporal", 
            "cloudflare_sample",
            "nasa_web_logs"
        ]
        
        # Policies to compare
        self.baseline_policies = [
            "LRU",
            "LFU", 
            "FIFO",
            "Random",
            "SizeBased",
            "HybridLRUSize",
            "AdaptiveLRU",
            "FrequencyAwareLRU"
        ]
        
        # DRL policy variants
        self.drl_policies = [
            {"name": "DRL-Cache-K8", "max_k": 8},
            {"name": "DRL-Cache-K16", "max_k": 16},
            {"name": "DRL-Cache-K32", "max_k": 32},
        ]
        
        # Number of repetitions for statistical significance
        self.num_repetitions = 3
        
        # Warm-up period (fraction of requests)
        self.warmup_fraction = 0.1
        
        # Maximum requests to process (for faster testing)
        self.max_requests = 100000
        
        # Output directory
        self.output_dir = "results"


class BenchmarkRunner:
    """Main benchmark runner."""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.downloader = DatasetDownloader()
        
        # Create output directory
        Path(self.config.output_dir).mkdir(exist_ok=True)
        
        # Results storage
        self.results = []
        
    def prepare_datasets(self):
        """Download and prepare all required datasets."""
        print("Preparing datasets...")
        
        for dataset_name in self.config.datasets:
            print(f"  Processing {dataset_name}...")
            success = self.downloader.download_dataset(dataset_name)
            if success:
                print(f"  [OK] {dataset_name} ready")
            else:
                print(f"  [FAILED] {dataset_name} failed")
                
    def create_policies(self) -> Dict[str, Any]:
        """Create all policy instances for benchmarking."""
        policies = {}
        
        # Baseline policies
        policy_classes = {
            "LRU": LRUPolicy,
            "LFU": LFUPolicy,
            "FIFO": FIFOPolicy,
            "Random": RandomPolicy,
            "SizeBased": SizeBasedPolicy,
            "HybridLRUSize": HybridLRUSizePolicy,
            "AdaptiveLRU": AdaptiveLRUPolicy,
            "FrequencyAwareLRU": FrequencyAwareLRUPolicy,
        }
        
        for name in self.config.baseline_policies:
            if name in policy_classes:
                policies[name] = policy_classes[name]()
        
        # DRL policies
        for drl_config in self.config.drl_policies:
            policies[drl_config["name"]] = DRLCachePolicy(
                max_k=drl_config["max_k"],
                inference_timeout_us=500
            )
        
        return policies
    
    def run_single_experiment(self, 
                            dataset_name: str,
                            cache_size: int,
                            policy_name: str,
                            policy,
                            repetition: int) -> Dict[str, Any]:
        """Run a single benchmark experiment."""
        
        try:
            # Load dataset
            requests = self.downloader.load_dataset(dataset_name)
            
            # Limit number of requests for faster testing
            if len(requests) > self.config.max_requests:
                # Take a representative sample
                step = len(requests) // self.config.max_requests
                requests = requests[::step][:self.config.max_requests]
            
            # Sort by timestamp
            requests.sort(key=lambda x: x.timestamp)
            
            # For optimal offline policy, precompute future accesses
            if "Optimal" in policy_name:
                future_accesses = self._compute_future_accesses(requests)
                policy = OptimalOfflinePolicy(future_accesses)
            
            # Run simulation
            start_time = time.time()
            stats = run_simulation(requests, cache_size, policy)
            execution_time = time.time() - start_time
            
            # Collect results
            result = {
                'dataset': dataset_name,
                'cache_size_mb': cache_size // (1024 * 1024),
                'policy': policy_name,
                'repetition': repetition,
                'execution_time_sec': execution_time,
                
                # Basic metrics
                'total_requests': stats.total_requests,
                'cache_hits': stats.cache_hits,
                'cache_misses': stats.cache_misses,
                'hit_ratio': stats.hit_ratio,
                'byte_hit_ratio': stats.byte_hit_ratio,
                
                # Performance metrics
                'total_evictions': stats.total_evictions,
                'bytes_evicted': stats.bytes_evicted,
                'avg_response_time': stats.avg_response_time,
                'total_cost': stats.total_cost,
                
                # Throughput
                'requests_per_second': stats.total_requests / execution_time,
            }
            
            # Add policy-specific metrics
            if hasattr(policy, 'get_performance_stats'):
                policy_stats = policy.get_performance_stats()
                for key, value in policy_stats.items():
                    if isinstance(value, (int, float)):
                        result[f'policy_{key}'] = value
            
            return result
            
        except Exception as e:
            print(f"Error in experiment {dataset_name}-{policy_name}: {e}")
            return {
                'dataset': dataset_name,
                'cache_size_mb': cache_size // (1024 * 1024),
                'policy': policy_name,
                'repetition': repetition,
                'error': str(e)
            }
    
    def _compute_future_accesses(self, requests: List[CacheRequest]) -> Dict[str, List[float]]:
        """Compute future access times for each object (for optimal policy)."""
        future_accesses = {}
        
        for request in requests:
            if request.key not in future_accesses:
                future_accesses[request.key] = []
            future_accesses[request.key].append(request.timestamp)
        
        return future_accesses
    
    def run_all_experiments(self, parallel: bool = True, max_workers: int = 4):
        """Run all benchmark experiments."""
        print("Starting comprehensive benchmark experiments...")
        
        # Create all policy instances
        policies = self.create_policies()
        
        # Generate all experiment combinations
        experiments = []
        
        for dataset_name in self.config.datasets:
            for cache_size in self.config.cache_sizes:
                for policy_name, policy in policies.items():
                    for repetition in range(self.config.num_repetitions):
                        experiments.append({
                            'dataset_name': dataset_name,
                            'cache_size': cache_size,
                            'policy_name': policy_name,
                            'policy': policy,
                            'repetition': repetition
                        })
        
        print(f"Total experiments to run: {len(experiments)}")
        
        # Run experiments
        if parallel and len(experiments) > 1:
            self._run_experiments_parallel(experiments, max_workers)
        else:
            self._run_experiments_sequential(experiments)
        
        # Save results
        self._save_results()
        
        print(f"Benchmark completed! Results saved to {self.config.output_dir}")
    
    def _run_experiments_sequential(self, experiments: List[Dict]):
        """Run experiments sequentially."""
        for exp in tqdm(experiments, desc="Running experiments"):
            result = self.run_single_experiment(
                exp['dataset_name'],
                exp['cache_size'],
                exp['policy_name'],
                exp['policy'],
                exp['repetition']
            )
            self.results.append(result)
    
    def _run_experiments_parallel(self, experiments: List[Dict], max_workers: int):
        """Run experiments in parallel."""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_exp = {
                executor.submit(
                    self.run_single_experiment,
                    exp['dataset_name'],
                    exp['cache_size'], 
                    exp['policy_name'],
                    exp['policy'],
                    exp['repetition']
                ): exp for exp in experiments
            }
            
            # Collect results
            for future in tqdm(as_completed(future_to_exp), 
                             total=len(experiments),
                             desc="Running experiments"):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    self.results.append(result)
                except Exception as e:
                    exp = future_to_exp[future]
                    print(f"Experiment failed: {exp['dataset_name']}-{exp['policy_name']}: {e}")
    
    def _save_results(self):
        """Save benchmark results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save raw results as JSON
        json_file = Path(self.config.output_dir) / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save as CSV for easy analysis
        csv_file = Path(self.config.output_dir) / f"benchmark_results_{timestamp}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_file, index=False)
        
        # Save summary statistics
        self._save_summary_stats(timestamp)
        
        print(f"Results saved:")
        print(f"  Raw data: {json_file}")
        print(f"  CSV data: {csv_file}")
    
    def _save_summary_stats(self, timestamp: str):
        """Generate and save summary statistics."""
        df = pd.DataFrame(self.results)
        
        if df.empty or 'error' in df.columns:
            print("No valid results to summarize")
            return
        
        # Group by dataset, cache size, and policy
        summary_stats = []
        
        for (dataset, cache_size, policy), group in df.groupby(['dataset', 'cache_size_mb', 'policy']):
            stats = {
                'dataset': dataset,
                'cache_size_mb': cache_size,
                'policy': policy,
                
                # Hit ratio statistics
                'hit_ratio_mean': group['hit_ratio'].mean(),
                'hit_ratio_std': group['hit_ratio'].std(),
                'hit_ratio_min': group['hit_ratio'].min(),
                'hit_ratio_max': group['hit_ratio'].max(),
                
                # Byte hit ratio statistics  
                'byte_hit_ratio_mean': group['byte_hit_ratio'].mean(),
                'byte_hit_ratio_std': group['byte_hit_ratio'].std(),
                
                # Response time statistics
                'avg_response_time_mean': group['avg_response_time'].mean(),
                'avg_response_time_std': group['avg_response_time'].std(),
                
                # Cost statistics
                'total_cost_mean': group['total_cost'].mean(),
                'total_cost_std': group['total_cost'].std(),
                
                # Throughput statistics
                'requests_per_second_mean': group['requests_per_second'].mean(),
                'requests_per_second_std': group['requests_per_second'].std(),
                
                # Sample size
                'num_repetitions': len(group)
            }
            
            summary_stats.append(stats)
        
        # Save summary
        summary_file = Path(self.config.output_dir) / f"benchmark_summary_{timestamp}.csv"
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(summary_file, index=False)
        
        print(f"  Summary: {summary_file}")
        
        # Print top performers
        print("\nTop performers by hit ratio:")
        top_performers = summary_df.nlargest(5, 'hit_ratio_mean')
        print(top_performers[['dataset', 'cache_size_mb', 'policy', 'hit_ratio_mean']].to_string(index=False))


class QuickBenchmark:
    """Simplified benchmark for quick testing."""
    
    def __init__(self):
        self.downloader = DatasetDownloader()
    
    def run_quick_test(self):
        """Run a quick benchmark test."""
        print("Running quick benchmark test...")
        
        # Generate small synthetic dataset
        self.downloader.download_dataset("synthetic_zipf")
        requests = self.downloader.load_dataset("synthetic_zipf")
        requests = requests[:10000]  # Use only first 10k requests
        
        # Test with small cache
        cache_size = 50 * 1024 * 1024  # 50MB
        
        # Compare key policies
        policies = {
            "LRU": LRUPolicy(),
            "LFU": LFUPolicy(),
            "SizeBased": SizeBasedPolicy(),
            "DRL-Cache": DRLCachePolicy(max_k=16),
        }
        
        results = []
        
        for name, policy in policies.items():
            print(f"  Testing {name}...")
            start_time = time.time()
            stats = run_simulation(requests, cache_size, policy)
            execution_time = time.time() - start_time
            
            results.append({
                'policy': name,
                'hit_ratio': stats.hit_ratio,
                'byte_hit_ratio': stats.byte_hit_ratio,
                'execution_time': execution_time,
                'requests_per_sec': stats.total_requests / execution_time
            })
        
        # Print results
        df = pd.DataFrame(results)
        df = df.sort_values('hit_ratio', ascending=False)
        
        print("\nQuick Benchmark Results:")
        print("=" * 60)
        for _, row in df.iterrows():
            print(f"{row['policy']:20s} | Hit Ratio: {row['hit_ratio']:.3f} | "
                  f"Byte Hit Ratio: {row['byte_hit_ratio']:.3f} | "
                  f"RPS: {row['requests_per_sec']:.0f}")
        
        return df


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="DRL Cache Benchmark Runner")
    
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="Benchmark mode: quick test or full evaluation")
    parser.add_argument("--datasets", nargs="+", 
                       default=["synthetic_zipf", "synthetic_temporal"],
                       help="Datasets to benchmark")
    parser.add_argument("--cache-sizes", nargs="+", type=int,
                       default=[100, 500],
                       help="Cache sizes in MB")
    parser.add_argument("--repetitions", type=int, default=3,
                       help="Number of repetitions for each experiment")
    parser.add_argument("--max-requests", type=int, default=50000,
                       help="Maximum requests to process per dataset")
    parser.add_argument("--parallel", action="store_true",
                       help="Run experiments in parallel")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--output-dir", default="results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        # Quick test
        benchmark = QuickBenchmark()
        benchmark.run_quick_test()
        
    else:
        # Full benchmark
        config = BenchmarkConfig()
        config.datasets = args.datasets
        config.cache_sizes = [size * 1024 * 1024 for size in args.cache_sizes]  # Convert MB to bytes
        config.num_repetitions = args.repetitions
        config.max_requests = args.max_requests
        config.output_dir = args.output_dir
        
        runner = BenchmarkRunner(config)
        runner.prepare_datasets()
        runner.run_all_experiments(parallel=args.parallel, max_workers=args.workers)


if __name__ == "__main__":
    main()
