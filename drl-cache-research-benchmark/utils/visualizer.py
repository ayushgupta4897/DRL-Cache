"""
Publication-Quality Visualization for DRL Cache Benchmarks

Generates comprehensive plots and analysis suitable for research papers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    'grid.alpha': 0.3,
})

# Color palette for different policies
POLICY_COLORS = {
    'LRU': '#1f77b4',
    'LFU': '#ff7f0e', 
    'FIFO': '#2ca02c',
    'Random': '#d62728',
    'SizeBased': '#9467bd',
    'HybridLRUSize': '#8c564b',
    'AdaptiveLRU': '#e377c2',
    'FrequencyAwareLRU': '#7f7f7f',
    'DRL-Cache-K8': '#ff0000',
    'DRL-Cache-K16': '#cc0000', 
    'DRL-Cache-K32': '#990000',
    'Optimal-Offline': '#000000'
}


class BenchmarkVisualizer:
    """Creates publication-quality visualizations for benchmark results."""
    
    def __init__(self, results_file: str, output_dir: str = "plots"):
        """Initialize visualizer with benchmark results."""
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        if self.results_file.suffix == '.csv':
            self.df = pd.read_csv(self.results_file)
        else:
            # Assume JSON
            self.df = pd.read_json(self.results_file)
        
        # Clean data
        self._clean_data()
        
        print(f"Loaded {len(self.df)} benchmark results")
        print(f"Datasets: {self.df['dataset'].unique()}")
        print(f"Policies: {self.df['policy'].unique()}")
        print(f"Cache sizes: {sorted(self.df['cache_size_mb'].unique())} MB")
    
    def _clean_data(self):
        """Clean and prepare data for visualization."""
        # Remove failed experiments
        self.df = self.df[~self.df.get('error', pd.Series()).notna()]
        
        # Ensure numeric columns
        numeric_cols = ['hit_ratio', 'byte_hit_ratio', 'avg_response_time', 
                       'total_cost', 'requests_per_second']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Compute derived metrics
        if 'cache_misses' in self.df.columns and 'cache_hits' in self.df.columns:
            self.df['total_requests_calc'] = self.df['cache_hits'] + self.df['cache_misses']
        
        # Origin bandwidth savings (vs always going to origin)
        if 'byte_hit_ratio' in self.df.columns:
            self.df['bandwidth_savings'] = self.df['byte_hit_ratio']
    
    def generate_all_plots(self):
        """Generate all publication-quality plots."""
        print("Generating publication-quality plots...")
        
        # Main performance comparison plots
        self.plot_hit_ratio_comparison()
        self.plot_byte_hit_ratio_comparison() 
        self.plot_performance_vs_cache_size()
        self.plot_cost_analysis()
        
        # Detailed analysis plots
        self.plot_policy_performance_heatmap()
        self.plot_dataset_specific_analysis()
        self.plot_scalability_analysis()
        
        # DRL-specific analysis
        self.plot_drl_ablation_study()
        self.plot_inference_performance()
        
        # Statistical analysis
        self.plot_statistical_significance()
        
        # Interactive plots
        self.create_interactive_dashboard()
        
        print(f"All plots saved to {self.output_dir}")
    
    def plot_hit_ratio_comparison(self):
        """Plot hit ratio comparison across all policies."""
        # Aggregate across repetitions
        agg_df = self.df.groupby(['dataset', 'cache_size_mb', 'policy']).agg({
            'hit_ratio': ['mean', 'std'],
            'byte_hit_ratio': ['mean', 'std']
        }).reset_index()
        
        agg_df.columns = ['dataset', 'cache_size_mb', 'policy', 
                         'hit_ratio_mean', 'hit_ratio_std',
                         'byte_hit_ratio_mean', 'byte_hit_ratio_std']
        
        # Create subplot for each dataset
        datasets = agg_df['dataset'].unique()
        fig, axes = plt.subplots(2, len(datasets), figsize=(5*len(datasets), 10))
        if len(datasets) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, dataset in enumerate(datasets):
            dataset_df = agg_df[agg_df['dataset'] == dataset]
            
            # Hit ratio plot
            ax1 = axes[0, i] if len(datasets) > 1 else axes[0]
            
            # Group by policy and plot
            policies = dataset_df['policy'].unique()
            x_pos = np.arange(len(policies))
            
            for j, cache_size in enumerate(sorted(dataset_df['cache_size_mb'].unique())):
                cache_df = dataset_df[dataset_df['cache_size_mb'] == cache_size]
                cache_df = cache_df.sort_values('policy')
                
                bars = ax1.bar(x_pos + j*0.2, cache_df['hit_ratio_mean'], 
                              width=0.2, label=f'{cache_size}MB',
                              yerr=cache_df['hit_ratio_std'], capsize=3,
                              alpha=0.8)
            
            ax1.set_xlabel('Cache Policy')
            ax1.set_ylabel('Hit Ratio')
            ax1.set_title(f'Hit Ratio - {dataset}')
            ax1.set_xticks(x_pos + 0.2)
            ax1.set_xticklabels(policies, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Byte hit ratio plot
            ax2 = axes[1, i] if len(datasets) > 1 else axes[1]
            
            for j, cache_size in enumerate(sorted(dataset_df['cache_size_mb'].unique())):
                cache_df = dataset_df[dataset_df['cache_size_mb'] == cache_size]
                cache_df = cache_df.sort_values('policy')
                
                ax2.bar(x_pos + j*0.2, cache_df['byte_hit_ratio_mean'],
                       width=0.2, label=f'{cache_size}MB',
                       yerr=cache_df['byte_hit_ratio_std'], capsize=3,
                       alpha=0.8)
            
            ax2.set_xlabel('Cache Policy')
            ax2.set_ylabel('Byte Hit Ratio')
            ax2.set_title(f'Byte Hit Ratio - {dataset}')
            ax2.set_xticks(x_pos + 0.2)
            ax2.set_xticklabels(policies, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hit_ratio_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'hit_ratio_comparison.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_performance_vs_cache_size(self):
        """Plot performance metrics vs cache size."""
        # Aggregate data
        agg_df = self.df.groupby(['dataset', 'cache_size_mb', 'policy']).agg({
            'hit_ratio': 'mean',
            'byte_hit_ratio': 'mean',
            'avg_response_time': 'mean',
            'requests_per_second': 'mean'
        }).reset_index()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Identify DRL policies and baselines
        drl_policies = [p for p in agg_df['policy'].unique() if 'DRL' in p]
        baseline_policies = [p for p in agg_df['policy'].unique() if 'DRL' not in p]
        
        for dataset in agg_df['dataset'].unique():
            dataset_df = agg_df[agg_df['dataset'] == dataset]
            
            # Hit ratio vs cache size
            ax = axes[0, 0]
            for policy in baseline_policies[:4]:  # Show top 4 baselines
                policy_df = dataset_df[dataset_df['policy'] == policy]
                ax.plot(policy_df['cache_size_mb'], policy_df['hit_ratio'], 
                       'o-', label=f'{policy}', alpha=0.7)
            
            for policy in drl_policies:
                policy_df = dataset_df[dataset_df['policy'] == policy]
                ax.plot(policy_df['cache_size_mb'], policy_df['hit_ratio'], 
                       's-', label=f'{policy}', linewidth=2, markersize=8)
            
            ax.set_xlabel('Cache Size (MB)')
            ax.set_ylabel('Hit Ratio')
            ax.set_title('Hit Ratio vs Cache Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Byte hit ratio vs cache size  
            ax = axes[0, 1]
            for policy in baseline_policies[:4]:
                policy_df = dataset_df[dataset_df['policy'] == policy]
                ax.plot(policy_df['cache_size_mb'], policy_df['byte_hit_ratio'],
                       'o-', label=f'{policy}', alpha=0.7)
            
            for policy in drl_policies:
                policy_df = dataset_df[dataset_df['policy'] == policy]
                ax.plot(policy_df['cache_size_mb'], policy_df['byte_hit_ratio'],
                       's-', label=f'{policy}', linewidth=2, markersize=8)
            
            ax.set_xlabel('Cache Size (MB)')
            ax.set_ylabel('Byte Hit Ratio')
            ax.set_title('Byte Hit Ratio vs Cache Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Response time vs cache size
            ax = axes[1, 0]
            for policy in baseline_policies[:4]:
                policy_df = dataset_df[dataset_df['policy'] == policy]
                ax.plot(policy_df['cache_size_mb'], policy_df['avg_response_time'] * 1000,
                       'o-', label=f'{policy}', alpha=0.7)
            
            for policy in drl_policies:
                policy_df = dataset_df[dataset_df['policy'] == policy]
                ax.plot(policy_df['cache_size_mb'], policy_df['avg_response_time'] * 1000,
                       's-', label=f'{policy}', linewidth=2, markersize=8)
            
            ax.set_xlabel('Cache Size (MB)')
            ax.set_ylabel('Avg Response Time (ms)')
            ax.set_title('Response Time vs Cache Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Throughput vs cache size
            ax = axes[1, 1]
            for policy in baseline_policies[:4]:
                policy_df = dataset_df[dataset_df['policy'] == policy]
                ax.plot(policy_df['cache_size_mb'], policy_df['requests_per_second'],
                       'o-', label=f'{policy}', alpha=0.7)
            
            for policy in drl_policies:
                policy_df = dataset_df[dataset_df['policy'] == policy]
                ax.plot(policy_df['cache_size_mb'], policy_df['requests_per_second'],
                       's-', label=f'{policy}', linewidth=2, markersize=8)
            
            ax.set_xlabel('Cache Size (MB)')
            ax.set_ylabel('Requests per Second')
            ax.set_title('Throughput vs Cache Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_vs_cache_size.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'performance_vs_cache_size.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_policy_performance_heatmap(self):
        """Create heatmap showing policy performance across datasets."""
        # Aggregate data
        agg_df = self.df.groupby(['dataset', 'policy']).agg({
            'hit_ratio': 'mean'
        }).reset_index()
        
        # Pivot for heatmap
        heatmap_data = agg_df.pivot(index='policy', columns='dataset', values='hit_ratio')
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   cbar_kws={'label': 'Hit Ratio'},
                   linewidths=0.5, linecolor='white')
        
        plt.title('Policy Performance Heatmap (Hit Ratio)', fontsize=16, fontweight='bold')
        plt.xlabel('Dataset', fontweight='bold')
        plt.ylabel('Cache Policy', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'policy_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'policy_performance_heatmap.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_drl_ablation_study(self):
        """Analyze effect of different K values for DRL Cache."""
        drl_df = self.df[self.df['policy'].str.contains('DRL', na=False)].copy()
        
        if drl_df.empty:
            print("No DRL results found for ablation study")
            return
        
        # Extract K value from policy name
        drl_df['k_value'] = drl_df['policy'].str.extract(r'K(\d+)').astype(int)
        
        # Aggregate by K value
        agg_df = drl_df.groupby(['dataset', 'cache_size_mb', 'k_value']).agg({
            'hit_ratio': ['mean', 'std'],
            'policy_avg_inference_time_us': ['mean', 'std'],
            'policy_fallback_rate': ['mean', 'std']
        }).reset_index()
        
        agg_df.columns = ['dataset', 'cache_size_mb', 'k_value',
                         'hit_ratio_mean', 'hit_ratio_std',
                         'inference_time_mean', 'inference_time_std',
                         'fallback_rate_mean', 'fallback_rate_std']
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Hit ratio vs K
        ax = axes[0]
        for dataset in agg_df['dataset'].unique():
            dataset_df = agg_df[agg_df['dataset'] == dataset]
            avg_df = dataset_df.groupby('k_value').agg({
                'hit_ratio_mean': 'mean',
                'hit_ratio_std': 'mean'
            }).reset_index()
            
            ax.errorbar(avg_df['k_value'], avg_df['hit_ratio_mean'],
                       yerr=avg_df['hit_ratio_std'], 
                       'o-', label=dataset, capsize=5, markersize=8)
        
        ax.set_xlabel('K (Number of Candidates)', fontweight='bold')
        ax.set_ylabel('Hit Ratio', fontweight='bold')
        ax.set_title('Hit Ratio vs K Value', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([8, 16, 32])
        
        # Inference time vs K
        ax = axes[1]
        for dataset in agg_df['dataset'].unique():
            dataset_df = agg_df[agg_df['dataset'] == dataset]
            avg_df = dataset_df.groupby('k_value').agg({
                'inference_time_mean': 'mean',
                'inference_time_std': 'mean'
            }).reset_index()
            
            ax.errorbar(avg_df['k_value'], avg_df['inference_time_mean'],
                       yerr=avg_df['inference_time_std'],
                       'o-', label=dataset, capsize=5, markersize=8)
        
        ax.set_xlabel('K (Number of Candidates)', fontweight='bold')
        ax.set_ylabel('Avg Inference Time (μs)', fontweight='bold')
        ax.set_title('Inference Time vs K Value', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([8, 16, 32])
        
        # Fallback rate vs K
        ax = axes[2]
        for dataset in agg_df['dataset'].unique():
            dataset_df = agg_df[agg_df['dataset'] == dataset]
            avg_df = dataset_df.groupby('k_value').agg({
                'fallback_rate_mean': 'mean',
                'fallback_rate_std': 'mean'
            }).reset_index()
            
            ax.errorbar(avg_df['k_value'], avg_df['fallback_rate_mean'] * 100,
                       yerr=avg_df['fallback_rate_std'] * 100,
                       'o-', label=dataset, capsize=5, markersize=8)
        
        ax.set_xlabel('K (Number of Candidates)', fontweight='bold')
        ax.set_ylabel('Fallback Rate (%)', fontweight='bold')
        ax.set_title('Fallback Rate vs K Value', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([8, 16, 32])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'drl_ablation_study.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'drl_ablation_study.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_statistical_significance(self):
        """Plot statistical significance of improvements."""
        from scipy import stats
        
        # Compare DRL Cache vs best baseline for each dataset
        results = []
        
        for dataset in self.df['dataset'].unique():
            for cache_size in self.df['cache_size_mb'].unique():
                dataset_df = self.df[(self.df['dataset'] == dataset) & 
                                   (self.df['cache_size_mb'] == cache_size)]
                
                # Get DRL results
                drl_results = dataset_df[dataset_df['policy'].str.contains('DRL-Cache-K16', na=False)]
                if len(drl_results) < 2:
                    continue
                
                # Find best baseline
                baseline_policies = dataset_df[~dataset_df['policy'].str.contains('DRL', na=False)]
                if len(baseline_policies) == 0:
                    continue
                
                best_baseline_policy = baseline_policies.groupby('policy')['hit_ratio'].mean().idxmax()
                best_baseline_results = baseline_policies[baseline_policies['policy'] == best_baseline_policy]
                
                if len(best_baseline_results) < 2:
                    continue
                
                # Perform t-test
                drl_hit_ratios = drl_results['hit_ratio'].values
                baseline_hit_ratios = best_baseline_results['hit_ratio'].values
                
                t_stat, p_value = stats.ttest_ind(drl_hit_ratios, baseline_hit_ratios)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(drl_hit_ratios) - 1) * drl_hit_ratios.std() ** 2 + 
                                    (len(baseline_hit_ratios) - 1) * baseline_hit_ratios.std() ** 2) / 
                                   (len(drl_hit_ratios) + len(baseline_hit_ratios) - 2))
                cohens_d = (drl_hit_ratios.mean() - baseline_hit_ratios.mean()) / pooled_std
                
                results.append({
                    'dataset': dataset,
                    'cache_size_mb': cache_size,
                    'drl_mean': drl_hit_ratios.mean(),
                    'baseline_mean': baseline_hit_ratios.mean(),
                    'improvement': drl_hit_ratios.mean() - baseline_hit_ratios.mean(),
                    'improvement_pct': (drl_hit_ratios.mean() - baseline_hit_ratios.mean()) / baseline_hit_ratios.mean() * 100,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05,
                    'baseline_policy': best_baseline_policy
                })
        
        if not results:
            print("Not enough data for statistical significance analysis")
            return
        
        results_df = pd.DataFrame(results)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Improvement vs dataset
        ax = axes[0, 0]
        colors = ['green' if sig else 'orange' for sig in results_df['significant']]
        bars = ax.bar(range(len(results_df)), results_df['improvement_pct'], color=colors, alpha=0.7)
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Hit Ratio Improvement vs Best Baseline')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add significance annotations
        for i, (bar, p_val) in enumerate(zip(bars, results_df['p_value'])):
            height = bar.get_height()
            sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   sig_text, ha='center', va='bottom', fontweight='bold')
        
        # Effect size distribution
        ax = axes[0, 1]
        ax.hist(results_df['cohens_d'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0.2, color='red', linestyle='--', label='Small effect')
        ax.axvline(x=0.5, color='orange', linestyle='--', label='Medium effect')
        ax.axvline(x=0.8, color='green', linestyle='--', label='Large effect')
        ax.set_xlabel("Cohen's d")
        ax.set_ylabel('Frequency')
        ax.set_title('Effect Size Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # P-value distribution
        ax = axes[1, 0]
        ax.hist(results_df['p_value'], bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
        ax.axvline(x=0.01, color='orange', linestyle='--', label='α = 0.01')
        ax.set_xlabel('P-value')
        ax.set_ylabel('Frequency')
        ax.set_title('P-value Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary table
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        summary_data = [
            ['Total Experiments', len(results_df)],
            ['Significant (p < 0.05)', sum(results_df['significant'])],
            ['Mean Improvement', f"{results_df['improvement_pct'].mean():.2f}%"],
            ['Max Improvement', f"{results_df['improvement_pct'].max():.2f}%"],
            ['Mean Effect Size', f"{results_df['cohens_d'].mean():.3f}"],
            ['Large Effects (d > 0.8)', sum(results_df['cohens_d'] > 0.8)]
        ]
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax.set_title('Statistical Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_significance.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'statistical_significance.pdf', bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        results_df.to_csv(self.output_dir / 'statistical_analysis.csv', index=False)
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard."""
        # Aggregate data
        agg_df = self.df.groupby(['dataset', 'cache_size_mb', 'policy']).agg({
            'hit_ratio': 'mean',
            'byte_hit_ratio': 'mean',
            'avg_response_time': 'mean',
            'requests_per_second': 'mean'
        }).reset_index()
        
        # Create interactive plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hit Ratio', 'Byte Hit Ratio', 'Response Time (ms)', 'Throughput (RPS)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Color map for policies
        color_map = {policy: POLICY_COLORS.get(policy, f'rgb({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)})') 
                    for policy in agg_df['policy'].unique()}
        
        for policy in agg_df['policy'].unique():
            policy_df = agg_df[agg_df['policy'] == policy]
            
            # Hit ratio
            fig.add_trace(
                go.Scatter(x=policy_df['cache_size_mb'], y=policy_df['hit_ratio'],
                          mode='lines+markers', name=policy, 
                          line=dict(color=color_map[policy]),
                          hovertemplate='Cache Size: %{x}MB<br>Hit Ratio: %{y:.3f}'),
                row=1, col=1
            )
            
            # Byte hit ratio  
            fig.add_trace(
                go.Scatter(x=policy_df['cache_size_mb'], y=policy_df['byte_hit_ratio'],
                          mode='lines+markers', name=policy,
                          line=dict(color=color_map[policy]), showlegend=False,
                          hovertemplate='Cache Size: %{x}MB<br>Byte Hit Ratio: %{y:.3f}'),
                row=1, col=2
            )
            
            # Response time
            fig.add_trace(
                go.Scatter(x=policy_df['cache_size_mb'], y=policy_df['avg_response_time']*1000,
                          mode='lines+markers', name=policy,
                          line=dict(color=color_map[policy]), showlegend=False,
                          hovertemplate='Cache Size: %{x}MB<br>Response Time: %{y:.2f}ms'),
                row=2, col=1
            )
            
            # Throughput
            fig.add_trace(
                go.Scatter(x=policy_df['cache_size_mb'], y=policy_df['requests_per_second'],
                          mode='lines+markers', name=policy,
                          line=dict(color=color_map[policy]), showlegend=False,
                          hovertemplate='Cache Size: %{x}MB<br>Throughput: %{y:.0f} RPS'),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="DRL Cache Benchmark Results - Interactive Dashboard",
            height=800,
            showlegend=True,
            font=dict(size=12)
        )
        
        fig.update_xaxes(title_text="Cache Size (MB)")
        fig.update_yaxes(title_text="Hit Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Byte Hit Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Response Time (ms)", row=2, col=1)
        fig.update_yaxes(title_text="Requests per Second", row=2, col=2)
        
        # Save interactive plot
        fig.write_html(str(self.output_dir / 'interactive_dashboard.html'))
        fig.write_image(str(self.output_dir / 'dashboard_static.png'), width=1200, height=800)
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        # Calculate key metrics
        agg_df = self.df.groupby(['dataset', 'policy']).agg({
            'hit_ratio': ['mean', 'std', 'min', 'max'],
            'byte_hit_ratio': ['mean', 'std', 'min', 'max'],
            'avg_response_time': ['mean', 'std'],
            'requests_per_second': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        agg_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_df.columns]
        
        # Find best performers
        best_hit_ratio = agg_df.loc[agg_df['hit_ratio_mean'].idxmax()]
        best_throughput = agg_df.loc[agg_df['requests_per_second_mean'].idxmax()]
        
        # Generate report
        report = f"""
# DRL Cache Benchmark Results Summary

## Key Findings

### Best Hit Ratio Performance
- **Policy**: {best_hit_ratio['policy']}
- **Dataset**: {best_hit_ratio['dataset']}
- **Hit Ratio**: {best_hit_ratio['hit_ratio_mean']:.3f} ± {best_hit_ratio['hit_ratio_std']:.3f}
- **Byte Hit Ratio**: {best_hit_ratio['byte_hit_ratio_mean']:.3f} ± {best_hit_ratio['byte_hit_ratio_std']:.3f}

### Best Throughput Performance  
- **Policy**: {best_throughput['policy']}
- **Dataset**: {best_throughput['dataset']}
- **Throughput**: {best_throughput['requests_per_second_mean']:.0f} ± {best_throughput['requests_per_second_std']:.0f} RPS

## Performance Summary by Policy

"""
        
        # Add per-policy summary
        for policy in sorted(agg_df['policy'].unique()):
            policy_df = agg_df[agg_df['policy'] == policy]
            avg_hit_ratio = policy_df['hit_ratio_mean'].mean()
            avg_throughput = policy_df['requests_per_second_mean'].mean()
            
            report += f"### {policy}\n"
            report += f"- Average Hit Ratio: {avg_hit_ratio:.3f}\n"
            report += f"- Average Throughput: {avg_throughput:.0f} RPS\n\n"
        
        # Save report
        with open(self.output_dir / 'benchmark_summary_report.md', 'w') as f:
            f.write(report)
        
        print(f"Summary report saved to {self.output_dir / 'benchmark_summary_report.md'}")


def main():
    """Main function for testing visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate benchmark visualizations")
    parser.add_argument("results_file", help="Path to benchmark results CSV/JSON file")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = BenchmarkVisualizer(args.results_file, args.output_dir)
    
    # Generate all plots
    viz.generate_all_plots()
    viz.generate_summary_report()


if __name__ == "__main__":
    main()
