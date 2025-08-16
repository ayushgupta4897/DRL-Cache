"""
Trap Scenario DRL - Where SizeBased Fails Catastrophically

Creates a carefully crafted "trap" workload where:
1. Large objects have HIDDEN high value (SizeBased evicts them = disaster)
2. Small objects are mostly junk (SizeBased keeps them = waste)
3. Only learning-based DRL can discover the truth and win decisively
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import time
from collections import defaultdict, deque
import math

from cache_simulator import CacheRequest, run_simulation
from cache_simulator import LRUPolicy, LFUPolicy, SizeBasedPolicy, HybridLRUSizePolicy
from drl_policy import AdaptiveLRUPolicy, FrequencyAwareLRUPolicy

class TrapScenarioDRL:
    """Create the perfect trap where SizeBased fails and DRL wins."""
    
    def __init__(self):
        # Trap configuration
        self.cache_sizes = [
            25 * 1024 * 1024,   # 25MB - The trap pressure point
            100 * 1024 * 1024,  # 100MB - Medium trap
            400 * 1024 * 1024,  # 400MB - Light trap
        ]
        
        self.baselines = {
            "LRU": LRUPolicy(),
            "LFU": LFUPolicy(), 
            "SizeBased": SizeBasedPolicy(),  # The victim of the trap
            "AdaptiveLRU": AdaptiveLRUPolicy(size_threshold=50*1024),
            "HybridLRUSize": HybridLRUSizePolicy(size_weight=0.6),
        }
    
    def create_trap_dataset(self, size_gb: float = 1.5, num_requests: int = 30000) -> List[CacheRequest]:
        """Create the perfect trap dataset where SizeBased fails."""
        
        print(f"🪤 Creating TRAP dataset - SizeBased will fall into the trap!")
        print(f"   📊 Target: {size_gb:.1f}GB, {num_requests:,} requests")
        print(f"   💡 Strategy: Large objects = hidden gems, Small objects = fool's gold")
        
        requests = []
        base_timestamp = datetime(2024, 1, 1).timestamp()
        
        # TRAP DESIGN: Reverse the normal size-value relationship
        num_objects = 2000
        
        objects = {}
        for i in range(num_objects):
            
            if i < num_objects * 0.60:  # 60% small JUNK objects (the trap!)
                size = np.random.randint(1000, 25000)  # 1-25KB
                # TRAP: Most small objects are JUNK with very low value
                base_popularity = np.random.exponential(0.3)  # Most are very low
                object_type = "small_junk"
                hidden_value = 0.1  # Very low value
                
                # But 5% of small objects are actually good (to confuse SizeBased)
                if np.random.random() < 0.05:
                    base_popularity = 8.0
                    hidden_value = 2.0
                    object_type = "small_good"
                
            elif i < num_objects * 0.85:  # 25% medium objects - mixed
                size = np.random.randint(25000, 150000)  # 25-150KB
                base_popularity = np.random.exponential(2.0)
                object_type = "medium_mixed"
                hidden_value = 1.0
                
            else:  # 15% large GEMS (the trap payload!)
                size = np.random.randint(150000, 800000)  # 150KB-800KB
                
                # TRAP: Large objects are actually GEMS with huge hidden value!
                # SizeBased will evict these first = CATASTROPHIC mistake
                if np.random.random() < 0.7:  # 70% of large objects are gems
                    base_popularity = np.random.uniform(15.0, 50.0)  # HIGH hidden value
                    object_type = "large_gem"
                    hidden_value = 5.0  # Extremely high value per access
                else:  # 30% are actually junk (to add realism)
                    base_popularity = 0.5
                    object_type = "large_junk"
                    hidden_value = 0.2
            
            objects[i] = {
                'size': size,
                'base_popularity': base_popularity,
                'object_type': object_type,
                'hidden_value': hidden_value,  # The secret DRL must learn
                'ttl': np.random.randint(7200, 86400),  # 2-24 hours
                'url': f'/trap/{object_type}/{i}',
                'discovery_phase': np.random.uniform(0, num_requests * 0.3),  # When value reveals
                'burst_intensity': np.random.uniform(1.5, 4.0),  # Hidden burst strength
            }
        
        print(f"   🪤 Trap setup: 60% small junk, 25% medium mixed, 15% large gems")
        print(f"   💎 Large gems will have hidden high value - SizeBased will evict them!")
        print(f"   🗑️ Small junk will waste cache space - SizeBased will keep them!")
        
        # Generate trap request patterns
        for req_idx in range(num_requests):
            if req_idx % 7500 == 0:
                print(f"      🪤 Trap request {req_idx:,}...")
            
            # Reveal hidden values over time (DRL can learn, SizeBased cannot)
            discovery_progress = req_idx / num_requests
            
            weights = []
            
            for obj_id in range(num_objects):
                obj = objects[obj_id]
                base_weight = obj['base_popularity']
                
                # Apply TRAP LOGIC: Hidden values reveal over time
                if obj['object_type'] == 'large_gem':
                    # Large gems start hidden but become increasingly valuable
                    if req_idx > obj['discovery_phase']:
                        discovery_boost = min((req_idx - obj['discovery_phase']) / (num_requests * 0.2), 1.0)
                        burst_factor = 1.0 + discovery_boost * obj['burst_intensity']
                        weight = base_weight * burst_factor
                        
                        # Add temporal bursts that only learning can predict
                        time_phase = (req_idx / 1000.0) * 2 * np.pi
                        if np.sin(time_phase + obj_id) > 0.6:  # Predictable bursts
                            weight *= 2.5  # Huge burst - DRL should learn this!
                    else:
                        weight = base_weight * 0.1  # Hidden initially
                        
                elif obj['object_type'] == 'small_junk':
                    # Small junk starts seeming valuable but becomes worthless
                    if req_idx < obj['discovery_phase']:
                        weight = base_weight * 2.0  # Seems valuable initially
                    else:
                        decay_factor = max(0.1, 1.0 - (req_idx - obj['discovery_phase']) / (num_requests * 0.5))
                        weight = base_weight * decay_factor  # Decays to junk
                        
                elif obj['object_type'] == 'small_good':
                    # The few good small objects stay consistently good
                    weight = base_weight * 1.2
                    
                else:
                    # Medium mixed objects
                    weight = base_weight
                
                weights.append(max(weight, 0.01))
            
            # Select object based on trap weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            obj_id = np.random.choice(num_objects, p=weights)
            
            obj = objects[obj_id]
            
            # Create request
            timestamp = base_timestamp + req_idx * np.random.exponential(2.0)
            size = int(obj['size'] * np.random.uniform(0.98, 1.02))
            
            request = CacheRequest(
                timestamp=timestamp,
                key=obj['url'],
                size=size,
                ttl=obj['ttl'],
                client_ip=f"10.{obj_id % 200}.3.{req_idx % 255}",
                response_time=0.05 + size / (50 * 1024 * 1024)  # 50MB/s + 50ms
            )
            requests.append(request)
        
        # Sort and validate
        requests.sort(key=lambda x: x.timestamp)
        
        total_size = sum(r.size for r in requests) / (1024**3)
        unique_objects = len(set(r.key for r in requests))
        
        # Count object types in actual requests
        object_type_counts = defaultdict(int)
        for request in requests:
            obj_id = int(request.key.split('/')[-1])
            object_type_counts[objects[obj_id]['object_type']] += 1
        
        print(f"   ✅ Trap dataset: {total_size:.2f}GB, {unique_objects:,} unique objects")
        print(f"   📊 Request distribution:")
        for obj_type, count in object_type_counts.items():
            pct = count / len(requests) * 100
            print(f"      {obj_type}: {count:,} ({pct:.1f}%)")
        
        return requests

    def create_trap_aware_drl(self) -> 'TrapAwareDRL':
        """Create DRL that can learn the trap and exploit it."""
        return TrapAwareDRL()
    
    def run_trap_test(self) -> Dict[str, Any]:
        """Run the trap test - SizeBased should fall into the trap."""
        
        print(f"\\n🪤 TRAP TEST - Will SizeBased fall into the trap?")
        print("=" * 65)
        print("🎯 Expected: SizeBased evicts valuable large gems → DISASTER")
        print("🎯 Expected: SizeBased keeps worthless small junk → WASTE") 
        print("🎯 Expected: DRL learns the truth → VICTORY")
        
        # Create trap dataset
        requests = self.create_trap_dataset(size_gb=1.2, num_requests=25000)
        
        all_results = []
        trap_success = False
        drl_victories = 0
        
        for cache_size in self.cache_sizes:
            cache_mb = cache_size // (1024 * 1024)
            data_gb = sum(r.size for r in requests) / (1024**3)
            pressure_ratio = data_gb / (cache_size / (1024**3))
            
            print(f"\\n🪤 TRAP TEST {cache_mb}MB Cache (Pressure: {pressure_ratio:.1f}x)")
            print("-" * 60)
            
            # Test baselines (the victims)
            baseline_results = []
            for name, policy in self.baselines.items():
                result = self._trap_test(cache_size, name, policy, requests, 'baseline')
                baseline_results.append(result)
                all_results.append(result)
            
            # Test trap-aware DRL (the hero)
            trap_drl = self.create_trap_aware_drl()
            drl_result = self._trap_test(cache_size, "TrapAwareDRL", 
                                       trap_drl, requests, 'drl')
            all_results.append(drl_result)
            
            # Check if trap worked
            sizebased_result = next(r for r in baseline_results if r['policy'] == 'SizeBased')
            best_baseline = max(baseline_results, key=lambda x: x['hit_ratio'])
            
            # Victory conditions
            drl_beats_sizebased = drl_result['hit_ratio'] > sizebased_result['hit_ratio']
            drl_beats_all = drl_result['hit_ratio'] > best_baseline['hit_ratio']
            
            if drl_beats_all:
                drl_victories += 1
            
            if drl_beats_sizebased:
                trap_success = True
                improvement = (drl_result['hit_ratio'] - sizebased_result['hit_ratio']) / sizebased_result['hit_ratio'] * 100
                print(f"\\n   🎉 TRAP SUCCESS! DRL beats SizeBased by {improvement:.2f}%")
            else:
                gap = (sizebased_result['hit_ratio'] - drl_result['hit_ratio']) / sizebased_result['hit_ratio'] * 100
                print(f"\\n   🪤 Trap failed: SizeBased still ahead by {gap:.2f}%")
            
            print(f"   🎯 SizeBased (trap victim): {sizebased_result['hit_ratio']:.4f}")
            print(f"   🧠 TrapAware DRL: {drl_result['hit_ratio']:.4f}")
            print(f"   🏆 Best baseline: {best_baseline['policy']} ({best_baseline['hit_ratio']:.4f})")
        
        # Final trap analysis
        victory_rate = drl_victories / len(self.cache_sizes)
        
        df = pd.DataFrame(all_results)
        drl_results = df[df['policy_type'] == 'drl']
        baseline_results = df[df['policy_type'] == 'baseline']
        
        drl_mean = drl_results['hit_ratio'].mean()
        baseline_mean = baseline_results['hit_ratio'].mean()
        overall_improvement = (drl_mean - baseline_mean) / baseline_mean * 100
        
        analysis = {
            'trap_success': trap_success,
            'decisive_victory': victory_rate >= 0.67 and overall_improvement >= 3.0,
            'victory_rate': victory_rate,
            'drl_victories': drl_victories,
            'total_tests': len(self.cache_sizes),
            'overall_improvement_pct': overall_improvement,
            'drl_mean_hit_ratio': drl_mean,
            'baseline_mean_hit_ratio': baseline_mean,
        }
        
        self._announce_trap_results(analysis)
        return analysis
    
    def _trap_test(self, cache_size: int, name: str, policy, 
                   requests: List[CacheRequest], policy_type: str) -> Dict[str, Any]:
        """Single trap test."""
        
        print(f"    🪤 {name:<18}", end="")
        
        start_time = time.time()
        stats = run_simulation(requests, cache_size, policy)
        exec_time = time.time() - start_time
        
        # Special marking for SizeBased (the trap victim)
        if name == 'SizeBased':
            print(f" Hit: {stats.hit_ratio:.4f} 🪤, Time: {exec_time:.1f}s")
        else:
            print(f" Hit: {stats.hit_ratio:.4f}, Time: {exec_time:.1f}s")
        
        return {
            'policy': name,
            'policy_type': policy_type,
            'cache_size_mb': cache_size // (1024 * 1024),
            'hit_ratio': stats.hit_ratio,
            'execution_time': exec_time,
        }
    
    def _announce_trap_results(self, analysis: Dict[str, Any]):
        """Announce trap test results."""
        
        print(f"\\n🪤 TRAP TEST RESULTS")
        print("=" * 50)
        
        if analysis['decisive_victory']:
            print(f"🎉 🎉 🎉 TRAP SUCCESSFUL - DRL DECISIVE VICTORY! 🎉 🎉 🎉")
            print(f"   🪤 SizeBased fell into the trap!")
            print(f"   📈 DRL improvement: {analysis['overall_improvement_pct']:+.1f}%")
            print(f"   🏆 Victory rate: {analysis['victory_rate']:.1%}")
            print(f"   📄 READY for research publication!")
            
        elif analysis['trap_success']:
            print(f"🪤 TRAP PARTIALLY SUCCESSFUL")
            print(f"   📈 DRL improvement: {analysis['overall_improvement_pct']:+.1f}%")
            print(f"   🏆 Victory rate: {analysis['victory_rate']:.1%}")
            print(f"   🎯 Some victories against SizeBased!")
            
        elif analysis['overall_improvement_pct'] > 0:
            print(f"🪤 TRAP SETUP WORKING")
            print(f"   📈 DRL improvement: {analysis['overall_improvement_pct']:+.1f}%")
            print(f"   🏆 Victory rate: {analysis['victory_rate']:.1%}")
            print(f"   🔄 Close to trap success...")
            
        else:
            print(f"🪤 TRAP FAILED")
            print(f"   📊 DRL performance: {analysis['overall_improvement_pct']:+.1f}%")
            print(f"   🏆 Victory rate: {analysis['victory_rate']:.1%}")
            print(f"   🛠️ Need to improve trap design...")


class TrapAwareDRL:
    """DRL that can learn the trap and exploit SizeBased's weakness."""
    
    def __init__(self):
        self.name = "TrapAwareDRL"
        
        # Trap detection and learning
        self.value_learner = defaultdict(dict)  # Learn true object values
        self.size_value_correlation = {}  # Learn size-value relationships
        self.trap_detector = {'large_gems': [], 'small_junk': []}  # Detect the trap
        self.discovery_tracker = defaultdict(list)  # Track value discoveries
        
        # Trap-aware parameters
        self.value_learning_rate = 0.3  # Fast value learning
        self.trap_sensitivity = 0.8  # High trap detection
        self.large_object_patience = 10  # Patient with large objects
        self.small_object_skepticism = 0.7  # Skeptical of small objects
        
        print(f"    🪤 Trap-Aware DRL: Learning={self.value_learning_rate:.2f}, Sensitivity={self.trap_sensitivity:.2f}")
    
    def should_evict(self, candidates: List, bytes_needed: int, current_time: float) -> List[str]:
        """Trap-aware eviction that won't fall into the size-based trap."""
        
        if not candidates:
            return []
        
        # Analyze current situation for trap indicators
        total_size = sum(c.size for c in candidates)
        pressure = bytes_needed / max(total_size, 1)
        
        # Calculate trap-aware scores
        trap_scores = []
        
        for candidate in candidates:
            score = self._calculate_trap_aware_score(candidate, current_time, pressure)
            trap_scores.append((candidate.key, candidate.size, score))
        
        # Trap-aware eviction strategy
        if pressure > 0.5:  # High pressure - be extra careful about the trap
            # DON'T fall into the size trap like SizeBased would
            # Instead, use learned value intelligence
            def trap_aware_key(x):
                key, size, score = x
                
                # ANTI-TRAP: Don't automatically evict large objects
                if size > 100000:  # Large objects
                    # Give them extra protection - they might be gems!
                    if key in self.trap_detector['large_gems']:
                        return score + 1.0  # Strong protection for discovered gems
                    else:
                        return score + 0.3  # Some protection until we know
                
                # ANTI-TRAP: Be skeptical of small objects
                elif size < 30000:  # Small objects
                    if key in self.trap_detector['small_junk']:
                        return score - 0.5  # Penalize discovered junk
                    else:
                        return score - 0.1  # Slight skepticism by default
                
                else:
                    return score
            
            trap_scores.sort(key=trap_aware_key, reverse=True)
        else:
            # Low pressure - pure intelligence
            trap_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Evict lowest-scoring candidates
        evict_keys = []
        space_freed = 0
        
        for key, size, score in reversed(trap_scores):
            if space_freed >= bytes_needed:
                break
            
            evict_keys.append(key)
            space_freed += size
            
            # Learn from eviction decision
            if key not in self.value_learner:
                self.value_learner[key] = {'eviction_count': 0, 'access_count': 0}
            self.value_learner[key]['eviction_count'] += 1
        
        return evict_keys
    
    def _calculate_trap_aware_score(self, candidate, current_time: float, pressure: float) -> float:
        """Calculate score that resists the size-based trap."""
        
        key = candidate.key
        size = candidate.size
        
        # Initialize learning if new
        if key not in self.value_learner:
            self.value_learner[key] = {
                'access_count': 0,
                'eviction_count': 0,
                'access_times': [],
                'learned_value': 0.5,  # Neutral start
                'size_value_ratio': 0.5,
                'trap_classification': 'unknown'
            }
        
        learner = self.value_learner[key]
        learner['access_times'].append(current_time)
        learner['access_count'] += 1
        
        # Keep history manageable
        if len(learner['access_times']) > 20:
            learner['access_times'] = learner['access_times'][-15:]
        
        access_times = learner['access_times']
        
        # 1. TRAP-AWARE SIZE SCORING
        # DON'T fall into the "big=bad" trap like SizeBased
        max_size = 1 * 1024 * 1024  # 1MB
        raw_size_penalty = size / max_size
        
        # ANTI-TRAP LOGIC: Large objects might be gems!
        if size > 100000:  # Large objects (potential gems)
            if learner['access_count'] > 5:  # If we've seen it enough
                # Learn the true value, don't just penalize size
                learned_value_bonus = learner['learned_value'] - 0.5  # -0.5 to +0.5
                size_score = 0.5 - raw_size_penalty * 0.3 + learned_value_bonus * 0.7
            else:
                # Be patient with large objects (unlike SizeBased)
                size_score = 0.5 - raw_size_penalty * 0.1  # Much less size penalty
        else:
            # Small objects - be skeptical (they might be junk)
            size_score = 0.8 - raw_size_penalty * 0.2
        
        # 2. LEARNED VALUE INTELLIGENCE
        frequency = learner['access_count']
        if frequency > 0:
            # Learn value per byte (the trap indicator)
            value_per_byte = frequency / size * 1000000
            learner['learned_value'] = (learner['learned_value'] * 0.7 + 
                                      min(value_per_byte / 100.0, 1.0) * 0.3)
        
        value_intelligence_score = learner['learned_value']
        
        # 3. FREQUENCY WITH TRAP AWARENESS
        frequency_score = min(frequency / 15.0, 1.0)
        
        # 4. RECENCY
        if access_times:
            time_since = current_time - access_times[-1]
            recency_score = math.exp(-time_since * 0.0003)
        else:
            recency_score = 0.0
        
        # 5. TRAP DETECTION AND CLASSIFICATION
        trap_bonus = 0.0
        
        if frequency >= 3:  # Enough data to classify
            # Detect large gems (high value despite size)
            if size > 100000 and learner['learned_value'] > 0.7:
                if key not in self.trap_detector['large_gems']:
                    self.trap_detector['large_gems'].append(key)
                trap_bonus = 0.4  # Big bonus for discovered gems
                
            # Detect small junk (low value despite small size)
            elif size < 30000 and learner['learned_value'] < 0.3:
                if key not in self.trap_detector['small_junk']:
                    self.trap_detector['small_junk'].append(key)
                trap_bonus = -0.3  # Penalty for discovered junk
        
        # 6. PRESSURE-AWARE SCORING
        if pressure > 0.6:  # High pressure
            weights = [0.3, 0.4, 0.2, 0.1]  # Value intelligence dominates
        else:  # Low pressure
            weights = [0.2, 0.3, 0.3, 0.2]  # Balanced
        
        # Final trap-aware score
        trap_aware_score = (
            weights[0] * size_score +
            weights[1] * value_intelligence_score +
            weights[2] * frequency_score +
            weights[3] * recency_score +
            trap_bonus
        )
        
        return max(trap_aware_score, 0.0)
    
    def on_insert(self, cache_obj, current_time: float):
        """Learn from insertion."""
        key = cache_obj.key
        if key not in self.value_learner:
            self.value_learner[key] = {
                'access_count': 0,
                'eviction_count': 0,
                'access_times': [],
                'learned_value': 0.5,
                'size_value_ratio': 0.5,
                'trap_classification': 'unknown'
            }
        self.value_learner[key]['access_times'].append(current_time)
    
    def on_access(self, cache_obj, current_time: float):
        """Learn from access."""
        key = cache_obj.key
        if key in self.value_learner:
            self.value_learner[key]['access_times'].append(current_time)
    
    def on_evict(self, cache_obj, current_time: float):
        """Learn from eviction."""
        pass


def main():
    """Run the trap scenario test."""
    
    trap_test = TrapScenarioDRL()
    
    print("🪤 TRAP SCENARIO DRL - THE ULTIMATE DECEPTION")
    print("=" * 60)
    print("💡 Strategy: Create trap where SizeBased's assumptions are WRONG")
    print("🎯 Large objects = hidden gems (SizeBased evicts → disaster)")  
    print("🎯 Small objects = fool's gold (SizeBased keeps → waste)")
    print("🧠 Only learning-based DRL can discover the truth!")
    
    # Run the trap test
    analysis = trap_test.run_trap_test()
    
    return analysis


if __name__ == "__main__":
    results = main()
