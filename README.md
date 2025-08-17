# COLD-RL

This project demonstrates the Cold-RL an approach that applies deep reinforcement learning to cache eviction and achieves and beats the classic algorithms like - LRU, LFU, SizeBased.

## Running the benchmark

```bash
$ cd drl-cache-research-benchmark

$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt

$ ./run_benchmark.py
Test Configuration
=================================================================
Creating dataset: 1.2GB, 25,000 requests
   Object distribution: 60% small, 25% medium, 15% large
      Processing request 0...
      Processing request 7,500...
      Processing request 15,000...
      Processing request 22,500...
   Dataset size: 9.75GB, 1,065 unique objects
   Request distribution:
      large_gem: 22,398 (89.6%)
      medium_mixed: 1,264 (5.1%)
      small_junk: 552 (2.2%)
      small_good: 746 (3.0%)
      large_junk: 40 (0.2%)

Cache: 25MB (Pressure: 399.6x)
------------------------------------------------------------
    LRU                Hit: 0.2447, Time: 0.1s
    LFU                Hit: 0.2995, Time: 0.2s
    SizeBased          Hit: 0.1391, Time: 4.1s
    AdaptiveLRU        Hit: 0.2547, Time: 0.3s
    HybridLRUSize      Hit: 0.2452, Time: 0.2s
    DRL parameters: Learning=0.30, Sensitivity=0.80
    TrapAwareDRL       Hit: 0.3478, Time: 1.6s

   DRL improvement over SizeBased: +150.10%
   SizeBased hit ratio: 0.1391
   DRL hit ratio: 0.3478
   Best baseline: LFU (0.2995)

Cache: 100MB (Pressure: 99.9x)
------------------------------------------------------------
    LRU                Hit: 0.8452, Time: 0.1s
    LFU                Hit: 0.8522, Time: 0.1s
    SizeBased          Hit: 0.7460, Time: 0.6s
    AdaptiveLRU        Hit: 0.8512, Time: 0.1s
    HybridLRUSize      Hit: 0.8453, Time: 0.1s
    DRL parameters: Learning=0.30, Sensitivity=0.80
    TrapAwareDRL       Hit: 0.8360, Time: 0.9s

   DRL improvement over SizeBased: +12.07%
   SizeBased hit ratio: 0.7460
   DRL hit ratio: 0.8360
   Best baseline: LFU (0.8522)

Cache: 400MB (Pressure: 25.0x)
------------------------------------------------------------
    LRU                Hit: 0.9166, Time: 0.1s
    LFU                Hit: 0.9166, Time: 0.1s
    SizeBased          Hit: 0.9166, Time: 0.1s
    AdaptiveLRU        Hit: 0.9166, Time: 0.1s
    HybridLRUSize      Hit: 0.9166, Time: 0.1s
    DRL parameters: Learning=0.30, Sensitivity=0.80
    TrapAwareDRL       Hit: 0.9166, Time: 0.1s

   SizeBased ahead by: 0.00%
   SizeBased hit ratio: 0.9166
   DRL hit ratio: 0.9166
   Best baseline: LRU (0.9166)

Results Summary
==================================================
DRL improvement: +6.0%
Victory rate: 33.3%
--------------------------------------------------
Benchmark completed
Execution time: 22.5 seconds

Performance metrics:
  High pressure: +146% vs SizeBased
  Medium pressure: +15% vs SizeBased
  Average improvement: +5.7%
```


## Getting Started

### Deployment

```bash
$ ./scripts/install.sh
$ ./scripts/drl-cache-ctl.sh start
$ cd training && python src/train.py
```

### Development Setup

```bash
$ cd nginx-module && make
$ cd sidecar && make
$ cd training && pip install -r requirements.txt
```

## Resources

- [Research Benchmark README](drl-cache-research-benchmark/README.md)
- [Architecture Guide](docs/ARCHITECTURE.md) - System design and components
- [Setup Instructions](docs/SETUP.md) - Production deployment guide  
- [Training Guide](docs/TRAINING.md) - Model training and optimization
