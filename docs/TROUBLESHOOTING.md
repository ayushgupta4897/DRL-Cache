# DRL Cache Troubleshooting Guide

This comprehensive troubleshooting guide helps diagnose and resolve common issues with DRL Cache deployment and operation.

## Table of Contents

1. [Quick Diagnostic Commands](#quick-diagnostic-commands)
2. [Installation Issues](#installation-issues)
3. [Runtime Issues](#runtime-issues)
4. [Performance Issues](#performance-issues)
5. [Model Issues](#model-issues)
6. [Configuration Issues](#configuration-issues)
7. [Monitoring and Debugging](#monitoring-and-debugging)

## Quick Diagnostic Commands

Start troubleshooting with these commands to get an overview of system status:

```bash
# Check overall system status
sudo ./scripts/drl-cache-ctl.sh status

# Test all components
sudo ./scripts/drl-cache-ctl.sh test

# View recent logs
sudo ./scripts/drl-cache-ctl.sh logs 100

# Check system resources
sudo ./scripts/drl-cache-ctl.sh metrics
```

## Installation Issues

### NGINX Module Compilation Fails

**Symptoms:**
```
error: 'ngx_http_cache_t' has no member named 'sh'
make: *** [nginx-module] Error 1
```

**Causes & Solutions:**

1. **NGINX version incompatibility**
   ```bash
   # Check NGINX version
   nginx -v
   
   # DRL Cache requires NGINX 1.18+
   # Download compatible version:
   wget http://nginx.org/download/nginx-1.24.0.tar.gz
   ```

2. **Missing development headers**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libpcre3-dev libssl-dev zlib1g-dev
   
   # CentOS/RHEL
   sudo yum install pcre-devel openssl-devel zlib-devel
   ```

3. **NGINX not configured with module support**
   ```bash
   # Reconfigure NGINX with dynamic modules
   cd /usr/local/src/nginx-1.24.0
   ./configure --with-compat --add-dynamic-module=/path/to/DRL-Cache/nginx-module
   make modules
   ```

### ONNX Runtime Not Found

**Symptoms:**
```
fatal error: onnxruntime_cxx_api.h: No such file or directory
```

**Solutions:**

1. **Install ONNX Runtime (Ubuntu/Debian)**
   ```bash
   sudo apt-get update
   sudo apt-get install libonnxruntime-dev
   ```

2. **Manual installation (CentOS/RHEL)**
   ```bash
   # Download and install ONNX Runtime
   ONNX_VERSION="1.16.3"
   wget "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
   tar xzf onnxruntime-linux-x64-${ONNX_VERSION}.tgz
   sudo cp -r onnxruntime-linux-x64-${ONNX_VERSION}/include/* /usr/local/include/
   sudo cp -r onnxruntime-linux-x64-${ONNX_VERSION}/lib/* /usr/local/lib/
   sudo ldconfig
   ```

3. **Custom ONNX Runtime path**
   ```bash
   # Set custom path during compilation
   cd sidecar
   make ONNX_RUNTIME_ROOT=/opt/onnxruntime release
   ```

### Python Dependencies Issues

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement torch>=1.13.0
```

**Solutions:**

1. **Update Python and pip**
   ```bash
   python3 --version  # Ensure 3.8+
   python3 -m pip install --upgrade pip
   ```

2. **Install with specific index (if behind firewall)**
   ```bash
   pip install -i https://pypi.org/simple/ -r requirements.txt
   ```

3. **Use conda instead of pip**
   ```bash
   conda create -n drl-cache python=3.9
   conda activate drl-cache
   conda install pytorch torchvision torchaudio -c pytorch
   pip install -r requirements.txt
   ```

## Runtime Issues

### Sidecar Service Won't Start

**Symptoms:**
```
systemctl status drl-cache-sidecar
● drl-cache-sidecar.service - DRL Cache Sidecar
   Loaded: loaded
   Active: failed (Result: exit-code)
```

**Diagnostic Steps:**

1. **Check detailed service logs**
   ```bash
   journalctl -u drl-cache-sidecar -f --no-pager
   sudo cat /var/log/drl-cache/sidecar.log
   ```

2. **Test sidecar binary manually**
   ```bash
   sudo -u drl-cache /usr/local/bin/drl-cache-sidecar --help
   sudo -u drl-cache /usr/local/bin/drl-cache-sidecar \
       --config /etc/drl-cache/sidecar.conf \
       --socket /tmp/test.sock \
       --model /opt/drl-cache/models/policy.onnx
   ```

**Common Issues & Solutions:**

1. **Permission denied**
   ```bash
   # Check user and permissions
   id drl-cache
   ls -la /run/drl-cache/
   
   # Fix permissions
   sudo chown -R drl-cache:drl-cache /run/drl-cache /var/log/drl-cache
   sudo chmod 755 /run/drl-cache
   ```

2. **Model file missing/corrupted**
   ```bash
   # Check model file
   ls -la /opt/drl-cache/models/policy.onnx
   
   # Validate ONNX model
   python3 -c "import onnx; onnx.checker.check_model(onnx.load('/opt/drl-cache/models/policy.onnx'))"
   
   # Create default model if needed
   sudo ./scripts/drl-cache-ctl.sh train /var/log/nginx/access.log
   ```

3. **Socket path issues**
   ```bash
   # Check socket directory exists and is writable
   ls -ld /run/drl-cache/
   
   # Test socket creation manually
   sudo -u drl-cache touch /run/drl-cache/test.sock
   ```

### NGINX Module Not Loading

**Symptoms:**
```
nginx: [emerg] dlopen() "/etc/nginx/modules/ngx_http_drl_cache_module.so" failed
```

**Diagnostic Steps:**

1. **Check module file**
   ```bash
   ls -la /etc/nginx/modules/ngx_http_drl_cache_module.so
   file /etc/nginx/modules/ngx_http_drl_cache_module.so
   ```

2. **Check NGINX configuration**
   ```bash
   nginx -T | grep -i drl
   nginx -t  # Test configuration syntax
   ```

3. **Check library dependencies**
   ```bash
   ldd /etc/nginx/modules/ngx_http_drl_cache_module.so
   ```

**Solutions:**

1. **Module architecture mismatch**
   ```bash
   # Check NGINX architecture
   nginx -V 2>&1 | grep -o 'configure arguments:.*'
   
   # Recompile module with correct flags
   cd nginx-module
   make clean
   make NGINX_DIR=/usr/local/src/nginx-1.24.0
   ```

2. **Missing runtime libraries**
   ```bash
   # Install missing libraries
   sudo apt-get install libc6-dev  # Ubuntu/Debian
   sudo yum install glibc-devel     # CentOS/RHEL
   ```

### Socket Connection Errors

**Symptoms:**
```
[error] DRL inference failed: Connection refused
[warn] falling back to LRU eviction
```

**Diagnostic Steps:**

1. **Check socket file**
   ```bash
   ls -la /run/drl-cache/sidecar.sock
   file /run/drl-cache/sidecar.sock  # Should show "socket"
   ```

2. **Test socket connectivity**
   ```bash
   # Basic connectivity test
   echo | nc -U /run/drl-cache/sidecar.sock
   
   # Check socket listeners
   sudo ss -xl | grep drl-cache
   ```

3. **Monitor socket traffic**
   ```bash
   # Monitor with strace
   sudo strace -e trace=connect,write,read -p $(pgrep nginx) 2>&1 | grep sock
   ```

**Solutions:**

1. **Socket permission issues**
   ```bash
   # Check socket permissions
   ls -la /run/drl-cache/sidecar.sock
   
   # Fix permissions (socket should be accessible to nginx user)
   sudo chmod 666 /run/drl-cache/sidecar.sock
   ```

2. **Socket path mismatch**
   ```bash
   # Check paths match in both configs
   grep socket_path /etc/drl-cache/sidecar.conf
   grep drl_cache_socket /etc/nginx/nginx.conf
   ```

3. **SELinux blocking connections (CentOS/RHEL)**
   ```bash
   # Check SELinux denials
   sudo sealert -a /var/log/audit/audit.log
   
   # Create SELinux policy (if needed)
   sudo setsebool -P httpd_can_network_connect 1
   ```

## Performance Issues

### High Inference Latency

**Symptoms:**
```
[warn] DRL inference timeout after 500μs, using LRU fallback
drl_cache_fallback_rate: 0.15  # High fallback rate
```

**Diagnostic Steps:**

1. **Check inference timing**
   ```bash
   # Monitor inference latency
   grep "inference_time" /var/log/drl-cache/sidecar.log | tail -20
   
   # Get timing statistics
   sudo ./scripts/drl-cache-ctl.sh metrics | grep latency
   ```

2. **Check system load**
   ```bash
   # CPU and memory usage
   htop -p $(pgrep drl-cache-sidecar)
   
   # I/O wait
   iostat -x 1 5
   ```

**Solutions:**

1. **Increase timeout (temporary fix)**
   ```nginx
   # Increase timeout in nginx.conf
   drl_cache_timeout 1000us;  # 1ms instead of 500μs
   ```

2. **Optimize model size**
   ```bash
   # Check current model size
   ls -lh /opt/drl-cache/models/policy.onnx
   
   # Retrain with quantization
   python src/train.py --quantize-int8 --optimize-model
   ```

3. **Tune sidecar configuration**
   ```ini
   # In sidecar.conf
   num_threads = 2              # Increase threads (but use sparingly)
   socket_buffer_size = 131072  # Increase buffer
   use_cpu_only = true          # Ensure CPU-only for consistency
   ```

### High Memory Usage

**Symptoms:**
```
drl-cache-sidecar: 500MB RSS (expected ~50MB)
```

**Diagnostic Steps:**

1. **Memory analysis**
   ```bash
   # Detailed memory breakdown
   sudo pmap -x $(pgrep drl-cache-sidecar)
   
   # Check for memory leaks
   valgrind --tool=memcheck --leak-check=full drl-cache-sidecar --help
   ```

**Solutions:**

1. **Check model loading**
   ```bash
   # Verify model size is reasonable
   ls -lh /opt/drl-cache/models/policy.onnx  # Should be <10MB
   
   # Check for multiple model instances
   sudo lsof -p $(pgrep drl-cache-sidecar) | grep policy.onnx
   ```

2. **Restart sidecar periodically**
   ```bash
   # Add to crontab for daily restart (if needed)
   echo "0 2 * * * systemctl restart drl-cache-sidecar" | sudo crontab -
   ```

### Cache Hit Ratio Regression

**Symptoms:**
```
Hit ratio dropped from 85% to 65% after DRL Cache deployment
```

**Diagnostic Steps:**

1. **Compare LRU vs DRL performance**
   ```bash
   # Enable shadow mode for comparison
   # In nginx.conf: drl_cache_shadow on;
   
   # Analyze logs after collecting data
   grep -E "(drl_decision|lru_fallback)" /var/log/drl-cache/sidecar.log | \
   awk '{decisions[$3]++} END {for(d in decisions) print d, decisions[d]}'
   ```

2. **Check model training data quality**
   ```bash
   # Verify training data timeframe matches current traffic
   python src/data_pipeline.py --analyze /var/log/nginx/access.log
   ```

**Solutions:**

1. **Retrain model with recent data**
   ```bash
   # Train with last 7 days of logs
   sudo ./scripts/drl-cache-ctl.sh train /var/log/nginx/access.log
   ```

2. **Adjust model parameters**
   ```yaml
   # In training.yaml
   reward:
     size_penalty_lambda: 0.02  # Reduce size penalty
   simulation:
     k_candidates: 8  # Reduce complexity
   ```

## Model Issues

### Model Loading Failures

**Symptoms:**
```
[ERROR] Failed to load ONNX model: Invalid model format
```

**Diagnostic Steps:**

1. **Validate ONNX model**
   ```bash
   # Check model structure
   python3 -c "
   import onnx
   model = onnx.load('/opt/drl-cache/models/policy.onnx')
   print('Model inputs:', [i.name for i in model.graph.input])
   print('Model outputs:', [o.name for o in model.graph.output])
   onnx.checker.check_model(model)
   print('Model is valid')
   "
   ```

2. **Test model inference**
   ```bash
   # Test with dummy data
   python3 -c "
   import onnxruntime as ort
   import numpy as np
   session = ort.InferenceSession('/opt/drl-cache/models/policy.onnx')
   dummy_input = np.random.randn(1, 192).astype(np.float32)
   output = session.run(None, {'input': dummy_input})
   print('Inference successful, output shape:', output[0].shape)
   "
   ```

**Solutions:**

1. **Regenerate model**
   ```bash
   # Retrain and export new model
   cd training
   python src/train.py --export-only --resume models/checkpoints/best_model.pt
   ```

2. **Check ONNX version compatibility**
   ```bash
   # Verify ONNX Runtime version
   python3 -c "import onnxruntime; print(onnxruntime.__version__)"
   
   # Export with compatible opset version
   # In training config: export.onnx_opset_version: 11
   ```

### Model Performance Degradation

**Symptoms:**
```
Model accuracy decreased over time
Increased random-looking eviction decisions
```

**Solutions:**

1. **Implement model retraining schedule**
   ```bash
   # Add weekly retraining cron job
   echo "0 2 * * 0 cd /opt/drl-cache && ./scripts/drl-cache-ctl.sh train /var/log/nginx/access.log" | sudo crontab -
   ```

2. **Monitor model drift**
   ```python
   # Add to monitoring script
   def check_model_drift():
       recent_decisions = parse_recent_logs()
       if decision_entropy(recent_decisions) > THRESHOLD:
           trigger_retraining()
   ```

## Configuration Issues

### Incorrect Cache Behavior

**Symptoms:**
```
Objects being evicted immediately
Cache not respecting TTL settings
Unexpected cache misses
```

**Diagnostic Steps:**

1. **Verify cache configuration**
   ```bash
   # Check cache path and settings
   nginx -T | grep -A 10 proxy_cache_path
   
   # Check disk space
   df -h /var/cache/nginx/
   ```

2. **Analyze cache keys**
   ```bash
   # Check cache key generation
   grep "cache_key" /var/log/nginx/access.log | head -10
   ```

**Solutions:**

1. **Fix cache path permissions**
   ```bash
   sudo chown -R nginx:nginx /var/cache/nginx/
   sudo chmod -R 755 /var/cache/nginx/
   ```

2. **Verify cache key consistency**
   ```nginx
   # Ensure consistent cache key generation
   proxy_cache_key "$scheme$request_method$host$request_uri$is_args$args";
   ```

### Feature Extraction Issues

**Symptoms:**
```
[WARN] Invalid feature values detected
[ERROR] Feature extraction failed for candidate
```

**Solutions:**

1. **Check feature normalization**
   ```bash
   # Verify feature statistics
   python3 -c "
   from training.src.data_pipeline import DataPipeline
   from training.src.config import DataConfig, FeatureConfig
   pipeline = DataPipeline(DataConfig(), FeatureConfig())
   # Analyze feature distributions
   "
   ```

2. **Reset feature statistics**
   ```bash
   # If features seem corrupted, retrain
   sudo ./scripts/drl-cache-ctl.sh train /var/log/nginx/access.log
   ```

## Monitoring and Debugging

### Enable Debug Logging

1. **NGINX debug logging**
   ```nginx
   # Add to nginx.conf
   error_log /var/log/nginx/error.log debug;
   
   # Or specific to DRL cache
   error_log /var/log/nginx/drl_debug.log debug;
   ```

2. **Sidecar verbose logging**
   ```ini
   # In sidecar.conf
   enable_debug_logging = true
   enable_profiling = true
   ```

3. **Module-specific debugging**
   ```bash
   # Set debug environment variable
   export NGX_DEBUG=1
   sudo -E systemctl restart nginx
   ```

### Performance Profiling

1. **Profile sidecar performance**
   ```bash
   # Install profiling tools
   sudo apt-get install linux-tools-generic
   
   # Profile CPU usage
   sudo perf record -p $(pgrep drl-cache-sidecar)
   sudo perf report
   ```

2. **Memory profiling**
   ```bash
   # Use valgrind for memory analysis
   valgrind --tool=massif drl-cache-sidecar --config /etc/drl-cache/sidecar.conf
   ```

### Custom Monitoring Scripts

1. **Hit ratio monitoring**
   ```bash
   #!/bin/bash
   # monitor_hit_ratio.sh
   LOG_FILE="/var/log/nginx/access.log"
   WINDOW_SIZE=1000
   
   tail -n $WINDOW_SIZE $LOG_FILE | awk '
   $9=="HIT" { hits++ }
   $9=="MISS" { misses++ }
   END { 
       total = hits + misses
       if (total > 0)
           printf "Recent hit ratio: %.2f%% (%d/%d)\n", hits/total*100, hits, total
   }'
   ```

2. **Inference latency monitoring**
   ```bash
   #!/bin/bash
   # monitor_inference_latency.sh
   tail -f /var/log/drl-cache/sidecar.log | grep "inference_time" | \
   awk '{print $NF}' | awk '{sum+=$1; count++; if(count%100==0) print "Avg latency:", sum/count "μs"}'
   ```

If you're still experiencing issues after following this guide, please:

1. **Collect diagnostic information:**
   ```bash
   sudo ./scripts/drl-cache-ctl.sh status > debug_info.txt
   sudo ./scripts/drl-cache-ctl.sh logs 500 >> debug_info.txt
   nginx -V >> debug_info.txt 2>&1
   ```

2. **Check the GitHub issues** for similar problems
3. **Report the issue** with the collected diagnostic information

This troubleshooting guide covers the most common issues encountered in DRL Cache deployments. Regular monitoring and proactive maintenance will help prevent many of these issues.
