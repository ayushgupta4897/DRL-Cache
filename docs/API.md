# DRL Cache API Reference

This document provides a complete reference for all DRL Cache configuration directives, runtime APIs, and integration interfaces.

## Table of Contents

1. [NGINX Configuration Directives](#nginx-configuration-directives)
2. [Sidecar Configuration](#sidecar-configuration)
3. [Training Configuration](#training-configuration)
4. [Runtime APIs](#runtime-apis)
5. [Monitoring Endpoints](#monitoring-endpoints)
6. [IPC Protocol](#ipc-protocol)

## NGINX Configuration Directives

All DRL Cache directives can be used in `http`, `server`, or `location` contexts unless otherwise specified.

### Core Directives

#### `drl_cache`
**Syntax:** `drl_cache on|off;`  
**Default:** `drl_cache off;`  
**Context:** `http`, `server`, `location`

Enables or disables DRL Cache for the current context.

```nginx
http {
    drl_cache on;  # Enable globally
    
    server {
        drl_cache off;  # Disable for this server
        
        location /api/ {
            drl_cache on;  # Re-enable for API endpoints
        }
    }
}
```

#### `drl_cache_k`
**Syntax:** `drl_cache_k number;`  
**Default:** `drl_cache_k 16;`  
**Context:** `http`, `server`, `location`

Sets the number of LRU tail candidates to consider for eviction. Valid range: 1-32.

```nginx
drl_cache_k 16;  # Consider 16 candidates
```

**Performance Impact:**
- Lower values: Faster inference, less context
- Higher values: Slower inference, better decisions
- Recommended: 16 for most workloads, 8 for high-traffic, 32 for analysis

#### `drl_cache_socket`
**Syntax:** `drl_cache_socket path;`  
**Default:** `drl_cache_socket /tmp/drl-cache.sock;`  
**Context:** `http`, `server`, `location`

Specifies the Unix domain socket path for sidecar communication.

```nginx
drl_cache_socket /run/drl-cache/sidecar.sock;
```

**Security Notes:**
- Socket file permissions are managed by the sidecar
- Ensure NGINX worker processes can access the socket path
- Use `/run` or `/tmp` for temporary sockets

#### `drl_cache_timeout`
**Syntax:** `drl_cache_timeout time;`  
**Default:** `drl_cache_timeout 500us;`  
**Context:** `http`, `server`, `location`

Maximum time to wait for ML inference before falling back to LRU.

```nginx
drl_cache_timeout 500us;   # 500 microseconds
drl_cache_timeout 1ms;     # 1 millisecond
```

**Tuning Guidelines:**
- **Latency-critical**: 200-300μs
- **Balanced**: 500μs (default)
- **Throughput-focused**: 1000μs

### Advanced Directives

#### `drl_cache_fallback`
**Syntax:** `drl_cache_fallback lru|fifo|random;`  
**Default:** `drl_cache_fallback lru;`  
**Context:** `http`, `server`, `location`

Fallback eviction strategy when ML inference fails or times out.

```nginx
drl_cache_fallback lru;     # Least Recently Used (recommended)
drl_cache_fallback fifo;    # First In, First Out
drl_cache_fallback random;  # Random eviction
```

#### `drl_cache_shadow`
**Syntax:** `drl_cache_shadow on|off;`  
**Default:** `drl_cache_shadow off;`  
**Context:** `http`, `server`, `location`

Shadow mode: logs ML decisions but uses LRU for actual eviction.

```nginx
drl_cache_shadow on;  # Test mode - log decisions only
```

**Use Cases:**
- **Testing**: Validate ML decisions before production
- **A/B Testing**: Compare ML vs LRU performance
- **Debugging**: Analyze decision patterns

#### `drl_cache_min_free`
**Syntax:** `drl_cache_min_free size;`  
**Default:** `drl_cache_min_free 512m;`  
**Context:** `http`, `server`, `location`

Minimum free space to maintain in cache before triggering eviction.

```nginx
drl_cache_min_free 512m;    # 512 megabytes
drl_cache_min_free 1g;      # 1 gigabyte
```

#### `drl_cache_feature_mask`
**Syntax:** `drl_cache_feature_mask feature1,feature2,...;`  
**Default:** `drl_cache_feature_mask age,size,hits,iat,ttl,rtt;` (all features)  
**Context:** `http`, `server`, `location`

Selectively enable/disable features for ablation studies.

```nginx
# All features (default)
drl_cache_feature_mask age,size,hits,iat,ttl,rtt;

# Size-blind eviction (for testing)
drl_cache_feature_mask age,hits,iat,ttl,rtt;

# Only basic features
drl_cache_feature_mask age,size,hits;
```

**Feature Descriptions:**
- `age`: Time since object creation
- `size`: Object size (log-scaled)
- `hits`: Cache hit count
- `iat`: Inter-arrival time (time since last access)
- `ttl`: Time-to-live remaining
- `rtt`: Last upstream response time

### Debug and Monitoring Directives

#### Debug Headers
Add these headers to responses for debugging (remove in production):

```nginx
# Cache status information
add_header X-Cache-Status $upstream_cache_status always;
add_header X-Cache-Key $cache_key always;

# DRL Cache specific
add_header X-DRL-Enabled $drl_cache_enabled always;
add_header X-DRL-Fallback $drl_cache_fallback always;
add_header X-DRL-Inference-Time $drl_cache_inference_time always;
```

## Sidecar Configuration

The sidecar uses an INI-style configuration file (default: `/etc/drl-cache/sidecar.conf`).

### Basic Settings

```ini
# Socket path for NGINX communication
socket_path = /run/drl-cache/sidecar.sock

# ONNX model file path
model_path = /opt/drl-cache/models/policy.onnx

# Number of worker threads
num_threads = 1
```

### Performance Tuning

```ini
# Socket buffer size (bytes)
socket_buffer_size = 65536

# Maximum concurrent requests
max_concurrent_requests = 256

# Force CPU-only inference (disable GPU)
use_cpu_only = true
```

### Model Management

```ini
# Enable automatic model reloading
enable_model_hotswap = true

# Check for model changes every N seconds
model_check_interval_sec = 60
```

### Logging

```ini
# Enable detailed logging
enable_logging = true

# Debug logging (verbose, only for development)
enable_debug_logging = false

# Enable performance profiling
enable_profiling = false
```

### Command Line Options

The sidecar also accepts command line arguments:

```bash
drl-cache-sidecar [OPTIONS]

Options:
  -s, --socket PATH      Unix socket path
  -m, --model PATH       ONNX model path
  -t, --threads NUM      Worker threads (1-16)
  -c, --config FILE      Configuration file
  -d, --daemon           Run as daemon
  -v, --verbose          Verbose logging
  -h, --help             Show help
```

## Training Configuration

Training configuration uses YAML format. See [Training Guide](TRAINING.md) for detailed explanation.

### Key Configuration Sections

```yaml
# Model architecture
model:
  input_dim: 6
  max_k: 32
  hidden_dim: 256
  
# Training parameters
training:
  learning_rate: 3.0e-4
  batch_size: 4096
  num_epochs: 100
  
# Data processing
data:
  log_path: "/var/log/nginx/access.log"
  log_format: "nginx_combined"
  
# Export settings
export:
  onnx_opset_version: 11
  optimize_model: true
  quantize_int8: true
```

## Runtime APIs

### Control Script API

The `drl-cache-ctl.sh` script provides a management interface:

```bash
# Service management
./drl-cache-ctl.sh start
./drl-cache-ctl.sh stop
./drl-cache-ctl.sh restart
./drl-cache-ctl.sh status

# Configuration management
./drl-cache-ctl.sh reload
./drl-cache-ctl.sh test

# Model management
./drl-cache-ctl.sh update-model /path/to/new/model.onnx

# Monitoring
./drl-cache-ctl.sh logs [lines] [follow]
./drl-cache-ctl.sh metrics

# Training
./drl-cache-ctl.sh train /var/log/nginx/access.log [output-dir]
```

### Systemd Integration

```bash
# Service control
systemctl start drl-cache-sidecar
systemctl stop drl-cache-sidecar
systemctl restart drl-cache-sidecar
systemctl status drl-cache-sidecar

# Enable/disable automatic startup
systemctl enable drl-cache-sidecar
systemctl disable drl-cache-sidecar

# View logs
journalctl -u drl-cache-sidecar -f
```

## Monitoring Endpoints

### NGINX Status Endpoint

Add to your NGINX configuration:

```nginx
server {
    listen 8080;
    server_name localhost;
    
    location = /nginx-status {
        stub_status on;
        access_log off;
        allow 127.0.0.1;
        deny all;
    }
    
    location = /drl-cache-status {
        # Custom DRL cache status
        return 200 '{"status":"ok","module":"drl-cache"}';
        add_header Content-Type application/json;
    }
}
```

### Sidecar Health Check

The sidecar provides a simple health check:

```bash
# Test sidecar connectivity
echo | nc -U /run/drl-cache/sidecar.sock

# Check process status
pgrep -f drl-cache-sidecar
```

### Log Analysis Queries

#### Hit Ratio Analysis
```bash
# Overall hit ratio from NGINX access logs
awk '
$9=="HIT" { hits++ } 
$9=="MISS" { misses++ } 
END { 
    total = hits + misses
    if (total > 0) 
        printf "Hit Ratio: %.2f%% (%d hits, %d misses)\n", 
               hits/total*100, hits, misses 
}' /var/log/nginx/access.log
```

#### Cache Performance by Endpoint
```bash
# Hit ratios by URL path
awk '{
    url = $7; status = $9
    if (status == "HIT") hits[url]++
    else if (status == "MISS") misses[url]++
} 
END {
    for (url in hits) {
        total = hits[url] + misses[url]
        printf "%s: %.1f%% (%d/%d)\n", url, hits[url]/total*100, hits[url], total
    }
}' /var/log/nginx/access.log
```

#### DRL vs LRU Performance
```bash
# Compare ML decisions vs fallback usage
grep -E "(drl_decision|lru_fallback)" /var/log/drl-cache/sidecar.log | \
awk '{print $5}' | sort | uniq -c
```

## IPC Protocol

### Message Format

The communication between NGINX module and sidecar uses a binary protocol over Unix domain sockets.

#### Request Message

```c
struct DRLCacheIPCRequest {
    // Header (8 bytes)
    uint32_t version;        // Protocol version (current: 1)
    uint16_t k;             // Number of candidates (1-32)
    uint16_t feature_dims;  // Features per candidate (6)
    
    // Payload (variable size: k * feature_dims * 4 bytes)
    float features[k * feature_dims];  // Row-major feature matrix
} __attribute__((packed));
```

#### Response Message

```c
struct DRLCacheIPCResponse {
    uint32_t eviction_mask;  // Bitmask: bit i = 1 means evict candidate i
} __attribute__((packed));
```

#### Example

For K=3 candidates:
```
Request:
  version: 1
  k: 3
  feature_dims: 6
  features: [candidate0_features(6), candidate1_features(6), candidate2_features(6)]

Response:
  eviction_mask: 0x00000005  // Binary: 101 -> evict candidates 0 and 2
```

### Error Handling

#### Timeout Behavior
- **Timeout exceeded**: Module falls back to LRU immediately
- **Socket error**: Module logs error and uses LRU fallback
- **Invalid response**: Module discards response and uses LRU

#### Protocol Versioning
- **Version mismatch**: Sidecar rejects request with error
- **Backward compatibility**: Newer sidecars support older protocol versions
- **Feature expansion**: Additional features can be added without breaking compatibility

### Performance Characteristics

#### Typical Message Sizes
- **Request**: 8 + (K × 6 × 4) bytes = 8 + 24K bytes
- **Response**: 4 bytes
- **K=16**: Request=392 bytes, Response=4 bytes

#### Latency Breakdown
```
Component              Typical    p95     p99
Feature extraction     20μs      50μs    100μs
Serialization         10μs      20μs     40μs
Socket write/read     30μs      80μs    200μs  
ONNX inference       120μs     300μs    600μs
Deserialization        5μs      10μs     20μs
Total                185μs     460μs    960μs
```

## Environment Variables

### NGINX Module
```bash
# Override socket path
export DRL_CACHE_SOCKET=/custom/path/socket

# Enable debug mode
export DRL_CACHE_DEBUG=1

# Set default timeout (microseconds)
export DRL_CACHE_TIMEOUT=1000
```

### Sidecar
```bash
# Configuration file path
export DRL_CACHE_CONFIG=/etc/drl-cache/sidecar.conf

# Model file path
export DRL_CACHE_MODEL=/opt/models/policy.onnx

# Number of threads
export DRL_CACHE_THREADS=2

# Enable verbose logging
export DRL_CACHE_VERBOSE=1
```

### Training
```bash
# Training data path
export DRL_TRAINING_DATA=/data/access.log

# Output directory
export DRL_OUTPUT_DIR=/models/training

# GPU device (for training)
export CUDA_VISIBLE_DEVICES=0
```

## Integration Examples

### Docker Compose

```yaml
version: '3.8'
services:
  nginx:
    build: .
    ports:
      - "80:80"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf
      - cache-socket:/tmp
    depends_on:
      - drl-sidecar
      
  drl-sidecar:
    build: ./sidecar
    volumes:
      - ./models:/opt/models
      - ./config/sidecar.conf:/etc/drl-cache/sidecar.conf
      - cache-socket:/tmp
    command: >
      /usr/local/bin/drl-cache-sidecar
      --socket /tmp/drl-cache.sock
      --model /opt/models/policy.onnx

volumes:
  cache-socket:
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: drl-cache-config
data:
  nginx.conf: |
    load_module modules/ngx_http_drl_cache_module.so;
    http {
        drl_cache on;
        drl_cache_socket /tmp/drl-cache.sock;
        # ... rest of config
    }
    
  sidecar.conf: |
    socket_path = /tmp/drl-cache.sock
    model_path = /opt/models/policy.onnx
    num_threads = 1
```

## Troubleshooting API Issues

### Common Error Codes

| Error | Description | Solution |
|-------|-------------|----------|
| `ENOENT` | Socket file not found | Check sidecar is running and socket path |
| `ECONNREFUSED` | Connection refused | Verify socket permissions and sidecar status |
| `ETIMEDOUT` | Request timeout | Increase timeout or check sidecar performance |
| `EINVAL` | Invalid request format | Check protocol version compatibility |

### Debugging Commands

```bash
# Test socket connectivity
echo -e '\x01\x00\x00\x00\x02\x00\x06\x00' | nc -U /run/drl-cache/sidecar.sock | hexdump -C

# Monitor socket traffic
sudo strace -e trace=network -p $(pgrep nginx)

# Check sidecar resource usage
top -p $(pgrep drl-cache-sidecar)

# Validate ONNX model
python -c "import onnx; onnx.checker.check_model(onnx.load('/opt/models/policy.onnx'))"
```

This API reference provides complete documentation for configuring and integrating DRL Cache in production environments.
