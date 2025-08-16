# DRL Cache Setup Guide

This guide walks you through the complete setup of DRL Cache, from system requirements to running your first reinforcement learning-optimized cache.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Manual Installation](#manual-installation)
4. [Configuration](#configuration)
5. [Training Your First Model](#training-your-first-model)
6. [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)

## System Requirements

### Hardware Requirements

- **CPU**: x86_64 architecture, 4+ cores recommended
- **Memory**: 8GB+ RAM (16GB recommended for training)
- **Storage**: 20GB+ free space
- **Network**: Stable internet connection for downloading dependencies

### Software Requirements

#### Operating System
- Ubuntu 20.04+ / Debian 11+
- CentOS 8+ / RHEL 8+
- Other Linux distributions (may require manual dependency installation)

#### Core Dependencies
- **NGINX**: 1.18+ (compiled with module support)
- **ONNX Runtime**: 1.12+ 
- **Python**: 3.8+
- **GCC/G++**: 9+ (for compilation)
- **CMake**: 3.16+

#### Optional Dependencies
- **CUDA**: 11.0+ (for GPU training)
- **Docker**: For containerized deployment
- **Grafana**: For advanced monitoring

## Quick Start

The fastest way to get DRL Cache running is using the automated installation script:

### 1. Download and Run Installer

```bash
# Clone the repository
git clone https://github.com/your-org/DRL-Cache.git
cd DRL-Cache

# Run the automated installer (requires sudo)
sudo ./scripts/install.sh
```

The installer will:
- Install system dependencies
- Download and compile NGINX with the DRL Cache module
- Build the ONNX inference sidecar
- Set up system services
- Create a default model
- Configure the Python training environment

### 2. Configure NGINX

Review and customize the generated NGINX configuration:

```bash
# Review the example configuration
sudo nano /etc/nginx/nginx.conf.drl-cache-example

# Backup your current NGINX config
sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup

# Merge DRL Cache configuration with your existing setup
# At minimum, add this line to the main context:
echo "load_module modules/ngx_http_drl_cache_module.so;" | sudo tee -a /etc/nginx/nginx.conf
```

### 3. Start Services

```bash
# Start the DRL Cache sidecar
sudo systemctl start drl-cache-sidecar
sudo systemctl enable drl-cache-sidecar

# Test NGINX configuration and restart
sudo nginx -t
sudo systemctl restart nginx
```

### 4. Verify Installation

```bash
# Check status
sudo ./scripts/drl-cache-ctl.sh status

# Test connectivity
sudo ./scripts/drl-cache-ctl.sh test

# View logs
sudo ./scripts/drl-cache-ctl.sh logs
```

If everything is working, you should see:
- Sidecar service running
- Socket file created at `/run/drl-cache/sidecar.sock`
- NGINX successfully loading the DRL Cache module
- No errors in the logs

## Manual Installation

If you prefer manual installation or need to customize the process:

### 1. Install System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential gcc g++ cmake make wget curl git \
    libpcre3-dev libssl-dev zlib1g-dev libxml2-dev \
    libxslt1-dev libgd-dev libgeoip-dev libonnxruntime-dev \
    python3 python3-pip python3-venv
```

#### CentOS/RHEL:
```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
    gcc gcc-c++ cmake3 make wget curl git \
    pcre-devel openssl-devel zlib-devel libxml2-devel \
    libxslt-devel gd-devel GeoIP-devel \
    python3 python3-pip

# Install ONNX Runtime manually (see docs/DEPENDENCIES.md)
```

### 2. Download NGINX Source

```bash
# Download NGINX source code
NGINX_VERSION=1.24.0
wget http://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz
tar xzf nginx-${NGINX_VERSION}.tar.gz
sudo mv nginx-${NGINX_VERSION} /usr/local/src/
```

### 3. Build NGINX Module

```bash
cd nginx-module
make NGINX_DIR=/usr/local/src/nginx-${NGINX_VERSION} build-dynamic

# Install the module
sudo mkdir -p /etc/nginx/modules
sudo cp objs/ngx_http_drl_cache_module.so /etc/nginx/modules/
```

### 4. Build ONNX Sidecar

```bash
cd sidecar
make release

# Install the binary
sudo cp bin/drl-cache-sidecar /usr/local/bin/
sudo chmod +x /usr/local/bin/drl-cache-sidecar
```

### 5. Set Up System User and Directories

```bash
# Create system user
sudo useradd -r -s /bin/false -c "DRL Cache Service" drl-cache

# Create directories
sudo mkdir -p /run/drl-cache /var/log/drl-cache /etc/drl-cache
sudo mkdir -p /opt/drl-cache/{models,logs,data}
sudo chown -R drl-cache:drl-cache /run/drl-cache /var/log/drl-cache /opt/drl-cache
```

### 6. Install Configuration Files

```bash
# Copy configuration templates
sudo cp config/sidecar.conf /etc/drl-cache/
sudo cp config/nginx.conf /etc/nginx/nginx.conf.drl-cache-example
```

### 7. Create Systemd Service

```bash
sudo tee /etc/systemd/system/drl-cache-sidecar.service << 'EOF'
[Unit]
Description=DRL Cache Sidecar
After=network.target

[Service]
Type=simple
User=drl-cache
Group=drl-cache
ExecStart=/usr/local/bin/drl-cache-sidecar \
    --config /etc/drl-cache/sidecar.conf \
    --socket /run/drl-cache/sidecar.sock \
    --model /opt/drl-cache/models/policy.onnx
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable drl-cache-sidecar
```

## Configuration

### NGINX Configuration

#### Basic Setup

Add the DRL Cache module to your NGINX configuration:

```nginx
# Load the module (add to main context)
load_module modules/ngx_http_drl_cache_module.so;

http {
    # Set up cache path
    proxy_cache_path /var/cache/nginx/drl-cache
                     levels=1:2
                     keys_zone=drl_cache:512m
                     max_size=50g
                     inactive=12h
                     use_temp_path=off;

    # Enable DRL Cache
    drl_cache on;
    drl_cache_k 16;
    drl_cache_socket /run/drl-cache/sidecar.sock;
    drl_cache_timeout 500us;
    drl_cache_fallback lru;

    server {
        location /api/ {
            proxy_cache drl_cache;
            proxy_cache_key "$scheme$request_method$host$request_uri";
            proxy_cache_valid 200 302 10m;
            proxy_cache_use_stale error timeout updating;
            
            proxy_pass http://backend;
        }
    }
}
```

#### Advanced Configuration

For production deployments, consider these additional settings:

```nginx
# Feature selection for ablation studies
drl_cache_feature_mask age,size,hits,iat,ttl,rtt;

# Shadow mode for testing (logs decisions but uses LRU)
drl_cache_shadow on;

# Minimum free space buffer
drl_cache_min_free 1g;

# Debug headers (remove in production)
add_header X-Cache-Status $upstream_cache_status always;
add_header X-DRL-Fallback $drl_cache_fallback always;
```

### Sidecar Configuration

Edit `/etc/drl-cache/sidecar.conf`:

```ini
# Basic settings
socket_path = /run/drl-cache/sidecar.sock
model_path = /opt/drl-cache/models/policy.onnx
num_threads = 1

# Performance tuning
socket_buffer_size = 65536
max_concurrent_requests = 256
use_cpu_only = true

# Model management
enable_model_hotswap = true
model_check_interval_sec = 60

# Logging
enable_logging = true
```

## Training Your First Model

### 1. Set Up Training Environment

```bash
cd training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Access Logs

Ensure your NGINX is logging the required fields:

```nginx
log_format cache_combined '$remote_addr - $remote_user [$time_local] '
                         '"$request" $status $body_bytes_sent '
                         '"$http_referer" "$http_user_agent" '
                         '"$upstream_cache_status" "$cache_key" '
                         '"$upstream_response_time" "$cache_ttl"';

access_log /var/log/nginx/access.log cache_combined;
```

### 3. Configure Training

Edit `config/training.yaml` to match your environment:

```yaml
data:
  log_path: "/var/log/nginx/access.log"
  log_format: "nginx_combined"
  
simulation:
  max_size_gb: 50.0  # Match your cache size
  k_candidates: 16   # Match nginx config
  
training:
  batch_size: 4096
  num_epochs: 50
  learning_rate: 3.0e-4
```

### 4. Start Training

```bash
# Train with default configuration
python src/train.py --log-path /var/log/nginx/access.log

# Or with custom config
python src/train.py --config config/training.yaml --log-path /var/log/nginx/access.log
```

Training will:
- Parse your access logs
- Simulate cache behavior
- Train a dueling DQN model
- Export the model to ONNX format

### 5. Deploy the New Model

```bash
# Update the model (this will hot-swap it)
sudo ./scripts/drl-cache-ctl.sh update-model training/outputs/models/policy.onnx

# Monitor the deployment
sudo ./scripts/drl-cache-ctl.sh logs 50 true
```

## Monitoring and Troubleshooting

### Health Checks

```bash
# Check system status
sudo ./scripts/drl-cache-ctl.sh status

# Test connectivity
sudo ./scripts/drl-cache-ctl.sh test

# View performance metrics
sudo ./scripts/drl-cache-ctl.sh metrics
```

### Log Analysis

#### Sidecar Logs
```bash
# Real-time sidecar logs
sudo tail -f /var/log/drl-cache/sidecar.log

# Search for errors
sudo grep ERROR /var/log/drl-cache/sidecar.log
```

#### NGINX Logs
```bash
# DRL Cache related errors
sudo grep -i "drl\|cache" /var/log/nginx/error.log

# Cache hit/miss analysis
sudo awk '{print $9}' /var/log/nginx/access.log | sort | uniq -c
```

### Common Issues

#### "Socket connection failed"
```bash
# Check socket permissions
ls -la /run/drl-cache/

# Ensure sidecar is running
sudo systemctl status drl-cache-sidecar

# Check socket path in configuration
grep socket_path /etc/drl-cache/sidecar.conf
```

#### "Model file not found"
```bash
# Check model file exists
ls -la /opt/drl-cache/models/policy.onnx

# Create default model if missing
cd /opt/drl-cache/models
python3 -c "
import torch
import torch.nn as nn

class DefaultModel(nn.Module):
    def forward(self, x): return torch.zeros(x.size(0), 32)

model = DefaultModel()
torch.onnx.export(model, torch.randn(1, 192), 'policy.onnx')
print('Default model created')
"
```

#### "NGINX module not loading"
```bash
# Check module exists
ls -la /etc/nginx/modules/ngx_http_drl_cache_module.so

# Test nginx configuration
sudo nginx -t

# Check nginx error logs
sudo tail /var/log/nginx/error.log
```

### Performance Monitoring

#### Cache Performance
```bash
# Hit ratio analysis from access logs
awk '$9=="HIT" {hits++} $9=="MISS" {misses++} END {
  print "Hit ratio:", hits/(hits+misses)*100"%"
}' /var/log/nginx/access.log
```

#### System Resources
```bash
# Monitor sidecar resource usage
top -p $(pgrep drl-cache-sidecar)

# Check memory usage
free -h

# Monitor inference latency
sudo ./scripts/drl-cache-ctl.sh metrics | grep latency
```

### Production Deployment Tips

1. **Start in Shadow Mode**: Use `drl_cache_shadow on` to test decisions without affecting cache behavior
2. **Gradual Rollout**: Deploy to a small percentage of traffic first
3. **Monitor Key Metrics**: Watch hit ratio, latency, and error rates closely
4. **Have Fallback Ready**: Ensure LRU fallback is working correctly
5. **Regular Model Updates**: Retrain models weekly or when traffic patterns change

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand how DRL Cache works
- See [API.md](API.md) for configuration reference
- Check [TRAINING.md](TRAINING.md) for advanced training techniques
- Visit [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed problem resolution

## Getting Help

- **Issues**: Report bugs and issues on GitHub
- **Discussions**: Join our community discussions
- **Documentation**: Full documentation at docs/
- **Support**: Enterprise support available
