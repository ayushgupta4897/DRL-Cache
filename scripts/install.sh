#!/bin/bash
# DRL Cache Installation Script
# 
# This script automates the installation of DRL Cache components:
# - NGINX module compilation and installation
# - ONNX sidecar compilation and installation  
# - Python training environment setup
# - System service configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NGINX_VERSION=${NGINX_VERSION:-1.24.0}
NGINX_SOURCE_DIR=/usr/local/src/nginx-${NGINX_VERSION}
INSTALL_PREFIX=${INSTALL_PREFIX:-/usr/local}
DRL_CACHE_USER=${DRL_CACHE_USER:-drl-cache}
SIDECAR_SERVICE_NAME=drl-cache-sidecar

# Function definitions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

detect_os() {
    if [[ -f /etc/redhat-release ]]; then
        OS="centos"
        PKG_MANAGER="yum"
    elif [[ -f /etc/lsb-release ]]; then
        OS="ubuntu"
        PKG_MANAGER="apt-get"
    elif [[ -f /etc/debian_version ]]; then
        OS="debian"
        PKG_MANAGER="apt-get"
    else
        log_error "Unsupported operating system"
        exit 1
    fi
    
    log_info "Detected OS: $OS"
}

install_dependencies() {
    log_info "Installing system dependencies..."
    
    if [[ "$PKG_MANAGER" == "apt-get" ]]; then
        apt-get update
        apt-get install -y \
            build-essential \
            gcc \
            g++ \
            cmake \
            make \
            wget \
            curl \
            git \
            libpcre3-dev \
            libssl-dev \
            zlib1g-dev \
            libxml2-dev \
            libxslt1-dev \
            libgd-dev \
            libgeoip-dev \
            libonnxruntime-dev \
            python3 \
            python3-pip \
            python3-venv
    
    elif [[ "$PKG_MANAGER" == "yum" ]]; then
        yum groupinstall -y "Development Tools"
        yum install -y \
            gcc \
            gcc-c++ \
            cmake3 \
            make \
            wget \
            curl \
            git \
            pcre-devel \
            openssl-devel \
            zlib-devel \
            libxml2-devel \
            libxslt-devel \
            gd-devel \
            GeoIP-devel \
            python3 \
            python3-pip
        
        # Install ONNX Runtime manually for CentOS
        if ! command -v onnxruntime >/dev/null 2>&1; then
            log_warn "ONNX Runtime not found in package manager. Installing from source..."
            install_onnx_runtime_centos
        fi
    fi
}

install_onnx_runtime_centos() {
    log_info "Installing ONNX Runtime for CentOS..."
    
    ONNX_VERSION="1.16.3"
    ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
    
    cd /tmp
    wget $ONNX_URL
    tar xzf onnxruntime-linux-x64-${ONNX_VERSION}.tgz
    
    cp -r onnxruntime-linux-x64-${ONNX_VERSION}/include/* /usr/local/include/
    cp -r onnxruntime-linux-x64-${ONNX_VERSION}/lib/* /usr/local/lib/
    
    echo "/usr/local/lib" > /etc/ld.so.conf.d/onnxruntime.conf
    ldconfig
    
    rm -rf /tmp/onnxruntime-*
}

download_nginx_source() {
    log_info "Downloading NGINX source code..."
    
    if [[ ! -d "$NGINX_SOURCE_DIR" ]]; then
        mkdir -p $(dirname "$NGINX_SOURCE_DIR")
        cd $(dirname "$NGINX_SOURCE_DIR")
        
        wget http://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz
        tar xzf nginx-${NGINX_VERSION}.tar.gz
        rm nginx-${NGINX_VERSION}.tar.gz
    else
        log_info "NGINX source already exists: $NGINX_SOURCE_DIR"
    fi
}

build_nginx_module() {
    log_info "Building NGINX module..."
    
    cd $(dirname "$0")/..
    DRL_CACHE_ROOT=$(pwd)
    
    cd nginx-module
    make NGINX_DIR="$NGINX_SOURCE_DIR" build-dynamic
    
    # Install module
    mkdir -p /etc/nginx/modules
    cp objs/ngx_http_drl_cache_module.so /etc/nginx/modules/
    
    log_info "NGINX module installed to /etc/nginx/modules/"
}

build_sidecar() {
    log_info "Building ONNX sidecar..."
    
    cd $(dirname "$0")/../sidecar
    make release
    
    # Install sidecar
    cp bin/drl-cache-sidecar $INSTALL_PREFIX/bin/
    chmod +x $INSTALL_PREFIX/bin/drl-cache-sidecar
    
    log_info "Sidecar installed to $INSTALL_PREFIX/bin/"
}

create_user() {
    log_info "Creating DRL Cache user..."
    
    if ! id "$DRL_CACHE_USER" &>/dev/null; then
        useradd -r -s /bin/false -c "DRL Cache Service" "$DRL_CACHE_USER"
        log_info "Created user: $DRL_CACHE_USER"
    else
        log_info "User already exists: $DRL_CACHE_USER"
    fi
}

create_directories() {
    log_info "Creating directories..."
    
    # Runtime directories
    mkdir -p /run/drl-cache
    mkdir -p /var/log/drl-cache
    mkdir -p /etc/drl-cache
    mkdir -p /opt/drl-cache/{models,logs,data}
    
    # Set ownership
    chown -R $DRL_CACHE_USER:$DRL_CACHE_USER /run/drl-cache
    chown -R $DRL_CACHE_USER:$DRL_CACHE_USER /var/log/drl-cache
    chown -R $DRL_CACHE_USER:$DRL_CACHE_USER /opt/drl-cache
}

install_config_files() {
    log_info "Installing configuration files..."
    
    cd $(dirname "$0")/../config
    
    # Copy configuration files
    cp sidecar.conf /etc/drl-cache/
    cp nginx.conf /etc/nginx/nginx.conf.drl-cache-example
    
    # Adjust permissions
    chmod 644 /etc/drl-cache/sidecar.conf
    chmod 644 /etc/nginx/nginx.conf.drl-cache-example
    
    log_info "Configuration files installed"
    log_warn "Review /etc/nginx/nginx.conf.drl-cache-example and merge with your nginx.conf"
}

create_systemd_service() {
    log_info "Creating systemd service..."
    
    cat > /etc/systemd/system/${SIDECAR_SERVICE_NAME}.service << EOF
[Unit]
Description=DRL Cache Sidecar
After=network.target
Wants=network.target

[Service]
Type=simple
User=$DRL_CACHE_USER
Group=$DRL_CACHE_USER
ExecStart=$INSTALL_PREFIX/bin/drl-cache-sidecar \\
    --config /etc/drl-cache/sidecar.conf \\
    --socket /run/drl-cache/sidecar.sock \\
    --model /opt/drl-cache/models/policy.onnx
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=process
Restart=on-failure
RestartSec=10s

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/run/drl-cache /var/log/drl-cache /opt/drl-cache

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable $SIDECAR_SERVICE_NAME
    
    log_info "Systemd service created: $SIDECAR_SERVICE_NAME"
}

setup_training_environment() {
    log_info "Setting up training environment..."
    
    cd $(dirname "$0")/../training
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Install requirements
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Create sample configuration
    python src/config.py
    
    deactivate
    
    log_info "Training environment setup complete"
    log_info "Activate with: source $(pwd)/venv/bin/activate"
}

create_default_model() {
    log_info "Creating default model..."
    
    # Create a simple default model file
    cat > /opt/drl-cache/models/default_policy.py << 'EOF'
# Default policy for DRL Cache
# This creates a simple random policy as a starting point
import torch
import torch.nn as nn
import numpy as np

class DefaultPolicy(nn.Module):
    def __init__(self, input_dim=6, max_k=32):
        super().__init__()
        self.input_dim = input_dim
        self.max_k = max_k
        
        # Simple linear model
        self.net = nn.Sequential(
            nn.Linear(input_dim * max_k, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_k)
        )
    
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    model = DefaultPolicy()
    dummy_input = torch.randn(1, 6 * 32)  # Batch size 1, max_k=32, 6 features
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "/opt/drl-cache/models/policy.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("Default ONNX model created at /opt/drl-cache/models/policy.onnx")
EOF

    # Create the default ONNX model
    cd /opt/drl-cache/models
    python3 default_policy.py
    chown $DRL_CACHE_USER:$DRL_CACHE_USER policy.onnx
    
    log_info "Default model created"
}

verify_installation() {
    log_info "Verifying installation..."
    
    # Check NGINX module
    if [[ -f /etc/nginx/modules/ngx_http_drl_cache_module.so ]]; then
        log_info "✓ NGINX module installed"
    else
        log_error "✗ NGINX module missing"
        return 1
    fi
    
    # Check sidecar binary
    if [[ -f $INSTALL_PREFIX/bin/drl-cache-sidecar ]]; then
        log_info "✓ Sidecar binary installed"
    else
        log_error "✗ Sidecar binary missing"
        return 1
    fi
    
    # Check ONNX model
    if [[ -f /opt/drl-cache/models/policy.onnx ]]; then
        log_info "✓ Default ONNX model created"
    else
        log_error "✗ ONNX model missing"
        return 1
    fi
    
    # Check systemd service
    if systemctl list-unit-files | grep -q $SIDECAR_SERVICE_NAME; then
        log_info "✓ Systemd service installed"
    else
        log_error "✗ Systemd service missing"
        return 1
    fi
    
    # Test sidecar startup
    log_info "Testing sidecar startup..."
    if $INSTALL_PREFIX/bin/drl-cache-sidecar --help >/dev/null 2>&1; then
        log_info "✓ Sidecar binary functional"
    else
        log_error "✗ Sidecar binary not functional"
        return 1
    fi
    
    log_info "Installation verification complete!"
}

print_post_install_info() {
    log_info "Installation complete! Next steps:"
    echo ""
    echo "1. Configure NGINX:"
    echo "   - Review /etc/nginx/nginx.conf.drl-cache-example"
    echo "   - Add 'load_module modules/ngx_http_drl_cache_module.so;' to nginx.conf"
    echo "   - Configure cache settings and DRL cache directives"
    echo ""
    echo "2. Start the sidecar service:"
    echo "   sudo systemctl start $SIDECAR_SERVICE_NAME"
    echo "   sudo systemctl status $SIDECAR_SERVICE_NAME"
    echo ""
    echo "3. Reload NGINX:"
    echo "   sudo nginx -t  # Test configuration"
    echo "   sudo systemctl reload nginx"
    echo ""
    echo "4. Train a custom model (optional):"
    echo "   cd $(dirname "$0")/../training"
    echo "   source venv/bin/activate"
    echo "   python src/train.py --log-path /var/log/nginx/access.log"
    echo ""
    echo "5. Monitor logs:"
    echo "   sudo tail -f /var/log/drl-cache/sidecar.log"
    echo "   sudo tail -f /var/log/nginx/error.log"
    echo ""
    echo "Configuration files:"
    echo "   - Sidecar: /etc/drl-cache/sidecar.conf"
    echo "   - NGINX example: /etc/nginx/nginx.conf.drl-cache-example"
    echo "   - Model: /opt/drl-cache/models/policy.onnx"
}

# Main installation flow
main() {
    log_info "Starting DRL Cache installation..."
    
    check_root
    detect_os
    
    install_dependencies
    download_nginx_source
    
    build_nginx_module
    build_sidecar
    
    create_user
    create_directories
    install_config_files
    create_systemd_service
    create_default_model
    
    setup_training_environment
    
    verify_installation
    print_post_install_info
    
    log_info "DRL Cache installation completed successfully!"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
