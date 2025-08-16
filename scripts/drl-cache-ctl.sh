#!/bin/bash
# DRL Cache Control Script
# 
# This script provides easy management of DRL Cache components:
# - Start/stop/restart sidecar service
# - Monitor system status and performance
# - Update models and configurations
# - Collect logs and metrics

set -e

# Configuration
SIDECAR_SERVICE="drl-cache-sidecar"
NGINX_SERVICE="nginx"
SOCKET_PATH="/run/drl-cache/sidecar.sock"
MODEL_PATH="/opt/drl-cache/models/policy.onnx"
CONFIG_PATH="/etc/drl-cache/sidecar.conf"
LOG_DIR="/var/log/drl-cache"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

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

check_permissions() {
    if [[ $EUID -ne 0 ]] && [[ "$1" != "status" ]] && [[ "$1" != "logs" ]] && [[ "$1" != "metrics" ]]; then
        log_error "This command requires root privileges"
        exit 1
    fi
}

service_status() {
    local service=$1
    if systemctl is-active --quiet "$service"; then
        echo "running"
    elif systemctl is-enabled --quiet "$service" 2>/dev/null; then
        echo "stopped"
    else
        echo "disabled"
    fi
}

show_status() {
    log_info "DRL Cache System Status"
    echo "=========================="
    
    # Sidecar service status
    sidecar_status=$(service_status "$SIDECAR_SERVICE")
    echo "Sidecar Service:    $sidecar_status"
    
    # NGINX service status  
    nginx_status=$(service_status "$NGINX_SERVICE")
    echo "NGINX Service:      $nginx_status"
    
    # Socket file
    if [[ -S "$SOCKET_PATH" ]]; then
        echo "Socket File:        exists"
    else
        echo "Socket File:        missing"
    fi
    
    # Model file
    if [[ -f "$MODEL_PATH" ]]; then
        model_size=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null || echo "unknown")
        model_date=$(stat -c%y "$MODEL_PATH" 2>/dev/null || stat -f%Sm "$MODEL_PATH" 2>/dev/null || echo "unknown")
        echo "Model File:         exists (${model_size} bytes, ${model_date})"
    else
        echo "Model File:         missing"
    fi
    
    # System resources
    if command -v ps >/dev/null; then
        sidecar_pid=$(pgrep drl-cache-sidecar 2>/dev/null || echo "")
        if [[ -n "$sidecar_pid" ]]; then
            cpu_usage=$(ps -p $sidecar_pid -o %cpu= 2>/dev/null | tr -d ' ' || echo "unknown")
            mem_usage=$(ps -p $sidecar_pid -o %mem= 2>/dev/null | tr -d ' ' || echo "unknown")
            echo "Resource Usage:     CPU: ${cpu_usage}%, Memory: ${mem_usage}%"
        fi
    fi
    
    echo ""
}

start_service() {
    log_info "Starting DRL Cache services..."
    
    # Start sidecar
    systemctl start "$SIDECAR_SERVICE"
    
    # Wait for socket to be ready
    local retries=10
    while [[ $retries -gt 0 ]] && [[ ! -S "$SOCKET_PATH" ]]; do
        sleep 1
        retries=$((retries - 1))
    done
    
    if [[ -S "$SOCKET_PATH" ]]; then
        log_info "Sidecar started successfully"
    else
        log_error "Sidecar failed to create socket"
        return 1
    fi
    
    # Test NGINX configuration
    if nginx -t 2>/dev/null; then
        if [[ "$(service_status "$NGINX_SERVICE")" == "running" ]]; then
            systemctl reload "$NGINX_SERVICE"
            log_info "NGINX configuration reloaded"
        else
            systemctl start "$NGINX_SERVICE"
            log_info "NGINX started"
        fi
    else
        log_warn "NGINX configuration test failed, not restarting NGINX"
    fi
    
    show_status
}

stop_service() {
    log_info "Stopping DRL Cache services..."
    
    # Stop sidecar
    systemctl stop "$SIDECAR_SERVICE" || true
    
    # Remove socket file if it exists
    [[ -S "$SOCKET_PATH" ]] && rm -f "$SOCKET_PATH"
    
    log_info "Services stopped"
    show_status
}

restart_service() {
    log_info "Restarting DRL Cache services..."
    stop_service
    sleep 2
    start_service
}

reload_config() {
    log_info "Reloading configuration..."
    
    # Test configuration
    if ! drl-cache-sidecar --config "$CONFIG_PATH" --test 2>/dev/null; then
        log_warn "Configuration test not available, proceeding anyway"
    fi
    
    # Reload sidecar
    if systemctl is-active --quiet "$SIDECAR_SERVICE"; then
        systemctl reload "$SIDECAR_SERVICE" || systemctl restart "$SIDECAR_SERVICE"
        log_info "Sidecar configuration reloaded"
    fi
    
    # Test and reload NGINX
    if nginx -t 2>/dev/null; then
        systemctl reload "$NGINX_SERVICE"
        log_info "NGINX configuration reloaded"
    else
        log_error "NGINX configuration test failed"
        return 1
    fi
}

update_model() {
    local new_model_path="$1"
    
    if [[ -z "$new_model_path" ]]; then
        log_error "Usage: $0 update-model <path-to-new-model.onnx>"
        return 1
    fi
    
    if [[ ! -f "$new_model_path" ]]; then
        log_error "Model file not found: $new_model_path"
        return 1
    fi
    
    log_info "Updating model..."
    
    # Backup current model
    if [[ -f "$MODEL_PATH" ]]; then
        cp "$MODEL_PATH" "${MODEL_PATH}.backup.$(date +%Y%m%d_%H%M%S)"
        log_info "Current model backed up"
    fi
    
    # Copy new model
    cp "$new_model_path" "$MODEL_PATH"
    chown drl-cache:drl-cache "$MODEL_PATH"
    chmod 644 "$MODEL_PATH"
    
    # Signal sidecar to reload model (if it supports hot reloading)
    if systemctl is-active --quiet "$SIDECAR_SERVICE"; then
        systemctl reload "$SIDECAR_SERVICE" || systemctl restart "$SIDECAR_SERVICE"
        log_info "Model updated and sidecar reloaded"
    else
        log_info "Model updated (sidecar not running)"
    fi
    
    show_status
}

show_logs() {
    local lines="${1:-100}"
    local follow="${2:-false}"
    
    log_info "Showing DRL Cache logs (last $lines lines)"
    echo "========================================="
    
    local tail_opts="-n $lines"
    [[ "$follow" == "true" ]] && tail_opts="$tail_opts -f"
    
    if [[ -f "$LOG_DIR/sidecar.log" ]]; then
        echo "--- Sidecar Logs ---"
        tail $tail_opts "$LOG_DIR/sidecar.log"
    fi
    
    echo ""
    echo "--- NGINX Error Logs (DRL Cache related) ---"
    if [[ -f /var/log/nginx/error.log ]]; then
        tail $tail_opts /var/log/nginx/error.log | grep -i "drl\|cache" || echo "No DRL Cache related errors found"
    fi
    
    echo ""
    echo "--- Systemd Journal ---"
    journalctl -u "$SIDECAR_SERVICE" -n "$lines" --no-pager
}

show_metrics() {
    log_info "DRL Cache Metrics"
    echo "=================="
    
    # Service uptime
    if systemctl is-active --quiet "$SIDECAR_SERVICE"; then
        uptime=$(systemctl show "$SIDECAR_SERVICE" --property=ActiveEnterTimestamp --value)
        echo "Sidecar Uptime:     $(date -d "$uptime" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "unknown")"
    fi
    
    # Socket connections (if netstat/ss available)
    if command -v ss >/dev/null 2>&1; then
        connections=$(ss -x src "$SOCKET_PATH" 2>/dev/null | wc -l)
        echo "Socket Connections: $((connections - 1))"  # Subtract header line
    fi
    
    # Cache statistics from NGINX (if available)
    if command -v curl >/dev/null 2>&1; then
        cache_stats=$(curl -s http://localhost:8080/nginx-status 2>/dev/null | grep "cache" || echo "")
        if [[ -n "$cache_stats" ]]; then
            echo "NGINX Cache Stats:  $cache_stats"
        fi
    fi
    
    # Log analysis
    if [[ -f "$LOG_DIR/sidecar.log" ]]; then
        echo ""
        echo "Recent Activity:"
        echo "---------------"
        
        # Count log entries by level in last 1000 lines
        tail -n 1000 "$LOG_DIR/sidecar.log" | awk '
        /INFO/  { info++ }
        /WARN/  { warn++ }
        /ERROR/ { error++ }
        END {
            printf "  INFO:  %d\n", info+0
            printf "  WARN:  %d\n", warn+0
            printf "  ERROR: %d\n", error+0
        }'
        
        # Show recent errors if any
        recent_errors=$(tail -n 100 "$LOG_DIR/sidecar.log" | grep ERROR | tail -n 3)
        if [[ -n "$recent_errors" ]]; then
            echo ""
            echo "Recent Errors:"
            echo "$recent_errors"
        fi
    fi
}

test_connection() {
    log_info "Testing DRL Cache connectivity..."
    
    # Check socket
    if [[ ! -S "$SOCKET_PATH" ]]; then
        log_error "Socket not found: $SOCKET_PATH"
        return 1
    fi
    
    log_info "✓ Socket file exists"
    
    # Test socket connectivity (basic test)
    if timeout 5 bash -c "</dev/tcp/localhost/80" 2>/dev/null; then
        log_info "✓ Network connectivity OK"
    fi
    
    # Test NGINX configuration
    if nginx -t 2>/dev/null; then
        log_info "✓ NGINX configuration valid"
    else
        log_error "✗ NGINX configuration invalid"
        return 1
    fi
    
    # Test model file
    if [[ -f "$MODEL_PATH" ]]; then
        log_info "✓ Model file exists"
        
        # Basic ONNX model validation (if onnx tools available)
        if command -v python3 >/dev/null && python3 -c "import onnx; onnx.load('$MODEL_PATH')" 2>/dev/null; then
            log_info "✓ Model file is valid ONNX"
        else
            log_warn "? Model validation skipped (onnx package not available)"
        fi
    else
        log_error "✗ Model file missing: $MODEL_PATH"
        return 1
    fi
    
    log_info "Connection test completed successfully"
}

train_model() {
    local log_path="$1"
    local output_dir="${2:-/opt/drl-cache/training}"
    
    if [[ -z "$log_path" ]]; then
        log_error "Usage: $0 train <nginx-access-log-path> [output-dir]"
        return 1
    fi
    
    if [[ ! -f "$log_path" ]]; then
        log_error "Access log file not found: $log_path"
        return 1
    fi
    
    log_info "Starting model training..."
    log_info "Log file: $log_path"
    log_info "Output directory: $output_dir"
    
    # Create output directory
    mkdir -p "$output_dir"
    chown drl-cache:drl-cache "$output_dir"
    
    # Run training (assuming training environment is set up)
    local training_script="/opt/drl-cache/training/src/train.py"
    if [[ -f "$training_script" ]]; then
        cd /opt/drl-cache/training
        
        # Activate virtual environment if it exists
        if [[ -f "venv/bin/activate" ]]; then
            source venv/bin/activate
        fi
        
        python "$training_script" \
            --log-path "$log_path" \
            --output-dir "$output_dir" \
            --config config/training.yaml
        
        # If training was successful, update the model
        local new_model="$output_dir/models/policy.onnx"
        if [[ -f "$new_model" ]]; then
            update_model "$new_model"
        else
            log_warn "Training completed but no ONNX model found at $new_model"
        fi
    else
        log_error "Training script not found: $training_script"
        return 1
    fi
}

show_help() {
    cat << EOF
DRL Cache Control Script

Usage: $0 <command> [options]

Commands:
    status              Show system status
    start               Start DRL Cache services
    stop                Stop DRL Cache services  
    restart             Restart DRL Cache services
    reload              Reload configuration
    update-model <file> Update ONNX model file
    logs [lines] [follow] Show logs (default: 100 lines)
    metrics             Show performance metrics
    test                Test connectivity and configuration
    train <log> [dir]   Train new model from access logs
    help                Show this help message

Examples:
    $0 status
    $0 start
    $0 logs 50 true
    $0 update-model /path/to/new/model.onnx
    $0 train /var/log/nginx/access.log
    
Log locations:
    - Sidecar: $LOG_DIR/sidecar.log
    - NGINX: /var/log/nginx/error.log
    - System: journalctl -u $SIDECAR_SERVICE

EOF
}

# Main command dispatcher
main() {
    local command="$1"
    shift
    
    case "$command" in
        "status")
            show_status
            ;;
        "start")
            check_permissions "$command"
            start_service
            ;;
        "stop")
            check_permissions "$command"
            stop_service
            ;;
        "restart")
            check_permissions "$command"
            restart_service
            ;;
        "reload")
            check_permissions "$command"
            reload_config
            ;;
        "update-model")
            check_permissions "$command"
            update_model "$1"
            ;;
        "logs")
            show_logs "$1" "$2"
            ;;
        "metrics")
            show_metrics
            ;;
        "test")
            test_connection
            ;;
        "train")
            check_permissions "$command"
            train_model "$1" "$2"
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        "")
            log_error "No command specified"
            show_help
            exit 1
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
