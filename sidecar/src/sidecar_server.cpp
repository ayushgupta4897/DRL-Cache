#include "drl_cache_sidecar.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <cerrno>
#include <cstring>
#include <iostream>

SidecarServer::SidecarServer(const std::string& socket_path, const std::string& model_path)
    : socket_path_(socket_path)
    , listen_fd_(-1)
    , running_(false)
    , model_reload_requested_(false)
{
    model_ = std::make_unique<DRLCacheModel>(model_path);
}

SidecarServer::~SidecarServer() {
    stop();
    cleanup_socket();
}

bool SidecarServer::start(int num_threads) {
    if (running_.load()) {
        log_warn("Server already running");
        return false;
    }
    
    // Load the initial model
    if (!model_->load()) {
        log_error("Failed to load initial model");
        return false;
    }
    stats_.model_loads++;
    
    // Set up the Unix domain socket
    if (!setup_socket()) {
        log_error("Failed to set up socket");
        return false;
    }
    
    // Start worker threads
    running_ = true;
    worker_threads_.clear();
    worker_threads_.reserve(num_threads);
    
    for (int i = 0; i < num_threads; ++i) {
        worker_threads_.emplace_back(&SidecarServer::worker_thread_main, this);
    }
    
    log_info("Sidecar server started with " + std::to_string(num_threads) + " worker threads");
    return true;
}

void SidecarServer::stop() {
    if (!running_.load()) {
        return;
    }
    
    log_info("Stopping sidecar server...");
    running_ = false;
    
    // Wake up worker threads by connecting and closing
    if (listen_fd_ != -1) {
        shutdown(listen_fd_, SHUT_RDWR);
    }
    
    // Wait for worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    cleanup_socket();
    log_info("Sidecar server stopped");
}

bool SidecarServer::setup_socket() {
    // Remove existing socket file
    unlink(socket_path_.c_str());
    
    // Create socket
    listen_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd_ == -1) {
        log_error("Failed to create socket: " + std::string(strerror(errno)));
        return false;
    }
    
    // Set socket options
    int opt = 1;
    if (setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == -1) {
        log_warn("Failed to set SO_REUSEADDR");
    }
    
    // Configure socket address
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    
    if (socket_path_.length() >= sizeof(addr.sun_path)) {
        log_error("Socket path too long: " + socket_path_);
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);
    
    // Bind socket
    if (bind(listen_fd_, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        log_error("Failed to bind socket: " + std::string(strerror(errno)));
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    
    // Listen for connections
    if (listen(listen_fd_, DRL_CACHE_LISTEN_BACKLOG) == -1) {
        log_error("Failed to listen on socket: " + std::string(strerror(errno)));
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    
    // Set socket to non-blocking for graceful shutdown
    int flags = fcntl(listen_fd_, F_GETFL, 0);
    if (flags != -1) {
        fcntl(listen_fd_, F_SETFL, flags | O_NONBLOCK);
    }
    
    log_info("Socket listening on: " + socket_path_);
    return true;
}

void SidecarServer::worker_thread_main() {
    log_info("Worker thread started");
    
    while (running_.load()) {
        // Accept connections with timeout
        struct pollfd pfd;
        pfd.fd = listen_fd_;
        pfd.events = POLLIN;
        
        int poll_result = poll(&pfd, 1, 1000); // 1 second timeout
        if (poll_result < 0) {
            if (errno != EINTR) {
                log_error("Poll error: " + std::string(strerror(errno)));
                stats_.connection_errors++;
            }
            continue;
        } else if (poll_result == 0) {
            // Timeout - check for shutdown or model reload
            if (model_reload_requested_.load()) {
                hot_swap_model();
            }
            continue;
        }
        
        // Accept connection
        int client_fd = accept(listen_fd_, nullptr, nullptr);
        if (client_fd == -1) {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                log_error("Accept error: " + std::string(strerror(errno)));
                stats_.connection_errors++;
            }
            continue;
        }
        
        // Set client socket to non-blocking with timeout
        utils::set_socket_nonblocking(client_fd);
        utils::set_socket_timeout(client_fd, 1000); // 1 second timeout
        
        // Handle client request
        bool success = handle_client(client_fd);
        close(client_fd);
        
        if (!success) {
            stats_.connection_errors++;
        }
    }
    
    log_info("Worker thread stopped");
}

bool SidecarServer::handle_client(int client_fd) {
    DRLCacheIPCRequest request;
    DRLCacheIPCResponse response;
    
    auto start_time = utils::get_timestamp_us();
    
    // Receive request
    ssize_t bytes_received = recv(client_fd, &request, sizeof(request), 0);
    if (bytes_received != sizeof(request)) {
        log_warn("Invalid request size: " + std::to_string(bytes_received));
        return false;
    }
    
    stats_.total_requests++;
    
    // Validate request
    if (request.header.version != DRL_CACHE_IPC_VERSION) {
        log_warn("Invalid protocol version: " + std::to_string(request.header.version));
        return false;
    }
    
    if (request.header.k == 0 || request.header.k > DRL_CACHE_MAX_K) {
        log_warn("Invalid K value: " + std::to_string(request.header.k));
        return false;
    }
    
    if (request.header.feature_dims != DRL_CACHE_FEATURE_COUNT) {
        log_warn("Invalid feature dimensions: " + std::to_string(request.header.feature_dims));
        return false;
    }
    
    // Process inference request
    bool inference_success = process_inference_request(request, response);
    
    auto inference_time = utils::get_timestamp_us() - start_time;
    
    if (inference_success) {
        stats_.successful_inferences++;
        stats_.total_inference_time_us += inference_time;
        
        uint64_t current_max = stats_.max_inference_time_us.load();
        while (inference_time > current_max) {
            if (stats_.max_inference_time_us.compare_exchange_weak(current_max, inference_time)) {
                break;
            }
        }
    } else {
        stats_.failed_inferences++;
        // Return empty mask on failure
        response.eviction_mask = 0;
    }
    
    // Send response
    ssize_t bytes_sent = send(client_fd, &response, sizeof(response), 0);
    if (bytes_sent != sizeof(response)) {
        log_warn("Failed to send response");
        return false;
    }
    
    return true;
}

bool SidecarServer::process_inference_request(const DRLCacheIPCRequest& request, 
                                            DRLCacheIPCResponse& response) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    if (!model_ || !model_->is_loaded()) {
        log_error("Model not loaded");
        return false;
    }
    
    // Run inference
    return model_->predict(request.features, request.header.k, response.eviction_mask);
}

bool SidecarServer::reload_model(const std::string& new_model_path) {
    pending_model_path_ = new_model_path;
    model_reload_requested_ = true;
    log_info("Model reload requested: " + new_model_path);
    return true;
}

bool SidecarServer::hot_swap_model() {
    if (!model_reload_requested_.load()) {
        // Check if current model file has changed
        std::lock_guard<std::mutex> lock(model_mutex_);
        if (model_ && model_->is_loaded()) {
            return model_->reload_if_changed();
        }
        return false;
    }
    
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    std::string model_path = pending_model_path_.empty() ? 
        model_->get_path() : pending_model_path_;
    
    // Create new model
    auto new_model = std::make_unique<DRLCacheModel>(model_path);
    if (!new_model->load()) {
        log_error("Failed to load new model: " + model_path);
        model_reload_requested_ = false;
        return false;
    }
    
    // Atomic swap
    model_.swap(new_model);
    stats_.model_loads++;
    
    model_reload_requested_ = false;
    pending_model_path_.clear();
    
    log_info("Model hot-swapped successfully: " + model_path);
    return true;
}

void SidecarServer::cleanup_socket() {
    if (listen_fd_ != -1) {
        close(listen_fd_);
        listen_fd_ = -1;
    }
    
    // Remove socket file
    unlink(socket_path_.c_str());
}

void SidecarServer::log_info(const std::string& message) {
    std::cout << "[INFO] " << message << std::endl;
}

void SidecarServer::log_warn(const std::string& message) {
    std::cerr << "[WARN] " << message << std::endl;
}

void SidecarServer::log_error(const std::string& message) {
    std::cerr << "[ERROR] " << message << std::endl;
}
