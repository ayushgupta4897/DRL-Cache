#pragma once

#include <memory>
#include <vector>
#include <string>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdint>
#include <sys/socket.h>
#include <sys/un.h>

// ONNX Runtime includes
#include <onnxruntime_cxx_api.h>

// Configuration constants
constexpr uint32_t DRL_CACHE_IPC_VERSION = 1;
constexpr uint16_t DRL_CACHE_MAX_K = 32;
constexpr uint16_t DRL_CACHE_FEATURE_COUNT = 6;
constexpr size_t DRL_CACHE_SOCKET_PATH_MAX = 256;
constexpr size_t DRL_CACHE_DEFAULT_MODEL_SIZE = 8 * 1024; // 8KB
constexpr int DRL_CACHE_LISTEN_BACKLOG = 128;

// IPC message structures (must match nginx module)
#pragma pack(push, 1)
struct DRLCacheIPCHeader {
    uint32_t version;
    uint16_t k;
    uint16_t feature_dims;
};

struct DRLCacheIPCRequest {
    DRLCacheIPCHeader header;
    float features[DRL_CACHE_MAX_K * DRL_CACHE_FEATURE_COUNT];
};

struct DRLCacheIPCResponse {
    uint32_t eviction_mask;
};
#pragma pack(pop)

// Performance statistics
struct SidecarStats {
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> successful_inferences{0};
    std::atomic<uint64_t> failed_inferences{0};
    std::atomic<uint64_t> total_inference_time_us{0};
    std::atomic<uint64_t> max_inference_time_us{0};
    std::atomic<uint64_t> model_loads{0};
    std::atomic<uint64_t> connection_errors{0};
    
    void reset() {
        total_requests = 0;
        successful_inferences = 0;
        failed_inferences = 0;
        total_inference_time_us = 0;
        max_inference_time_us = 0;
        model_loads = 0;
        connection_errors = 0;
    }
    
    double get_success_rate() const {
        uint64_t total = total_requests.load();
        return total > 0 ? (double)successful_inferences.load() / total : 0.0;
    }
    
    double get_avg_inference_time_us() const {
        uint64_t count = successful_inferences.load();
        return count > 0 ? (double)total_inference_time_us.load() / count : 0.0;
    }
};

// ONNX Model wrapper for dueling DQN inference
class DRLCacheModel {
private:
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    Ort::MemoryInfo memory_info_;
    
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    
    std::string model_path_;
    std::chrono::system_clock::time_point last_modified_;
    
public:
    explicit DRLCacheModel(const std::string& model_path);
    ~DRLCacheModel() = default;
    
    // Load/reload model
    bool load();
    bool reload_if_changed();
    
    // Inference
    bool predict(const float* features, uint16_t k, uint32_t& eviction_mask);
    
    // Model info
    const std::string& get_path() const { return model_path_; }
    bool is_loaded() const { return session_ != nullptr; }
    
private:
    bool load_model_info();
    std::chrono::system_clock::time_point get_file_modified_time() const;
};

// Unix domain socket server
class SidecarServer {
private:
    std::string socket_path_;
    int listen_fd_;
    std::atomic<bool> running_;
    std::vector<std::thread> worker_threads_;
    
    std::unique_ptr<DRLCacheModel> model_;
    SidecarStats stats_;
    
    // Hot-swappable model path
    std::atomic<bool> model_reload_requested_;
    std::string pending_model_path_;
    std::mutex model_mutex_;
    
public:
    explicit SidecarServer(const std::string& socket_path, const std::string& model_path);
    ~SidecarServer();
    
    // Server lifecycle
    bool start(int num_threads = 1);
    void stop();
    bool is_running() const { return running_.load(); }
    
    // Model management
    bool reload_model(const std::string& new_model_path);
    bool hot_swap_model(); // Atomic model swap
    
    // Statistics
    const SidecarStats& get_stats() const { return stats_; }
    void reset_stats() { stats_.reset(); }
    
private:
    bool setup_socket();
    void worker_thread_main();
    bool handle_client(int client_fd);
    bool process_inference_request(const DRLCacheIPCRequest& request, 
                                  DRLCacheIPCResponse& response);
    
    void cleanup_socket();
    
    // Logging
    void log_info(const std::string& message);
    void log_warn(const std::string& message);
    void log_error(const std::string& message);
};

// Configuration management
struct SidecarConfig {
    std::string socket_path = "/tmp/drl-cache.sock";
    std::string model_path = "./models/policy.onnx";
    int num_threads = 1;
    bool enable_logging = true;
    bool enable_model_hotswap = true;
    int model_check_interval_sec = 60;
    
    // Performance tuning
    int socket_buffer_size = 64 * 1024; // 64KB
    int max_concurrent_requests = 256;
    bool use_cpu_only = true;
    
    bool load_from_file(const std::string& config_file);
    void print() const;
};

// Signal handling for graceful shutdown
class SignalHandler {
private:
    static std::atomic<bool> shutdown_requested_;
    static SidecarServer* server_instance_;
    
public:
    static void setup();
    static void handle_signal(int sig);
    static bool should_shutdown() { return shutdown_requested_.load(); }
    static void set_server_instance(SidecarServer* server) { server_instance_ = server; }
};

// Utility functions
namespace utils {
    // Time utilities
    inline uint64_t get_timestamp_us() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();
    }
    
    // String utilities
    std::vector<std::string> split(const std::string& str, char delimiter);
    std::string trim(const std::string& str);
    bool ends_with(const std::string& str, const std::string& suffix);
    
    // File utilities
    bool file_exists(const std::string& path);
    std::chrono::system_clock::time_point get_file_mtime(const std::string& path);
    size_t get_file_size(const std::string& path);
    
    // Network utilities
    bool set_socket_timeout(int fd, int timeout_ms);
    bool set_socket_nonblocking(int fd);
    bool set_socket_buffer_size(int fd, int size);
}

// Version information
constexpr const char* DRL_CACHE_SIDECAR_VERSION = "1.0.0";
constexpr const char* DRL_CACHE_BUILD_DATE = __DATE__ " " __TIME__;

// Feature names for debugging
extern const char* FEATURE_NAMES[DRL_CACHE_FEATURE_COUNT];
