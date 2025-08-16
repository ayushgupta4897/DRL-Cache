#include "drl_cache_sidecar.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <sys/socket.h>
#include <cerrno>
#include <csignal>
#include <iostream>

// Signal handling
std::atomic<bool> SignalHandler::shutdown_requested_{false};
SidecarServer* SignalHandler::server_instance_{nullptr};

void SignalHandler::setup() {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::signal(SIGPIPE, SIG_IGN); // Ignore broken pipes
}

void SignalHandler::handle_signal(int sig) {
    const char* signal_name = (sig == SIGINT) ? "SIGINT" : 
                             (sig == SIGTERM) ? "SIGTERM" : "UNKNOWN";
    
    std::cout << "\nReceived " << signal_name << ", initiating graceful shutdown..." << std::endl;
    shutdown_requested_ = true;
    
    if (server_instance_) {
        server_instance_->stop();
    }
}

// Configuration management
bool SidecarConfig::load_from_file(const std::string& config_file) {
    if (!utils::file_exists(config_file)) {
        std::cerr << "Config file not found: " << config_file << std::endl;
        return false;
    }
    
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << config_file << std::endl;
        return false;
    }
    
    std::string line;
    int line_number = 0;
    
    while (std::getline(file, line)) {
        line_number++;
        
        // Skip comments and empty lines
        line = utils::trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Parse key=value pairs
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            std::cerr << "Invalid config line " << line_number << ": " << line << std::endl;
            continue;
        }
        
        std::string key = utils::trim(line.substr(0, eq_pos));
        std::string value = utils::trim(line.substr(eq_pos + 1));
        
        // Remove quotes if present
        if (value.length() >= 2 && 
            ((value[0] == '"' && value.back() == '"') ||
             (value[0] == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.length() - 2);
        }
        
        // Set configuration values
        if (key == "socket_path") {
            socket_path = value;
        } else if (key == "model_path") {
            model_path = value;
        } else if (key == "num_threads") {
            num_threads = std::stoi(value);
        } else if (key == "enable_logging") {
            enable_logging = (value == "true" || value == "1");
        } else if (key == "enable_model_hotswap") {
            enable_model_hotswap = (value == "true" || value == "1");
        } else if (key == "model_check_interval_sec") {
            model_check_interval_sec = std::stoi(value);
        } else if (key == "socket_buffer_size") {
            socket_buffer_size = std::stoi(value);
        } else if (key == "max_concurrent_requests") {
            max_concurrent_requests = std::stoi(value);
        } else if (key == "use_cpu_only") {
            use_cpu_only = (value == "true" || value == "1");
        } else {
            std::cerr << "Unknown config key: " << key << std::endl;
        }
    }
    
    return true;
}

void SidecarConfig::print() const {
    std::cout << "  Socket path: " << socket_path << std::endl;
    std::cout << "  Model path: " << model_path << std::endl;
    std::cout << "  Threads: " << num_threads << std::endl;
    std::cout << "  Logging: " << (enable_logging ? "enabled" : "disabled") << std::endl;
    std::cout << "  Model hotswap: " << (enable_model_hotswap ? "enabled" : "disabled") << std::endl;
    std::cout << "  Socket buffer: " << socket_buffer_size << " bytes" << std::endl;
    std::cout << "  Max concurrent: " << max_concurrent_requests << std::endl;
    std::cout << "  CPU only: " << (use_cpu_only ? "yes" : "no") << std::endl;
}

// Utility functions
namespace utils {

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(trim(token));
    }
    
    return tokens;
}

std::string trim(const std::string& str) {
    const std::string whitespace = " \t\n\r\f\v";
    
    size_t start = str.find_first_not_of(whitespace);
    if (start == std::string::npos) {
        return "";
    }
    
    size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.length() > str.length()) {
        return false;
    }
    
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

bool file_exists(const std::string& path) {
    return std::filesystem::exists(path);
}

std::chrono::system_clock::time_point get_file_mtime(const std::string& path) {
    try {
        auto ftime = std::filesystem::last_write_time(path);
        auto system_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
        return system_time;
    } catch (...) {
        return std::chrono::system_clock::time_point{};
    }
}

size_t get_file_size(const std::string& path) {
    try {
        return std::filesystem::file_size(path);
    } catch (...) {
        return 0;
    }
}

bool set_socket_timeout(int fd, int timeout_ms) {
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == -1) {
        return false;
    }
    
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) == -1) {
        return false;
    }
    
    return true;
}

bool set_socket_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) {
        return false;
    }
    
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK) != -1;
}

bool set_socket_buffer_size(int fd, int size) {
    if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size)) == -1) {
        return false;
    }
    
    if (setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &size, sizeof(size)) == -1) {
        return false;
    }
    
    return true;
}

} // namespace utils
