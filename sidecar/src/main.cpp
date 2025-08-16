#include "drl_cache_sidecar.h"
#include <iostream>
#include <getopt.h>
#include <csignal>
#include <unistd.h>

void print_usage(const char* program_name) {
    std::cout << "DRL Cache Sidecar v" << DRL_CACHE_SIDECAR_VERSION << "\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -s, --socket PATH      Unix socket path (default: /tmp/drl-cache.sock)\n";
    std::cout << "  -m, --model PATH       ONNX model path (default: ./models/policy.onnx)\n";
    std::cout << "  -t, --threads NUM      Worker threads (default: 1)\n";
    std::cout << "  -c, --config FILE      Configuration file\n";
    std::cout << "  -d, --daemon           Run as daemon\n";
    std::cout << "  -v, --verbose          Verbose logging\n";
    std::cout << "  -h, --help             Show this help\n";
    std::cout << "\nEnvironment variables:\n";
    std::cout << "  DRL_CACHE_SOCKET       Unix socket path\n";
    std::cout << "  DRL_CACHE_MODEL        ONNX model path\n";
    std::cout << "  DRL_CACHE_THREADS      Number of worker threads\n";
    std::cout << "\nBuilt on " << DRL_CACHE_BUILD_DATE << "\n";
}

void print_banner() {
    std::cout << R"(
╔══════════════════════════════════════╗
║        DRL Cache Sidecar v)" << DRL_CACHE_SIDECAR_VERSION << R"(        ║
║  Reinforcement Learning Cache Policy  ║
║         ONNX Inference Server         ║
╚══════════════════════════════════════╝
)" << std::endl;
}

int main(int argc, char* argv[]) {
    SidecarConfig config;
    bool daemon_mode = false;
    bool verbose = false;
    std::string config_file;
    
    // Parse command line arguments
    static struct option long_options[] = {
        {"socket",   required_argument, 0, 's'},
        {"model",    required_argument, 0, 'm'},
        {"threads",  required_argument, 0, 't'},
        {"config",   required_argument, 0, 'c'},
        {"daemon",   no_argument,       0, 'd'},
        {"verbose",  no_argument,       0, 'v'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int c;
    while ((c = getopt_long(argc, argv, "s:m:t:c:dvh", long_options, nullptr)) != -1) {
        switch (c) {
        case 's':
            config.socket_path = optarg;
            break;
        case 'm':
            config.model_path = optarg;
            break;
        case 't':
            config.num_threads = std::stoi(optarg);
            if (config.num_threads < 1 || config.num_threads > 16) {
                std::cerr << "Error: threads must be between 1 and 16\n";
                return 1;
            }
            break;
        case 'c':
            config_file = optarg;
            break;
        case 'd':
            daemon_mode = true;
            break;
        case 'v':
            verbose = true;
            config.enable_logging = true;
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Load configuration file if specified
    if (!config_file.empty()) {
        if (!config.load_from_file(config_file)) {
            std::cerr << "Error: Failed to load config file: " << config_file << std::endl;
            return 1;
        }
    }
    
    // Override with environment variables
    const char* env_socket = getenv("DRL_CACHE_SOCKET");
    if (env_socket) {
        config.socket_path = env_socket;
    }
    
    const char* env_model = getenv("DRL_CACHE_MODEL");
    if (env_model) {
        config.model_path = env_model;
    }
    
    const char* env_threads = getenv("DRL_CACHE_THREADS");
    if (env_threads) {
        config.num_threads = std::stoi(env_threads);
    }
    
    // Validate configuration
    if (!utils::file_exists(config.model_path)) {
        std::cerr << "Error: Model file not found: " << config.model_path << std::endl;
        return 1;
    }
    
    if (config.socket_path.empty()) {
        std::cerr << "Error: Socket path cannot be empty" << std::endl;
        return 1;
    }
    
    // Daemonize if requested
    if (daemon_mode) {
        if (daemon(0, 0) != 0) {
            std::cerr << "Error: Failed to daemonize" << std::endl;
            return 1;
        }
    }
    
    // Print startup banner
    if (!daemon_mode) {
        print_banner();
        std::cout << "Configuration:" << std::endl;
        config.print();
        std::cout << std::endl;
    }
    
    // Set up signal handling
    SignalHandler::setup();
    
    try {
        // Create and start server
        SidecarServer server(config.socket_path, config.model_path);
        SignalHandler::set_server_instance(&server);
        
        std::cout << "Starting sidecar server..." << std::endl;
        if (!server.start(config.num_threads)) {
            std::cerr << "Error: Failed to start server" << std::endl;
            return 1;
        }
        
        std::cout << "Sidecar server started successfully" << std::endl;
        std::cout << "Listening on: " << config.socket_path << std::endl;
        std::cout << "Model: " << config.model_path << std::endl;
        std::cout << "Threads: " << config.num_threads << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;
        
        // Main event loop
        auto last_stats_time = std::chrono::steady_clock::now();
        const auto stats_interval = std::chrono::seconds(60); // Print stats every 60 seconds
        
        while (server.is_running() && !SignalHandler::should_shutdown()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            // Periodic statistics logging
            auto now = std::chrono::steady_clock::now();
            if (verbose && now - last_stats_time >= stats_interval) {
                const auto& stats = server.get_stats();
                std::cout << "Stats: requests=" << stats.total_requests.load()
                         << " success_rate=" << (stats.get_success_rate() * 100.0) << "%"
                         << " avg_latency=" << stats.get_avg_inference_time_us() << "us"
                         << " max_latency=" << stats.max_inference_time_us.load() << "us"
                         << std::endl;
                last_stats_time = now;
            }
            
            // Check for model updates if hotswap is enabled
            if (config.enable_model_hotswap) {
                server.hot_swap_model();
            }
        }
        
        std::cout << "\nShutting down sidecar server..." << std::endl;
        server.stop();
        
        // Print final statistics
        const auto& final_stats = server.get_stats();
        std::cout << "\nFinal Statistics:" << std::endl;
        std::cout << "  Total requests: " << final_stats.total_requests.load() << std::endl;
        std::cout << "  Success rate: " << (final_stats.get_success_rate() * 100.0) << "%" << std::endl;
        std::cout << "  Average latency: " << final_stats.get_avg_inference_time_us() << " μs" << std::endl;
        std::cout << "  Max latency: " << final_stats.max_inference_time_us.load() << " μs" << std::endl;
        std::cout << "  Model loads: " << final_stats.model_loads.load() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Sidecar server stopped cleanly" << std::endl;
    return 0;
}
