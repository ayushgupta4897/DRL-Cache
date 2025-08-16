#include "drl_cache_sidecar.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>

const char* FEATURE_NAMES[DRL_CACHE_FEATURE_COUNT] = {
    "age_sec",
    "size_kb", 
    "hit_count",
    "inter_arrival_dt",
    "ttl_left_sec",
    "last_origin_rtt_us"
};

DRLCacheModel::DRLCacheModel(const std::string& model_path)
    : model_path_(model_path)
    , memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , input_shape_{1, DRL_CACHE_MAX_K * DRL_CACHE_FEATURE_COUNT}
    , output_shape_{1, DRL_CACHE_MAX_K}
{
    // Initialize ONNX Runtime environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DRLCacheSidecar");
}

bool DRLCacheModel::load() {
    try {
        std::cout << "Loading ONNX model: " << model_path_ << std::endl;
        
        // Check if model file exists
        if (!utils::file_exists(model_path_)) {
            std::cerr << "Error: Model file does not exist: " << model_path_ << std::endl;
            return false;
        }
        
        // Get file modification time
        last_modified_ = get_file_modified_time();
        
        // Create session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1); // Single-threaded for low latency
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        
        // Disable logging for performance
        session_options.SetLogSeverityLevel(3); // Only errors
        
        // Load the model
        session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), session_options);
        
        // Load model metadata
        if (!load_model_info()) {
            std::cerr << "Error: Failed to load model info" << std::endl;
            session_.reset();
            return false;
        }
        
        std::cout << "Model loaded successfully" << std::endl;
        std::cout << "  Input shape: [" << input_shape_[0] << ", " << input_shape_[1] << "]" << std::endl;
        std::cout << "  Output shape: [" << output_shape_[0] << ", " << output_shape_[1] << "]" << std::endl;
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        session_.reset();
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        session_.reset();
        return false;
    }
}

bool DRLCacheModel::load_model_info() {
    if (!session_) {
        return false;
    }
    
    try {
        // Get input info
        size_t num_inputs = session_->GetInputCount();
        if (num_inputs != 1) {
            std::cerr << "Expected 1 input, got " << num_inputs << std::endl;
            return false;
        }
        
        // Clear previous names
        for (auto* name : input_names_) {
            delete[] name;
        }
        input_names_.clear();
        
        // Get input name
        Ort::AllocatorWithDefaultOptions allocator;
        char* input_name = session_->GetInputName(0, allocator);
        char* input_name_copy = new char[strlen(input_name) + 1];
        strcpy(input_name_copy, input_name);
        input_names_.push_back(input_name_copy);
        
        // Get input shape
        auto input_type_info = session_->GetInputTypeInfo(0);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = tensor_info.GetShape();
        
        // Validate input shape
        if (input_shape_.size() != 2) {
            std::cerr << "Expected 2D input tensor, got " << input_shape_.size() << "D" << std::endl;
            return false;
        }
        
        // Allow batch size to be dynamic (-1) or 1
        if (input_shape_[0] != -1 && input_shape_[0] != 1) {
            std::cerr << "Expected batch size 1 or dynamic, got " << input_shape_[0] << std::endl;
            return false;
        }
        input_shape_[0] = 1; // Set to 1 for inference
        
        // Validate feature dimension
        int64_t expected_features = DRL_CACHE_MAX_K * DRL_CACHE_FEATURE_COUNT;
        if (input_shape_[1] != expected_features) {
            std::cerr << "Expected " << expected_features << " features, got " << input_shape_[1] << std::endl;
            return false;
        }
        
        // Get output info
        size_t num_outputs = session_->GetOutputCount();
        if (num_outputs != 1) {
            std::cerr << "Expected 1 output, got " << num_outputs << std::endl;
            return false;
        }
        
        // Clear previous names
        for (auto* name : output_names_) {
            delete[] name;
        }
        output_names_.clear();
        
        // Get output name
        char* output_name = session_->GetOutputName(0, allocator);
        char* output_name_copy = new char[strlen(output_name) + 1];
        strcpy(output_name_copy, output_name);
        output_names_.push_back(output_name_copy);
        
        // Get output shape
        auto output_type_info = session_->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        output_shape_ = output_tensor_info.GetShape();
        
        // Validate output shape
        if (output_shape_.size() != 2) {
            std::cerr << "Expected 2D output tensor, got " << output_shape_.size() << "D" << std::endl;
            return false;
        }
        
        if (output_shape_[0] != -1 && output_shape_[0] != 1) {
            std::cerr << "Expected output batch size 1 or dynamic, got " << output_shape_[0] << std::endl;
            return false;
        }
        output_shape_[0] = 1; // Set to 1 for inference
        
        // Output should have K values (Q-values for each candidate)
        if (output_shape_[1] != DRL_CACHE_MAX_K) {
            std::cerr << "Expected " << DRL_CACHE_MAX_K << " output values, got " << output_shape_[1] << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Error getting model info: " << e.what() << std::endl;
        return false;
    }
}

bool DRLCacheModel::predict(const float* features, uint16_t k, uint32_t& eviction_mask) {
    if (!session_ || !features || k == 0 || k > DRL_CACHE_MAX_K) {
        return false;
    }
    
    try {
        // Prepare input tensor - pad with zeros if k < MAX_K
        std::vector<float> input_data(DRL_CACHE_MAX_K * DRL_CACHE_FEATURE_COUNT, 0.0f);
        
        // Copy actual features
        size_t copy_size = k * DRL_CACHE_FEATURE_COUNT * sizeof(float);
        memcpy(input_data.data(), features, copy_size);
        
        // Create input tensor
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            input_data.data(),
            input_data.size(),
            input_shape_.data(),
            input_shape_.size()
        );
        
        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            1,
            output_names_.data(),
            1
        );
        
        // Extract output
        if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
            return false;
        }
        
        auto& output_tensor = output_tensors[0];
        const float* output_data = output_tensor.GetTensorData<float>();
        
        // Convert Q-values to eviction mask
        // For dueling DQN, we get Q-values for eviction action
        // Higher Q-value means more likely to evict
        eviction_mask = 0;
        
        for (uint16_t i = 0; i < k; ++i) {
            // Use threshold of 0.0 - positive Q-value means evict
            if (output_data[i] > 0.0f) {
                eviction_mask |= (1U << i);
            }
        }
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX inference error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return false;
    }
}

bool DRLCacheModel::reload_if_changed() {
    if (model_path_.empty()) {
        return false;
    }
    
    try {
        auto current_time = get_file_modified_time();
        if (current_time != last_modified_) {
            std::cout << "Model file changed, reloading..." << std::endl;
            return load();
        }
        return true;
    } catch (...) {
        return false;
    }
}

std::chrono::system_clock::time_point DRLCacheModel::get_file_modified_time() const {
    try {
        auto ftime = std::filesystem::last_write_time(model_path_);
        auto system_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
        return system_time;
    } catch (...) {
        return std::chrono::system_clock::time_point{};
    }
}
