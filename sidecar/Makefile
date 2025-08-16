# DRL Cache Sidecar Makefile

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -pthread
DEBUG_FLAGS = -g -O0 -DDEBUG
RELEASE_FLAGS = -O3 -DNDEBUG -flto

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BIN_DIR = bin
OBJ_DIR = obj

# ONNX Runtime configuration
ONNX_RUNTIME_ROOT ?= /usr/local
ONNX_INCLUDE = $(ONNX_RUNTIME_ROOT)/include
ONNX_LIB = $(ONNX_RUNTIME_ROOT)/lib

# Check for system-installed ONNX Runtime
ifeq ($(wildcard /usr/include/onnxruntime_cxx_api.h),)
    ifeq ($(wildcard $(ONNX_INCLUDE)/onnxruntime_cxx_api.h),)
        $(warning ONNX Runtime not found. Install with: sudo apt-get install libonnxruntime-dev)
        $(warning Or set ONNX_RUNTIME_ROOT to custom installation path)
    endif
endif

# Include paths
INCLUDES = -I$(INCLUDE_DIR) -I$(ONNX_INCLUDE)

# Libraries
LIBS = -lonnxruntime -lpthread

# Library paths
LDFLAGS = -L$(ONNX_LIB)

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
TARGET = $(BIN_DIR)/drl-cache-sidecar

# Default target
all: $(TARGET)

# Create directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(OBJECTS) $(LDFLAGS) $(LIBS) -o $@

# Debug build
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean $(TARGET)

# Release build
release: CXXFLAGS += $(RELEASE_FLAGS)
release: clean $(TARGET)

# Static build (embed dependencies)
static: LDFLAGS += -static-libgcc -static-libstdc++
static: $(TARGET)

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Install system-wide
install: $(TARGET)
	@echo "Installing DRL Cache Sidecar..."
	sudo mkdir -p /usr/local/bin
	sudo cp $(TARGET) /usr/local/bin/
	sudo chmod +x /usr/local/bin/drl-cache-sidecar
	@echo "Installed to /usr/local/bin/drl-cache-sidecar"

# Uninstall
uninstall:
	sudo rm -f /usr/local/bin/drl-cache-sidecar

# Development targets
lint: $(SOURCES)
	@echo "Running static analysis..."
	cppcheck --enable=all --suppress=missingIncludeSystem $(SRC_DIR)/
	@echo "Checking code style..."
	clang-format --dry-run --Werror $(SRC_DIR)/*.cpp $(INCLUDE_DIR)/*.h

format:
	@echo "Formatting source code..."
	clang-format -i $(SRC_DIR)/*.cpp $(INCLUDE_DIR)/*.h

# Performance profiling
profile: CXXFLAGS += -pg -O2
profile: $(TARGET)

# Memory debugging
valgrind: debug
	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all $(TARGET) --help

# Test with dummy model
test: $(TARGET)
	@echo "Testing sidecar binary..."
	$(TARGET) --help
	@echo "Test completed successfully"

# Benchmark inference speed
benchmark: $(TARGET)
	@echo "Running inference benchmark..."
	# This would require a test model and benchmark script
	# ./scripts/benchmark_inference.py

# Check dependencies
deps:
	@echo "Checking dependencies..."
	@echo "C++ Compiler: $(shell $(CXX) --version | head -n1)"
	@echo "ONNX Runtime headers: $(if $(wildcard $(ONNX_INCLUDE)/onnxruntime_cxx_api.h),✓ Found,✗ Missing)"
	@echo "ONNX Runtime library: $(if $(wildcard $(ONNX_LIB)/libonnxruntime.so),✓ Found,$(if $(wildcard $(ONNX_LIB)/libonnxruntime.a),✓ Found (static),✗ Missing))"
	@echo "Pthread support: $(if $(shell echo '#include <pthread.h>' | $(CXX) -x c++ -c - -o /dev/null 2>/dev/null && echo ok),✓ Found,✗ Missing)"

# Install ONNX Runtime (Ubuntu/Debian)
install-onnx-ubuntu:
	@echo "Installing ONNX Runtime for Ubuntu/Debian..."
	sudo apt-get update
	sudo apt-get install -y libonnxruntime-dev

# Install ONNX Runtime (CentOS/RHEL)
install-onnx-centos:
	@echo "Installing ONNX Runtime for CentOS/RHEL..."
	@echo "Manual installation required - download from:"
	@echo "https://github.com/microsoft/onnxruntime/releases"

# Package for distribution
package: release
	@echo "Creating distribution package..."
	mkdir -p dist/drl-cache-sidecar/bin
	mkdir -p dist/drl-cache-sidecar/config
	cp $(TARGET) dist/drl-cache-sidecar/bin/
	cp -r $(INCLUDE_DIR) dist/drl-cache-sidecar/
	echo "DRL Cache Sidecar v$(shell $(TARGET) --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' || echo '1.0.0')" > dist/drl-cache-sidecar/README.txt
	tar czf drl-cache-sidecar-$(shell date +%Y%m%d).tar.gz -C dist drl-cache-sidecar
	rm -rf dist

# Docker build
docker:
	docker build -t drl-cache-sidecar .

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build sidecar (default)"
	@echo "  debug            - Build debug version"
	@echo "  release          - Build optimized version"
	@echo "  static           - Build with static linking"
	@echo "  clean            - Clean build artifacts"
	@echo "  install          - Install system-wide"
	@echo "  uninstall        - Remove system installation"
	@echo "  lint             - Run static analysis"
	@echo "  format           - Format source code"
	@echo "  test             - Test build"
	@echo "  deps             - Check dependencies"
	@echo "  package          - Create distribution package"
	@echo ""
	@echo "Variables:"
	@echo "  ONNX_RUNTIME_ROOT - Path to ONNX Runtime (default: /usr/local)"
	@echo "  CXX              - C++ compiler (default: g++)"

# Show build info
info:
	@echo "DRL Cache Sidecar Build Information:"
	@echo "  Target: $(TARGET)"
	@echo "  Compiler: $(CXX)"
	@echo "  Flags: $(CXXFLAGS)"
	@echo "  ONNX Runtime: $(ONNX_RUNTIME_ROOT)"
	@echo "  Sources: $(SOURCES)"
	@echo "  Objects: $(OBJECTS)"

.PHONY: all debug release static clean install uninstall lint format test benchmark deps \
        install-onnx-ubuntu install-onnx-centos package docker help info
