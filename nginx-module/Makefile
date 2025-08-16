# NGINX DRL Cache Module Makefile

# Default NGINX source directory (can be overridden)
NGINX_DIR ?= /usr/local/src/nginx

# Module source files
MODULE_SRCS = src/ngx_http_drl_cache_module.c \
             src/drl_cache_features.c \
             src/drl_cache_ipc.c

MODULE_DEPS = src/ngx_http_drl_cache_module.h

# Build targets
.PHONY: all clean install configure build-dynamic build-static

all: build-dynamic

# Configure NGINX to build as dynamic module
configure:
	@if [ ! -d "$(NGINX_DIR)" ]; then \
		echo "Error: NGINX_DIR $(NGINX_DIR) not found"; \
		echo "Set NGINX_DIR to your nginx source directory"; \
		exit 1; \
	fi
	@echo "Configuring nginx with DRL cache module..."
	cd $(NGINX_DIR) && ./configure \
		--add-dynamic-module=$(PWD) \
		--with-http_realip_module \
		--with-http_stub_status_module \
		--with-http_auth_request_module \
		--with-http_secure_link_module \
		--with-threads \
		--with-file-aio

# Build dynamic module
build-dynamic: configure
	@echo "Building dynamic module..."
	cd $(NGINX_DIR) && make modules
	@echo "Copying module to objs/"
	mkdir -p objs
	cp $(NGINX_DIR)/objs/ngx_http_drl_cache_module.so objs/

# Build static module (embedded in nginx)
build-static:
	@if [ ! -d "$(NGINX_DIR)" ]; then \
		echo "Error: NGINX_DIR $(NGINX_DIR) not found"; \
		exit 1; \
	fi
	@echo "Configuring nginx with static DRL cache module..."
	cd $(NGINX_DIR) && ./configure \
		--add-module=$(PWD) \
		--with-http_realip_module \
		--with-http_stub_status_module \
		--with-http_auth_request_module \
		--with-http_secure_link_module \
		--with-threads \
		--with-file-aio
	cd $(NGINX_DIR) && make

# Install module system-wide
install: build-dynamic
	@echo "Installing DRL cache module..."
	sudo mkdir -p /etc/nginx/modules
	sudo cp objs/ngx_http_drl_cache_module.so /etc/nginx/modules/
	@echo "Module installed to /etc/nginx/modules/"
	@echo "Add 'load_module modules/ngx_http_drl_cache_module.so;' to nginx.conf"

# Clean build artifacts
clean:
	rm -rf objs/
	@if [ -d "$(NGINX_DIR)" ]; then \
		cd $(NGINX_DIR) && make clean || true; \
	fi

# Development helpers

# Check syntax and static analysis
lint: $(MODULE_SRCS) $(MODULE_DEPS)
	@echo "Running static analysis..."
	cppcheck --enable=all --suppress=missingIncludeSystem src/
	@echo "Checking code style..."
	find src/ -name "*.c" -o -name "*.h" | xargs clang-format --dry-run --Werror

# Format source code
format:
	@echo "Formatting source code..."
	find src/ -name "*.c" -o -name "*.h" | xargs clang-format -i

# Run tests (requires nginx test framework)
test: build-dynamic
	@echo "Running module tests..."
	@if [ -d "tests" ]; then \
		cd tests && prove .; \
	else \
		echo "No tests directory found"; \
	fi

# Debug build with symbols
debug: CFLAGS += -g -O0 -DDEBUG
debug: build-dynamic

# Release build with optimizations
release: CFLAGS += -O2 -DNDEBUG
release: build-dynamic

# Show module information
info:
	@echo "DRL Cache Module Information:"
	@echo "  Module name: ngx_http_drl_cache_module"
	@echo "  Source files: $(MODULE_SRCS)"
	@echo "  Dependencies: $(MODULE_DEPS)"
	@echo "  NGINX_DIR: $(NGINX_DIR)"
	@echo "  Build type: dynamic module"

# Package module for distribution
package: clean build-dynamic
	@echo "Creating distribution package..."
	tar czf ../ngx_http_drl_cache_module-$(shell date +%Y%m%d).tar.gz \
		--exclude='.git' \
		--exclude='*.tar.gz' \
		.

# Auto-detect nginx installation
detect-nginx:
	@echo "Detecting NGINX installation..."
	@if command -v nginx >/dev/null 2>&1; then \
		echo "NGINX binary: $$(which nginx)"; \
		echo "NGINX version: $$(nginx -v 2>&1)"; \
		echo "NGINX config: $$(nginx -t 2>&1 | grep 'configuration file' | cut -d' ' -f4)"; \
		echo "NGINX modules dir: $$(nginx -V 2>&1 | grep -o 'modules-path=[^[:space:]]*' | cut -d= -f2)"; \
	else \
		echo "NGINX not found in PATH"; \
	fi

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build dynamic module (default)"
	@echo "  configure    - Configure NGINX build"
	@echo "  build-dynamic - Build as dynamic module"
	@echo "  build-static - Build as static module"
	@echo "  install      - Install module system-wide"
	@echo "  clean        - Clean build artifacts"
	@echo "  lint         - Run static analysis"
	@echo "  format       - Format source code"
	@echo "  test         - Run tests"
	@echo "  debug        - Build debug version"
	@echo "  release      - Build optimized version"
	@echo "  info         - Show module information"
	@echo "  package      - Create distribution package"
	@echo "  detect-nginx - Detect NGINX installation"
	@echo ""
	@echo "Variables:"
	@echo "  NGINX_DIR    - Path to NGINX source (default: /usr/local/src/nginx)"
