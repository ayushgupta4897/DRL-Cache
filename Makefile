# DRL-Cache Build System

.PHONY: all clean nginx-module sidecar training install-module

# Default target
all: nginx-module sidecar

# Build NGINX module
nginx-module:
	@echo "Building NGINX dynamic module..."
	cd nginx-module && $(MAKE)

# Build ONNX sidecar
sidecar:
	@echo "Building ONNX sidecar..."
	cd sidecar && $(MAKE)

# Install Python training dependencies
training:
	@echo "Installing training dependencies..."
	cd training && pip install -r requirements.txt

# Install NGINX module system-wide
install-module: nginx-module
	@echo "Installing NGINX module..."
	sudo cp nginx-module/objs/ngx_http_drl_cache_module.so /etc/nginx/modules/
	@echo "Module installed. Add 'load_module modules/ngx_http_drl_cache_module.so;' to nginx.conf"

# Clean all builds
clean:
	cd nginx-module && $(MAKE) clean || true
	cd sidecar && $(MAKE) clean || true
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -exec rm -rf {} + || true

# Development helpers
test: all
	@echo "Running tests..."
	cd training && python -m pytest tests/

lint:
	cd nginx-module && cppcheck --enable=all src/
	cd sidecar && cppcheck --enable=all src/
	cd training && flake8 --max-line-length=100 src/

# Package for distribution
dist: clean all
	tar czf drl-cache-$(shell date +%Y%m%d).tar.gz \
		--exclude='.git' \
		--exclude='*.tar.gz' \
		--exclude='logs/' \
		--exclude='models/' \
		.
