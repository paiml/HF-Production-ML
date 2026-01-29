# APR Explorer - WASM Build
#
# Architecture: probar zero-business-logic-in-JS pattern
# - All APR parsing in pure Rust (testable via cargo test)
# - MockDom for browser-independent testing
# - Driver pattern for unified TUI/WASM testing
# - Minimal bootstrap JS only for WASM init
#
# Prerequisites:
#   cargo install wasm-pack
#   rustup target add wasm32-unknown-unknown

.PHONY: build dev clean serve test size setup sample-apr

# Build WASM with wasm-pack (requires wasm feature)
build:
	cargo install wasm-pack --quiet
	wasm-pack build --target web --out-dir www/pkg --release -- --features wasm

# Development build (faster, with debug info)
dev:
	cargo install wasm-pack --quiet
	wasm-pack build --target web --out-dir www/pkg --dev -- --features wasm

# Run tests (pure Rust, no browser needed)
test:
	cargo test

# Run tests with verbose output
test-verbose:
	cargo test -- --nocapture

# Run local development server
serve: build
	@echo "Starting server at http://localhost:8080"
	cd www && python3 -m http.server 8080

# Serve without rebuilding
serve-only:
	@echo "Starting server at http://localhost:8080"
	cd www && python3 -m http.server 8080

# Clean build artifacts
clean:
	cargo clean
	rm -rf www/pkg

# Check WASM size
size: build
	@echo "WASM binary size:"
	@ls -lh www/pkg/*.wasm
	@echo ""
	@echo "Compressed (gzip):"
	@gzip -c www/pkg/apr_explorer_bg.wasm | wc -c | awk '{print $$1/1024 " KB"}'

# Install dependencies
setup:
	cargo install wasm-pack
	rustup target add wasm32-unknown-unknown

# Lint and format
lint:
	cargo fmt --check
	cargo clippy -- -D warnings

# Format code
fmt:
	cargo fmt

# Generate sample APR file for testing
sample-apr:
	@echo "Creating sample APR file..."
	@python3 -c '\
import struct, json, os;\
\
metadata = json.dumps({"model_type": "demo", "architecture": "mlp", "hidden_size": 128}).encode();\
meta_padded = metadata + b"\x00" * (64 - len(metadata) % 64) if len(metadata) % 64 != 0 else metadata;\
\
tensors = [{"name": "weights.0", "dtype": "f32", "shape": [128, 128], "offset": 0, "size": 65536},\
           {"name": "bias.0", "dtype": "f32", "shape": [128], "offset": 65536, "size": 512}];\
index = json.dumps(tensors).encode();\
idx_padded = index + b"\x00" * (64 - len(index) % 64) if len(index) % 64 != 0 else index;\
\
meta_offset = 64;\
idx_offset = meta_offset + len(meta_padded);\
data_offset = idx_offset + len(idx_padded);\
\
tensor_data = os.urandom(66048);\
\
header = b"APR\x00" + bytes([2, 0]) + struct.pack("<H", 0);\
header += struct.pack("<QQQQQQ", meta_offset, len(metadata), idx_offset, len(index), data_offset, 66048);\
header += b"\x00" * (64 - len(header));\
\
with open("www/sample.apr", "wb") as f:\
    f.write(header + meta_padded + idx_padded + tensor_data);\
print("Created www/sample.apr (", len(header + meta_padded + idx_padded + tensor_data), "bytes)");\
'
