# Production ML Demos - Course 5
# Sovereign AI Stack: trueno, aprender, realizar, batuta

.PHONY: all help setup lint test check clean
.PHONY: demo-test demo-all
.PHONY: demo-tgi-architecture demo-continuous-batching demo-kv-cache
.PHONY: demo-serving-api demo-streaming demo-throughput-bench
.PHONY: demo-quantization demo-apr-format demo-pruning
.PHONY: demo-distillation demo-flash-attention demo-speculative-decode demo-tensor-parallel
.PHONY: demo-wasm-inference demo-lambda-handler demo-presentar-ui
.PHONY: demo-latency-profile demo-model-registry demo-hardware-detect
.PHONY: benchmark-cpu benchmark-gpu benchmark-all
.PHONY: falsify compliance profile brick-score
.PHONY: export-arrow docs serve

# Directories
WEEK1_DIR := demos/week1
WEEK2_DIR := demos/week2
WEEK3_DIR := demos/week3
BRICK_DIR := brick
DOCS_DIR := docs

# Default target
all: lint test demo-test

# Help
help:
	@printf 'Production ML Demos - Course 5\n'
	@printf '================================\n\n'
	@printf 'Setup:\n'
	@printf '  make setup          Install dependencies\n\n'
	@printf 'Quality:\n'
	@printf '  make lint           Run formatter and linter\n'
	@printf '  make test           Run test suite\n'
	@printf '  make check          Run all checks\n'
	@printf '  make compliance     PMAT compliance check\n\n'
	@printf 'Demos (Week 1 - Inference Serving):\n'
	@printf '  make demo-tgi-architecture      TGI architecture concepts\n'
	@printf '  make demo-continuous-batching   PagedAttention simulation\n'
	@printf '  make demo-kv-cache              KV cache management\n'
	@printf '  make demo-serving-api           REST API patterns\n'
	@printf '  make demo-streaming             SSE streaming\n'
	@printf '  make demo-throughput-bench      Throughput measurement\n\n'
	@printf 'Demos (Week 2 - Model Optimization):\n'
	@printf '  make demo-quantization          Q4_K, Q5_K, Q6_K comparison\n'
	@printf '  make demo-apr-format            APR native format\n'
	@printf '  make demo-pruning               Weight pruning\n'
	@printf '  make demo-distillation          Knowledge distillation\n'
	@printf '  make demo-flash-attention       Flash attention math\n'
	@printf '  make demo-speculative-decode    Speculative decoding\n'
	@printf '  make demo-tensor-parallel       Multi-GPU splitting\n\n'
	@printf 'Demos (Week 3 - Edge Deployment):\n'
	@printf '  make demo-wasm-inference        WASM deployment\n'
	@printf '  make demo-lambda-handler        Serverless patterns\n'
	@printf '  make demo-presentar-ui          Presentar WASM UI\n'
	@printf '  make demo-latency-profile       P99 measurement\n'
	@printf '  make demo-model-registry        Pacha push/pull\n'
	@printf '  make demo-hardware-detect       Capability detection\n\n'
	@printf 'Benchmarks:\n'
	@printf '  make benchmark-cpu     CPU benchmarks\n'
	@printf '  make benchmark-gpu     GPU benchmarks (CUDA)\n'
	@printf '  make benchmark-all     All benchmarks\n\n'
	@printf 'Validation:\n'
	@printf '  make falsify           100-point falsification\n'
	@printf '  make demo-test         All demos (CI mode)\n'
	@printf '  make export-arrow      Export benchmark results\n'

# Setup
setup:
	@printf '=== Installing Dependencies ===\n'
	cargo install trueno --locked || true
	cargo install aprender --locked || true
	cargo install realizar --locked || true
	cargo install batuta --locked || true
	cargo install pmat --locked || true
	cargo install bashrs --locked || true
	cargo install cargo-llvm-cov --locked || true
	pmat hooks install || true
	@printf '=== Setup Complete ===\n'

# Linting
lint:
	@printf '=== Linting ===\n'
	cargo fmt --check
	cargo clippy -- -D warnings
	@printf '=== Lint Complete ===\n'

# Testing
test:
	@printf '=== Running Tests ===\n'
	cargo test --all-features
	@printf '=== Tests Complete ===\n'

# Coverage
coverage:
	@printf '=== Computing Coverage ===\n'
	cargo llvm-cov --html --output-dir target/coverage
	@printf '=== Coverage report: target/coverage/html/index.html ===\n'

# Check all
check: lint test compliance
	@printf '=== All Checks Complete ===\n'

#
# Week 1 Demos: Inference Serving
#

demo-tgi-architecture:
	@printf '=== Demo: TGI Architecture ===\n'
	cd $(WEEK1_DIR) && cargo run --bin demo-tgi-architecture

demo-continuous-batching:
	@printf '=== Demo: Continuous Batching ===\n'
	cd $(WEEK1_DIR) && cargo run --bin demo-continuous-batching

demo-kv-cache:
	@printf '=== Demo: KV Cache ===\n'
	cd $(WEEK1_DIR) && cargo run --bin demo-kv-cache

demo-serving-api:
	@printf '=== Demo: Serving API ===\n'
	cd $(WEEK1_DIR) && cargo run --bin demo-serving-api

demo-streaming:
	@printf '=== Demo: Streaming ===\n'
	cd $(WEEK1_DIR) && cargo run --bin demo-streaming

demo-throughput-bench:
	@printf '=== Demo: Throughput Benchmark ===\n'
	cd $(WEEK1_DIR) && cargo run --bin demo-throughput-bench

#
# Week 2 Demos: Model Optimization
#

demo-quantization:
	@printf '=== Demo: Quantization ===\n'
	cd $(WEEK2_DIR) && cargo run --bin demo-quantization

demo-apr-format:
	@printf '=== Demo: APR Model Format ===\n'
	cd $(WEEK2_DIR) && cargo run --bin demo-apr-format

demo-pruning:
	@printf '=== Demo: Pruning ===\n'
	cd $(WEEK2_DIR) && cargo run --bin demo-pruning

demo-distillation:
	@printf '=== Demo: Distillation ===\n'
	cd $(WEEK2_DIR) && cargo run --bin demo-distillation

demo-flash-attention:
	@printf '=== Demo: Flash Attention ===\n'
	cd $(WEEK2_DIR) && cargo run --bin demo-flash-attention

demo-speculative-decode:
	@printf '=== Demo: Speculative Decoding ===\n'
	cd $(WEEK2_DIR) && cargo run --bin demo-speculative-decode

demo-tensor-parallel:
	@printf '=== Demo: Tensor Parallelism ===\n'
	cd $(WEEK2_DIR) && cargo run --bin demo-tensor-parallel

#
# Week 3 Demos: Edge Deployment
#

demo-wasm-inference:
	@printf '=== Demo: WASM Inference ===\n'
	cd $(WEEK3_DIR) && cargo run --bin demo-wasm-inference

demo-lambda-handler:
	@printf '=== Demo: Lambda Handler ===\n'
	cd $(WEEK3_DIR) && cargo run --bin demo-lambda-handler

demo-presentar-ui:
	@printf '=== Demo: Presentar UI ===\n'
	cd $(WEEK3_DIR) && cargo run --bin demo-presentar-ui

demo-latency-profile:
	@printf '=== Demo: Latency Profile ===\n'
	cd $(WEEK3_DIR) && cargo run --bin demo-latency-profile

demo-model-registry:
	@printf '=== Demo: Model Registry ===\n'
	cd $(WEEK3_DIR) && cargo run --bin demo-model-registry

demo-hardware-detect:
	@printf '=== Demo: Hardware Detection ===\n'
	cd $(WEEK3_DIR) && cargo run --bin demo-hardware-detect

#
# Demo Test (CI Mode)
#

demo-test-week1:
	@printf '=== Week 1 Demos (CI) ===\n'
	cd $(WEEK1_DIR) && cargo run --bin demo-tgi-architecture -- --stdout
	cd $(WEEK1_DIR) && cargo run --bin demo-continuous-batching -- --stdout
	cd $(WEEK1_DIR) && cargo run --bin demo-kv-cache -- --stdout
	cd $(WEEK1_DIR) && cargo run --bin demo-serving-api -- --stdout
	cd $(WEEK1_DIR) && cargo run --bin demo-streaming -- --stdout
	cd $(WEEK1_DIR) && cargo run --bin demo-throughput-bench -- --stdout

demo-test-week2:
	@printf '=== Week 2 Demos (CI) ===\n'
	cd $(WEEK2_DIR) && cargo run --bin demo-quantization -- --stdout
	cd $(WEEK2_DIR) && cargo run --bin demo-apr-format -- --stdout
	cd $(WEEK2_DIR) && cargo run --bin demo-pruning -- --stdout
	cd $(WEEK2_DIR) && cargo run --bin demo-distillation -- --stdout
	cd $(WEEK2_DIR) && cargo run --bin demo-flash-attention -- --stdout
	cd $(WEEK2_DIR) && cargo run --bin demo-speculative-decode -- --stdout
	cd $(WEEK2_DIR) && cargo run --bin demo-tensor-parallel -- --stdout

demo-test-week3:
	@printf '=== Week 3 Demos (CI) ===\n'
	cd $(WEEK3_DIR) && cargo run --bin demo-wasm-inference -- --stdout
	cd $(WEEK3_DIR) && cargo run --bin demo-lambda-handler -- --stdout
	cd $(WEEK3_DIR) && cargo run --bin demo-presentar-ui -- --stdout
	cd $(WEEK3_DIR) && cargo run --bin demo-latency-profile -- --stdout
	cd $(WEEK3_DIR) && cargo run --bin demo-model-registry -- --stdout
	cd $(WEEK3_DIR) && cargo run --bin demo-hardware-detect -- --stdout

demo-test: demo-test-week1 demo-test-week2 demo-test-week3
	@printf '=== All Demos Passed ===\n'

demo-all: demo-tgi-architecture demo-continuous-batching demo-kv-cache \
          demo-serving-api demo-streaming demo-throughput-bench \
          demo-quantization demo-apr-format demo-pruning \
          demo-distillation demo-flash-attention demo-speculative-decode demo-tensor-parallel \
          demo-wasm-inference demo-lambda-handler demo-presentar-ui \
          demo-latency-profile demo-model-registry demo-hardware-detect

#
# Benchmarks
#

benchmark-cpu:
	@printf '=== CPU Benchmarks ===\n'
	cd $(WEEK1_DIR) && cargo run --release --bin demo-throughput-bench -- --benchmark --iterations 1000 --cpu
	@printf '=== CPU Benchmarks Complete ===\n'

benchmark-gpu:
	@printf '=== GPU Benchmarks ===\n'
	cd $(WEEK1_DIR) && cargo run --release --features cuda --bin demo-throughput-bench -- --benchmark --iterations 1000 --gpu
	@printf '=== GPU Benchmarks Complete ===\n'

benchmark-all: benchmark-cpu benchmark-gpu
	@printf '=== All Benchmarks Complete ===\n'

#
# Profiling
#

profile:
	@printf '=== Profiling ===\n'
	mkdir -p $(BRICK_DIR)/profiles
	cd $(WEEK1_DIR) && cargo run --release --bin demo-throughput-bench -- --profile --output ../../$(BRICK_DIR)/profiles/latest.json
	@printf '=== Profile saved to $(BRICK_DIR)/profiles/latest.json ===\n'

brick-score:
	@printf '=== Computing Brick Score ===\n'
	pmat brick-score --input $(BRICK_DIR)/profiles/latest.json
	@printf '=== Brick Score Complete ===\n'

#
# Compliance
#

compliance:
	@printf '=== PMAT Compliance ===\n'
	pmat comply check
	@printf '=== Compliance Complete ===\n'

#
# Falsification
#

falsify:
	@printf '=== Running 100-Point Falsification ===\n'
	@printf '\n--- Section 7.1: Inference Performance (25 pts) ---\n'
	cd $(WEEK1_DIR) && cargo run --release --bin demo-throughput-bench -- --benchmark --verify
	@printf '\n--- Section 7.2: Model Optimization (20 pts) ---\n'
	cd $(WEEK2_DIR) && cargo run --release --bin demo-quantization -- --verify
	@printf '\n--- Section 7.3: Edge Deployment (15 pts) ---\n'
	cd $(WEEK3_DIR) && cargo run --release --bin demo-wasm-inference -- --verify
	cd $(WEEK3_DIR) && cargo run --release --bin demo-lambda-handler -- --verify
	@printf '\n--- Section 7.4: API Correctness (15 pts) ---\n'
	cd $(WEEK1_DIR) && cargo run --release --bin demo-serving-api -- --test-openai-compat
	@printf '\n--- Section 7.5: Reproducibility (15 pts) ---\n'
	cargo build --locked
	cd $(WEEK1_DIR) && cargo run --bin demo-throughput-bench -- --seed 42 --verify-determinism
	@printf '\n--- Section 7.6: Code Quality (10 pts) ---\n'
	cargo llvm-cov --fail-under-lines 95
	cargo clippy -- -D warnings
	@printf '\n=== Falsification Complete ===\n'

#
# Export
#

export-arrow:
	@printf '=== Exporting Benchmark Results ===\n'
	mkdir -p benchmarks
	cd $(WEEK1_DIR) && cargo run --release --bin demo-throughput-bench -- --export-arrow ../../benchmarks/results.arrow
	sha256sum benchmarks/results.arrow > benchmarks/checksums.sha256
	@printf '=== Export Complete: benchmarks/results.arrow ===\n'

#
# Documentation
#

docs:
	@printf '=== Building Documentation ===\n'
	mdbook build $(DOCS_DIR) || true
	@printf '=== Docs built to $(DOCS_DIR)/book ===\n'

serve:
	@printf '=== Serving Documentation ===\n'
	mdbook serve $(DOCS_DIR) || python3 -m http.server 8080 --directory $(DOCS_DIR)

#
# Clean
#

clean:
	@printf '=== Cleaning ===\n'
	cargo clean
	rm -rf $(BRICK_DIR)/profiles/*.json
	rm -rf $(DOCS_DIR)/book
	rm -rf benchmarks/*.arrow
	rm -rf target/coverage
	@printf '=== Clean Complete ===\n'
