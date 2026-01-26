# Course 5: Production ML with Hugging Face

## Overview

**Duration**: 3 weeks (~12 hours)
**Stack**: Sovereign AI (trueno 0.13, aprender 0.24, realizar 0.6, batuta 0.5)
**Hub Integration**: paiml/prod-ml-demos, paiml/prod-ml-benchmarks
**Specification**: [docs/specifications/hf-ml-prod-demos.md](specifications/hf-ml-prod-demos.md)

## Course Description

Deploy ML models to production using the Sovereign Rust Stack. Master inference serving with continuous batching, model optimization through quantization and pruning, and edge deployment via WASM and serverless functions. Pure Rust implementation with zero external runtime dependencies.

## Learning Outcomes

By completing this course, learners will be able to:

1. Deploy inference servers with continuous batching and streaming responses
2. Optimize models through quantization (Q4_K, Q5_K, Q6_K) achieving ≥3.5x compression
3. Implement Flash Attention and speculative decoding for 2x+ speedup
4. Deploy to edge environments (WASM, Lambda) with sub-10ms latency
5. Build production-grade APIs with OpenAI compatibility

## Model Tiers

| Tier | Model | VRAM | Use Case |
|------|-------|------|----------|
| Tiny | phi-2 (2.7B Q4_K) | 4GB | Development, edge |
| Small | llama-7b (Q4_K) | 8GB | Standard inference |
| Medium | llama-13b (Q4_K) | 16GB | High quality |
| Large | llama-70b (Q4_K) | 48GB | Production scale |

**Selection Criteria**:
- Choose Tiny for demos and edge deployment
- Choose Small for standard production workloads
- Choose Medium/Large when quality is paramount

---

## Week 1: Inference Serving

Master production inference serving with continuous batching, KV cache management, and streaming APIs.

### 1.0 Model Format Fundamentals (Rosetta)

Understanding model portability before deployment.

**Video Series**: 10-13 minutes total

| # | Topic | Diagram | Duration |
|---|-------|---------|----------|
| 1 | Model Anatomy | `model-anatomy.svg` | 2-3 min |
| 2 | Why Self-Contained | `self-contained-models-why.svg` | 4-5 min |
| 3 | Conversion Flow | `safetensors-to-apr-flow.svg` | 3-4 min |

**Key Concepts**:
- 4 components: weights, tokenizer, config, metadata
- SafeTensors (fragmented) → GGUF (bundled) → APR (bundled + verified)
- Production constraints: CDN, WASM sandbox, offline, version mismatch
- Rosetta conversion pipeline: parse → validate → embed → compress → sign

**Scripts**: `docs/scripts/model-anatomy.md`, `self-contained-models-why.md`, `safetensors-to-apr-flow.md`

### 1.0.1 ML Observability (Tracing)

Understanding tracing for production inference debugging.

**Video Series**: 4-6 minutes total

| # | Topic | Diagram | Duration |
|---|-------|---------|----------|
| 1 | Why Trace | `tracing-why.svg` | 2-3 min |
| 2 | Three Layers | `tracing-layers.svg` | 2-3 min |

**Key Concepts**:
- Four problems: NaN propagation, latency spikes, quality drift, memory bloat
- Four trace types: Layer, Brick, Diff, Syscall
- Three levels: Request (prod safe) → Layer (offline) → System (deep dive)
- Jidoka: stop the line when anomaly detected

**Scripts**: `docs/scripts/tracing-why.md`, `tracing-layers.md`

### 1.1 TGI Architecture Concepts

Understanding production inference server architecture.

**Demo**: `make demo-tgi-architecture`

```bash
cd demos/week1 && cargo run --bin demo-tgi-architecture
```

Key concepts:
- Request queue and batch scheduler
- Tokenizer integration (BPE/SentencePiece)
- Model executor with trueno backend
- Output decoder with streaming

### 1.2 Continuous Batching

Dynamic batching for maximum throughput.

**Demo**: `make demo-continuous-batching`

```bash
cd demos/week1 && cargo run --bin demo-continuous-batching -- --block-size 16
```

Key concepts:
- PagedAttention memory model
- Block table management
- Defragmentation strategies
- Utilization optimization

### 1.3 KV Cache Management

Efficient key-value cache for autoregressive generation.

**Demo**: `make demo-kv-cache`

```bash
cd demos/week1 && cargo run --bin demo-kv-cache -- --max-seq-len 4096
```

Key concepts:
- Cache growth patterns
- Eviction policies
- Memory pressure handling
- Prefix caching

### 1.4 REST API Design

OpenAI-compatible serving endpoints.

**Demo**: `make demo-serving-api`

```bash
cd demos/week1 && cargo run --bin demo-serving-api -- --port 8080
```

Endpoints:
- `POST /v1/completions` - Text completion
- `POST /v1/chat/completions` - Chat completion
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

### 1.5 Streaming Responses

Server-sent events for real-time token delivery.

**Demo**: `make demo-streaming`

```bash
cd demos/week1 && cargo run --bin demo-streaming -- --measure-ttft
```

Key concepts:
- TTFT (Time To First Token)
- Token-by-token delivery
- Connection management
- Backpressure handling

### 1.6 Throughput Benchmarking

Comprehensive performance measurement.

**Demo**: `make demo-throughput-bench`

```bash
cd demos/week1 && cargo run --bin demo-throughput-bench -- --benchmark --iterations 1000
```

Metrics captured:
- Tokens per second (throughput)
- P50, P90, P99 latency
- Memory utilization
- Batch efficiency

**Target**: ≥250 tok/s on RTX 4090

---

## Week 2: Model Optimization

Compress and accelerate models while preserving quality.

### 2.1 Quantization Methods

Reduce model size with minimal quality loss.

**Demo**: `make demo-quantization`

```bash
cd demos/week2 && cargo run --bin demo-quantization -- --method q4_k --eval-ppl
```

Methods supported:
- Q8_0: 8-bit symmetric
- Q6_K: 6-bit K-quants
- Q5_K: 5-bit K-quants
- Q4_K: 4-bit K-quants (recommended)

**Target**: ≥3.5x compression, ≤10% perplexity increase

### 2.2 APR Model Format

Native cross-platform model format with zero-copy loading.

**Demo**: `make demo-apr-format`

```bash
cd demos/week2 && cargo run --bin demo-apr-format -- --verify
```

Key concepts:
- SafeTensors-compatible serialization
- Zero-copy memory-mapped loading
- Optional encryption and Ed25519 signing
- Cross-platform (x86_64, ARM64, WASM)

### 2.3 Weight Pruning

Remove redundant weights for efficiency.

**Demo**: `make demo-pruning`

```bash
cd demos/week2 && cargo run --bin demo-pruning -- --target-sparsity 0.5
```

Methods:
- Magnitude pruning
- Structured pruning (channels)
- Gradual pruning schedule

**Target**: ≥50% sparsity, ≤5% accuracy loss

### 2.4 Knowledge Distillation

Transfer knowledge to smaller models.

**Demo**: `make demo-distillation`

```bash
cd demos/week2 && cargo run --bin demo-distillation -- --teacher large --student small
```

Key concepts:
- Temperature scaling
- Soft label training
- Hidden state matching

### 2.5 Flash Attention

Memory-efficient attention computation.

**Demo**: `make demo-flash-attention`

```bash
cd demos/week2 && cargo run --bin demo-flash-attention -- --seq-len 8192
```

Key concepts:
- Tiled computation
- O(N) memory complexity
- Block-sparse patterns

**Target**: 2x speedup over naive attention

### 2.6 Speculative Decoding

Accelerate generation with draft models.

**Demo**: `make demo-speculative-decode`

```bash
cd demos/week2 && cargo run --bin demo-speculative-decode -- --draft-tokens 4
```

Key concepts:
- Draft model selection
- Verification pass
- Acceptance rate optimization

**Target**: 2x effective throughput

### 2.7 Tensor Parallelism

Split models across GPUs.

**Demo**: `make demo-tensor-parallel`

```bash
cd demos/week2 && cargo run --bin demo-tensor-parallel -- --gpus 2
```

Key concepts:
- Column/row parallel linear layers
- All-reduce communication
- Load balancing

**Target**: ≥1.8x scaling per GPU

---

## Week 3: Edge Deployment

Deploy to resource-constrained environments.

### 3.1 WASM Inference

Browser-based model execution.

**Demo**: `make demo-wasm-inference`

```bash
cd demos/week3 && cargo run --bin demo-wasm-inference -- --benchmark
```

Key concepts:
- WASM32 compilation
- SIMD via wasm32 intrinsics
- Memory management in browser

**Target**: ≤5MB binary, ≤50ms inference

### 3.2 Serverless Deployment

Lambda function patterns.

**Demo**: `make demo-lambda-handler`

```bash
cd demos/week3 && cargo run --bin demo-lambda-handler -- --cold-start
```

Key concepts:
- Cold vs warm start optimization
- Embedded model loading
- Response streaming

**Target**: ≤500ms cold start, ≤10ms warm

### 3.3 Presentar WASM UI

Interactive ML demos with the sovereign presentar framework.

**Demo**: `make demo-presentar-ui`

```bash
cd demos/week3 && cargo run --bin demo-presentar-ui -- --port 7860
```

Key concepts:
- YAML-driven declarative UI
- Measure-Layout-Paint widget paradigm
- Unidirectional data flow
- Zero Python dependencies
- 60fps GPU-accelerated rendering via WebGPU
- Built-in WCAG 2.1 AA accessibility

### 3.4 Latency Profiling

P99 latency measurement and optimization.

**Demo**: `make demo-latency-profile`

```bash
cd demos/week3 && cargo run --bin demo-latency-profile -- --percentiles
```

Metrics:
- P50, P90, P95, P99, P99.9
- Latency breakdown by stage
- Bottleneck identification

**Target**: P99 ≤10ms for classical ML

### 3.5 Model Registry

Secure model versioning with Pacha.

**Demo**: `make demo-model-registry`

```bash
cd demos/week3 && cargo run --bin demo-model-registry
```

Operations:
- `push` - Upload with signing
- `pull` - Download with verification
- `sign` - Add Ed25519 signature
- `verify` - Check signature validity

### 3.6 Hardware Detection

Runtime capability discovery.

**Demo**: `make demo-hardware-detect`

```bash
cd demos/week3 && cargo run --bin demo-hardware-detect
```

Detects:
- CPU: SSE2, AVX, AVX2, AVX-512, NEON
- GPU: CUDA compute capability, VRAM
- Memory: Available RAM, swap

---

## Labs

| Week | Lab | Tier | Description |
|------|-----|------|-------------|
| 1 | Deploy TGI-Style Server | Small | Production server with continuous batching |
| 1 | Benchmark Throughput | Small | Measure and optimize tokens/second |
| 1 | Streaming Client | Tiny | Build SSE client with error handling |
| 2 | Quantize to Q4_K | Medium | Compress model with quality validation |
| 2 | APR Model Export | Small | Save/load with signing and encryption |
| 2 | Implement Flash Attention | Small | Memory-efficient attention layer |
| 2 | Speculative Decoding | Small | Draft-verify acceleration |
| 3 | WASM Deployment | Tiny | Browser-based classifier |
| 3 | Lambda Function | Tiny | Serverless inference endpoint |
| 3 | Full Pipeline | Small | End-to-end production deployment |

## Demo Commands

```bash
# Week 1: Inference Serving
make demo-tgi-architecture
make demo-continuous-batching
make demo-kv-cache
make demo-serving-api
make demo-streaming
make demo-throughput-bench

# Week 2: Model Optimization
make demo-quantization
make demo-apr-format
make demo-pruning
make demo-distillation
make demo-flash-attention
make demo-speculative-decode
make demo-tensor-parallel

# Week 3: Edge Deployment
make demo-wasm-inference
make demo-lambda-handler
make demo-presentar-ui
make demo-latency-profile
make demo-model-registry
make demo-hardware-detect

# Run all demos (CI mode)
make demo-test

# Benchmark suite
make benchmark-cpu
make benchmark-gpu

# Falsification validation
make falsify
```

## Prerequisites

- Rust 1.83.0 or later
- 16GB RAM minimum (64GB recommended)
- 8GB VRAM for GPU demos (24GB recommended)
- Linux x86_64 or macOS ARM64

## Resources

### Sovereign AI Stack

- [trueno](https://crates.io/crates/trueno) - SIMD/GPU compute primitives
- [aprender](https://crates.io/crates/aprender) - ML algorithms
- [realizar](https://crates.io/crates/realizar) - Inference server
- [batuta](https://crates.io/crates/batuta) - Orchestration

### HuggingFace

- [Text Generation Inference](https://huggingface.co/docs/text-generation-inference)
- [Inference Endpoints](https://huggingface.co/docs/inference-endpoints)
- [Optimum](https://huggingface.co/docs/optimum)
- [Transformers.js](https://huggingface.co/docs/transformers.js)

### Papers (Key Citations)

1. Kwon et al. (2023) - PagedAttention (SOSP '23)
2. Dao et al. (2022) - FlashAttention (NeurIPS '22)
3. Leviathan et al. (2023) - Speculative Decoding (ICML '23)
4. Frantar et al. (2022) - GPTQ Quantization
5. Dettmers et al. (2023) - QLoRA 4-bit Training

---

## Quality Gates

```bash
# Before committing
make lint          # cargo fmt + clippy
make test          # cargo test --all-features
make demo-test     # All demos with --stdout
make compliance    # pmat comply check

# Before release
make benchmark-all
make falsify       # 100-point validation
make export-arrow  # Benchmark results
```

**Minimum Requirements**:
- Test coverage ≥95%
- Zero clippy warnings
- All demos pass CI mode
- Falsification score ≥90/100
