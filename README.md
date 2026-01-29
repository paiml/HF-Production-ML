# Production ML with Hugging Face

**Course 5 of 5: Deploying ML Models with the Sovereign AI Stack**

[![CI](https://github.com/paiml/HF-Production-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/HF-Production-ML/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.83%2B-orange.svg)](https://www.rust-lang.org/)

A production ML course using pure Rust - no Python runtime required.

## Quick Start

```bash
git clone https://github.com/paiml/HF-Production-ML.git
cd HF-Production-ML
make setup
make check
```

## Course Structure

| Module | Focus | Duration |
|--------|-------|----------|
| Week 1 | Model Formats (GGUF, SafeTensors, APR) | ~3 hours |
| Week 2 | MLOps (CI/CD, Observability, Security) | ~4 hours |
| Week 3 | Projects (Depyler, Whisper, APR Tools) | ~3 hours |
| Capstone | Multi-Format Deployment | 4-6 hours |

## Sovereign AI Stack

| Crate | Purpose |
|-------|---------|
| [trueno](https://crates.io/crates/trueno) | SIMD/GPU compute |
| [aprender](https://crates.io/crates/aprender) | ML algorithms, APR format |
| [realizar](https://crates.io/crates/realizar) | Inference server |
| [batuta](https://crates.io/crates/batuta) | Orchestration |
| [presentar](https://crates.io/crates/presentar) | WASM UI |

## Demos

```bash
# Week 1: Inference Serving
make demo-tgi-architecture
make demo-continuous-batching
make demo-kv-cache

# Week 2: Model Optimization
make demo-quantization
make demo-flash-attention
make demo-speculative-decode

# Week 3: Edge Deployment
make demo-wasm-inference
make demo-lambda-handler
make demo-hardware-detect
```

Run `make help` for all available commands.

## Documentation

- [Course Introduction](docs/readings/0.0-course-introduction-resources.md)
- [Lesson Structure](docs/lesson-structure.md)
- [Capstone Project](docs/readings/4.0-capstone-multi-format-deployment.md)
- [Full Specification](docs/specifications/hf-ml-prod-demos.md)

## License

Apache-2.0
