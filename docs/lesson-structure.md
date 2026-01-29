# Production ML with Hugging Face - Lesson Structure

**3 Weeks × 3 Lessons = 9 Lessons Total**

---

## Week 1: Model Formats

**Description:** Understanding ML model formats and the Sovereign AI Stack. Learn GGUF, SafeTensors, and APR formats for different deployment targets.

**Learning Objectives:**
- Understand the Sovereign AI Stack architecture (trueno, aprender, realizar, batuta)
- Convert between model formats: GGUF, SafeTensors, and APR
- Select appropriate formats for different deployment targets

---

### Lesson 1.1: Course Introduction & HuggingFace Publishing

| Filename | Title |
|----------|-------|
| `1.0-course-intro.mp4` | Course Introduction |
| `1.3-hg-model-publish.mp4` | HuggingFace Model Publishing |

**Key Concepts:**
- Pure Rust stack, no Python runtime
- HuggingFace as model source
- `apr pull` command
- Model artifacts: weights, tokenizer, config, metadata

---

### Lesson 1.2: Model Types & APR Format

| Filename | Title |
|----------|-------|
| `1.4-model-types-hf.mp4` | Model Types on HuggingFace |
| `1.5-apr-format.mp4` | APR Format Deep Dive |

**Key Concepts:**
- GGUF: CPU inference, quantization, consumer hardware
- SafeTensors: Safe loading, HF standard
- APR: WASM-first, scales to CUDA
- Sub-millisecond browser inference

---

### Lesson 1.3: Format Comparison & Model Tracing

| Filename | Title |
|----------|-------|
| `1.6-mode-format-compare.mp4` | Model Format Comparison |
| `1.9-why-trace-models.mp4` | Why Trace Models |

**Key Concepts:**
- Weights + tokenizer + config + metadata
- Format selection by deployment target
- Tracing catches problems at source
- Layer-level observability

---

## Week 2: MLOps Foundations

**Description:** Production infrastructure for ML systems. CI/CD pipelines, observability, security, and deployment patterns.

**Learning Objectives:**
- Implement CI/CD pipelines for ML models
- Set up observability with logs, metrics, and traces
- Apply cryptographic model signing for security
- Choose optimal deployment patterns (embedded, external, mmap)

---

### Lesson 2.1: Model Registry & CI/CD Pipeline

| Filename | Title |
|----------|-------|
| `2.1-model-registery.mp4` | Model Registry Architecture |
| `2.2-cicd-pipeline-ml.mp4` | CI/CD Pipeline for ML |

**Key Concepts:**
- One address, all artifacts
- Multiple format support (GGUF, SafeTensors, APR)
- MLOps = DevOps + ML complexity
- Continuous improvement pipelines

---

### Lesson 2.2: Observability & Security

| Filename | Title |
|----------|-------|
| `2.3-model-observability-stack.mp4` | Model Observability Stack |
| `2.4-model-signing-security.mp4` | Model Signing & Security |

**Key Concepts:**
- Log events: model load, layer dimensions, errors
- Metrics: latency, throughput, memory
- Traces: request flow, layer timing
- Sign from training to deployment
- Detect poisoned training data

---

### Lesson 2.3: Deployment Patterns & Infrastructure

| Filename | Title |
|----------|-------|
| `2.5-binary-deployment-patterns.mp4` | Binary Deployment Patterns |
| `2.6-inference-server-architecture.mp4` | Inference Server Architecture |
| `2.8-corpus-management-dataops.mp4` | Corpus Management & DataOps |
| `2.9-cost-performance-decision-matrix.mp4` | Cost-Performance Decision Matrix |

**Key Concepts:**
- Binary deployment enables local automation
- `include_bytes!` for embedded models
- Inference server: health checks, round robin, queuing
- Versioned training data like code
- WASM: free, slow | CPU: cheap, medium | GPU: expensive, fast

---

## Week 3: Project Showcase

**Description:** Real-world projects built with the Sovereign AI Stack. Depyler transpiler, Whisper speech-to-text, and APR ecosystem tools.

**Learning Objectives:**
- Build self-improving systems with compiler-in-the-loop (CITL) training
- Deploy speech-to-text models to browser and CLI
- Use APR ecosystem tools: apr-chat, apr-run, apr-server
- Convert between model formats using the Rosetta module

---

### Lesson 3.1: Depyler - Python to Rust Transpiler

| Filename | Title |
|----------|-------|
| `3.2-four-projects-one-stack.mp4` | Four Projects, One Stack |
| `3.3-depyler-deep-dive.mp4` | Depyler Deep Dive |
| `3.4-depyler-oracle.mp4` | Depyler Oracle Training |
| `3.5-depyler-single-shot-compile.mp4` | Depyler Single-Shot Compile |

**Key Concepts:**
- Python → AST → Rust → Compile
- Oracle learns project-specific error-fix patterns
- HuggingFace corpus: Python-Rust pairs
- CITL: compiler teaches the model
- 40% → 80% single-shot compile rate

---

### Lesson 3.2: Whisper.apr - Speech-to-Text

| Filename | Title |
|----------|-------|
| `3.6-whisper-overiew.mp4` | Whisper.apr Overview |
| `3.7-whisper-code-walkthrough.mp4` | Whisper Code Walkthrough |
| `3.8-whisper-demo.mp4` | Whisper Demo |

**Key Concepts:**
- Model conversion to smaller format
- WebAssembly browser inference
- CLI for local transcription
- Rust wrapper simplifies LLM usage
- Makefile as abstraction layer

---

### Lesson 3.3: APR Ecosystem & Course Conclusion

| Filename | Title |
|----------|-------|
| `3.9-apr-format-rosetta.mp4` | APR Format Rosetta Stone |
| `3.10-apr-hub-spoke.mp4` | APR Hub & Spoke Architecture |
| `3.11-apr-chat-demo.mp4` | APR Chat Demo |
| `3.20-course-conclusion.mp4` | Course Conclusion |

**Key Concepts:**
- GGUF ↔ SafeTensors ↔ APR (all bidirectional)
- Local sovereign AI stack
- Chat, single-shot, server modes
- Week 1: formats, Week 2: MLOps, Week 3: projects
- Ship it!

---

## Video Mapping Summary

| Lesson | Filename | Title |
|--------|----------|-------|
| 1.1 | `1.0-course-intro.mp4` | Course Introduction |
| 1.1 | `1.3-hg-model-publish.mp4` | HuggingFace Model Publishing |
| 1.2 | `1.4-model-types-hf.mp4` | Model Types on HuggingFace |
| 1.2 | `1.5-apr-format.mp4` | APR Format Deep Dive |
| 1.3 | `1.6-mode-format-compare.mp4` | Model Format Comparison |
| 1.3 | `1.9-why-trace-models.mp4` | Why Trace Models |
| 2.1 | `2.1-model-registery.mp4` | Model Registry Architecture |
| 2.1 | `2.2-cicd-pipeline-ml.mp4` | CI/CD Pipeline for ML |
| 2.2 | `2.3-model-observability-stack.mp4` | Model Observability Stack |
| 2.2 | `2.4-model-signing-security.mp4` | Model Signing & Security |
| 2.3 | `2.5-binary-deployment-patterns.mp4` | Binary Deployment Patterns |
| 2.3 | `2.6-inference-server-architecture.mp4` | Inference Server Architecture |
| 2.3 | `2.8-corpus-management-dataops.mp4` | Corpus Management & DataOps |
| 2.3 | `2.9-cost-performance-decision-matrix.mp4` | Cost-Performance Decision Matrix |
| 3.1 | `3.2-four-projects-one-stack.mp4` | Four Projects, One Stack |
| 3.1 | `3.3-depyler-deep-dive.mp4` | Depyler Deep Dive |
| 3.1 | `3.4-depyler-oracle.mp4` | Depyler Oracle Training |
| 3.1 | `3.5-depyler-single-shot-compile.mp4` | Depyler Single-Shot Compile |
| 3.2 | `3.6-whisper-overiew.mp4` | Whisper.apr Overview |
| 3.2 | `3.7-whisper-code-walkthrough.mp4` | Whisper Code Walkthrough |
| 3.2 | `3.8-whisper-demo.mp4` | Whisper Demo |
| 3.3 | `3.9-apr-format-rosetta.mp4` | APR Format Rosetta Stone |
| 3.3 | `3.10-apr-hub-spoke.mp4` | APR Hub & Spoke Architecture |
| 3.3 | `3.11-apr-chat-demo.mp4` | APR Chat Demo |
| 3.3 | `3.20-course-conclusion.mp4` | Course Conclusion |

**Total: 25 videos → 9 lessons → 3 modules**

---

## Module 4: Capstone Project (Standalone)

**Description:** Final project deploying Qwen2.5-Coder-0.5B across all three model formats. Students demonstrate mastery of format conversion, CLI deployment, server deployment, and performance benchmarking.

**Learning Objectives:**
- Convert models between GGUF, SafeTensors, and APR formats using Rosetta
- Deploy models via CLI (`apr run`, `apr chat`) and REST API (`apr serve`)
- Benchmark performance across formats (throughput, latency, memory)
- Select optimal formats for edge, cloud, and browser deployment scenarios

---

### Lesson 4.1: Multi-Format Model Deployment

| Resource | Description |
|----------|-------------|
| `4.0-capstone-multi-format-deployment.md` | Complete capstone project guide |

**Project Components:**

| Part | Focus | Deliverable |
|------|-------|-------------|
| Part 1 | Model Acquisition & Conversion | Three model files (GGUF, SafeTensors, APR) |
| Part 2 | CLI Deployment | Working inference scripts |
| Part 3 | Server Deployment | REST APIs on three ports |
| Part 4 | Performance Benchmarking | Benchmark JSON files |
| Part 5 | Format Comparison | Output equivalence verification |
| Part 6 | Deployment Scenarios | Edge, Cloud, WASM recommendations |
| Part 7 | Final Report | `report.md` with analysis |

**Grading Rubric:**

| Component | Points |
|-----------|--------|
| Format Conversion | 20 |
| CLI Deployment | 20 |
| Server Deployment | 20 |
| Benchmarking | 25 |
| Report Quality | 15 |
| **Total** | **100** |

Pass threshold: 80 points

---

## Complete Course Summary

| Module | Focus | Duration | Lessons |
|--------|-------|----------|---------|
| Week 1 | Model Formats | ~3 hours | 3 |
| Week 2 | MLOps Foundations | ~4 hours | 3 |
| Week 3 | Project Showcase | ~3 hours | 3 |
| Module 4 | Capstone Project | 4-6 hours | 1 |

**Total Course:** ~14 hours, 10 lessons, 4 modules
