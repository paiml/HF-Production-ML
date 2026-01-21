# Production ML with Hugging Face - Course 5

## Project Overview

**Course**: Production ML with Hugging Face (Course 5 of 5)
**Stack**: Sovereign AI (trueno, aprender, realizar, batuta)
**Language**: Pure Rust (no Python runtime dependencies)
**Specification**: `docs/specifications/hf-ml-prod-demos.md`

## Quick Reference

```bash
# Setup
make setup

# Run demos
make demo-tgi-architecture    # Week 1: Inference serving
make demo-quantization        # Week 2: Model optimization
make demo-wasm-inference      # Week 3: Edge deployment

# Quality gates
make check                    # lint + test + compliance
make falsify                  # 100-point validation

# Benchmarks
make benchmark-cpu
make benchmark-gpu
```

## Architecture

### Sovereign Stack Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| trueno | 0.13.0 | SIMD/GPU compute primitives |
| aprender | 0.24.1 | ML algorithms, .apr format |
| realizar | 0.6.8 | Inference server, API |
| batuta | 0.5.0 | Orchestration, Pacha registry |
| presentar | 0.3.2 | WASM UI framework (Gradio replacement) |

### Demo Structure

```
demos/
├── week1/  # Inference Serving (6 demos)
├── week2/  # Model Optimization (7 demos)
└── week3/  # Edge Deployment (6 demos)
```

Each demo supports:
- `--stdout` for CI mode (non-interactive)
- `--benchmark` for performance measurement
- `--verify` for falsification validation

## Development Guidelines

### Demo Pattern

```rust
use clap::Parser;
use prod_ml_demos::{DemoArgs, DemoResult};

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    demo: DemoArgs,

    // Demo-specific args
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let result = run_demo(&args)?;

    match args.demo.output {
        OutputFormat::Tui => render_tui(&result)?,
        OutputFormat::Stdout => print_stdout(&result),
        OutputFormat::Json => println!("{}", serde_json::to_string(&result)?),
    }
    Ok(())
}
```

### SVG Diagrams

- Resolution: 1920x1080 (16:9)
- Background: `#020617` (Slate 950)
- Panels: `#0f172a` (Slate 900)
- Colors:
  - Green `#22c55e`: Active/trainable
  - Blue `#1e3a8a`: Frozen weights
  - Purple `#7c3aed`: Baseline model
  - Teal `#14b8a6`: Attention
  - Orange `#f59e0b`: FFN
- Fonts: Inter (UI), JetBrains Mono (code)

### Model Format

Use `.apr` (aprender native format) exclusively:

```rust
// Save
model.save_safetensors("model.apr")?;

// Load (zero-copy)
let model = Model::load_safetensors("model.apr")?;

// With signing
model.save_signed("model.apr", &identity)?;
```

**Do NOT use**: ONNX, pickle, HDF5, or other formats.

## Quality Requirements

### Falsification Criteria (100 points)

| Section | Points | Key Metrics |
|---------|--------|-------------|
| 7.1 Inference Performance | 25 | ≥250 tok/s, P99 ≤10ms |
| 7.2 Model Optimization | 20 | ≥3.5x compression |
| 7.3 Edge Deployment | 15 | ≤50ms WASM, ≤500ms Lambda cold |
| 7.4 API Correctness | 15 | OpenAI compatible |
| 7.5 Reproducibility | 15 | Deterministic outputs |
| 7.6 Code Quality | 10 | ≥95% coverage |

Pass threshold: ≥90 points

### Pre-Commit Checklist

```bash
make lint          # cargo fmt + clippy
make test          # All tests pass
make demo-test     # All demos in CI mode
make compliance    # PMAT compliance
```

## File Organization

```
HF-Production-ML/
├── docs/
│   ├── outline.md                      # Course navigation
│   ├── specifications/
│   │   └── hf-ml-prod-demos.md        # Full specification
│   └── images/
│       ├── week1/                      # SVG diagrams
│       ├── week2/
│       └── week3/
├── demos/
│   ├── week1/                          # Inference serving demos
│   ├── week2/                          # Optimization demos
│   └── week3/                          # Edge deployment demos
├── brick/
│   └── profiles/                       # BrickProfiler output
├── benchmarks/                         # Exported results
├── Cargo.toml                          # Workspace config
├── Makefile                            # Build targets
├── .pmat-metrics.toml                  # Quality thresholds
└── CLAUDE.md                           # This file
```

## Citations

Key papers referenced in demos:

1. **PagedAttention** (Kwon et al., 2023) - Continuous batching
2. **FlashAttention** (Dao et al., 2022) - Memory-efficient attention
3. **Speculative Decoding** (Leviathan et al., 2023) - Draft-verify
4. **GPTQ** (Frantar et al., 2022) - Weight quantization
5. **QLoRA** (Dettmers et al., 2023) - 4-bit NormalFloat

Full citations in `docs/specifications/hf-ml-prod-demos.md` Section 8.

## Troubleshooting

### CUDA not detected
```bash
# Check trueno hardware detection
cargo run --bin demo-hardware-detect
```

### Demo hangs
```bash
# Run in stdout mode for CI
cargo run --bin demo-name -- --stdout
```

### Version mismatch
```bash
# Verify locked versions
cargo tree | grep -E "trueno|aprender|realizar|batuta"

# Update to specification versions
cargo update trueno aprender realizar batuta
```
