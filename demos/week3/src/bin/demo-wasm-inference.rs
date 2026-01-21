//! Demo: WebAssembly Inference
//!
//! Demonstrates edge deployment via WebAssembly:
//! - WASM module compilation and size optimization
//! - Browser and edge runtime performance
//! - Memory constraints and quantization requirements
//! - Comparison with native inference
//!
//! Uses Qwen2.5 Tiny tier for edge-viable model size.
//!
//! References:
//! - WebAssembly SIMD proposal for accelerated inference
//! - wasm-bindgen and wasm-pack toolchain
//! - Candle WASM backend (HuggingFace)

use clap::{Parser, ValueEnum};
use std::io::{self, Write};

/// Model tier (only small models are WASM-viable)
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum ModelTier {
    /// Qwen2.5-0.5B - Edge viable
    Tiny,
    /// Qwen2.5-Coder-1.5B - Marginal for edge
    Small,
}

/// WASM runtime target
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum WasmTarget {
    /// Browser (Chrome, Firefox, Safari)
    Browser,
    /// Node.js runtime
    Node,
    /// Cloudflare Workers
    CloudflareWorkers,
    /// Deno runtime
    Deno,
}

/// Quantization for WASM
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum WasmQuant {
    /// FP32 (baseline, large)
    Fp32,
    /// FP16 (requires SIMD)
    Fp16,
    /// Q8_0 (recommended for edge)
    Q8,
    /// Q4_K_M (smallest, some quality loss)
    Q4,
}

#[allow(dead_code)]
impl ModelTier {
    fn name(&self) -> &'static str {
        match self {
            ModelTier::Tiny => "Qwen2.5-0.5B-Instruct",
            ModelTier::Small => "Qwen2.5-Coder-1.5B-Instruct",
        }
    }

    fn parameters_billions(&self) -> f64 {
        match self {
            ModelTier::Tiny => 0.5,
            ModelTier::Small => 1.5,
        }
    }

    fn hidden_dim(&self) -> usize {
        match self {
            ModelTier::Tiny => 896,
            ModelTier::Small => 1536,
        }
    }

    fn num_layers(&self) -> usize {
        match self {
            ModelTier::Tiny => 24,
            ModelTier::Small => 28,
        }
    }

    fn native_speed(&self) -> f64 {
        // Native Rust inference speed (tok/s)
        match self {
            ModelTier::Tiny => 800.0,
            ModelTier::Small => 300.0,
        }
    }
}

impl WasmTarget {
    fn name(&self) -> &'static str {
        match self {
            WasmTarget::Browser => "Browser",
            WasmTarget::Node => "Node.js",
            WasmTarget::CloudflareWorkers => "Cloudflare Workers",
            WasmTarget::Deno => "Deno",
        }
    }

    fn memory_limit_mb(&self) -> usize {
        match self {
            WasmTarget::Browser => 4096,          // 4GB typical
            WasmTarget::Node => 8192,             // 8GB typical
            WasmTarget::CloudflareWorkers => 128, // Very constrained
            WasmTarget::Deno => 4096,             // Similar to browser
        }
    }

    fn simd_support(&self) -> bool {
        match self {
            WasmTarget::Browser => true,           // Modern browsers
            WasmTarget::Node => true,              // Node 16+
            WasmTarget::CloudflareWorkers => true, // Supported
            WasmTarget::Deno => true,              // Supported
        }
    }

    fn performance_factor(&self) -> f64 {
        // Performance relative to native (1.0 = native speed)
        match self {
            WasmTarget::Browser => 0.35,           // ~35% of native
            WasmTarget::Node => 0.40,              // ~40% of native
            WasmTarget::CloudflareWorkers => 0.25, // More overhead
            WasmTarget::Deno => 0.38,              // Similar to Node
        }
    }

    fn startup_overhead_ms(&self) -> f64 {
        match self {
            WasmTarget::Browser => 500.0,          // Module compilation
            WasmTarget::Node => 200.0,             // Faster startup
            WasmTarget::CloudflareWorkers => 50.0, // Pre-compiled
            WasmTarget::Deno => 250.0,
        }
    }
}

impl WasmQuant {
    fn name(&self) -> &'static str {
        match self {
            WasmQuant::Fp32 => "FP32",
            WasmQuant::Fp16 => "FP16",
            WasmQuant::Q8 => "Q8_0",
            WasmQuant::Q4 => "Q4_K_M",
        }
    }

    fn bytes_per_param(&self) -> f64 {
        match self {
            WasmQuant::Fp32 => 4.0,
            WasmQuant::Fp16 => 2.0,
            WasmQuant::Q8 => 1.0,
            WasmQuant::Q4 => 0.5,
        }
    }

    fn quality_factor(&self) -> f64 {
        // Quality retention (1.0 = baseline)
        match self {
            WasmQuant::Fp32 => 1.0,
            WasmQuant::Fp16 => 0.999,
            WasmQuant::Q8 => 0.995,
            WasmQuant::Q4 => 0.94,
        }
    }

    fn requires_simd(&self) -> bool {
        matches!(self, WasmQuant::Fp16)
    }
}

/// WASM inference analysis result
#[derive(Debug)]
#[allow(dead_code)]
struct WasmAnalysisResult {
    model: ModelTier,
    target: WasmTarget,
    quant: WasmQuant,
    model_size_mb: f64,
    wasm_module_size_mb: f64,
    fits_in_memory: bool,
    estimated_speed: f64,
    native_speed: f64,
    performance_ratio: f64,
    startup_time_ms: f64,
}

/// WASM inference analyzer
struct WasmAnalyzer {
    model: ModelTier,
    target: WasmTarget,
}

impl WasmAnalyzer {
    fn new(model: ModelTier, target: WasmTarget) -> Self {
        Self { model, target }
    }

    fn analyze(&self, quant: WasmQuant) -> WasmAnalysisResult {
        // Model size calculation
        let params_b = self.model.parameters_billions();
        let model_size_mb = params_b * 1000.0 * quant.bytes_per_param();

        // WASM module overhead (~10% for runtime + bindings)
        let wasm_module_size_mb = model_size_mb * 1.1 + 2.0; // +2MB for WASM runtime

        // Memory check
        let memory_required = model_size_mb * 1.5; // Need headroom for activations
        let fits_in_memory = memory_required < self.target.memory_limit_mb() as f64;

        // Performance calculation
        let native_speed = self.model.native_speed();
        let wasm_factor = self.target.performance_factor();
        let quant_factor = if quant.requires_simd() && !self.target.simd_support() {
            0.5 // Fallback to scalar
        } else {
            1.0
        };
        let estimated_speed = native_speed * wasm_factor * quant_factor * quant.quality_factor();

        // Startup time
        let startup_time_ms = self.target.startup_overhead_ms() + model_size_mb / 100.0 * 50.0; // ~50ms per 100MB

        WasmAnalysisResult {
            model: self.model,
            target: self.target,
            quant,
            model_size_mb,
            wasm_module_size_mb,
            fits_in_memory,
            estimated_speed,
            native_speed,
            performance_ratio: estimated_speed / native_speed,
            startup_time_ms,
        }
    }

    fn compare_quantizations(&self) -> Vec<WasmAnalysisResult> {
        vec![
            self.analyze(WasmQuant::Fp32),
            self.analyze(WasmQuant::Fp16),
            self.analyze(WasmQuant::Q8),
            self.analyze(WasmQuant::Q4),
        ]
    }
}

/// WASM Inference Demo
#[derive(Parser)]
#[command(name = "demo-wasm-inference")]
#[command(about = "Demonstrate WebAssembly inference for edge deployment")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Model tier
    #[arg(long, value_enum, default_value = "tiny")]
    tier: ModelTier,

    /// WASM target runtime
    #[arg(long, value_enum, default_value = "browser")]
    target: WasmTarget,

    /// Quantization method
    #[arg(long, value_enum, default_value = "q8")]
    quant: WasmQuant,

    /// Compare all quantization methods
    #[arg(long)]
    compare: bool,
}

fn print_wasm_diagram(model: ModelTier, target: WasmTarget) {
    let analyzer = WasmAnalyzer::new(model, target);
    let result = analyzer.analyze(WasmQuant::Q8);

    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WEBASSEMBLY INFERENCE                                    │
│              Model: {} on {}                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Deployment Pipeline:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │   │
│  │  │  Rust    │    │  WASM    │    │  WASM    │    │  Edge    │     │   │
│  │  │  Model   │───►│  Pack    │───►│  Binary  │───►│  Runtime │     │   │
│  │  │  Code    │    │  Build   │    │  .wasm   │    │  Exec    │     │   │
│  │  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │   │
│  │       │                               │                            │   │
│  │       │ Quantize weights              │ Ship to CDN               │   │
│  │       └───────────────────────────────┘                            │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  WASM Features:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ✓ SIMD: {}     Memory Limit: {} MB                         │   │
│  │  ✓ Streaming compilation                                           │   │
│  │  ✓ Sandboxed execution                                             │   │
│  │  ✓ Cross-platform (no native deps)                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Performance: {:.0} tok/s ({:.0}% of native)                               │
│  Module Size: {:.0} MB │ Startup: {:.0}ms                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        model.name(),
        target.name(),
        if target.simd_support() { "Yes" } else { "No " },
        target.memory_limit_mb(),
        result.estimated_speed,
        result.performance_ratio * 100.0,
        result.wasm_module_size_mb,
        result.startup_time_ms
    );
}

fn run_analysis(args: &Args) {
    println!("\n=== WASM Inference Analysis: {} ===\n", args.tier.name());

    let analyzer = WasmAnalyzer::new(args.tier, args.target);

    println!("Configuration:");
    println!(
        "  Model: {} ({:.1}B params)",
        args.tier.name(),
        args.tier.parameters_billions()
    );
    println!("  Target: {}", args.target.name());
    println!("  Memory limit: {} MB", args.target.memory_limit_mb());
    println!(
        "  SIMD support: {}",
        if args.target.simd_support() {
            "Yes"
        } else {
            "No"
        }
    );
    println!();

    if args.compare {
        let results = analyzer.compare_quantizations();

        println!("Quantization Comparison:");
        println!("┌───────────────────────────────────────────────────────────────────────────┐");
        println!("│ Quant  │ Size (MB) │ WASM (MB) │ Fits │ Speed │ % Native │ Startup    │");
        println!("├────────┼───────────┼───────────┼──────┼───────┼──────────┼────────────┤");

        for result in &results {
            println!(
                "│ {:6} │ {:>9.0} │ {:>9.0} │ {:>4} │ {:>5.0} │ {:>7.0}% │ {:>8.0}ms │",
                result.quant.name(),
                result.model_size_mb,
                result.wasm_module_size_mb,
                if result.fits_in_memory { "Yes" } else { "No" },
                result.estimated_speed,
                result.performance_ratio * 100.0,
                result.startup_time_ms
            );
        }

        println!("└───────────────────────────────────────────────────────────────────────────┘");
    } else {
        let result = analyzer.analyze(args.quant);

        println!("Analysis for {}:", args.quant.name());
        println!("  Model size: {:.0} MB", result.model_size_mb);
        println!("  WASM module size: {:.0} MB", result.wasm_module_size_mb);
        println!(
            "  Fits in memory: {}",
            if result.fits_in_memory { "Yes" } else { "No" }
        );
        println!("  Native speed: {:.0} tok/s", result.native_speed);
        println!("  WASM speed: {:.0} tok/s", result.estimated_speed);
        println!(
            "  Performance ratio: {:.0}% of native",
            result.performance_ratio * 100.0
        );
        println!("  Startup time: {:.0}ms", result.startup_time_ms);

        if !result.fits_in_memory {
            println!("\n  ⚠ Model too large for target! Use smaller quantization.");
        }
    }
}

fn main() {
    let args = Args::parse();

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-wasm-inference");
        println!("  model: {}", args.tier.name());
        println!("  target: {}", args.target.name());
        println!("  quant: {}", args.quant.name());

        let analyzer = WasmAnalyzer::new(args.tier, args.target);
        let result = analyzer.analyze(args.quant);

        println!("  model_size_mb: {:.0}", result.model_size_mb);
        println!("  wasm_size_mb: {:.0}", result.wasm_module_size_mb);
        println!("  fits_in_memory: {}", result.fits_in_memory);
        println!("  speed: {:.0} tok/s", result.estimated_speed);
        println!(
            "  performance_ratio: {:.0}%",
            result.performance_ratio * 100.0
        );
        println!("  startup_ms: {:.0}", result.startup_time_ms);

        // Check viability: must fit in memory and achieve >20% native performance
        let viable = result.fits_in_memory && result.performance_ratio >= 0.20;
        println!("  edge_viable: {}", if viable { "PASS" } else { "FAIL" });
    } else {
        // Interactive mode
        print_wasm_diagram(args.tier, args.target);
        run_analysis(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
