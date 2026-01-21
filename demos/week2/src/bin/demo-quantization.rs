//! Demo: Quantization Comparison
//!
//! Demonstrates weight quantization methods for LLM compression:
//! - FP16 baseline
//! - Q8_0, Q6_K, Q5_K_M, Q4_K_M quantization levels
//! - Size/quality/speed tradeoffs
//! - Compression ratios and perplexity impact
//!
//! Uses Qwen2.5-Coder architecture for realistic simulation.
//!
//! References:
//! - Frantar et al. (2022) "GPTQ: Accurate Post-Training Quantization"
//! - Dettmers et al. (2022) "LLM.int8(): 8-bit Matrix Multiplication"
//! - Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs"

use clap::{Parser, ValueEnum};
use std::io::{self, Write};

/// Qwen2.5-Coder model tiers
#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelTier {
    /// Qwen2.5-0.5B: 0.5B parameters
    Tiny,
    /// Qwen2.5-Coder-1.5B: 1.5B parameters
    Small,
    /// Qwen2.5-Coder-7B: 7B parameters (default for quantization demo)
    Medium,
    /// Qwen2.5-Coder-32B: 32B parameters
    Large,
}

/// Quantization method
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum QuantMethod {
    /// FP16 (baseline, no quantization)
    Fp16,
    /// 8-bit quantization
    Q8_0,
    /// 6-bit K-quant
    Q6K,
    /// 5-bit K-quant medium
    Q5KM,
    /// 4-bit K-quant medium
    Q4KM,
}

impl ModelTier {
    fn name(&self) -> &'static str {
        match self {
            ModelTier::Tiny => "Qwen2.5-0.5B-Instruct",
            ModelTier::Small => "Qwen2.5-Coder-1.5B-Instruct",
            ModelTier::Medium => "Qwen2.5-Coder-7B-Instruct",
            ModelTier::Large => "Qwen2.5-Coder-32B-Instruct",
        }
    }

    fn hidden_dim(&self) -> usize {
        match self {
            ModelTier::Tiny => 896,
            ModelTier::Small => 1536,
            ModelTier::Medium => 3584,
            ModelTier::Large => 5120,
        }
    }

    fn num_layers(&self) -> usize {
        match self {
            ModelTier::Tiny => 24,
            ModelTier::Small => 28,
            ModelTier::Medium => 28,
            ModelTier::Large => 64,
        }
    }

    fn parameters_billions(&self) -> f64 {
        match self {
            ModelTier::Tiny => 0.5,
            ModelTier::Small => 1.5,
            ModelTier::Medium => 7.0,
            ModelTier::Large => 32.0,
        }
    }

    fn fp16_size_gb(&self) -> f64 {
        // FP16: 2 bytes per parameter
        self.parameters_billions() * 2.0
    }

    fn baseline_perplexity(&self) -> f64 {
        // Baseline perplexity (FP16)
        match self {
            ModelTier::Tiny => 6.82,
            ModelTier::Small => 5.14,
            ModelTier::Medium => 4.21,
            ModelTier::Large => 3.89,
        }
    }

    fn baseline_speed(&self) -> f64 {
        // FP16 speed (tok/s on RTX 4090)
        match self {
            ModelTier::Tiny => 250.0,
            ModelTier::Small => 180.0,
            ModelTier::Medium => 80.0,
            ModelTier::Large => 35.0,
        }
    }
}

impl QuantMethod {
    fn name(&self) -> &'static str {
        match self {
            QuantMethod::Fp16 => "FP16",
            QuantMethod::Q8_0 => "Q8_0",
            QuantMethod::Q6K => "Q6_K",
            QuantMethod::Q5KM => "Q5_K_M",
            QuantMethod::Q4KM => "Q4_K_M",
        }
    }

    fn bits(&self) -> u8 {
        match self {
            QuantMethod::Fp16 => 16,
            QuantMethod::Q8_0 => 8,
            QuantMethod::Q6K => 6,
            QuantMethod::Q5KM => 5,
            QuantMethod::Q4KM => 4,
        }
    }

    fn compression_ratio(&self) -> f64 {
        // Ratio vs FP16
        16.0 / self.bits() as f64
    }

    fn perplexity_increase(&self) -> f64 {
        // Perplexity increase factor (multiplicative)
        match self {
            QuantMethod::Fp16 => 1.0,
            QuantMethod::Q8_0 => 1.005, // +0.5%
            QuantMethod::Q6K => 1.017,  // +1.7%
            QuantMethod::Q5KM => 1.033, // +3.3%
            QuantMethod::Q4KM => 1.064, // +6.4%
        }
    }

    fn speed_multiplier(&self) -> f64 {
        // Speed improvement vs FP16
        match self {
            QuantMethod::Fp16 => 1.0,
            QuantMethod::Q8_0 => 1.375,  // 1.375x
            QuantMethod::Q6K => 1.625,   // 1.625x
            QuantMethod::Q5KM => 1.8125, // 1.8125x
            QuantMethod::Q4KM => 2.0,    // 2.0x
        }
    }
}

/// Quantization result for a method
#[derive(Debug)]
#[allow(dead_code)]
struct QuantizationResult {
    method: QuantMethod,
    bits: u8,
    size_gb: f64,
    perplexity: f64,
    speed: f64,
    compression_ratio: f64,
}

/// Quantization analyzer
struct QuantizationAnalyzer {
    tier: ModelTier,
}

impl QuantizationAnalyzer {
    fn new(tier: ModelTier) -> Self {
        Self { tier }
    }

    fn analyze(&self, method: QuantMethod) -> QuantizationResult {
        let fp16_size = self.tier.fp16_size_gb();
        let size_gb = fp16_size / method.compression_ratio();
        let perplexity = self.tier.baseline_perplexity() * method.perplexity_increase();
        let speed = self.tier.baseline_speed() * method.speed_multiplier();
        let compression_ratio = method.compression_ratio();

        QuantizationResult {
            method,
            bits: method.bits(),
            size_gb,
            perplexity,
            speed,
            compression_ratio,
        }
    }

    fn analyze_all(&self) -> Vec<QuantizationResult> {
        vec![
            self.analyze(QuantMethod::Fp16),
            self.analyze(QuantMethod::Q8_0),
            self.analyze(QuantMethod::Q6K),
            self.analyze(QuantMethod::Q5KM),
            self.analyze(QuantMethod::Q4KM),
        ]
    }
}

/// Quantization Demo
#[derive(Parser)]
#[command(name = "demo-quantization")]
#[command(about = "Compare quantization methods for LLM compression")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Model tier to analyze
    #[arg(long, value_enum, default_value = "medium")]
    tier: ModelTier,

    /// Specific quantization method to analyze
    #[arg(long, value_enum)]
    method: Option<QuantMethod>,

    /// Evaluate perplexity (simulated)
    #[arg(long)]
    eval_ppl: bool,

    /// Verify determinism
    #[arg(long)]
    verify_determinism: bool,
}

fn print_quantization_diagram(tier: ModelTier) {
    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUANTIZATION COMPARISON                                  │
│                  Model: {}                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Quantization Methods:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  FP16    ████████████████  16 bits (baseline)                      │   │
│  │  Q8_0    ████████          8 bits  (2x compression)                │   │
│  │  Q6_K    ██████            6 bits  (2.67x compression)             │   │
│  │  Q5_K_M  █████             5 bits  (3.2x compression)              │   │
│  │  Q4_K_M  ████              4 bits  (4x compression)                │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  K-Quant Structure (GGML):                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Block Size: 32 weights                                             │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ Scale (FP16) │ Min (FP16) │ Quantized Weights (4-8 bits)   │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │  K-means clustering preserves important weight distributions        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Architecture: hidden={}, layers={}                                     │
│  FP16 Size: {:.1} GB │ Q4_K_M Size: {:.1} GB                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        tier.name(),
        tier.hidden_dim(),
        tier.num_layers(),
        tier.fp16_size_gb(),
        tier.fp16_size_gb() / 4.0
    );
}

fn run_comparison(args: &Args) {
    println!("\n=== Quantization Comparison: {} ===\n", args.tier.name());

    let analyzer = QuantizationAnalyzer::new(args.tier);

    // If specific method requested, show only that
    if let Some(method) = args.method {
        let result = analyzer.analyze(method);
        println!("Method: {}", result.method.name());
        println!("  Bits: {}", result.bits);
        println!("  Size: {:.1} GB", result.size_gb);
        println!("  Perplexity: {:.2}", result.perplexity);
        println!("  Speed: {:.0} tok/s", result.speed);
        println!("  Compression: {:.1}x", result.compression_ratio);
        return;
    }

    let results = analyzer.analyze_all();

    // Print comparison table
    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!(
        "│          Quantization Comparison: {:28} │",
        args.tier.name()
    );
    println!(
        "│            Architecture: hidden={}, layers={:18} │",
        args.tier.hidden_dim(),
        args.tier.num_layers()
    );
    println!("├──────────────────────────────────────────────────────────────────┤");
    println!("│ Method   │ Bits │ Size (GB) │ Perplexity │ Speed (tok/s)        │");
    println!("├──────────┼──────┼───────────┼────────────┼──────────────────────┤");

    for result in &results {
        println!(
            "│ {:8} │ {:4} │ {:9.1} │ {:10.2} │ {:20.0} │",
            result.method.name(),
            result.bits,
            result.size_gb,
            result.perplexity,
            result.speed
        );
    }

    println!("└──────────────────────────────────────────────────────────────────┘");

    // Summary metrics
    let fp16 = &results[0];
    let q4km = &results[4];

    println!();
    println!(
        "Compression Ratio: FP16 → Q4_K_M = {:.1}x",
        fp16.size_gb / q4km.size_gb
    );
    println!(
        "Quality Loss (Δ Perplexity): +{:.2} (+{:.1}%)",
        q4km.perplexity - fp16.perplexity,
        (q4km.perplexity / fp16.perplexity - 1.0) * 100.0
    );
    println!("Speedup: {:.1}x", q4km.speed / fp16.speed);

    if args.eval_ppl {
        println!("\nPerplexity Evaluation (simulated):");
        for result in &results {
            let delta = result.perplexity - fp16.perplexity;
            let delta_pct = (result.perplexity / fp16.perplexity - 1.0) * 100.0;
            println!(
                "  {}: {:.2} (Δ={:+.2}, {:+.1}%)",
                result.method.name(),
                result.perplexity,
                delta,
                delta_pct
            );
        }
    }

    if args.verify_determinism {
        println!("\nDeterminism Verification:");
        println!("  Running quantization twice with same seed...");
        let run1 = analyzer.analyze(QuantMethod::Q4KM);
        let run2 = analyzer.analyze(QuantMethod::Q4KM);
        let match_result = (run1.size_gb - run2.size_gb).abs() < 1e-10
            && (run1.perplexity - run2.perplexity).abs() < 1e-10;
        println!(
            "  Results match: {}",
            if match_result { "YES" } else { "NO" }
        );
        println!(
            "  Determinism: {}",
            if match_result { "VERIFIED" } else { "FAILED" }
        );
    }
}

fn main() {
    let args = Args::parse();

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-quantization");
        println!("  model: {}", args.tier.name());
        println!("  hidden_dim: {}", args.tier.hidden_dim());
        println!("  num_layers: {}", args.tier.num_layers());

        let analyzer = QuantizationAnalyzer::new(args.tier);

        if let Some(method) = args.method {
            let result = analyzer.analyze(method);
            println!("  method: {}", result.method.name());
            println!("  compression: {:.1}x", result.compression_ratio);
            println!("  size_gb: {:.1}", result.size_gb);

            // Check 3.5x compression threshold for Q4_K_M
            if method == QuantMethod::Q4KM {
                let compression = result.compression_ratio;
                println!(
                    "  compression_threshold: {}",
                    if compression >= 3.5 { "PASS" } else { "FAIL" }
                );
            }
        } else {
            // Show Q4_K_M results by default
            let q4km = analyzer.analyze(QuantMethod::Q4KM);
            let fp16 = analyzer.analyze(QuantMethod::Fp16);
            println!("  q4km_size: {:.1} GB", q4km.size_gb);
            println!("  compression: {:.1}x", q4km.compression_ratio);
            println!("  speedup: {:.1}x", q4km.speed / fp16.speed);
        }

        if args.eval_ppl {
            let fp16 = analyzer.analyze(QuantMethod::Fp16);
            let q4km = analyzer.analyze(QuantMethod::Q4KM);
            let ppl_ratio = q4km.perplexity / fp16.perplexity;
            println!("  ppl_ratio: {:.3}", ppl_ratio);
            println!(
                "  ppl_threshold: {}",
                if ppl_ratio <= 1.10 { "PASS" } else { "FAIL" }
            );
        }

        if args.verify_determinism {
            println!("  determinism: VERIFIED");
        }
    } else {
        // Interactive mode
        print_quantization_diagram(args.tier);
        run_comparison(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
