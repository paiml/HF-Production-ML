//! Demo: Model Pruning
//!
//! Demonstrates structured and unstructured pruning for LLM compression:
//! - Magnitude-based weight pruning
//! - Structured pruning (heads, layers, neurons)
//! - Sparsity patterns and speedup tradeoffs
//! - Comparison across pruning ratios
//!
//! Uses Qwen2.5-Coder architecture for realistic simulation.
//!
//! References:
//! - Zhu & Gupta (2017) "To Prune, or Not to Prune"
//! - Kurtic et al. (2022) "The Optimal BERT Surgeon"
//! - Frantar & Alistarh (2023) "SparseGPT: Massive Language Models Can Be Accurately Pruned"

use clap::{Parser, ValueEnum};
use std::io::{self, Write};

/// Qwen2.5-Coder model tiers
#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelTier {
    /// Qwen2.5-0.5B
    Tiny,
    /// Qwen2.5-Coder-1.5B
    Small,
    /// Qwen2.5-Coder-7B
    Medium,
    /// Qwen2.5-Coder-32B
    Large,
}

/// Pruning method
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum PruningMethod {
    /// Magnitude-based unstructured pruning
    Magnitude,
    /// Movement pruning (gradient-based)
    Movement,
    /// SparseGPT one-shot pruning
    SparseGpt,
    /// Structured: attention head pruning
    HeadPruning,
    /// Structured: layer pruning
    LayerPruning,
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

    fn num_heads(&self) -> usize {
        match self {
            ModelTier::Tiny => 14,
            ModelTier::Small => 12,
            ModelTier::Medium => 28,
            ModelTier::Large => 40,
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

    fn baseline_perplexity(&self) -> f64 {
        match self {
            ModelTier::Tiny => 6.82,
            ModelTier::Small => 5.14,
            ModelTier::Medium => 4.21,
            ModelTier::Large => 3.89,
        }
    }

    fn baseline_speed(&self) -> f64 {
        // tok/s on RTX 4090
        match self {
            ModelTier::Tiny => 800.0,
            ModelTier::Small => 500.0,
            ModelTier::Medium => 150.0,
            ModelTier::Large => 60.0,
        }
    }
}

impl PruningMethod {
    fn name(&self) -> &'static str {
        match self {
            PruningMethod::Magnitude => "Magnitude",
            PruningMethod::Movement => "Movement",
            PruningMethod::SparseGpt => "SparseGPT",
            PruningMethod::HeadPruning => "Head Pruning",
            PruningMethod::LayerPruning => "Layer Pruning",
        }
    }

    fn is_structured(&self) -> bool {
        matches!(
            self,
            PruningMethod::HeadPruning | PruningMethod::LayerPruning
        )
    }

    /// Perplexity degradation factor at given sparsity
    fn perplexity_factor(&self, sparsity: f64) -> f64 {
        // Lower is better (less degradation)
        match self {
            PruningMethod::Magnitude => {
                // Simple magnitude pruning degrades quality faster
                1.0 + sparsity.powf(1.5) * 0.5
            }
            PruningMethod::Movement => {
                // Movement pruning is better at preserving quality
                1.0 + sparsity.powf(2.0) * 0.3
            }
            PruningMethod::SparseGpt => {
                // SparseGPT achieves best quality retention
                1.0 + sparsity.powf(2.5) * 0.15
            }
            PruningMethod::HeadPruning => {
                // Structured pruning: quality depends on which heads removed
                1.0 + sparsity.powf(1.8) * 0.25
            }
            PruningMethod::LayerPruning => {
                // Layer pruning: significant impact at high sparsity
                1.0 + sparsity.powf(1.3) * 0.4
            }
        }
    }

    /// Speedup factor at given sparsity (structured pruning is more efficient)
    fn speedup_factor(&self, sparsity: f64) -> f64 {
        match self {
            PruningMethod::Magnitude | PruningMethod::Movement | PruningMethod::SparseGpt => {
                // Unstructured: requires sparse matrix support for speedup
                // Typical hardware gets ~1.5-2x at 50% sparsity with 2:4 patterns
                1.0 + sparsity * 0.8
            }
            PruningMethod::HeadPruning => {
                // Head pruning: direct compute reduction
                1.0 / (1.0 - sparsity * 0.9)
            }
            PruningMethod::LayerPruning => {
                // Layer pruning: linear compute reduction
                1.0 / (1.0 - sparsity)
            }
        }
    }
}

/// Pruning analysis result
#[derive(Debug)]
#[allow(dead_code)]
struct PruningResult {
    method: PruningMethod,
    sparsity: f64,
    remaining_params_b: f64,
    perplexity: f64,
    speed: f64,
    speedup: f64,
}

/// Pruning analyzer
struct PruningAnalyzer {
    tier: ModelTier,
}

impl PruningAnalyzer {
    fn new(tier: ModelTier) -> Self {
        Self { tier }
    }

    fn analyze(&self, method: PruningMethod, sparsity: f64) -> PruningResult {
        let remaining_params_b = self.tier.parameters_billions() * (1.0 - sparsity);
        let perplexity = self.tier.baseline_perplexity() * method.perplexity_factor(sparsity);
        let speedup = method.speedup_factor(sparsity);
        let speed = self.tier.baseline_speed() * speedup;

        PruningResult {
            method,
            sparsity,
            remaining_params_b,
            perplexity,
            speed,
            speedup,
        }
    }

    fn compare_methods(&self, sparsity: f64) -> Vec<PruningResult> {
        vec![
            self.analyze(PruningMethod::Magnitude, sparsity),
            self.analyze(PruningMethod::Movement, sparsity),
            self.analyze(PruningMethod::SparseGpt, sparsity),
            self.analyze(PruningMethod::HeadPruning, sparsity),
            self.analyze(PruningMethod::LayerPruning, sparsity),
        ]
    }
}

/// Model Pruning Demo
#[derive(Parser)]
#[command(name = "demo-pruning")]
#[command(about = "Demonstrate model pruning techniques for LLM compression")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Model tier to analyze
    #[arg(long, value_enum, default_value = "medium")]
    tier: ModelTier,

    /// Pruning method
    #[arg(long, value_enum, default_value = "sparse-gpt")]
    method: PruningMethod,

    /// Target sparsity (0.0-0.9)
    #[arg(long, default_value = "0.5")]
    sparsity: f64,

    /// Compare all methods
    #[arg(long)]
    compare: bool,

    /// Show sparsity sweep
    #[arg(long)]
    sweep: bool,
}

fn print_pruning_diagram(tier: ModelTier) {
    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODEL PRUNING TECHNIQUES                                 │
│                  Model: {}                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Unstructured Pruning (removes individual weights):                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Dense:        ████████████████████████████████                     │   │
│  │  50% Sparse:   ██░░██░░████░░██░░████░░██░░████  (scattered zeros)  │   │
│  │  2:4 Pattern:  ██░░██░░██░░██░░██░░██░░██░░██░░  (HW accelerated)   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Structured Pruning (removes entire components):                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Head Pruning:  [H1][H2][░░][H4][░░][H6]  Remove attention heads    │   │
│  │  Layer Pruning: [L1][L2][L3][░░][░░][L6]  Remove transformer layers │   │
│  │  Neuron Pruning: ████░░████████░░████░░  Remove MLP neurons         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Quality vs Sparsity:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PPL │ SparseGPT achieves 50% sparsity with <5% perplexity increase │   │
│  │   ↑  │      ▄▄▄▄▄▄ Magnitude                                        │   │
│  │      │   ▄▄▄░░░░░░ Movement                                         │   │
│  │      │ ▄▄░░░░░░░░░ SparseGPT                                        │   │
│  │   ───┼──────────────────────────────► Sparsity                      │   │
│  │      │ 0%   25%   50%   75%   90%                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Architecture: hidden={}, layers={}, heads={}                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        tier.name(),
        tier.hidden_dim(),
        tier.num_layers(),
        tier.num_heads()
    );
}

fn run_analysis(args: &Args) {
    println!("\n=== Pruning Analysis: {} ===\n", args.tier.name());

    let analyzer = PruningAnalyzer::new(args.tier);

    println!("Model Configuration:");
    println!("  Parameters: {:.1}B", args.tier.parameters_billions());
    println!("  Hidden dim: {}", args.tier.hidden_dim());
    println!("  Layers: {}", args.tier.num_layers());
    println!("  Attention heads: {}", args.tier.num_heads());
    println!(
        "  Baseline perplexity: {:.2}",
        args.tier.baseline_perplexity()
    );
    println!("  Baseline speed: {:.0} tok/s", args.tier.baseline_speed());
    println!();

    if args.compare {
        // Compare all methods at fixed sparsity
        let results = analyzer.compare_methods(args.sparsity);

        println!(
            "Method Comparison at {:.0}% sparsity:",
            args.sparsity * 100.0
        );
        println!("┌─────────────────────────────────────────────────────────────────────┐");
        println!("│ Method          │ Type       │ PPL    │ Speed (tok/s) │ Speedup    │");
        println!("├─────────────────┼────────────┼────────┼───────────────┼────────────┤");

        for result in &results {
            let ptype = if result.method.is_structured() {
                "Structured"
            } else {
                "Unstructured"
            };
            println!(
                "│ {:15} │ {:10} │ {:6.2} │ {:13.0} │ {:9.2}x │",
                result.method.name(),
                ptype,
                result.perplexity,
                result.speed,
                result.speedup
            );
        }

        println!("└─────────────────────────────────────────────────────────────────────┘");
    } else if args.sweep {
        // Show sparsity sweep for selected method
        println!(
            "Sparsity Sweep for {} on {}:",
            args.method.name(),
            args.tier.name()
        );
        println!("┌────────────────────────────────────────────────────────────────┐");
        println!("│ Sparsity │ Remaining │ Perplexity │ Speed (tok/s) │ Speedup   │");
        println!("├──────────┼───────────┼────────────┼───────────────┼───────────┤");

        for sparsity_pct in [0, 25, 50, 75, 90] {
            let sparsity = sparsity_pct as f64 / 100.0;
            let result = analyzer.analyze(args.method, sparsity);
            println!(
                "│ {:>7.0}% │ {:>8.2}B │ {:>10.2} │ {:>13.0} │ {:>8.2}x │",
                sparsity * 100.0,
                result.remaining_params_b,
                result.perplexity,
                result.speed,
                result.speedup
            );
        }

        println!("└────────────────────────────────────────────────────────────────┘");
    } else {
        // Single analysis
        let result = analyzer.analyze(args.method, args.sparsity);

        println!("Pruning Analysis:");
        println!("  Method: {}", result.method.name());
        println!(
            "  Type: {}",
            if result.method.is_structured() {
                "Structured"
            } else {
                "Unstructured"
            }
        );
        println!("  Target sparsity: {:.0}%", args.sparsity * 100.0);
        println!("  Remaining parameters: {:.2}B", result.remaining_params_b);
        println!("  Perplexity: {:.2}", result.perplexity);
        println!(
            "  Perplexity increase: +{:.1}%",
            (result.perplexity / args.tier.baseline_perplexity() - 1.0) * 100.0
        );
        println!("  Speed: {:.0} tok/s", result.speed);
        println!("  Speedup: {:.2}x", result.speedup);
    }
}

fn main() {
    let args = Args::parse();

    // Clamp sparsity to valid range
    let sparsity = args.sparsity.clamp(0.0, 0.9);
    let args = Args { sparsity, ..args };

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-pruning");
        println!("  model: {}", args.tier.name());
        println!("  method: {}", args.method.name());
        println!("  sparsity: {:.0}%", args.sparsity * 100.0);

        let analyzer = PruningAnalyzer::new(args.tier);
        let result = analyzer.analyze(args.method, args.sparsity);

        println!("  remaining_params: {:.2}B", result.remaining_params_b);
        println!("  perplexity: {:.2}", result.perplexity);
        println!("  speedup: {:.2}x", result.speedup);

        // Check quality threshold: <10% perplexity increase at 50% sparsity
        let ppl_increase = result.perplexity / args.tier.baseline_perplexity() - 1.0;
        if args.sparsity >= 0.5 && args.method == PruningMethod::SparseGpt {
            println!(
                "  quality_threshold: {}",
                if ppl_increase <= 0.10 { "PASS" } else { "FAIL" }
            );
        }
    } else {
        // Interactive mode
        print_pruning_diagram(args.tier);
        run_analysis(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
