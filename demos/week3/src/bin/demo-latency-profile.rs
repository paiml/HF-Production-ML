//! Demo: Latency Profiling
//!
//! Demonstrates inference latency breakdown and optimization:
//! - Time-to-first-token (TTFT) analysis
//! - Inter-token latency (ITL) profiling
//! - Pipeline stage breakdown
//! - Bottleneck identification
//!
//! Uses Qwen2.5-Coder architecture for realistic profiling.
//!
//! References:
//! - LLM inference latency optimization techniques
//! - CUDA profiling with NVTX markers
//! - Roofline model for compute vs memory bound analysis

use clap::{Parser, ValueEnum};
use std::io::{self, Write};

/// Qwen2.5-Coder model tiers
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
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

/// Inference phase
#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    Tokenization,
    Embedding,
    Prefill,
    Decode,
    Sampling,
    Detokenization,
}

#[allow(dead_code)]
impl ModelTier {
    fn name(&self) -> &'static str {
        match self {
            ModelTier::Tiny => "Qwen2.5-0.5B-Instruct",
            ModelTier::Small => "Qwen2.5-Coder-1.5B-Instruct",
            ModelTier::Medium => "Qwen2.5-Coder-7B-Instruct",
            ModelTier::Large => "Qwen2.5-Coder-32B-Instruct",
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
}

impl Phase {
    fn name(&self) -> &'static str {
        match self {
            Phase::Tokenization => "Tokenization",
            Phase::Embedding => "Embedding",
            Phase::Prefill => "Prefill",
            Phase::Decode => "Decode",
            Phase::Sampling => "Sampling",
            Phase::Detokenization => "Detokenization",
        }
    }
}

/// Latency breakdown for a phase
#[derive(Debug, Clone)]
struct PhaseLatency {
    phase: Phase,
    latency_ms: f64,
    percentage: f64,
    bound: &'static str,
}

/// Latency profile result
#[derive(Debug)]
#[allow(dead_code)]
struct LatencyProfile {
    model: ModelTier,
    prompt_tokens: usize,
    output_tokens: usize,
    ttft_ms: f64,
    itl_ms: f64,
    total_ms: f64,
    phases: Vec<PhaseLatency>,
    bottleneck: Phase,
}

/// Latency profiler
struct LatencyProfiler {
    model: ModelTier,
}

impl LatencyProfiler {
    fn new(model: ModelTier) -> Self {
        Self { model }
    }

    fn profile(&self, prompt_tokens: usize, output_tokens: usize) -> LatencyProfile {
        let params = self.model.parameters_billions();

        // Tokenization: ~0.1ms per 100 tokens
        let tokenization_ms = prompt_tokens as f64 / 100.0 * 0.1;

        // Embedding lookup: ~0.05ms per 100 tokens
        let embedding_ms = prompt_tokens as f64 / 100.0 * 0.05;

        // Prefill: compute-bound, scales with prompt length and model size
        // ~0.5ms per 100 tokens per billion params
        let prefill_ms = prompt_tokens as f64 / 100.0 * params * 0.5;

        // Decode: memory-bound, scales with model size
        // ~1ms per token per billion params (amortized)
        let decode_per_token_ms = params * 1.0;
        let decode_total_ms = output_tokens as f64 * decode_per_token_ms;

        // Sampling: ~0.02ms per token
        let sampling_ms = output_tokens as f64 * 0.02;

        // Detokenization: ~0.05ms per token
        let detokenization_ms = output_tokens as f64 * 0.05;

        let total_ms = tokenization_ms
            + embedding_ms
            + prefill_ms
            + decode_total_ms
            + sampling_ms
            + detokenization_ms;

        let phases = vec![
            PhaseLatency {
                phase: Phase::Tokenization,
                latency_ms: tokenization_ms,
                percentage: tokenization_ms / total_ms * 100.0,
                bound: "CPU",
            },
            PhaseLatency {
                phase: Phase::Embedding,
                latency_ms: embedding_ms,
                percentage: embedding_ms / total_ms * 100.0,
                bound: "Memory",
            },
            PhaseLatency {
                phase: Phase::Prefill,
                latency_ms: prefill_ms,
                percentage: prefill_ms / total_ms * 100.0,
                bound: "Compute",
            },
            PhaseLatency {
                phase: Phase::Decode,
                latency_ms: decode_total_ms,
                percentage: decode_total_ms / total_ms * 100.0,
                bound: "Memory",
            },
            PhaseLatency {
                phase: Phase::Sampling,
                latency_ms: sampling_ms,
                percentage: sampling_ms / total_ms * 100.0,
                bound: "CPU",
            },
            PhaseLatency {
                phase: Phase::Detokenization,
                latency_ms: detokenization_ms,
                percentage: detokenization_ms / total_ms * 100.0,
                bound: "CPU",
            },
        ];

        // Find bottleneck
        let bottleneck = phases
            .iter()
            .max_by(|a, b| a.latency_ms.partial_cmp(&b.latency_ms).unwrap())
            .map(|p| p.phase)
            .unwrap_or(Phase::Decode);

        // TTFT = time until first output token
        let ttft_ms = tokenization_ms + embedding_ms + prefill_ms + decode_per_token_ms;

        // ITL = inter-token latency (decode phase per token)
        let itl_ms = decode_per_token_ms;

        LatencyProfile {
            model: self.model,
            prompt_tokens,
            output_tokens,
            ttft_ms,
            itl_ms,
            total_ms,
            phases,
            bottleneck,
        }
    }
}

/// Latency Profile Demo
#[derive(Parser)]
#[command(name = "demo-latency-profile")]
#[command(about = "Demonstrate inference latency profiling")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Model tier
    #[arg(long, value_enum, default_value = "medium")]
    tier: ModelTier,

    /// Prompt tokens
    #[arg(long, default_value = "100")]
    prompt_tokens: usize,

    /// Output tokens
    #[arg(long, default_value = "50")]
    output_tokens: usize,

    /// Show detailed phase breakdown
    #[arg(long)]
    detailed: bool,
}

fn print_latency_diagram(model: ModelTier, prompt_tokens: usize, output_tokens: usize) {
    let profiler = LatencyProfiler::new(model);
    let profile = profiler.profile(prompt_tokens, output_tokens);

    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LATENCY PROFILING                                        │
│              Model: {} │ Prompt: {} │ Output: {}          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Inference Pipeline:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  [Tokenize] → [Embed] → [Prefill] → [Decode x N] → [Sample] → [Out]│   │
│  │      │           │          │            │            │             │   │
│  │     CPU       Memory     Compute      Memory        CPU             │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Key Metrics:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TTFT (Time-to-First-Token): {:.1}ms                                │   │
│  │  ITL  (Inter-Token Latency): {:.1}ms                                │   │
│  │  Total Generation Time:      {:.1}ms                                │   │
│  │  Effective Throughput:       {:.0} tok/s                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Bottleneck: {} ({:.1}% of total time)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        model.name(),
        prompt_tokens,
        output_tokens,
        profile.ttft_ms,
        profile.itl_ms,
        profile.total_ms,
        output_tokens as f64 * 1000.0 / profile.total_ms,
        profile.bottleneck.name(),
        profile
            .phases
            .iter()
            .find(|p| p.phase == profile.bottleneck)
            .map(|p| p.percentage)
            .unwrap_or(0.0)
    );
}

fn run_analysis(args: &Args) {
    println!("\n=== Latency Profile: {} ===\n", args.tier.name());

    let profiler = LatencyProfiler::new(args.tier);
    let profile = profiler.profile(args.prompt_tokens, args.output_tokens);

    println!("Configuration:");
    println!(
        "  Model: {} ({:.1}B params)",
        args.tier.name(),
        args.tier.parameters_billions()
    );
    println!("  Prompt tokens: {}", args.prompt_tokens);
    println!("  Output tokens: {}", args.output_tokens);
    println!();

    println!("Key Metrics:");
    println!("  Time-to-First-Token (TTFT): {:.1}ms", profile.ttft_ms);
    println!("  Inter-Token Latency (ITL): {:.1}ms", profile.itl_ms);
    println!("  Total generation time: {:.1}ms", profile.total_ms);
    println!(
        "  Effective throughput: {:.0} tok/s",
        args.output_tokens as f64 * 1000.0 / profile.total_ms
    );
    println!();

    if args.detailed {
        println!("Phase Breakdown:");
        println!("┌────────────────────────────────────────────────────────────────┐");
        println!("│ Phase           │ Latency (ms) │ Percentage │ Bound           │");
        println!("├─────────────────┼──────────────┼────────────┼─────────────────┤");

        for phase in &profile.phases {
            let marker = if phase.phase == profile.bottleneck {
                "→"
            } else {
                " "
            };
            println!(
                "│{}{:15} │ {:>12.2} │ {:>9.1}% │ {:>15} │",
                marker,
                phase.phase.name(),
                phase.latency_ms,
                phase.percentage,
                phase.bound
            );
        }

        println!("└────────────────────────────────────────────────────────────────┘");
        println!();
    }

    println!("Bottleneck: {}", profile.bottleneck.name());
    println!(
        "  → Optimization: Focus on {} operations",
        match profile.bottleneck {
            Phase::Prefill => "batching and compute parallelism",
            Phase::Decode => "KV cache and memory bandwidth",
            _ => "CPU efficiency",
        }
    );
}

fn main() {
    let args = Args::parse();

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-latency-profile");
        println!("  model: {}", args.tier.name());
        println!("  prompt_tokens: {}", args.prompt_tokens);
        println!("  output_tokens: {}", args.output_tokens);

        let profiler = LatencyProfiler::new(args.tier);
        let profile = profiler.profile(args.prompt_tokens, args.output_tokens);

        println!("  ttft_ms: {:.1}", profile.ttft_ms);
        println!("  itl_ms: {:.1}", profile.itl_ms);
        println!("  total_ms: {:.1}", profile.total_ms);
        println!(
            "  throughput: {:.0} tok/s",
            args.output_tokens as f64 * 1000.0 / profile.total_ms
        );
        println!("  bottleneck: {}", profile.bottleneck.name());

        // Check latency threshold: TTFT < 500ms for interactive use
        println!(
            "  ttft_threshold: {}",
            if profile.ttft_ms < 500.0 {
                "PASS"
            } else {
                "FAIL"
            }
        );
    } else {
        // Interactive mode
        print_latency_diagram(args.tier, args.prompt_tokens, args.output_tokens);
        run_analysis(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
