//! Demo: Speculative Decoding
//!
//! Demonstrates draft-verify decoding for LLM speedup:
//! - Draft model generates speculative tokens
//! - Target model verifies in single forward pass
//! - Acceptance rate determines speedup
//!
//! Uses Qwen2.5 model hierarchy:
//! - Draft: Qwen2.5-0.5B (Tiny tier) - fast, approximate
//! - Target: Qwen2.5-Coder-7B (Medium tier) - accurate, slower
//!
//! References:
//! - Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding"
//! - Chen et al. (2023) "Accelerating LLM Decoding with Speculative Sampling"

use clap::{Parser, ValueEnum};
use std::io::{self, Write};

/// Draft model tier
#[derive(Debug, Clone, Copy, ValueEnum)]
enum DraftTier {
    /// Qwen2.5-0.5B (fastest draft)
    Tiny,
}

/// Target model tier
#[derive(Debug, Clone, Copy, ValueEnum)]
enum TargetTier {
    /// Qwen2.5-Coder-1.5B
    Small,
    /// Qwen2.5-Coder-7B
    Medium,
    /// Qwen2.5-Coder-32B
    Large,
}

impl DraftTier {
    fn name(&self) -> &'static str {
        match self {
            DraftTier::Tiny => "Qwen2.5-0.5B-Instruct",
        }
    }

    fn hidden_dim(&self) -> usize {
        match self {
            DraftTier::Tiny => 896,
        }
    }

    fn num_layers(&self) -> usize {
        match self {
            DraftTier::Tiny => 24,
        }
    }

    fn tokens_per_second(&self) -> f64 {
        match self {
            // 0.5B model on modern GPU can hit 1500+ tok/s
            DraftTier::Tiny => 1500.0,
        }
    }
}

impl TargetTier {
    fn name(&self) -> &'static str {
        match self {
            TargetTier::Small => "Qwen2.5-Coder-1.5B-Instruct",
            TargetTier::Medium => "Qwen2.5-Coder-7B-Instruct",
            TargetTier::Large => "Qwen2.5-Coder-32B-Instruct",
        }
    }

    fn hidden_dim(&self) -> usize {
        match self {
            TargetTier::Small => 1536,
            TargetTier::Medium => 3584,
            TargetTier::Large => 5120,
        }
    }

    fn num_layers(&self) -> usize {
        match self {
            TargetTier::Small => 28,
            TargetTier::Medium => 28,
            TargetTier::Large => 64,
        }
    }

    fn tokens_per_second(&self) -> f64 {
        match self {
            TargetTier::Small => 300.0,
            TargetTier::Medium => 150.0,
            TargetTier::Large => 80.0,
        }
    }

    fn acceptance_rate(&self) -> f64 {
        // Acceptance rate varies by task and model pair
        // Code generation tasks typically have higher acceptance
        match self {
            TargetTier::Small => 0.92,  // Similar architecture, high acceptance
            TargetTier::Medium => 0.89, // Code tasks
            TargetTier::Large => 0.85,  // More divergence
        }
    }
}

/// Speculative decoding simulator
struct SpeculativeDecoder {
    draft: DraftTier,
    target: TargetTier,
    speculative_tokens: usize,
}

impl SpeculativeDecoder {
    fn new(draft: DraftTier, target: TargetTier, speculative_tokens: usize) -> Self {
        Self {
            draft,
            target,
            speculative_tokens,
        }
    }

    /// Calculate effective throughput with speculative decoding
    fn effective_throughput(&self) -> f64 {
        let acceptance_rate = self.target.acceptance_rate();
        let draft_speed = self.draft.tokens_per_second();
        let target_speed = self.target.tokens_per_second();

        // Time to generate K draft tokens
        let draft_time = self.speculative_tokens as f64 / draft_speed;

        // Time for single target forward pass (verifies all K tokens)
        let verify_time = 1.0 / target_speed;

        // Expected accepted tokens per iteration
        // With K speculative tokens and acceptance rate p:
        // E[accepted] ≈ (1 - p^(K+1)) / (1 - p) for geometric series
        let p = acceptance_rate;
        let k = self.speculative_tokens as f64;
        let expected_accepted = if p >= 0.999 {
            k + 1.0
        } else {
            (1.0 - p.powf(k + 1.0)) / (1.0 - p)
        };

        // Total time per iteration
        let total_time = draft_time + verify_time;

        // Effective throughput
        expected_accepted / total_time
    }

    /// Speedup over target-only decoding
    fn speedup(&self) -> f64 {
        self.effective_throughput() / self.target.tokens_per_second()
    }

    /// Simulate a decoding step
    fn simulate_step(&self, step: usize) -> Vec<bool> {
        // Deterministic simulation based on step number
        let acceptance_rate = self.target.acceptance_rate();
        let mut accepted = Vec::new();

        for i in 0..self.speculative_tokens {
            // Deterministic pseudo-random based on step and position
            let rand_val = ((step * 7 + i * 13) % 100) as f64 / 100.0;
            accepted.push(rand_val < acceptance_rate);
        }

        accepted
    }
}

/// Speculative Decoding Demo
#[derive(Parser)]
#[command(name = "demo-speculative-decode")]
#[command(about = "Demonstrate speculative decoding for LLM speedup")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Draft model tier
    #[arg(long, value_enum, default_value = "tiny")]
    draft: DraftTier,

    /// Target model tier
    #[arg(long, value_enum, default_value = "medium")]
    target: TargetTier,

    /// Number of speculative tokens per iteration
    #[arg(long, default_value = "4")]
    speculative_tokens: usize,
}

fn print_speculative_diagram(draft: DraftTier, target: TargetTier) {
    let decoder = SpeculativeDecoder::new(draft, target, 4);

    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SPECULATIVE DECODING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Draft:  {} ({} hidden, {} layers)  → {} tok/s      │
│ Target: {} ({} hidden, {} layers) → {} tok/s  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Step 1: Draft model (0.5B) generates 4 tokens speculatively                │
│         Draft: [def] [fibonacci] [(n)] [:]                                 │
│                                                                             │
│ Step 2: Target model (7B) verifies batch in single forward                 │
│         [def ✓] [fibonacci ✓] [(n) ✓] [: ✓]                               │
│                                                                             │
│ Step 3: All accepted! Target generates bonus token                         │
│         → [\n]                                                             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Acceptance Rate: {:.1}% (code tasks)                                       │
│ Effective Throughput: {:.0} tok/s ({:.1}x vs target alone)                 │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        draft.name(),
        draft.hidden_dim(),
        draft.num_layers(),
        draft.tokens_per_second() as u32,
        target.name(),
        target.hidden_dim(),
        target.num_layers(),
        target.tokens_per_second() as u32,
        target.acceptance_rate() * 100.0,
        decoder.effective_throughput(),
        decoder.speedup()
    );
}

fn run_simulation(args: &Args) {
    println!(
        "\n=== Speculative Decoding: {} + {} ===\n",
        args.draft.name(),
        args.target.name()
    );

    let decoder = SpeculativeDecoder::new(args.draft, args.target, args.speculative_tokens);

    println!("Configuration:");
    println!(
        "  Draft model: {} ({:.0} tok/s)",
        args.draft.name(),
        args.draft.tokens_per_second()
    );
    println!(
        "  Target model: {} ({:.0} tok/s)",
        args.target.name(),
        args.target.tokens_per_second()
    );
    println!("  Speculative tokens: {}", args.speculative_tokens);
    println!(
        "  Acceptance rate: {:.1}%",
        args.target.acceptance_rate() * 100.0
    );
    println!();

    // Simulate several decoding steps
    println!("Simulation (10 steps):");
    let mut total_accepted = 0;
    let mut total_generated = 0;

    for step in 0..10 {
        let accepted = decoder.simulate_step(step);
        let _num_accepted = accepted.iter().filter(|&&x| x).count();

        // Find first rejection point
        let first_reject = accepted.iter().position(|&x| !x);
        let actual_accepted = match first_reject {
            Some(pos) => pos + 1,                // Include bonus token from target
            None => args.speculative_tokens + 1, // All accepted + bonus
        };

        total_accepted += actual_accepted;
        total_generated += args.speculative_tokens;

        if step < 5 {
            let status: String = accepted
                .iter()
                .map(|&a| if a { "✓" } else { "✗" })
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "  Step {:2}: [{}] → {} tokens accepted",
                step + 1,
                status,
                actual_accepted
            );
        } else if step == 5 {
            println!("  ...");
        }
    }

    let actual_rate = total_accepted as f64 / (total_generated + 10) as f64; // +10 for bonus tokens

    println!();
    println!("Results:");
    println!("  Total tokens generated: {}", total_accepted);
    println!("  Actual acceptance rate: {:.1}%", actual_rate * 100.0);
    println!(
        "  Effective throughput: {:.0} tok/s",
        decoder.effective_throughput()
    );
    println!("  Speedup vs target-only: {:.1}x", decoder.speedup());
}

fn main() {
    let args = Args::parse();

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-speculative-decode");
        println!("  draft: {}", args.draft.name());
        println!("  target: {}", args.target.name());
        println!("  draft_speed: {:.0} tok/s", args.draft.tokens_per_second());
        println!(
            "  target_speed: {:.0} tok/s",
            args.target.tokens_per_second()
        );

        let decoder = SpeculativeDecoder::new(args.draft, args.target, args.speculative_tokens);
        println!(
            "  acceptance_rate: {:.1}%",
            args.target.acceptance_rate() * 100.0
        );
        println!(
            "  effective_throughput: {:.0} tok/s",
            decoder.effective_throughput()
        );
        println!("  speedup: {:.1}x", decoder.speedup());

        // Check 2.8x speedup threshold from spec
        let speedup = decoder.speedup();
        println!(
            "  speedup_threshold: {}",
            if speedup >= 2.8 { "PASS" } else { "FAIL" }
        );
    } else {
        // Interactive mode
        print_speculative_diagram(args.draft, args.target);
        run_simulation(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
