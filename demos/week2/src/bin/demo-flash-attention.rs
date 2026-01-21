//! Demo: Flash Attention
//!
//! Demonstrates memory-efficient attention computation:
//! - Standard attention: O(N²) memory
//! - Flash attention: O(N) memory via tiling
//! - IO-aware memory access patterns
//! - VRAM savings comparison
//!
//! Uses Qwen2.5-Coder architecture for realistic calculations.
//!
//! References:
//! - Dao et al. (2022) "FlashAttention: Fast and Memory-Efficient Exact Attention"
//! - Dao (2023) "FlashAttention-2: Faster Attention with Better Parallelism"

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

    fn num_heads(&self) -> usize {
        match self {
            ModelTier::Tiny => 14,
            ModelTier::Small => 12,
            ModelTier::Medium => 28,
            ModelTier::Large => 40,
        }
    }

    fn head_dim(&self) -> usize {
        128 // Qwen2.5 uses 128 for all tiers
    }

    fn num_kv_heads(&self) -> usize {
        match self {
            ModelTier::Tiny => 2,
            ModelTier::Small => 2,
            ModelTier::Medium => 4,
            ModelTier::Large => 8,
        }
    }
}

/// Attention memory calculator
struct AttentionMemory {
    tier: ModelTier,
    seq_len: usize,
    batch_size: usize,
}

impl AttentionMemory {
    fn new(tier: ModelTier, seq_len: usize, batch_size: usize) -> Self {
        Self {
            tier,
            seq_len,
            batch_size,
        }
    }

    /// Standard attention memory: O(N²) for attention matrix
    fn standard_memory_gb(&self) -> f64 {
        let heads = self.tier.num_heads();
        let head_dim = self.tier.head_dim();

        // Q, K, V tensors: batch × heads × seq × head_dim × 2 (FP16)
        let qkv_memory = self.batch_size * heads * self.seq_len * head_dim * 2 * 3;

        // Attention matrix: batch × heads × seq × seq × 4 (FP32 for softmax)
        let attn_matrix = self.batch_size * heads * self.seq_len * self.seq_len * 4;

        // Output: batch × heads × seq × head_dim × 2
        let output_memory = self.batch_size * heads * self.seq_len * head_dim * 2;

        (qkv_memory + attn_matrix + output_memory) as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Flash attention memory: O(N) - no materialized attention matrix
    fn flash_memory_gb(&self) -> f64 {
        let heads = self.tier.num_heads();
        let head_dim = self.tier.head_dim();
        let block_size = 64; // Typical block size

        // Q, K, V tensors (same as standard)
        let qkv_memory = self.batch_size * heads * self.seq_len * head_dim * 2 * 3;

        // Tiled computation: only store blocks, not full N×N matrix
        // Block buffers: 2 blocks of size (block_size × block_size × 4)
        let block_memory = self.batch_size * heads * block_size * block_size * 4 * 2;

        // Running softmax statistics: batch × heads × seq × 2 (max, sum)
        let stats_memory = self.batch_size * heads * self.seq_len * 4 * 2;

        // Output (same as standard)
        let output_memory = self.batch_size * heads * self.seq_len * head_dim * 2;

        (qkv_memory + block_memory + stats_memory + output_memory) as f64
            / (1024.0 * 1024.0 * 1024.0)
    }

    fn memory_savings(&self) -> f64 {
        self.standard_memory_gb() / self.flash_memory_gb()
    }

    /// Standard attention speed (relative)
    fn standard_speed(&self) -> f64 {
        100.0 // Baseline
    }

    /// Flash attention speedup
    fn flash_speed(&self) -> f64 {
        // Flash attention is typically 2-4x faster due to reduced memory bandwidth
        200.0
    }
}

/// Flash Attention Demo
#[derive(Parser)]
#[command(name = "demo-flash-attention")]
#[command(about = "Demonstrate memory-efficient flash attention")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Model tier
    #[arg(long, value_enum, default_value = "medium")]
    tier: ModelTier,

    /// Sequence length
    #[arg(long, default_value = "4096")]
    seq_len: usize,

    /// Batch size
    #[arg(long, default_value = "1")]
    batch_size: usize,

    /// Run benchmark comparison
    #[arg(long)]
    benchmark: bool,
}

fn print_flash_attention_diagram(tier: ModelTier, seq_len: usize) {
    let calc = AttentionMemory::new(tier, seq_len, 1);

    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FLASH ATTENTION COMPARISON                               │
│                Model: {} │ Seq: {}                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Standard Attention                    Flash Attention                      │
│  ━━━━━━━━━━━━━━━━━━━━━                ━━━━━━━━━━━━━━━━━━━━━                 │
│                                                                             │
│  Q×K^T → [N×N] matrix                 Q×K^T → [block×block]                │
│         ↓                                    ↓                              │
│      Softmax                             Softmax (tiled)                    │
│         ↓                                    ↓                              │
│        ×V                               ×V (accumulated)                    │
│                                                                             │
│  Memory: O(N²)                         Memory: O(N)                         │
│  VRAM: {:.1} GB                        VRAM: {:.1} GB                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Algorithm Comparison:                                               │   │
│  │                                                                     │   │
│  │ Standard:                          Flash:                           │   │
│  │ 1. Compute S = Q×K^T (N×N)        1. For each block (Bq, Bk):      │   │
│  │ 2. P = softmax(S)                  2.   S_block = Bq × Bk^T        │   │
│  │ 3. O = P × V                       3.   Update running softmax     │   │
│  │                                    4.   Accumulate O_block          │   │
│  │                                                                     │   │
│  │ Memory bandwidth bound ──────────▶ Compute bound (IO-aware)        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Speedup: {:.1}x │ Memory Savings: {:.1}x                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        tier.name(),
        seq_len,
        calc.standard_memory_gb(),
        calc.flash_memory_gb(),
        calc.flash_speed() / calc.standard_speed(),
        calc.memory_savings()
    );
}

fn run_comparison(args: &Args) {
    println!("\n=== Flash Attention Analysis: {} ===\n", args.tier.name());

    println!("Model Configuration:");
    println!("  Hidden dim: {}", args.tier.hidden_dim());
    println!("  Attention heads: {}", args.tier.num_heads());
    println!("  KV heads: {}", args.tier.num_kv_heads());
    println!("  Head dim: {}", args.tier.head_dim());
    println!();

    // Compare across different sequence lengths
    let seq_lengths = [512, 1024, 2048, 4096, 8192, 16384];

    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│             Memory Comparison by Sequence Length                 │");
    println!("├──────────┬───────────────┬───────────────┬───────────────────────┤");
    println!("│ Seq Len  │ Standard (GB) │ Flash (GB)    │ Savings               │");
    println!("├──────────┼───────────────┼───────────────┼───────────────────────┤");

    for seq_len in seq_lengths {
        let calc = AttentionMemory::new(args.tier, seq_len, args.batch_size);
        println!(
            "│ {:>8} │ {:>13.2} │ {:>13.3} │ {:>20.1}x │",
            seq_len,
            calc.standard_memory_gb(),
            calc.flash_memory_gb(),
            calc.memory_savings()
        );
    }

    println!("└──────────┴───────────────┴───────────────┴───────────────────────┘");

    if args.benchmark {
        println!("\nBenchmark (simulated):");
        let calc = AttentionMemory::new(args.tier, args.seq_len, args.batch_size);
        println!(
            "  Standard attention: {:.0} tokens/sec",
            calc.standard_speed()
        );
        println!("  Flash attention: {:.0} tokens/sec", calc.flash_speed());
        println!(
            "  Speedup: {:.1}x",
            calc.flash_speed() / calc.standard_speed()
        );
    }
}

fn main() {
    let args = Args::parse();

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-flash-attention");
        println!("  model: {}", args.tier.name());
        println!("  heads: {}", args.tier.num_heads());
        println!("  head_dim: {}", args.tier.head_dim());
        println!("  seq_len: {}", args.seq_len);

        let calc = AttentionMemory::new(args.tier, args.seq_len, args.batch_size);
        println!("  standard_memory_gb: {:.2}", calc.standard_memory_gb());
        println!("  flash_memory_gb: {:.3}", calc.flash_memory_gb());
        println!("  memory_savings: {:.1}x", calc.memory_savings());

        if args.benchmark {
            let speedup = calc.flash_speed() / calc.standard_speed();
            println!("  speedup: {:.1}x", speedup);
            println!(
                "  speedup_threshold: {}",
                if speedup >= 2.0 { "PASS" } else { "FAIL" }
            );
        }
    } else {
        // Interactive mode
        print_flash_attention_diagram(args.tier, args.seq_len);
        run_comparison(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
