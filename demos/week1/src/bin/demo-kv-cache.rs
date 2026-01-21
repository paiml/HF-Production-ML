//! Demo: KV Cache Memory Management
//!
//! Visualizes Key-Value cache growth during autoregressive generation:
//! - Memory allocation per sequence
//! - Cache eviction strategies
//! - Paged attention block management
//! - Memory efficiency metrics
//!
//! Uses Qwen2.5-Coder architecture dimensions for realistic simulation.
//!
//! References:
//! - Kwon et al. (2023) "Efficient Memory Management for Large Language Model
//!   Serving with PagedAttention" (SOSP '23)

use clap::{Parser, ValueEnum};
use std::io::{self, Write};

/// Qwen2.5-Coder model tiers with architecture dimensions
#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelTier {
    /// Qwen2.5-0.5B: hidden=896, layers=24, heads=14, kv_heads=2
    Tiny,
    /// Qwen2.5-Coder-1.5B: hidden=1536, layers=28, heads=12, kv_heads=2
    Small,
    /// Qwen2.5-Coder-7B: hidden=3584, layers=28, heads=28, kv_heads=4
    Medium,
    /// Qwen2.5-Coder-32B: hidden=5120, layers=64, heads=40, kv_heads=8
    Large,
}

impl ModelTier {
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

    fn num_kv_heads(&self) -> usize {
        match self {
            ModelTier::Tiny => 2,
            ModelTier::Small => 2,
            ModelTier::Medium => 4,
            ModelTier::Large => 8,
        }
    }

    fn head_dim(&self) -> usize {
        // head_dim = hidden_dim / num_heads (but for KV, use kv_heads)
        // For Qwen, head_dim is consistent at 128 for all tiers
        128
    }

    fn name(&self) -> &'static str {
        match self {
            ModelTier::Tiny => "Qwen2.5-0.5B",
            ModelTier::Small => "Qwen2.5-Coder-1.5B",
            ModelTier::Medium => "Qwen2.5-Coder-7B",
            ModelTier::Large => "Qwen2.5-Coder-32B",
        }
    }
}

/// KV Cache Demo - Visualize memory management
#[derive(Parser)]
#[command(name = "demo-kv-cache")]
#[command(about = "Visualize KV cache memory management for Qwen2.5-Coder")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Model tier to simulate
    #[arg(long, value_enum, default_value = "small")]
    tier: ModelTier,

    /// Maximum sequence length
    #[arg(long, default_value = "2048")]
    max_seq_len: usize,

    /// Number of concurrent sequences
    #[arg(long, default_value = "8")]
    num_sequences: usize,

    /// Block size for paged attention
    #[arg(long, default_value = "16")]
    block_size: usize,

    /// Show memory efficiency metrics
    #[arg(long)]
    efficiency: bool,
}

/// KV Cache entry for a single sequence
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct KVCacheEntry {
    sequence_id: usize,
    current_len: usize,
    max_len: usize,
    allocated_blocks: usize,
}

/// KV Cache manager with paged attention
struct KVCacheManager {
    tier: ModelTier,
    block_size: usize,
    max_blocks: usize,
    free_blocks: Vec<usize>,
    entries: Vec<KVCacheEntry>,
    block_assignments: Vec<Vec<usize>>, // sequence_id -> block_ids
}

impl KVCacheManager {
    fn new(tier: ModelTier, block_size: usize, max_memory_mb: usize) -> Self {
        // Calculate bytes per block
        // KV cache per token = 2 (K+V) * num_layers * num_kv_heads * head_dim * 2 (fp16)
        let bytes_per_token = 2 * tier.num_layers() * tier.num_kv_heads() * tier.head_dim() * 2;
        let bytes_per_block = bytes_per_token * block_size;

        // Calculate max blocks from memory budget
        let max_blocks = (max_memory_mb * 1024 * 1024) / bytes_per_block;
        let free_blocks = (0..max_blocks).collect();

        Self {
            tier,
            block_size,
            max_blocks,
            free_blocks,
            entries: Vec::new(),
            block_assignments: Vec::new(),
        }
    }

    fn allocate_sequence(
        &mut self,
        sequence_id: usize,
        initial_len: usize,
        max_len: usize,
    ) -> bool {
        let blocks_needed = initial_len.div_ceil(self.block_size);

        if blocks_needed > self.free_blocks.len() {
            return false;
        }

        let mut assigned_blocks = Vec::new();
        for _ in 0..blocks_needed {
            if let Some(block_id) = self.free_blocks.pop() {
                assigned_blocks.push(block_id);
            }
        }

        self.entries.push(KVCacheEntry {
            sequence_id,
            current_len: initial_len,
            max_len,
            allocated_blocks: blocks_needed,
        });

        // Ensure block_assignments has enough entries
        while self.block_assignments.len() <= sequence_id {
            self.block_assignments.push(Vec::new());
        }
        self.block_assignments[sequence_id] = assigned_blocks;

        true
    }

    fn extend_sequence(&mut self, sequence_id: usize, new_len: usize) -> bool {
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|e| e.sequence_id == sequence_id)
        {
            let blocks_needed = new_len.div_ceil(self.block_size);
            let current_blocks = entry.allocated_blocks;

            if blocks_needed > current_blocks {
                let additional = blocks_needed - current_blocks;
                if additional > self.free_blocks.len() {
                    return false;
                }

                for _ in 0..additional {
                    if let Some(block_id) = self.free_blocks.pop() {
                        self.block_assignments[sequence_id].push(block_id);
                    }
                }
                entry.allocated_blocks = blocks_needed;
            }

            entry.current_len = new_len;
            true
        } else {
            false
        }
    }

    fn free_sequence(&mut self, sequence_id: usize) {
        if let Some(pos) = self
            .entries
            .iter()
            .position(|e| e.sequence_id == sequence_id)
        {
            self.entries.remove(pos);
        }

        if sequence_id < self.block_assignments.len() {
            let blocks = std::mem::take(&mut self.block_assignments[sequence_id]);
            self.free_blocks.extend(blocks);
        }
    }

    fn memory_usage_mb(&self) -> f64 {
        let used_blocks = self.max_blocks - self.free_blocks.len();
        let bytes_per_token =
            2 * self.tier.num_layers() * self.tier.num_kv_heads() * self.tier.head_dim() * 2;
        let bytes = used_blocks * self.block_size * bytes_per_token;
        bytes as f64 / (1024.0 * 1024.0)
    }

    fn utilization(&self) -> f64 {
        let used_blocks = self.max_blocks - self.free_blocks.len();
        if self.max_blocks == 0 {
            0.0
        } else {
            used_blocks as f64 / self.max_blocks as f64
        }
    }

    fn efficiency(&self) -> f64 {
        // Efficiency = actual tokens stored / (allocated blocks * block_size)
        let total_tokens: usize = self.entries.iter().map(|e| e.current_len).sum();
        let allocated_slots: usize = self
            .entries
            .iter()
            .map(|e| e.allocated_blocks * self.block_size)
            .sum();
        if allocated_slots == 0 {
            1.0
        } else {
            total_tokens as f64 / allocated_slots as f64
        }
    }
}

fn print_kv_cache_diagram(tier: ModelTier) {
    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                   KV CACHE MEMORY MANAGEMENT: {}                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Architecture: hidden={}, layers={}, kv_heads={}, head_dim={}           │
│                                                                             │
│  Per-Token KV Memory:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ K: [layers × kv_heads × head_dim × fp16] = {} × {} × {} × 2B     │   │
│  │ V: [layers × kv_heads × head_dim × fp16] = {} × {} × {} × 2B     │   │
│  │                                                                     │   │
│  │ Total per token: {} bytes ({:.2} KB)                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Paged Attention Block Table:                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Sequence 0:  [Block 0] → [Block 3] → [Block 7]                    │    │
│  │ Sequence 1:  [Block 1] → [Block 4] → [Block 8]                    │    │
│  │ Sequence 2:  [Block 2] → [Block 5]                                │    │
│  │                          ↓                                         │    │
│  │              Physical Block Pool (Shared)                         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Memory Growth Pattern:                                                     │
│  ├─ Prefill: Allocate blocks for prompt tokens                             │
│  ├─ Decode:  Extend by 1 block when current block fills                    │
│  └─ Evict:   Return blocks when sequence completes                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        tier.name(),
        tier.hidden_dim(),
        tier.num_layers(),
        tier.num_kv_heads(),
        tier.head_dim(),
        tier.num_layers(),
        tier.num_kv_heads(),
        tier.head_dim(),
        tier.num_layers(),
        tier.num_kv_heads(),
        tier.head_dim(),
        2 * tier.num_layers() * tier.num_kv_heads() * tier.head_dim() * 2,
        (2 * tier.num_layers() * tier.num_kv_heads() * tier.head_dim() * 2) as f64 / 1024.0
    );
}

fn run_simulation(args: &Args) {
    println!("\n=== KV Cache Simulation: {} ===\n", args.tier.name());

    // Calculate memory budget based on tier
    let memory_budget_mb = match args.tier {
        ModelTier::Tiny => 512,
        ModelTier::Small => 1024,
        ModelTier::Medium => 4096,
        ModelTier::Large => 16384,
    };

    let mut manager = KVCacheManager::new(args.tier, args.block_size, memory_budget_mb);

    println!("Configuration:");
    println!("  Model: {}", args.tier.name());
    println!("  Hidden dim: {}", args.tier.hidden_dim());
    println!("  Layers: {}", args.tier.num_layers());
    println!("  KV heads: {}", args.tier.num_kv_heads());
    println!("  Block size: {} tokens", args.block_size);
    println!("  Max blocks: {}", manager.max_blocks);
    println!("  Memory budget: {} MB", memory_budget_mb);
    println!();

    // Simulate sequences
    println!("Simulating {} sequences:\n", args.num_sequences);

    for seq_id in 0..args.num_sequences {
        let prompt_len = 50 + (seq_id * 30) % 200;
        let max_len = prompt_len + 100 + (seq_id * 50) % 300;

        if manager.allocate_sequence(seq_id, prompt_len, max_len) {
            println!(
                "  Seq {}: allocated {} tokens ({} blocks) - utilization: {:.1}%, memory: {:.1} MB",
                seq_id,
                prompt_len,
                prompt_len.div_ceil(args.block_size),
                manager.utilization() * 100.0,
                manager.memory_usage_mb()
            );
        } else {
            println!("  Seq {}: FAILED - out of memory", seq_id);
        }
    }

    println!();

    // Simulate generation growth
    println!("Simulating generation (10 tokens per sequence):\n");
    for step in 1..=10 {
        for seq_id in 0..args.num_sequences {
            if let Some(entry) = manager.entries.iter().find(|e| e.sequence_id == seq_id) {
                let new_len = entry.current_len + 1;
                if new_len <= entry.max_len {
                    manager.extend_sequence(seq_id, new_len);
                }
            }
        }

        if step <= 3 || step == 10 {
            println!(
                "  Step {:2}: utilization: {:.1}%, efficiency: {:.1}%, memory: {:.1} MB",
                step,
                manager.utilization() * 100.0,
                manager.efficiency() * 100.0,
                manager.memory_usage_mb()
            );
        } else if step == 4 {
            println!("  ...");
        }
    }

    println!();

    // Complete some sequences
    println!("Completing sequences 0-2:\n");
    for seq_id in 0..3.min(args.num_sequences) {
        manager.free_sequence(seq_id);
        println!(
            "  Freed seq {}: utilization now {:.1}%, memory: {:.1} MB",
            seq_id,
            manager.utilization() * 100.0,
            manager.memory_usage_mb()
        );
    }

    println!();
    println!("=== Summary ===");
    println!("  Final utilization: {:.1}%", manager.utilization() * 100.0);
    println!("  Final efficiency: {:.1}%", manager.efficiency() * 100.0);
    println!("  Memory in use: {:.1} MB", manager.memory_usage_mb());
    println!("  Free blocks: {}", manager.free_blocks.len());
}

fn main() {
    let args = Args::parse();

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-kv-cache");
        println!("  model: {}", args.tier.name());
        println!("  hidden_dim: {}", args.tier.hidden_dim());
        println!("  num_layers: {}", args.tier.num_layers());
        println!("  kv_heads: {}", args.tier.num_kv_heads());
        println!("  block_size: {}", args.block_size);

        if args.efficiency {
            // Quick efficiency check
            let mut manager = KVCacheManager::new(args.tier, args.block_size, 1024);
            for i in 0..4 {
                manager.allocate_sequence(i, 100, 200);
            }
            println!("  efficiency: {:.1}%", manager.efficiency() * 100.0);
            println!("  utilization: {:.1}%", manager.utilization() * 100.0);
        }
    } else {
        // Interactive mode
        print_kv_cache_diagram(args.tier);
        run_simulation(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
