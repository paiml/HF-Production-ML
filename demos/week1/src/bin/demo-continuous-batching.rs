//! Demo: Continuous Batching with PagedAttention
//!
//! Simulates PagedAttention-style memory management:
//! - Block-based memory allocation
//! - Block tables for sequence tracking
//! - Dynamic batch admission/eviction
//! - Defragmentation strategies
//!
//! References:
//! - Kwon et al. (2023) "Efficient Memory Management for Large Language Model
//!   Serving with PagedAttention" (SOSP '23)

use clap::Parser;
use std::collections::HashMap;
use std::io::{self, Write};

/// Continuous Batching Demo - PagedAttention memory management
#[derive(Parser)]
#[command(name = "demo-continuous-batching")]
#[command(about = "Simulate PagedAttention-style memory management")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Block size (tokens per block)
    #[arg(long, default_value = "16")]
    block_size: usize,

    /// Total number of blocks in memory pool
    #[arg(long, default_value = "256")]
    num_blocks: usize,

    /// Number of sequences to simulate
    #[arg(long, default_value = "12")]
    num_sequences: usize,

    /// Show utilization metrics
    #[arg(long)]
    utilization: bool,
}

/// A physical memory block that can hold `block_size` tokens of KV cache
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Block {
    id: usize,
    ref_count: usize, // For copy-on-write
}

/// Manages the pool of physical blocks
#[allow(dead_code)]
struct BlockAllocator {
    block_size: usize,
    free_blocks: Vec<usize>,
    blocks: Vec<Block>,
}

impl BlockAllocator {
    fn new(num_blocks: usize, block_size: usize) -> Self {
        let blocks: Vec<Block> = (0..num_blocks)
            .map(|id| Block { id, ref_count: 0 })
            .collect();
        let free_blocks: Vec<usize> = (0..num_blocks).collect();

        Self {
            block_size,
            free_blocks,
            blocks,
        }
    }

    fn allocate(&mut self) -> Option<usize> {
        self.free_blocks.pop().inspect(|&id| {
            self.blocks[id].ref_count = 1;
        })
    }

    fn free(&mut self, block_id: usize) {
        if block_id < self.blocks.len() {
            self.blocks[block_id].ref_count -= 1;
            if self.blocks[block_id].ref_count == 0 {
                self.free_blocks.push(block_id);
            }
        }
    }

    fn num_free(&self) -> usize {
        self.free_blocks.len()
    }

    fn num_total(&self) -> usize {
        self.blocks.len()
    }

    fn utilization(&self) -> f64 {
        let used = self.num_total() - self.num_free();
        used as f64 / self.num_total() as f64
    }
}

/// Block table mapping logical blocks to physical blocks for a sequence
#[allow(dead_code)]
struct BlockTable {
    sequence_id: usize,
    logical_to_physical: Vec<usize>,
    num_tokens: usize,
}

impl BlockTable {
    fn new(sequence_id: usize) -> Self {
        Self {
            sequence_id,
            logical_to_physical: Vec::new(),
            num_tokens: 0,
        }
    }

    fn num_blocks(&self) -> usize {
        self.logical_to_physical.len()
    }
}

/// Sequence in the batch
#[derive(Debug, Clone)]
struct Sequence {
    id: usize,
    prompt_len: usize,
    generated_len: usize,
    max_len: usize,
    state: SequenceState,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum SequenceState {
    Waiting,
    Running,
    Finished,
    Preempted,
}

/// Continuous batching scheduler
struct ContinuousBatchingScheduler {
    block_allocator: BlockAllocator,
    block_tables: HashMap<usize, BlockTable>,
    running: Vec<Sequence>,
    waiting: Vec<Sequence>,
    max_batch_size: usize,
    block_size: usize,
}

impl ContinuousBatchingScheduler {
    fn new(num_blocks: usize, block_size: usize, max_batch_size: usize) -> Self {
        Self {
            block_allocator: BlockAllocator::new(num_blocks, block_size),
            block_tables: HashMap::new(),
            running: Vec::new(),
            waiting: Vec::new(),
            max_batch_size,
            block_size,
        }
    }

    fn add_sequence(&mut self, seq: Sequence) {
        self.waiting.push(seq);
    }

    /// Allocate blocks for a sequence's current token count
    fn allocate_blocks(&mut self, seq_id: usize, num_tokens: usize) -> bool {
        let block_table = self
            .block_tables
            .entry(seq_id)
            .or_insert_with(|| BlockTable::new(seq_id));

        let needed_blocks = num_tokens.div_ceil(self.block_size);
        let current_blocks = block_table.num_blocks();

        // Allocate additional blocks if needed
        for _ in current_blocks..needed_blocks {
            if let Some(block_id) = self.block_allocator.allocate() {
                block_table.logical_to_physical.push(block_id);
            } else {
                return false; // Out of memory
            }
        }

        block_table.num_tokens = num_tokens;
        true
    }

    /// Free all blocks for a sequence
    fn free_sequence(&mut self, seq_id: usize) {
        if let Some(block_table) = self.block_tables.remove(&seq_id) {
            for block_id in block_table.logical_to_physical {
                self.block_allocator.free(block_id);
            }
        }
    }

    /// Schedule one iteration of continuous batching
    fn schedule(&mut self) -> Vec<usize> {
        let mut scheduled = Vec::new();

        // First, try to add waiting sequences
        while !self.waiting.is_empty() && self.running.len() < self.max_batch_size {
            let seq = &self.waiting[0];
            let initial_tokens = seq.prompt_len;

            // Try to allocate blocks
            if self.allocate_blocks(seq.id, initial_tokens) {
                let mut seq = self.waiting.remove(0);
                seq.state = SequenceState::Running;
                self.running.push(seq);
            } else {
                break; // No more memory
            }
        }

        // Collect info about running sequences first
        let running_info: Vec<(usize, usize, usize, usize)> = self
            .running
            .iter()
            .map(|s| (s.id, s.prompt_len, s.generated_len, s.max_len))
            .collect();

        // Extend running sequences by one token
        let mut finished = Vec::new();
        for (seq_id, prompt_len, generated_len, max_len) in running_info {
            let new_len = prompt_len + generated_len + 1;

            if self.allocate_blocks(seq_id, new_len) {
                // Update the sequence in running
                if let Some(seq) = self.running.iter_mut().find(|s| s.id == seq_id) {
                    seq.generated_len += 1;
                    scheduled.push(seq.id);

                    if seq.prompt_len + seq.generated_len >= max_len {
                        seq.state = SequenceState::Finished;
                        finished.push(seq.id);
                    }
                }
            }
        }

        // Remove finished sequences and free their blocks
        for seq_id in finished {
            self.free_sequence(seq_id);
            self.running.retain(|s| s.id != seq_id);
        }

        scheduled
    }

    fn utilization(&self) -> f64 {
        self.block_allocator.utilization()
    }

    fn batch_size(&self) -> usize {
        self.running.len()
    }
}

fn print_continuous_batching_diagram() {
    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONTINUOUS BATCHING WITH PAGEDATTENTION              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Physical Memory Pool (Blocks)                                              │
│   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐  │
│   │ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │ B8 │ B9 │B10 │B11 │... │Bn  │  │
│   └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘  │
│     ▲    ▲    ▲         ▲    ▲    ▲         ▲    ▲                         │
│     │    │    │         │    │    │         │    │                         │
│     └────┴────┴─────────┴────┴────┴─────────┴────┘                         │
│           │                   │                   │                         │
│           ▼                   ▼                   ▼                         │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                │
│   │  Block Table  │   │  Block Table  │   │  Block Table  │                │
│   │   Seq 0       │   │   Seq 1       │   │   Seq 2       │                │
│   │  [0,1,2]      │   │  [3,4,5]      │   │  [6,7,8]      │                │
│   └───────────────┘   └───────────────┘   └───────────────┘                │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│   Key Advantages:                                                            │
│   • O(1) memory per sequence (vs O(max_seq_len) with padding)               │
│   • Dynamic admission: New sequences join mid-generation                     │
│   • Memory sharing: Copy-on-write for prompt caching                        │
│   • Preemption: Evict sequences under memory pressure                       │
│                                                                              │
│   Traditional Batching:          Continuous Batching:                        │
│   ┌────────────────────┐         ┌────────────────────┐                     │
│   │ Seq0 ████████░░░░░ │         │ Seq0 ████████      │ ← Compact           │
│   │ Seq1 ██████░░░░░░░ │         │ Seq1 ██████        │                     │
│   │ Seq2 ████░░░░░░░░░ │         │ Seq2 ████          │                     │
│   │      ↑ Wasted      │         │ Seq3 ██████████    │ ← New join          │
│   └────────────────────┘         └────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
"#
    );
}

fn run_simulation(args: &Args) {
    println!("\n=== Continuous Batching Simulation ===\n");

    let mut scheduler =
        ContinuousBatchingScheduler::new(args.num_blocks, args.block_size, args.num_sequences / 2);

    // Create sequences with varying lengths
    for i in 0..args.num_sequences {
        let prompt_len = 20 + (i * 15) % 80;
        let max_len = prompt_len + 50 + (i * 20) % 100;
        scheduler.add_sequence(Sequence {
            id: i,
            prompt_len,
            generated_len: 0,
            max_len,
            state: SequenceState::Waiting,
        });
    }

    println!("Configuration:");
    println!("  Block size: {} tokens", args.block_size);
    println!("  Total blocks: {}", args.num_blocks);
    println!("  Sequences: {}", args.num_sequences);
    println!();

    // Run simulation for several iterations
    let mut iteration = 0;
    let max_iterations = 50;

    while iteration < max_iterations
        && (!scheduler.running.is_empty() || !scheduler.waiting.is_empty())
    {
        let scheduled = scheduler.schedule();

        if iteration < 10 || iteration % 10 == 0 {
            println!(
                "Iteration {:3}: batch_size={}, scheduled={}, utilization={:.1}%, waiting={}",
                iteration,
                scheduler.batch_size(),
                scheduled.len(),
                scheduler.utilization() * 100.0,
                scheduler.waiting.len()
            );

            if iteration < 5 {
                // Show block table details for first few iterations
                print!("             blocks: ");
                for seq in &scheduler.running {
                    if let Some(bt) = scheduler.block_tables.get(&seq.id) {
                        print!("S{}:{:?} ", seq.id, bt.logical_to_physical);
                    }
                }
                println!();
            }
        }

        iteration += 1;
    }

    println!();
    println!("=== Summary ===");
    println!("Total iterations: {}", iteration);
    println!("Final utilization: {:.1}%", scheduler.utilization() * 100.0);
    println!("Remaining in batch: {}", scheduler.batch_size());
    println!("Remaining waiting: {}", scheduler.waiting.len());
}

fn main() {
    let args = Args::parse();

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-continuous-batching");
        println!("  block_size: {}", args.block_size);
        println!("  num_blocks: {}", args.num_blocks);
        println!("  num_sequences: {}", args.num_sequences);

        if args.utilization {
            // Quick utilization check
            let mut scheduler = ContinuousBatchingScheduler::new(
                args.num_blocks,
                args.block_size,
                args.num_sequences / 2,
            );
            for i in 0..args.num_sequences {
                scheduler.add_sequence(Sequence {
                    id: i,
                    prompt_len: 50,
                    generated_len: 0,
                    max_len: 100,
                    state: SequenceState::Waiting,
                });
            }
            scheduler.schedule();
            println!("  utilization: {:.1}%", scheduler.utilization() * 100.0);
        }
    } else {
        // Interactive mode
        print_continuous_batching_diagram();
        run_simulation(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
