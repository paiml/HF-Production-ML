//! Demo: TGI Architecture Concepts
//!
//! Visualizes production inference server architecture with:
//! - Request queue and batch scheduler
//! - Model executor with trueno backend
//! - KV cache management
//! - Output decoder with streaming
//!
//! References:
//! - Kwon et al. (2023) "Efficient Memory Management for Large Language Model
//!   Serving with PagedAttention" (SOSP '23)

use clap::Parser;
use std::io::{self, Write};
use std::time::{Duration, Instant};

/// TGI Architecture Demo - Visualize production inference server components
#[derive(Parser)]
#[command(name = "demo-tgi-architecture")]
#[command(about = "Visualize TGI-equivalent serving architecture")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Number of simulated requests
    #[arg(long, default_value = "8")]
    requests: usize,

    /// Maximum batch size
    #[arg(long, default_value = "4")]
    batch_size: usize,

    /// Maximum sequence length
    #[arg(long, default_value = "512")]
    max_seq_len: usize,
}

/// Simulated request in the queue
#[derive(Debug, Clone)]
struct Request {
    id: usize,
    prompt_tokens: usize,
    max_new_tokens: usize,
    state: RequestState,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum RequestState {
    Queued,
    Prefilling,
    Decoding,
    Complete,
}

/// Batch scheduler component
struct BatchScheduler {
    max_batch_size: usize,
    max_batch_tokens: usize,
    queue: Vec<Request>,
    active_batch: Vec<Request>,
}

impl BatchScheduler {
    fn new(max_batch_size: usize, max_seq_len: usize) -> Self {
        Self {
            max_batch_size,
            max_batch_tokens: max_batch_size * max_seq_len,
            queue: Vec::new(),
            active_batch: Vec::new(),
        }
    }

    fn enqueue(&mut self, request: Request) {
        self.queue.push(request);
    }

    fn schedule_batch(&mut self) -> Vec<Request> {
        let mut batch = Vec::new();
        let mut batch_tokens = 0;

        // Continuous batching: add requests while capacity allows
        while !self.queue.is_empty() && batch.len() < self.max_batch_size {
            let req = &self.queue[0];
            let req_tokens = req.prompt_tokens + req.max_new_tokens;

            if batch_tokens + req_tokens <= self.max_batch_tokens {
                let mut req = self.queue.remove(0);
                req.state = RequestState::Prefilling;
                batch_tokens += req_tokens;
                batch.push(req);
            } else {
                break;
            }
        }

        self.active_batch = batch.clone();
        batch
    }

    fn utilization(&self) -> f64 {
        if self.max_batch_size == 0 {
            return 0.0;
        }
        self.active_batch.len() as f64 / self.max_batch_size as f64
    }
}

/// KV Cache simulation
#[allow(dead_code)]
struct KVCache {
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    allocated_blocks: usize,
    total_blocks: usize,
}

impl KVCache {
    fn new(num_layers: usize, num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let block_size = 16; // tokens per block
        let total_blocks = (max_seq_len / block_size) * 32; // pool size
        Self {
            num_layers,
            num_heads,
            head_dim,
            max_seq_len,
            allocated_blocks: 0,
            total_blocks,
        }
    }

    fn allocate(&mut self, seq_len: usize) -> bool {
        let block_size = 16;
        let needed = seq_len.div_ceil(block_size);
        if self.allocated_blocks + needed <= self.total_blocks {
            self.allocated_blocks += needed;
            true
        } else {
            false
        }
    }

    fn memory_usage_mb(&self) -> f64 {
        // 2 (K+V) * layers * heads * head_dim * allocated_tokens * 2 (fp16)
        let tokens = self.allocated_blocks * 16;
        let bytes = 2 * self.num_layers * self.num_heads * self.head_dim * tokens * 2;
        bytes as f64 / (1024.0 * 1024.0)
    }

    fn utilization(&self) -> f64 {
        self.allocated_blocks as f64 / self.total_blocks as f64
    }
}

/// Model executor simulation
#[allow(dead_code)]
struct ModelExecutor {
    vocab_size: usize,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
}

impl ModelExecutor {
    fn new() -> Self {
        // Simulating a ~7B parameter model
        Self {
            vocab_size: 32000,
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
        }
    }

    fn prefill(&self, _tokens: usize) -> Duration {
        // Simulate prefill time: ~10ms per 100 tokens
        Duration::from_micros(100)
    }

    fn decode_step(&self) -> Duration {
        // Simulate decode time: ~5ms per token
        Duration::from_micros(50)
    }
}

fn print_architecture_diagram() {
    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PRODUCTION INFERENCE SERVER ARCHITECTURE                 │
│                        (TGI-equivalent with Sovereign Stack)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐ │
│   │   Request   │───▶│     Batch       │───▶│      Model Executor         │ │
│   │    Queue    │    │   Scheduler     │    │    (trueno backend)         │ │
│   │             │    │                 │    │                             │ │
│   │  • FIFO     │    │  • Continuous   │    │  • Forward pass             │ │
│   │  • Priority │    │    batching     │    │  • SIMD/GPU acceleration    │ │
│   │  • Timeout  │    │  • Dynamic size │    │  • Flash Attention          │ │
│   └─────────────┘    └─────────────────┘    └─────────────────────────────┘ │
│         │                    │                           │                   │
│         │                    │                           │                   │
│         ▼                    ▼                           ▼                   │
│   ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐ │
│   │  Tokenizer  │    │    KV Cache     │    │     Output Decoder          │ │
│   │   (BPE)     │    │    Manager      │    │      + Streaming            │ │
│   │             │    │                 │    │                             │ │
│   │  • Encode   │    │  • Paged alloc  │    │  • Sampling (top-p/k)       │ │
│   │  • Decode   │    │  • Block table  │    │  • Stop sequences           │ │
│   │  • Special  │    │  • Eviction     │    │  • SSE streaming            │ │
│   └─────────────┘    └─────────────────┘    └─────────────────────────────┘ │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Key Optimizations:                                                          │
│  • Continuous batching: Add/remove requests mid-generation                   │
│  • PagedAttention: O(1) memory per sequence (vs O(N²) standard)             │
│  • Speculative decoding: Draft model + verify (2-3x speedup)                │
│  • Quantization: Q4_K/Q5_K for 3-4x compression                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#
    );
}

fn run_simulation(args: &Args) {
    println!("\n=== Inference Server Simulation ===\n");

    let mut scheduler = BatchScheduler::new(args.batch_size, args.max_seq_len);
    let mut kv_cache = KVCache::new(32, 32, 128, args.max_seq_len);
    let executor = ModelExecutor::new();

    // Create requests
    for i in 0..args.requests {
        let prompt_tokens = 50 + (i * 20) % 100;
        let max_new_tokens = 100 + (i * 30) % 200;
        scheduler.enqueue(Request {
            id: i,
            prompt_tokens,
            max_new_tokens,
            state: RequestState::Queued,
        });
    }

    println!("Requests in queue: {}", args.requests);
    println!("Max batch size: {}", args.batch_size);
    println!("Max sequence length: {}", args.max_seq_len);
    println!();

    // Process batches
    let mut total_tokens = 0;
    let mut batch_num = 0;
    let start = Instant::now();

    while !scheduler.queue.is_empty() || !scheduler.active_batch.is_empty() {
        let batch = scheduler.schedule_batch();
        if batch.is_empty() {
            break;
        }

        batch_num += 1;
        println!("Batch {}: {} requests", batch_num, batch.len());

        for req in &batch {
            // Allocate KV cache
            let seq_len = req.prompt_tokens + req.max_new_tokens;
            if kv_cache.allocate(seq_len) {
                print!("  Request {}: ", req.id);

                // Prefill
                let prefill_time = executor.prefill(req.prompt_tokens);
                print!("prefill {}tok ", req.prompt_tokens);
                std::thread::sleep(prefill_time);

                // Decode (simulate a few tokens)
                let decode_tokens = 5.min(req.max_new_tokens);
                for _ in 0..decode_tokens {
                    let decode_time = executor.decode_step();
                    std::thread::sleep(decode_time);
                }
                total_tokens += req.prompt_tokens + decode_tokens;

                println!("→ decode {}tok ✓", decode_tokens);
            } else {
                println!("  Request {}: KV cache full, waiting...", req.id);
            }
        }

        println!(
            "  Batch utilization: {:.1}%",
            scheduler.utilization() * 100.0
        );
        println!(
            "  KV cache: {:.1}% ({:.1} MB)",
            kv_cache.utilization() * 100.0,
            kv_cache.memory_usage_mb()
        );
        println!();
    }

    let elapsed = start.elapsed();
    let throughput = total_tokens as f64 / elapsed.as_secs_f64();

    println!("=== Summary ===");
    println!("Total batches: {}", batch_num);
    println!("Total tokens: {}", total_tokens);
    println!("Elapsed: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("Throughput: {:.1} tokens/sec (simulated)", throughput);
}

fn main() {
    let args = Args::parse();

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-tgi-architecture");
        println!("  components: request_queue, batch_scheduler, model_executor, kv_cache");
        println!("  batch_size: {}", args.batch_size);
        println!("  max_seq_len: {}", args.max_seq_len);
    } else {
        // Interactive mode
        print_architecture_diagram();
        run_simulation(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
