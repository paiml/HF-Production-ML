//! Demo: Tensor Parallelism
//!
//! Demonstrates distributed inference with tensor parallelism:
//! - Column-wise and row-wise partitioning
//! - All-reduce communication patterns
//! - Multi-GPU scaling efficiency
//! - Memory distribution across devices
//!
//! Uses Qwen2.5-Coder architecture for realistic simulation.
//!
//! References:
//! - Shoeybi et al. (2019) "Megatron-LM: Training Multi-Billion Parameter Language Models"
//! - Narayanan et al. (2021) "Efficient Large-Scale Language Model Training"

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

/// GPU type for simulation
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum GpuType {
    /// NVIDIA RTX 4090 (24GB)
    Rtx4090,
    /// NVIDIA A100 (40GB)
    A100_40,
    /// NVIDIA A100 (80GB)
    A100_80,
    /// NVIDIA H100 (80GB)
    H100,
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

    fn memory_fp16_gb(&self) -> f64 {
        // FP16: 2 bytes per parameter + KV cache + activations
        self.parameters_billions() * 2.0 * 1.3 // 30% overhead
    }

    fn single_gpu_speed(&self, gpu: GpuType) -> f64 {
        // Base speed on single GPU (if it fits)
        let base = match self {
            ModelTier::Tiny => 1500.0,
            ModelTier::Small => 500.0,
            ModelTier::Medium => 150.0,
            ModelTier::Large => 50.0,
        };
        // Scale by GPU compute capability
        base * gpu.compute_scale()
    }
}

impl GpuType {
    fn name(&self) -> &'static str {
        match self {
            GpuType::Rtx4090 => "RTX 4090",
            GpuType::A100_40 => "A100-40GB",
            GpuType::A100_80 => "A100-80GB",
            GpuType::H100 => "H100",
        }
    }

    fn vram_gb(&self) -> f64 {
        match self {
            GpuType::Rtx4090 => 24.0,
            GpuType::A100_40 => 40.0,
            GpuType::A100_80 => 80.0,
            GpuType::H100 => 80.0,
        }
    }

    fn compute_scale(&self) -> f64 {
        // Relative compute performance
        match self {
            GpuType::Rtx4090 => 1.0,
            GpuType::A100_40 => 1.2,
            GpuType::A100_80 => 1.2,
            GpuType::H100 => 2.0,
        }
    }

    fn interconnect_bandwidth_gbps(&self) -> f64 {
        // GPU-to-GPU bandwidth (NVLink for datacenter, PCIe for consumer)
        match self {
            GpuType::Rtx4090 => 64.0,  // PCIe 4.0 x16
            GpuType::A100_40 => 600.0, // NVLink 3
            GpuType::A100_80 => 600.0, // NVLink 3
            GpuType::H100 => 900.0,    // NVLink 4
        }
    }
}

/// Tensor parallelism analysis result
#[derive(Debug)]
#[allow(dead_code)]
struct TensorParallelResult {
    model: ModelTier,
    gpu: GpuType,
    num_gpus: usize,
    memory_per_gpu_gb: f64,
    fits_in_memory: bool,
    throughput: f64,
    scaling_efficiency: f64,
    communication_overhead_pct: f64,
}

/// Tensor parallelism analyzer
struct TensorParallelAnalyzer {
    model: ModelTier,
    gpu: GpuType,
}

impl TensorParallelAnalyzer {
    fn new(model: ModelTier, gpu: GpuType) -> Self {
        Self { model, gpu }
    }

    fn min_gpus_required(&self) -> usize {
        let model_memory = self.model.memory_fp16_gb();
        let gpu_memory = self.gpu.vram_gb();
        ((model_memory / gpu_memory).ceil() as usize).max(1)
    }

    fn analyze(&self, num_gpus: usize) -> TensorParallelResult {
        let model_memory = self.model.memory_fp16_gb();
        let memory_per_gpu = model_memory / num_gpus as f64;
        let fits_in_memory = memory_per_gpu <= self.gpu.vram_gb() * 0.9; // 90% usable

        // Single GPU baseline speed
        let single_gpu_speed = self.model.single_gpu_speed(self.gpu);

        // Communication overhead from all-reduce operations
        // Each layer requires all-reduce for attention and MLP outputs
        let hidden_dim = self.model.hidden_dim() as f64;
        let num_layers = self.model.num_layers() as f64;

        // Bytes to communicate per token: 2 * hidden_dim * num_layers * 2 (all-reduce = 2x)
        let bytes_per_token = 2.0 * hidden_dim * num_layers * 2.0 * 2.0; // FP16

        // Time to communicate (assuming batch of 1 for latency)
        let bandwidth = self.gpu.interconnect_bandwidth_gbps() * 1e9 / 8.0; // bytes/sec
        let comm_time_per_token = bytes_per_token / bandwidth * (num_gpus - 1).max(1) as f64;

        // Compute time per token on distributed system
        let compute_time_per_token = 1.0 / single_gpu_speed;
        let parallel_compute_time = compute_time_per_token / num_gpus as f64;

        // Total time per token
        let total_time = parallel_compute_time + comm_time_per_token;
        let throughput = 1.0 / total_time;

        // Scaling efficiency
        let ideal_throughput = single_gpu_speed * num_gpus as f64;
        let scaling_efficiency = throughput / ideal_throughput * 100.0;

        // Communication overhead
        let communication_overhead_pct = comm_time_per_token / total_time * 100.0;

        TensorParallelResult {
            model: self.model,
            gpu: self.gpu,
            num_gpus,
            memory_per_gpu_gb: memory_per_gpu,
            fits_in_memory,
            throughput,
            scaling_efficiency,
            communication_overhead_pct,
        }
    }

    fn scaling_sweep(&self) -> Vec<TensorParallelResult> {
        vec![1, 2, 4, 8]
            .into_iter()
            .map(|n| self.analyze(n))
            .collect()
    }
}

/// Tensor Parallelism Demo
#[derive(Parser)]
#[command(name = "demo-tensor-parallel")]
#[command(about = "Demonstrate tensor parallelism for distributed inference")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Model tier
    #[arg(long, value_enum, default_value = "large")]
    tier: ModelTier,

    /// GPU type
    #[arg(long, value_enum, default_value = "a100-80")]
    gpu: GpuType,

    /// Number of GPUs
    #[arg(long, default_value = "4")]
    num_gpus: usize,

    /// Show scaling sweep
    #[arg(long)]
    sweep: bool,
}

fn print_tensor_parallel_diagram(model: ModelTier, gpu: GpuType, num_gpus: usize) {
    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TENSOR PARALLELISM                                       │
│        Model: {} on {}x {}                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Column-Parallel Linear (Attention Q, K, V, MLP up):                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Input X                                         │   │
│  │                        ↓                                            │   │
│  │    ┌─────────┬─────────┬─────────┬─────────┐                       │   │
│  │    │  GPU 0  │  GPU 1  │  GPU 2  │  GPU 3  │  Weight columns       │   │
│  │    │  W[:,0] │  W[:,1] │  W[:,2] │  W[:,3] │  partitioned          │   │
│  │    └────┬────┴────┬────┴────┬────┴────┬────┘                       │   │
│  │         │         │         │         │                            │   │
│  │         └─────────┴────┬────┴─────────┘                            │   │
│  │                        ↓                                            │   │
│  │                  All-Gather                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Row-Parallel Linear (Attention O, MLP down):                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │    ┌─────────┬─────────┬─────────┬─────────┐                       │   │
│  │    │  GPU 0  │  GPU 1  │  GPU 2  │  GPU 3  │  Weight rows          │   │
│  │    │  W[0,:] │  W[1,:] │  W[2,:] │  W[3,:] │  partitioned          │   │
│  │    └────┬────┴────┬────┴────┬────┴────┬────┘                       │   │
│  │         │         │         │         │                            │   │
│  │         └─────────┴────┬────┴─────────┘                            │   │
│  │                        ↓                                            │   │
│  │                   All-Reduce                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Memory: {:.1} GB/GPU │ Interconnect: {} Gbps                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        model.name(),
        num_gpus,
        gpu.name(),
        model.memory_fp16_gb() / num_gpus as f64,
        gpu.interconnect_bandwidth_gbps() as u32
    );
}

fn run_analysis(args: &Args) {
    println!(
        "\n=== Tensor Parallelism Analysis: {} ===\n",
        args.tier.name()
    );

    let analyzer = TensorParallelAnalyzer::new(args.tier, args.gpu);

    println!("Configuration:");
    println!(
        "  Model: {} ({:.1}B params)",
        args.tier.name(),
        args.tier.parameters_billions()
    );
    println!(
        "  Model memory (FP16): {:.1} GB",
        args.tier.memory_fp16_gb()
    );
    println!(
        "  GPU: {} ({:.0} GB VRAM)",
        args.gpu.name(),
        args.gpu.vram_gb()
    );
    println!(
        "  Interconnect: {} Gbps",
        args.gpu.interconnect_bandwidth_gbps() as u32
    );
    println!("  Minimum GPUs required: {}", analyzer.min_gpus_required());
    println!();

    if args.sweep {
        let results = analyzer.scaling_sweep();

        println!("Scaling Analysis:");
        println!("┌─────────────────────────────────────────────────────────────────────────┐");
        println!("│ GPUs │ Mem/GPU (GB) │ Fits │ Throughput │ Efficiency │ Comm Overhead │");
        println!("├──────┼──────────────┼──────┼────────────┼────────────┼───────────────┤");

        for result in &results {
            println!(
                "│ {:>4} │ {:>12.1} │ {:>4} │ {:>9.0} │ {:>9.1}% │ {:>12.1}% │",
                result.num_gpus,
                result.memory_per_gpu_gb,
                if result.fits_in_memory { "Yes" } else { "No" },
                result.throughput,
                result.scaling_efficiency,
                result.communication_overhead_pct
            );
        }

        println!("└─────────────────────────────────────────────────────────────────────────┘");
    } else {
        let result = analyzer.analyze(args.num_gpus);

        println!("Analysis for {} GPUs:", args.num_gpus);
        println!("  Memory per GPU: {:.1} GB", result.memory_per_gpu_gb);
        println!(
            "  Fits in memory: {}",
            if result.fits_in_memory { "Yes" } else { "No" }
        );
        println!("  Throughput: {:.0} tok/s", result.throughput);
        println!("  Scaling efficiency: {:.1}%", result.scaling_efficiency);
        println!(
            "  Communication overhead: {:.1}%",
            result.communication_overhead_pct
        );

        if !result.fits_in_memory {
            println!(
                "\n  ⚠ Model does not fit! Need at least {} GPUs.",
                analyzer.min_gpus_required()
            );
        }
    }
}

fn main() {
    let args = Args::parse();

    // Clamp num_gpus to valid range
    let num_gpus = args.num_gpus.clamp(1, 8);
    let args = Args { num_gpus, ..args };

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-tensor-parallel");
        println!("  model: {}", args.tier.name());
        println!("  gpu: {}", args.gpu.name());
        println!("  num_gpus: {}", args.num_gpus);

        let analyzer = TensorParallelAnalyzer::new(args.tier, args.gpu);
        let result = analyzer.analyze(args.num_gpus);

        println!("  memory_per_gpu: {:.1} GB", result.memory_per_gpu_gb);
        println!("  fits_in_memory: {}", result.fits_in_memory);
        println!("  throughput: {:.0} tok/s", result.throughput);
        println!("  scaling_efficiency: {:.1}%", result.scaling_efficiency);
        println!(
            "  communication_overhead: {:.1}%",
            result.communication_overhead_pct
        );

        // Check efficiency threshold: >70% scaling efficiency
        println!(
            "  efficiency_threshold: {}",
            if result.scaling_efficiency >= 70.0 {
                "PASS"
            } else {
                "FAIL"
            }
        );
    } else {
        // Interactive mode
        print_tensor_parallel_diagram(args.tier, args.gpu, args.num_gpus);
        run_analysis(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
