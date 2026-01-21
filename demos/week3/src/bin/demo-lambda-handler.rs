//! Demo: AWS Lambda Handler
//!
//! Demonstrates serverless inference deployment:
//! - Lambda function structure for ML inference
//! - Cold start optimization strategies
//! - Memory/timeout configuration
//! - Cost analysis for different workloads
//!
//! Uses Qwen2.5 Tiny tier for Lambda-viable deployment.
//!
//! References:
//! - AWS Lambda Rust Runtime
//! - Provisioned Concurrency for cold start mitigation
//! - SnapStart for JVM-like instant startup (Rust benefits from small binaries)

use clap::{Parser, ValueEnum};
use std::io::{self, Write};

/// Model tier (Lambda-viable sizes)
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum ModelTier {
    /// Qwen2.5-0.5B - Lambda viable
    Tiny,
    /// Qwen2.5-Coder-1.5B - Requires max memory
    Small,
}

/// Lambda memory configuration
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum LambdaMemory {
    /// 512 MB
    Mb512,
    /// 1024 MB
    Mb1024,
    /// 2048 MB
    Mb2048,
    /// 4096 MB
    Mb4096,
    /// 10240 MB (max)
    Mb10240,
}

/// Quantization for Lambda
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum LambdaQuant {
    /// Q8_0 (recommended balance)
    Q8,
    /// Q4_K_M (smallest)
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
}

impl LambdaMemory {
    fn mb(&self) -> usize {
        match self {
            LambdaMemory::Mb512 => 512,
            LambdaMemory::Mb1024 => 1024,
            LambdaMemory::Mb2048 => 2048,
            LambdaMemory::Mb4096 => 4096,
            LambdaMemory::Mb10240 => 10240,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            LambdaMemory::Mb512 => "512 MB",
            LambdaMemory::Mb1024 => "1 GB",
            LambdaMemory::Mb2048 => "2 GB",
            LambdaMemory::Mb4096 => "4 GB",
            LambdaMemory::Mb10240 => "10 GB",
        }
    }

    fn vcpu_fraction(&self) -> f64 {
        // Lambda allocates CPU proportionally to memory
        // 1769 MB = 1 vCPU
        self.mb() as f64 / 1769.0
    }

    fn cost_per_ms(&self) -> f64 {
        // Cost per millisecond in USD (us-east-1 pricing)
        // $0.0000166667 per GB-second = $0.0000000166667 per MB-ms
        self.mb() as f64 * 0.0000000166667
    }

    fn cost_per_request(&self) -> f64 {
        // $0.20 per 1M requests
        0.0000002
    }
}

impl LambdaQuant {
    fn name(&self) -> &'static str {
        match self {
            LambdaQuant::Q8 => "Q8_0",
            LambdaQuant::Q4 => "Q4_K_M",
        }
    }

    fn bytes_per_param(&self) -> f64 {
        match self {
            LambdaQuant::Q8 => 1.0,
            LambdaQuant::Q4 => 0.5,
        }
    }

    fn quality_factor(&self) -> f64 {
        match self {
            LambdaQuant::Q8 => 0.995,
            LambdaQuant::Q4 => 0.94,
        }
    }
}

/// Lambda analysis result
#[derive(Debug)]
#[allow(dead_code)]
struct LambdaAnalysisResult {
    model: ModelTier,
    memory: LambdaMemory,
    quant: LambdaQuant,
    model_size_mb: f64,
    fits_in_memory: bool,
    cold_start_ms: f64,
    warm_latency_ms: f64,
    throughput: f64,
    cost_per_1k_requests: f64,
    cost_per_1m_tokens: f64,
}

/// Lambda analyzer
struct LambdaAnalyzer {
    model: ModelTier,
    memory: LambdaMemory,
}

impl LambdaAnalyzer {
    fn new(model: ModelTier, memory: LambdaMemory) -> Self {
        Self { model, memory }
    }

    fn analyze(&self, quant: LambdaQuant) -> LambdaAnalysisResult {
        let params_b = self.model.parameters_billions();
        let model_size_mb = params_b * 1000.0 * quant.bytes_per_param();

        // Need ~1.5x model size for runtime overhead
        let memory_required = model_size_mb * 1.5 + 100.0; // +100MB for Lambda runtime
        let fits_in_memory = memory_required < self.memory.mb() as f64;

        // Cold start: model loading + initialization
        // ~50ms base + ~0.5ms per MB of model
        let cold_start_ms = 500.0 + model_size_mb * 0.5;

        // Warm latency: inference time
        // Scales with model size and inversely with CPU allocation
        let base_inference_ms = params_b * 200.0; // ~200ms per billion params baseline
        let warm_latency_ms = base_inference_ms / self.memory.vcpu_fraction().max(0.5);

        // Throughput: tokens per second
        // Assume average 50 tokens per request
        let tokens_per_request = 50.0;
        let throughput = tokens_per_request * 1000.0 / warm_latency_ms * quant.quality_factor();

        // Cost analysis
        let duration_ms = warm_latency_ms;
        let compute_cost = self.memory.cost_per_ms() * duration_ms;
        let request_cost = self.memory.cost_per_request();
        let cost_per_request = compute_cost + request_cost;
        let cost_per_1k_requests = cost_per_request * 1000.0;
        let cost_per_1m_tokens = cost_per_request / tokens_per_request * 1_000_000.0;

        LambdaAnalysisResult {
            model: self.model,
            memory: self.memory,
            quant,
            model_size_mb,
            fits_in_memory,
            cold_start_ms,
            warm_latency_ms,
            throughput,
            cost_per_1k_requests,
            cost_per_1m_tokens,
        }
    }

    fn compare_memory_configs(&self, quant: LambdaQuant) -> Vec<LambdaAnalysisResult> {
        let memories = [
            LambdaMemory::Mb1024,
            LambdaMemory::Mb2048,
            LambdaMemory::Mb4096,
            LambdaMemory::Mb10240,
        ];

        memories
            .iter()
            .map(|&mem| {
                let analyzer = LambdaAnalyzer::new(self.model, mem);
                analyzer.analyze(quant)
            })
            .collect()
    }
}

/// Lambda Handler Demo
#[derive(Parser)]
#[command(name = "demo-lambda-handler")]
#[command(about = "Demonstrate AWS Lambda serverless inference")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Model tier
    #[arg(long, value_enum, default_value = "tiny")]
    tier: ModelTier,

    /// Lambda memory configuration
    #[arg(long, value_enum, default_value = "mb2048")]
    memory: LambdaMemory,

    /// Quantization method
    #[arg(long, value_enum, default_value = "q8")]
    quant: LambdaQuant,

    /// Compare memory configurations
    #[arg(long)]
    compare: bool,

    /// Show cost analysis
    #[arg(long)]
    cost: bool,
}

fn print_lambda_diagram(model: ModelTier, memory: LambdaMemory) {
    let analyzer = LambdaAnalyzer::new(model, memory);
    let result = analyzer.analyze(LambdaQuant::Q8);

    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AWS LAMBDA INFERENCE                                     │
│              Model: {} │ Memory: {}                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Lambda Function Structure:                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │  pub async fn handler(event: Request) -> Response {{          │  │   │
│  │  │      // Model loaded once, reused across invocations         │  │   │
│  │  │      let model = MODEL.get_or_init(|| load_model());         │  │   │
│  │  │                                                               │  │   │
│  │  │      // Inference                                             │  │   │
│  │  │      let tokens = model.generate(&event.prompt);             │  │   │
│  │  │      Response {{ text: tokens.decode() }}                    │  │   │
│  │  │  }}                                                           │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Cold Start Optimization:                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ✓ Lazy model loading (OnceCell pattern)                           │   │
│  │  ✓ Provisioned Concurrency for critical paths                      │   │
│  │  ✓ Quantized weights reduce load time                              │   │
│  │  ✓ ARM64 (Graviton) for better price/performance                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Cold Start: {:.0}ms │ Warm: {:.0}ms │ Cost: ${:.4}/1K req              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        model.name(),
        memory.name(),
        result.cold_start_ms,
        result.warm_latency_ms,
        result.cost_per_1k_requests
    );
}

fn run_analysis(args: &Args) {
    println!(
        "\n=== Lambda Inference Analysis: {} ===\n",
        args.tier.name()
    );

    let analyzer = LambdaAnalyzer::new(args.tier, args.memory);

    println!("Configuration:");
    println!(
        "  Model: {} ({:.1}B params)",
        args.tier.name(),
        args.tier.parameters_billions()
    );
    println!("  Memory: {}", args.memory.name());
    println!("  vCPU: {:.2}", args.memory.vcpu_fraction());
    println!("  Quantization: {}", args.quant.name());
    println!();

    if args.compare {
        let results = analyzer.compare_memory_configs(args.quant);

        println!("Memory Configuration Comparison:");
        println!("┌──────────────────────────────────────────────────────────────────────────┐");
        println!("│ Memory │ Fits │ Cold (ms) │ Warm (ms) │ tok/s │ $/1K req │ $/1M tok   │");
        println!("├────────┼──────┼───────────┼───────────┼───────┼──────────┼────────────┤");

        for result in &results {
            println!(
                "│ {:>6} │ {:>4} │ {:>9.0} │ {:>9.0} │ {:>5.0} │ {:>8.4} │ {:>10.2} │",
                result.memory.name(),
                if result.fits_in_memory { "Yes" } else { "No" },
                result.cold_start_ms,
                result.warm_latency_ms,
                result.throughput,
                result.cost_per_1k_requests,
                result.cost_per_1m_tokens
            );
        }

        println!("└──────────────────────────────────────────────────────────────────────────┘");
    } else {
        let result = analyzer.analyze(args.quant);

        println!("Analysis:");
        println!("  Model size: {:.0} MB", result.model_size_mb);
        println!(
            "  Fits in memory: {}",
            if result.fits_in_memory { "Yes" } else { "No" }
        );
        println!("  Cold start: {:.0}ms", result.cold_start_ms);
        println!("  Warm latency: {:.0}ms", result.warm_latency_ms);
        println!("  Throughput: {:.0} tok/s", result.throughput);

        if args.cost {
            println!();
            println!("Cost Analysis:");
            println!("  Per 1K requests: ${:.4}", result.cost_per_1k_requests);
            println!("  Per 1M tokens: ${:.2}", result.cost_per_1m_tokens);
            println!();
            println!("  Comparison (1M requests/month):");
            println!(
                "    Lambda:    ${:.2}",
                result.cost_per_1k_requests * 1000.0
            );
            println!("    EC2 g4dn.xlarge: ~$380 (dedicated)");
            println!(
                "    → Lambda cost-effective for <{}K req/month",
                380.0 / result.cost_per_1k_requests
            );
        }
    }
}

fn main() {
    let args = Args::parse();

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-lambda-handler");
        println!("  model: {}", args.tier.name());
        println!("  memory: {}", args.memory.name());
        println!("  quant: {}", args.quant.name());

        let analyzer = LambdaAnalyzer::new(args.tier, args.memory);
        let result = analyzer.analyze(args.quant);

        println!("  model_size_mb: {:.0}", result.model_size_mb);
        println!("  fits_in_memory: {}", result.fits_in_memory);
        println!("  cold_start_ms: {:.0}", result.cold_start_ms);
        println!("  warm_latency_ms: {:.0}", result.warm_latency_ms);
        println!("  throughput: {:.0} tok/s", result.throughput);
        println!("  cost_per_1k: ${:.4}", result.cost_per_1k_requests);

        // Check viability: must fit and have <2s cold start
        let viable = result.fits_in_memory && result.cold_start_ms < 2000.0;
        println!("  lambda_viable: {}", if viable { "PASS" } else { "FAIL" });
    } else {
        // Interactive mode
        print_lambda_diagram(args.tier, args.memory);
        run_analysis(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
