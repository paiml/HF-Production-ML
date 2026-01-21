//! Demo: Model Registry (Batuta)
//!
//! Demonstrates model versioning and registry operations:
//! - Model metadata management
//! - Version control and artifact storage
//! - Quantization variant tracking
//! - Integration with Hugging Face Hub
//!
//! Uses Qwen2.5-Coder family for registry examples.
//!
//! References:
//! - MLflow Model Registry patterns
//! - Hugging Face Hub API
//! - Semantic versioning for ML models

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

/// Model stage in registry
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum ModelStage {
    /// Development/experimental
    Development,
    /// Staging for validation
    Staging,
    /// Production deployment
    Production,
    /// Archived/deprecated
    Archived,
}

/// Quantization format
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum QuantFormat {
    /// Full precision FP16
    Fp16,
    /// 8-bit quantized
    Q8,
    /// 6-bit quantized
    Q6,
    /// 5-bit quantized
    Q5,
    /// 4-bit quantized
    Q4,
}

/// Registry operation
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum RegistryOp {
    /// Register new model
    Register,
    /// Query model metadata
    Query,
    /// List all versions
    List,
    /// Promote to next stage
    Promote,
    /// Download artifact
    Download,
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

    fn hf_repo(&self) -> &'static str {
        match self {
            ModelTier::Tiny => "Qwen/Qwen2.5-0.5B-Instruct",
            ModelTier::Small => "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            ModelTier::Medium => "Qwen/Qwen2.5-Coder-7B-Instruct",
            ModelTier::Large => "Qwen/Qwen2.5-Coder-32B-Instruct",
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

    fn base_size_gb(&self) -> f64 {
        // FP16 size in GB
        self.parameters_billions() * 2.0
    }
}

impl ModelStage {
    fn name(&self) -> &'static str {
        match self {
            ModelStage::Development => "Development",
            ModelStage::Staging => "Staging",
            ModelStage::Production => "Production",
            ModelStage::Archived => "Archived",
        }
    }

    fn next(&self) -> Option<ModelStage> {
        match self {
            ModelStage::Development => Some(ModelStage::Staging),
            ModelStage::Staging => Some(ModelStage::Production),
            ModelStage::Production => Some(ModelStage::Archived),
            ModelStage::Archived => None,
        }
    }

    fn validation_required(&self) -> bool {
        matches!(self, ModelStage::Staging | ModelStage::Production)
    }
}

impl QuantFormat {
    fn name(&self) -> &'static str {
        match self {
            QuantFormat::Fp16 => "FP16",
            QuantFormat::Q8 => "Q8_0",
            QuantFormat::Q6 => "Q6_K",
            QuantFormat::Q5 => "Q5_K_M",
            QuantFormat::Q4 => "Q4_K_M",
        }
    }

    fn bits_per_weight(&self) -> f64 {
        match self {
            QuantFormat::Fp16 => 16.0,
            QuantFormat::Q8 => 8.0,
            QuantFormat::Q6 => 6.0,
            QuantFormat::Q5 => 5.0,
            QuantFormat::Q4 => 4.0,
        }
    }

    fn size_multiplier(&self) -> f64 {
        self.bits_per_weight() / 16.0
    }

    fn quality_retention(&self) -> f64 {
        match self {
            QuantFormat::Fp16 => 1.0,
            QuantFormat::Q8 => 0.995,
            QuantFormat::Q6 => 0.985,
            QuantFormat::Q5 => 0.97,
            QuantFormat::Q4 => 0.94,
        }
    }
}

impl RegistryOp {
    fn name(&self) -> &'static str {
        match self {
            RegistryOp::Register => "register",
            RegistryOp::Query => "query",
            RegistryOp::List => "list",
            RegistryOp::Promote => "promote",
            RegistryOp::Download => "download",
        }
    }
}

/// Model version metadata
#[derive(Debug)]
#[allow(dead_code)]
struct ModelVersion {
    tier: ModelTier,
    version: String,
    stage: ModelStage,
    quant: QuantFormat,
    size_gb: f64,
    sha256: String,
    created: String,
    metrics: ModelMetrics,
}

/// Model performance metrics
#[derive(Debug)]
#[allow(dead_code)]
struct ModelMetrics {
    humaneval_pass_1: f64,
    mbpp_pass_1: f64,
    avg_latency_ms: f64,
    tokens_per_second: f64,
}

/// Registry query result
#[derive(Debug)]
#[allow(dead_code)]
struct RegistryResult {
    operation: RegistryOp,
    model: ModelTier,
    versions: Vec<ModelVersion>,
    total_size_gb: f64,
    production_version: Option<String>,
}

/// Model registry
struct ModelRegistry {
    model: ModelTier,
}

impl ModelRegistry {
    fn new(model: ModelTier) -> Self {
        Self { model }
    }

    fn query(&self, quant: QuantFormat, stage: ModelStage) -> RegistryResult {
        let base_size = self.model.base_size_gb() * quant.size_multiplier();

        // Simulate version history
        let versions = vec![
            ModelVersion {
                tier: self.model,
                version: "1.0.0".to_string(),
                stage: ModelStage::Archived,
                quant,
                size_gb: base_size,
                sha256: "a1b2c3d4...".to_string(),
                created: "2024-09-01".to_string(),
                metrics: ModelMetrics {
                    humaneval_pass_1: 0.82,
                    mbpp_pass_1: 0.75,
                    avg_latency_ms: 45.0,
                    tokens_per_second: 35.0,
                },
            },
            ModelVersion {
                tier: self.model,
                version: "1.1.0".to_string(),
                stage: ModelStage::Production,
                quant,
                size_gb: base_size,
                sha256: "e5f6g7h8...".to_string(),
                created: "2024-11-01".to_string(),
                metrics: ModelMetrics {
                    humaneval_pass_1: 0.85,
                    mbpp_pass_1: 0.78,
                    avg_latency_ms: 42.0,
                    tokens_per_second: 38.0,
                },
            },
            ModelVersion {
                tier: self.model,
                version: "1.2.0-beta".to_string(),
                stage,
                quant,
                size_gb: base_size,
                sha256: "i9j0k1l2...".to_string(),
                created: "2025-01-15".to_string(),
                metrics: ModelMetrics {
                    humaneval_pass_1: 0.87,
                    mbpp_pass_1: 0.80,
                    avg_latency_ms: 40.0,
                    tokens_per_second: 40.0,
                },
            },
        ];

        let total_size_gb = versions.iter().map(|v| v.size_gb).sum();
        let production_version = versions
            .iter()
            .find(|v| v.stage == ModelStage::Production)
            .map(|v| v.version.clone());

        RegistryResult {
            operation: RegistryOp::Query,
            model: self.model,
            versions,
            total_size_gb,
            production_version,
        }
    }

    fn list_quant_variants(&self) -> Vec<(QuantFormat, f64)> {
        vec![
            QuantFormat::Fp16,
            QuantFormat::Q8,
            QuantFormat::Q6,
            QuantFormat::Q5,
            QuantFormat::Q4,
        ]
        .into_iter()
        .map(|q| (q, self.model.base_size_gb() * q.size_multiplier()))
        .collect()
    }
}

/// Model Registry Demo
#[derive(Parser)]
#[command(name = "demo-model-registry")]
#[command(about = "Demonstrate model registry operations")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Model tier
    #[arg(long, value_enum, default_value = "medium")]
    tier: ModelTier,

    /// Registry operation
    #[arg(long, value_enum, default_value = "query")]
    op: RegistryOp,

    /// Model stage
    #[arg(long, value_enum, default_value = "staging")]
    stage: ModelStage,

    /// Quantization format
    #[arg(long, value_enum, default_value = "q4")]
    quant: QuantFormat,

    /// Show all quantization variants
    #[arg(long)]
    variants: bool,
}

fn print_registry_diagram(model: ModelTier) {
    let registry = ModelRegistry::new(model);
    let result = registry.query(QuantFormat::Q4, ModelStage::Staging);

    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODEL REGISTRY (BATUTA)                                  │
│              Model: {}                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Registry Architecture:                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
│  │  │  HuggingFace │    │   Registry   │    │   Artifact   │         │   │
│  │  │     Hub      │───►│   Metadata   │───►│    Store     │         │   │
│  │  │   (Source)   │    │   (Index)    │    │   (S3/GCS)   │         │   │
│  │  └──────────────┘    └──────┬───────┘    └──────────────┘         │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │  Version Control: Semantic Versioning + Stage Transitions    │  │   │
│  │  │                                                              │  │   │
│  │  │  [Development]──►[Staging]──►[Production]──►[Archived]      │  │   │
│  │  │       │              │             │                         │  │   │
│  │  │    Iterate       Validate      Deploy                        │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Versions: {} │ Production: {} │ Total: {:.1} GB                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        model.name(),
        result.versions.len(),
        result.production_version.as_deref().unwrap_or("none"),
        result.total_size_gb
    );
}

fn run_analysis(args: &Args) {
    println!("\n=== Model Registry Analysis ===\n");

    let registry = ModelRegistry::new(args.tier);

    println!("Model:");
    println!("  Name: {}", args.tier.name());
    println!("  HuggingFace Repo: {}", args.tier.hf_repo());
    println!("  Parameters: {:.1}B", args.tier.parameters_billions());
    println!("  Base size (FP16): {:.1} GB", args.tier.base_size_gb());
    println!();

    if args.variants {
        println!("Quantization Variants:");
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!("│ Format │ Bits │ Size (GB) │ Quality │ Use Case                  │");
        println!("├────────┼──────┼───────────┼─────────┼───────────────────────────┤");

        for (quant, size) in registry.list_quant_variants() {
            let use_case = match quant {
                QuantFormat::Fp16 => "Training, fine-tuning",
                QuantFormat::Q8 => "High-quality inference",
                QuantFormat::Q6 => "Balanced quality/size",
                QuantFormat::Q5 => "Memory constrained",
                QuantFormat::Q4 => "Edge deployment",
            };
            println!(
                "│ {:6} │ {:>4} │ {:>9.1} │ {:>6.1}% │ {:25} │",
                quant.name(),
                quant.bits_per_weight() as usize,
                size,
                quant.quality_retention() * 100.0,
                use_case
            );
        }

        println!("└──────────────────────────────────────────────────────────────────┘");
        println!();
    }

    println!("Operation: {}", args.op.name());
    println!();

    match args.op {
        RegistryOp::Query | RegistryOp::List => {
            let result = registry.query(args.quant, args.stage);

            println!("Versions ({})::", args.quant.name());
            println!(
                "┌───────────────────────────────────────────────────────────────────────────┐"
            );
            println!("│ Version    │ Stage       │ Size (GB) │ HumanEval │ Latency │ Created    │");
            println!("├────────────┼─────────────┼───────────┼───────────┼─────────┼────────────┤");

            for v in &result.versions {
                let marker = if Some(v.version.clone()) == result.production_version {
                    "★"
                } else {
                    " "
                };
                println!(
                    "│{}{:10} │ {:11} │ {:>9.1} │ {:>8.1}% │ {:>6.0}ms │ {:10} │",
                    marker,
                    v.version,
                    v.stage.name(),
                    v.size_gb,
                    v.metrics.humaneval_pass_1 * 100.0,
                    v.metrics.avg_latency_ms,
                    v.created
                );
            }

            println!(
                "└───────────────────────────────────────────────────────────────────────────┘"
            );
            println!();
            println!(
                "Production version: {}",
                result.production_version.as_deref().unwrap_or("none")
            );
            println!("Total storage: {:.1} GB", result.total_size_gb);
        }
        RegistryOp::Promote => {
            if let Some(next_stage) = args.stage.next() {
                println!(
                    "Promoting {} from {} to {}",
                    args.tier.name(),
                    args.stage.name(),
                    next_stage.name()
                );
                if next_stage.validation_required() {
                    println!("  → Validation required before promotion");
                }
            } else {
                println!("Cannot promote from {} stage", args.stage.name());
            }
        }
        RegistryOp::Register => {
            let size = args.tier.base_size_gb() * args.quant.size_multiplier();
            println!("Registering new model:");
            println!("  Model: {}", args.tier.name());
            println!("  Quantization: {}", args.quant.name());
            println!("  Size: {:.1} GB", size);
            println!("  Initial stage: Development");
        }
        RegistryOp::Download => {
            let size = args.tier.base_size_gb() * args.quant.size_multiplier();
            println!("Download artifact:");
            println!("  Model: {}", args.tier.name());
            println!("  Quantization: {}", args.quant.name());
            println!("  Size: {:.1} GB", size);
            println!("  Estimated time: {:.0}s @ 100 MB/s", size * 1024.0 / 100.0);
        }
    }
}

fn main() {
    let args = Args::parse();

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-model-registry");
        println!("  model: {}", args.tier.name());
        println!("  hf_repo: {}", args.tier.hf_repo());
        println!("  operation: {}", args.op.name());
        println!("  stage: {}", args.stage.name());
        println!("  quant: {}", args.quant.name());

        let registry = ModelRegistry::new(args.tier);
        let result = registry.query(args.quant, args.stage);

        println!("  version_count: {}", result.versions.len());
        println!("  total_size_gb: {:.1}", result.total_size_gb);
        println!(
            "  production_version: {}",
            result.production_version.as_deref().unwrap_or("none")
        );
        println!(
            "  base_size_gb: {:.1}",
            args.tier.base_size_gb() * args.quant.size_multiplier()
        );
        println!(
            "  quality_retention: {:.1}%",
            args.quant.quality_retention() * 100.0
        );

        // Check registry viability: must have production version and valid sizes
        let viable = result.production_version.is_some() && result.total_size_gb > 0.0;
        println!("  registry_valid: {}", if viable { "PASS" } else { "FAIL" });
    } else {
        // Interactive mode
        print_registry_diagram(args.tier);
        run_analysis(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
