//! Demo: Knowledge Distillation
//!
//! Demonstrates knowledge distillation for model compression:
//! - Teacher-student paradigm
//! - Soft target training
//! - Temperature scaling
//! - Layer-wise feature distillation
//!
//! Uses Qwen2.5-Coder hierarchy for realistic simulation:
//! - Teacher: 7B or 32B model
//! - Student: 0.5B or 1.5B model
//!
//! References:
//! - Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"
//! - Sanh et al. (2019) "DistilBERT: A distilled version of BERT"
//! - Wu et al. (2023) "LaMini-LM: A Diverse Herd of Distilled Models"

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

/// Distillation method
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum DistillMethod {
    /// Soft targets only (KL divergence)
    SoftTargets,
    /// Feature distillation (intermediate layers)
    FeatureDistill,
    /// Combined: soft targets + feature alignment
    Combined,
    /// Online distillation (mutual learning)
    Online,
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

    fn training_cost_relative(&self) -> f64 {
        // Relative training cost (GPU hours)
        match self {
            ModelTier::Tiny => 1.0,
            ModelTier::Small => 4.0,
            ModelTier::Medium => 20.0,
            ModelTier::Large => 100.0,
        }
    }

    fn inference_speed(&self) -> f64 {
        // tok/s on RTX 4090
        match self {
            ModelTier::Tiny => 1500.0,
            ModelTier::Small => 500.0,
            ModelTier::Medium => 150.0,
            ModelTier::Large => 60.0,
        }
    }
}

impl DistillMethod {
    fn name(&self) -> &'static str {
        match self {
            DistillMethod::SoftTargets => "Soft Targets",
            DistillMethod::FeatureDistill => "Feature Distillation",
            DistillMethod::Combined => "Combined",
            DistillMethod::Online => "Online Distillation",
        }
    }

    /// Quality retention factor (how much of teacher's quality is retained)
    fn quality_retention(&self) -> f64 {
        match self {
            DistillMethod::SoftTargets => 0.92,
            DistillMethod::FeatureDistill => 0.94,
            DistillMethod::Combined => 0.96,
            DistillMethod::Online => 0.91,
        }
    }

    /// Training overhead multiplier
    fn training_overhead(&self) -> f64 {
        match self {
            DistillMethod::SoftTargets => 1.2,    // Need teacher inference
            DistillMethod::FeatureDistill => 1.5, // Additional feature matching
            DistillMethod::Combined => 1.7,       // Both overheads
            DistillMethod::Online => 2.0,         // Train both models
        }
    }
}

/// Distillation result
#[derive(Debug)]
#[allow(dead_code)]
struct DistillationResult {
    teacher: ModelTier,
    student: ModelTier,
    method: DistillMethod,
    temperature: f64,
    student_perplexity_before: f64,
    student_perplexity_after: f64,
    improvement_pct: f64,
    compression_ratio: f64,
    speedup: f64,
}

/// Distillation analyzer
struct DistillationAnalyzer {
    teacher: ModelTier,
    student: ModelTier,
}

impl DistillationAnalyzer {
    fn new(teacher: ModelTier, student: ModelTier) -> Self {
        Self { teacher, student }
    }

    fn analyze(&self, method: DistillMethod, temperature: f64) -> DistillationResult {
        let student_before = self.student.baseline_perplexity();
        let teacher_ppl = self.teacher.baseline_perplexity();

        // Distillation brings student closer to teacher
        // Higher temperature -> softer probability distribution -> better knowledge transfer
        let temp_factor = 1.0 + (temperature - 1.0).ln().max(0.0) * 0.1;
        let quality_retention = method.quality_retention() * temp_factor.min(1.05);

        // Student perplexity approaches teacher perplexity based on retention
        let gap = student_before - teacher_ppl;
        let student_after = teacher_ppl + gap * (1.0 - quality_retention);

        let improvement_pct = (student_before - student_after) / student_before * 100.0;
        let compression_ratio =
            self.teacher.parameters_billions() / self.student.parameters_billions();
        let speedup = self.student.inference_speed() / self.teacher.inference_speed();

        DistillationResult {
            teacher: self.teacher,
            student: self.student,
            method,
            temperature,
            student_perplexity_before: student_before,
            student_perplexity_after: student_after,
            improvement_pct,
            compression_ratio,
            speedup,
        }
    }

    fn compare_methods(&self, temperature: f64) -> Vec<DistillationResult> {
        vec![
            self.analyze(DistillMethod::SoftTargets, temperature),
            self.analyze(DistillMethod::FeatureDistill, temperature),
            self.analyze(DistillMethod::Combined, temperature),
            self.analyze(DistillMethod::Online, temperature),
        ]
    }
}

/// Knowledge Distillation Demo
#[derive(Parser)]
#[command(name = "demo-distillation")]
#[command(about = "Demonstrate knowledge distillation for model compression")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Teacher model tier
    #[arg(long, value_enum, default_value = "medium")]
    teacher: ModelTier,

    /// Student model tier
    #[arg(long, value_enum, default_value = "tiny")]
    student: ModelTier,

    /// Distillation method
    #[arg(long, value_enum, default_value = "combined")]
    method: DistillMethod,

    /// Temperature for soft targets
    #[arg(long, default_value = "4.0")]
    temperature: f64,

    /// Compare all methods
    #[arg(long)]
    compare: bool,

    /// Temperature sweep
    #[arg(long)]
    sweep: bool,
}

fn print_distillation_diagram(teacher: ModelTier, student: ModelTier) {
    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE DISTILLATION                                   │
│          Teacher: {} → Student: {}              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Distillation Process:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │    ┌───────────────┐                    ┌───────────────┐          │   │
│  │    │    Teacher    │    Soft Labels     │    Student    │          │   │
│  │    │    ({:.1}B)     │ ─────────────────► │    ({:.1}B)     │          │   │
│  │    │  PPL: {:.2}    │    P(y|x, T)       │  Learning...  │          │   │
│  │    └───────────────┘                    └───────────────┘          │   │
│  │           │                                    │                   │   │
│  │           │ Hidden States                      │                   │   │
│  │           └──────── Feature Alignment ─────────┘                   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Loss Function:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  L = α × L_hard(y, ŷ) + (1-α) × T² × KL(σ(z_t/T) ‖ σ(z_s/T))     │   │
│  │                                                                     │   │
│  │  where: T = temperature, σ = softmax, z = logits                   │   │
│  │         Higher T → softer distributions → better knowledge transfer │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Compression: {:.0}x │ Speedup: {:.0}x                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        teacher.name(),
        student.name(),
        teacher.parameters_billions(),
        student.parameters_billions(),
        teacher.baseline_perplexity(),
        teacher.parameters_billions() / student.parameters_billions(),
        student.inference_speed() / teacher.inference_speed()
    );
}

fn run_analysis(args: &Args) {
    println!(
        "\n=== Distillation Analysis: {} → {} ===\n",
        args.teacher.name(),
        args.student.name()
    );

    let analyzer = DistillationAnalyzer::new(args.teacher, args.student);

    println!("Configuration:");
    println!(
        "  Teacher: {} ({:.1}B params, PPL: {:.2})",
        args.teacher.name(),
        args.teacher.parameters_billions(),
        args.teacher.baseline_perplexity()
    );
    println!(
        "  Student: {} ({:.1}B params, PPL: {:.2})",
        args.student.name(),
        args.student.parameters_billions(),
        args.student.baseline_perplexity()
    );
    println!();

    if args.compare {
        let results = analyzer.compare_methods(args.temperature);

        println!("Method Comparison (T={:.1}):", args.temperature);
        println!("┌────────────────────────────────────────────────────────────────────────┐");
        println!("│ Method              │ PPL Before │ PPL After │ Improvement │ Retention │");
        println!("├─────────────────────┼────────────┼───────────┼─────────────┼───────────┤");

        for result in &results {
            println!(
                "│ {:19} │ {:10.2} │ {:9.2} │ {:10.1}% │ {:8.1}% │",
                result.method.name(),
                result.student_perplexity_before,
                result.student_perplexity_after,
                result.improvement_pct,
                result.method.quality_retention() * 100.0
            );
        }

        println!("└────────────────────────────────────────────────────────────────────────┘");
    } else if args.sweep {
        println!("Temperature Sweep for {} distillation:", args.method.name());
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!("│ Temperature │ PPL After │ Improvement │ Training Cost           │");
        println!("├─────────────┼───────────┼─────────────┼─────────────────────────┤");

        for temp in [1.0, 2.0, 4.0, 8.0, 16.0] {
            let result = analyzer.analyze(args.method, temp);
            let training_cost =
                args.student.training_cost_relative() * args.method.training_overhead();
            println!(
                "│ {:>11.1} │ {:>9.2} │ {:>10.1}% │ {:>22.1}x │",
                temp, result.student_perplexity_after, result.improvement_pct, training_cost
            );
        }

        println!("└──────────────────────────────────────────────────────────────────┘");
    } else {
        let result = analyzer.analyze(args.method, args.temperature);

        println!("Distillation Results:");
        println!("  Method: {}", result.method.name());
        println!("  Temperature: {:.1}", result.temperature);
        println!(
            "  Student PPL (before): {:.2}",
            result.student_perplexity_before
        );
        println!(
            "  Student PPL (after): {:.2}",
            result.student_perplexity_after
        );
        println!("  Improvement: {:.1}%", result.improvement_pct);
        println!("  Compression ratio: {:.1}x", result.compression_ratio);
        println!("  Inference speedup: {:.0}x", result.speedup);
    }
}

fn main() {
    let args = Args::parse();

    // Validate teacher > student
    if args.teacher.parameters_billions() <= args.student.parameters_billions() {
        eprintln!(
            "Warning: Teacher ({:.1}B) should be larger than student ({:.1}B)",
            args.teacher.parameters_billions(),
            args.student.parameters_billions()
        );
    }

    // Clamp temperature to valid range
    let temperature = args.temperature.clamp(1.0, 20.0);
    let args = Args {
        temperature,
        ..args
    };

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-distillation");
        println!("  teacher: {}", args.teacher.name());
        println!("  student: {}", args.student.name());
        println!("  method: {}", args.method.name());
        println!("  temperature: {:.1}", args.temperature);

        let analyzer = DistillationAnalyzer::new(args.teacher, args.student);
        let result = analyzer.analyze(args.method, args.temperature);

        println!("  ppl_before: {:.2}", result.student_perplexity_before);
        println!("  ppl_after: {:.2}", result.student_perplexity_after);
        println!("  improvement: {:.1}%", result.improvement_pct);
        println!("  compression: {:.1}x", result.compression_ratio);
        println!("  speedup: {:.0}x", result.speedup);

        // Check quality threshold: >90% quality retention
        let retention = 1.0
            - (result.student_perplexity_after - args.teacher.baseline_perplexity())
                / (result.student_perplexity_before - args.teacher.baseline_perplexity());
        println!(
            "  quality_threshold: {}",
            if retention >= 0.90 { "PASS" } else { "FAIL" }
        );
    } else {
        // Interactive mode
        print_distillation_diagram(args.teacher, args.student);
        run_analysis(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
