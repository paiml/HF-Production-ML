//! Week 1: Inference Serving Demos
//!
//! Demonstrations of production inference serving concepts:
//! - TGI architecture patterns
//! - Continuous batching with PagedAttention
//! - KV cache management
//! - REST API design (OpenAI-compatible)
//! - Streaming responses (SSE)
//! - Throughput benchmarking

use clap::Args;
use serde::{Deserialize, Serialize};

/// Common demo arguments shared across all demos
#[derive(Args, Debug, Clone)]
pub struct DemoArgs {
    /// Output format: tui (default), stdout (CI), json
    #[arg(long, default_value = "tui")]
    pub output: OutputFormat,

    /// Run in benchmark mode with N iterations
    #[arg(long)]
    pub benchmark: bool,

    /// Number of benchmark iterations
    #[arg(long, default_value = "1000")]
    pub iterations: usize,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Verify falsification criteria
    #[arg(long)]
    pub verify: bool,
}

/// Output format for demo results
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OutputFormat {
    #[default]
    Tui,
    Stdout,
    Json,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "tui" => Ok(Self::Tui),
            "stdout" => Ok(Self::Stdout),
            "json" => Ok(Self::Json),
            _ => Err(format!("Unknown output format: {s}")),
        }
    }
}

/// Demo result with provenance metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoResult {
    pub demo_name: String,
    pub success: bool,
    pub metrics: Vec<Metric>,
    pub provenance: Provenance,
}

/// Individual metric measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub value: f64,
    pub unit: String,
}

/// Provenance metadata for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub version: String,
    pub timestamp: String,
    pub git_sha: String,
    pub hardware: HardwareInfo,
    pub stack_versions: StackVersions,
}

/// Hardware information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu: String,
    pub gpu: Option<String>,
    pub memory_gb: u64,
}

/// Sovereign stack versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackVersions {
    pub trueno: String,
    pub aprender: String,
    pub realizar: String,
    pub batuta: String,
}

impl Default for StackVersions {
    fn default() -> Self {
        Self {
            trueno: "0.13.0".to_string(),
            aprender: "0.24.1".to_string(),
            realizar: "0.6.8".to_string(),
            batuta: "0.5.0".to_string(),
        }
    }
}

/// Render result as TUI (placeholder)
pub fn render_tui(result: &DemoResult) -> anyhow::Result<()> {
    println!("=== {} ===", result.demo_name);
    for metric in &result.metrics {
        println!("  {}: {:.2} {}", metric.name, metric.value, metric.unit);
    }
    Ok(())
}

/// Print result to stdout for CI
pub fn print_stdout(result: &DemoResult) {
    println!(
        "[{}] {}",
        if result.success { "PASS" } else { "FAIL" },
        result.demo_name
    );
    for metric in &result.metrics {
        println!("  {}: {:.2} {}", metric.name, metric.value, metric.unit);
    }
}
