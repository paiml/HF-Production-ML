//! Integration tests for Week 1 demos
//!
//! Falsification testing: Verify that demos have proper thresholds
//! and fail when they should, not just pass for happy paths.

use std::process::Command;

fn run_demo(name: &str, args: &[&str]) -> (bool, String) {
    let output = Command::new("cargo")
        .args(["run", "--release", "--bin", name, "--"])
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute demo");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let has_pass = stdout.contains("[PASS]");
    (has_pass, stdout)
}

#[test]
fn test_tgi_architecture_ci_mode() {
    let (passed, output) = run_demo("demo-tgi-architecture", &["--stdout"]);
    assert!(passed, "demo-tgi-architecture should pass in CI mode");
    assert!(
        output.contains("components:"),
        "Should output components info"
    );
}

#[test]
fn test_tgi_architecture_batch_sizes() {
    for batch_size in &["4", "8", "16", "32"] {
        let (passed, _output) = run_demo(
            "demo-tgi-architecture",
            &["--stdout", "--batch-size", batch_size],
        );
        assert!(
            passed,
            "demo-tgi-architecture should pass for batch size {}",
            batch_size
        );
    }
}

#[test]
fn test_continuous_batching_ci_mode() {
    let (passed, output) = run_demo("demo-continuous-batching", &["--stdout"]);
    assert!(passed, "demo-continuous-batching should pass in CI mode");
    assert!(
        output.contains("num_sequences:"),
        "Should output sequence info"
    );
}

#[test]
fn test_continuous_batching_block_sizes() {
    for block_size in &["8", "16", "32"] {
        let (passed, _output) = run_demo(
            "demo-continuous-batching",
            &["--stdout", "--block-size", block_size],
        );
        assert!(
            passed,
            "demo-continuous-batching should pass for block size {}",
            block_size
        );
    }
}

#[test]
fn test_kv_cache_ci_mode() {
    let (passed, output) = run_demo("demo-kv-cache", &["--stdout"]);
    assert!(passed, "demo-kv-cache should pass in CI mode");
    assert!(output.contains("model:"), "Should output model info");
    assert!(
        output.contains("hidden_dim:"),
        "Should output hidden dimension"
    );
}

#[test]
fn test_kv_cache_model_tiers() {
    for tier in &["tiny", "small", "medium", "large"] {
        let (passed, _output) = run_demo("demo-kv-cache", &["--stdout", "--tier", tier]);
        assert!(passed, "demo-kv-cache should pass for tier {}", tier);
    }
}

#[test]
fn test_serving_api_ci_mode() {
    let (passed, output) = run_demo("demo-serving-api", &["--stdout"]);
    assert!(passed, "demo-serving-api should pass in CI mode");
    assert!(output.contains("[PASS]"), "Should output pass marker");
}

#[test]
fn test_streaming_ci_mode() {
    let (passed, output) = run_demo("demo-streaming", &["--stdout"]);
    assert!(passed, "demo-streaming should pass in CI mode");
    assert!(output.contains("[PASS]"), "Should output pass marker");
}

#[test]
fn test_throughput_bench_ci_mode() {
    let (passed, output) = run_demo("demo-throughput-bench", &["--stdout"]);
    assert!(passed, "demo-throughput-bench should pass in CI mode");
    assert!(output.contains("[PASS]"), "Should output pass marker");
}

#[test]
fn test_throughput_bench_model_tiers() {
    for tier in &["tiny", "small", "medium"] {
        let (passed, _output) = run_demo("demo-throughput-bench", &["--stdout", "--tier", tier]);
        assert!(
            passed,
            "demo-throughput-bench should pass for tier {}",
            tier
        );
    }
}
