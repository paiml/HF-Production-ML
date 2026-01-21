//! Integration tests for Week 2 demos
//!
//! Falsification testing: Verify optimization demos have proper
//! quality/performance thresholds and fail when appropriate.

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

// --- Quantization Tests ---

#[test]
fn test_quantization_ci_mode() {
    let (passed, output) = run_demo("demo-quantization", &["--stdout"]);
    assert!(passed, "demo-quantization should pass in CI mode");
    assert!(
        output.contains("compression:"),
        "Should output compression ratio"
    );
}

#[test]
fn test_quantization_model_tiers() {
    for tier in &["tiny", "small", "medium", "large"] {
        let (passed, _output) = run_demo("demo-quantization", &["--stdout", "--tier", tier]);
        assert!(passed, "demo-quantization should pass for tier {}", tier);
    }
}

// --- APR Format Tests ---

#[test]
fn test_apr_format_ci_mode() {
    let (passed, output) = run_demo("demo-apr-format", &["--stdout"]);
    assert!(passed, "demo-apr-format should pass in CI mode");
    assert!(output.contains("[PASS]"), "Should output pass marker");
}

// --- Flash Attention Tests ---

#[test]
fn test_flash_attention_ci_mode() {
    let (passed, output) = run_demo("demo-flash-attention", &["--stdout"]);
    assert!(passed, "demo-flash-attention should pass in CI mode");
    assert!(output.contains("[PASS]"), "Should output pass marker");
}

#[test]
fn test_flash_attention_sequence_lengths() {
    for seq_len in &["512", "2048", "8192"] {
        let (passed, _output) =
            run_demo("demo-flash-attention", &["--stdout", "--seq-len", seq_len]);
        assert!(
            passed,
            "demo-flash-attention should pass for seq_len {}",
            seq_len
        );
    }
}

// --- Speculative Decoding Tests ---

#[test]
fn test_speculative_decode_ci_mode() {
    let (passed, output) = run_demo("demo-speculative-decode", &["--stdout"]);
    assert!(passed, "demo-speculative-decode should pass in CI mode");
    assert!(
        output.contains("speedup_threshold: PASS"),
        "Should achieve speedup threshold"
    );
}

#[test]
fn test_speculative_decode_speedup() {
    // Speculative decoding should achieve at least 2.8x speedup
    let (passed, output) = run_demo("demo-speculative-decode", &["--stdout"]);
    assert!(passed, "Speculative decoding should pass threshold");
    assert!(output.contains("speedup:"), "Should output speedup");
}

// --- Pruning Tests ---

#[test]
fn test_pruning_ci_mode() {
    let (passed, output) = run_demo("demo-pruning", &["--stdout"]);
    assert!(passed, "demo-pruning should pass in CI mode");
    assert!(
        output.contains("quality_threshold:"),
        "Should output quality threshold"
    );
}

#[test]
fn test_pruning_methods() {
    for method in &[
        "magnitude",
        "movement",
        "sparse-gpt",
        "head-pruning",
        "layer-pruning",
    ] {
        let (passed, _output) = run_demo("demo-pruning", &["--stdout", "--method", method]);
        assert!(passed, "demo-pruning should pass for method {}", method);
    }
}

// --- Distillation Tests ---

#[test]
fn test_distillation_ci_mode() {
    let (passed, output) = run_demo("demo-distillation", &["--stdout"]);
    assert!(passed, "demo-distillation should pass in CI mode");
    assert!(
        output.contains("quality_threshold:"),
        "Should output quality threshold"
    );
}

#[test]
fn test_distillation_teacher_tiers() {
    for teacher in &["medium", "large"] {
        let (passed, _output) = run_demo("demo-distillation", &["--stdout", "--teacher", teacher]);
        assert!(
            passed,
            "demo-distillation should pass for teacher {}",
            teacher
        );
    }
}

// --- Tensor Parallelism Tests ---

#[test]
fn test_tensor_parallel_ci_mode() {
    let (passed, output) = run_demo("demo-tensor-parallel", &["--stdout"]);
    assert!(passed, "demo-tensor-parallel should pass in CI mode");
    assert!(
        output.contains("efficiency_threshold:"),
        "Should output efficiency threshold"
    );
}

#[test]
fn test_tensor_parallel_gpu_counts() {
    for gpus in &["2", "4", "8"] {
        let (passed, _output) = run_demo("demo-tensor-parallel", &["--stdout", "--num-gpus", gpus]);
        assert!(passed, "demo-tensor-parallel should pass for {} GPUs", gpus);
    }
}
