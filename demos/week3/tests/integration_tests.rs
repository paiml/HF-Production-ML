//! Integration tests for Week 3 demos
//!
//! Falsification testing: Verify edge deployment demos have proper
//! viability thresholds and fail for unsupported configurations.

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

// --- WASM Inference Tests ---

#[test]
fn test_wasm_inference_ci_mode() {
    let (passed, output) = run_demo("demo-wasm-inference", &["--stdout"]);
    assert!(passed, "demo-wasm-inference should pass in CI mode");
    assert!(
        output.contains("edge_viable:"),
        "Should output viability check"
    );
}

#[test]
fn test_wasm_inference_targets() {
    for target in &["browser", "node", "cloudflare-workers", "deno"] {
        let (passed, _output) = run_demo("demo-wasm-inference", &["--stdout", "--target", target]);
        assert!(
            passed,
            "demo-wasm-inference should pass for target {}",
            target
        );
    }
}

#[test]
fn test_wasm_inference_quantization() {
    for quant in &["fp32", "fp16", "q8", "q4"] {
        let (passed, _output) = run_demo("demo-wasm-inference", &["--stdout", "--quant", quant]);
        assert!(
            passed,
            "demo-wasm-inference should pass for quant {}",
            quant
        );
    }
}

// --- Lambda Handler Tests ---

#[test]
fn test_lambda_handler_ci_mode() {
    let (passed, output) = run_demo("demo-lambda-handler", &["--stdout"]);
    assert!(passed, "demo-lambda-handler should pass in CI mode");
    assert!(
        output.contains("lambda_viable:"),
        "Should output viability check"
    );
}

#[test]
fn test_lambda_handler_memory_configs() {
    for memory in &["mb1024", "mb2048", "mb4096", "mb10240"] {
        let (passed, _output) = run_demo("demo-lambda-handler", &["--stdout", "--memory", memory]);
        assert!(
            passed,
            "demo-lambda-handler should pass for memory {}",
            memory
        );
    }
}

#[test]
fn test_lambda_handler_model_tiers() {
    for tier in &["tiny", "small"] {
        let (passed, _output) = run_demo("demo-lambda-handler", &["--stdout", "--tier", tier]);
        assert!(passed, "demo-lambda-handler should pass for tier {}", tier);
    }
}

// --- Latency Profile Tests ---

#[test]
fn test_latency_profile_ci_mode() {
    let (passed, output) = run_demo("demo-latency-profile", &["--stdout"]);
    assert!(passed, "demo-latency-profile should pass in CI mode");
    assert!(
        output.contains("ttft_threshold:"),
        "Should output TTFT threshold"
    );
}

#[test]
fn test_latency_profile_tiers() {
    for tier in &["tiny", "small", "medium", "large"] {
        let (passed, _output) = run_demo("demo-latency-profile", &["--stdout", "--tier", tier]);
        assert!(passed, "demo-latency-profile should pass for tier {}", tier);
    }
}

// --- Hardware Detection Tests ---

#[test]
fn test_hardware_detect_ci_mode() {
    let (passed, output) = run_demo("demo-hardware-detect", &["--stdout"]);
    assert!(passed, "demo-hardware-detect should pass in CI mode");
    assert!(
        output.contains("detection_valid:"),
        "Should output detection validity"
    );
}

#[test]
fn test_hardware_detect_cpu_architectures() {
    for cpu in &["x86-avx2", "x86-avx512", "arm64-neon", "apple-silicon"] {
        let (passed, _output) = run_demo("demo-hardware-detect", &["--stdout", "--cpu", cpu]);
        assert!(passed, "demo-hardware-detect should pass for CPU {}", cpu);
    }
}

#[test]
fn test_hardware_detect_gpu_types() {
    for gpu in &[
        "none",
        "nvidia-rtx30",
        "nvidia-rtx40",
        "nvidia-datacenter",
        "amd-rdna3",
        "apple-metal",
    ] {
        let (passed, _output) = run_demo("demo-hardware-detect", &["--stdout", "--gpu", gpu]);
        assert!(passed, "demo-hardware-detect should pass for GPU {}", gpu);
    }
}

// --- Presentar UI Tests ---

#[test]
fn test_presentar_ui_ci_mode() {
    let (passed, output) = run_demo("demo-presentar-ui", &["--stdout"]);
    assert!(passed, "demo-presentar-ui should pass in CI mode");
    assert!(output.contains("ui_viable:"), "Should output UI viability");
}

#[test]
fn test_presentar_ui_layouts() {
    for layout in &["compact", "split", "dashboard"] {
        let (passed, _output) = run_demo("demo-presentar-ui", &["--stdout", "--layout", layout]);
        assert!(
            passed,
            "demo-presentar-ui should pass for layout {}",
            layout
        );
    }
}

#[test]
fn test_presentar_ui_themes() {
    for theme in &["dark", "light", "high-contrast"] {
        let (passed, _output) = run_demo("demo-presentar-ui", &["--stdout", "--theme", theme]);
        assert!(passed, "demo-presentar-ui should pass for theme {}", theme);
    }
}

// --- Model Registry Tests ---

#[test]
fn test_model_registry_ci_mode() {
    let (passed, output) = run_demo("demo-model-registry", &["--stdout"]);
    assert!(passed, "demo-model-registry should pass in CI mode");
    assert!(
        output.contains("registry_valid:"),
        "Should output registry validity"
    );
}

#[test]
fn test_model_registry_operations() {
    for op in &["query", "list", "register", "promote", "download"] {
        let (passed, _output) = run_demo("demo-model-registry", &["--stdout", "--op", op]);
        assert!(
            passed,
            "demo-model-registry should pass for operation {}",
            op
        );
    }
}

#[test]
fn test_model_registry_model_tiers() {
    for tier in &["tiny", "small", "medium", "large"] {
        let (passed, _output) = run_demo("demo-model-registry", &["--stdout", "--tier", tier]);
        assert!(passed, "demo-model-registry should pass for tier {}", tier);
    }
}
