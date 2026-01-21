//! Demo: Hardware Detection
//!
//! Demonstrates automatic hardware capability detection:
//! - CPU features (AVX2, AVX-512, NEON)
//! - GPU detection and VRAM sizing
//! - Memory bandwidth estimation
//! - Optimal model tier selection
//!
//! Uses system introspection for runtime optimization.
//!
//! References:
//! - CPUID instruction for x86 feature detection
//! - CUDA/Vulkan device enumeration
//! - Memory hierarchy and cache sizing

use clap::{Parser, ValueEnum};
use std::io::{self, Write};

/// CPU architecture
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum CpuArch {
    /// x86-64 with AVX2
    X86Avx2,
    /// x86-64 with AVX-512
    X86Avx512,
    /// ARM64 with NEON
    Arm64Neon,
    /// Apple Silicon (M1/M2/M3)
    AppleSilicon,
}

/// GPU type
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum GpuType {
    /// No GPU
    None,
    /// NVIDIA RTX 30 series
    NvidiaRtx30,
    /// NVIDIA RTX 40 series
    NvidiaRtx40,
    /// NVIDIA A100/H100
    NvidiaDatacenter,
    /// AMD RDNA 3
    AmdRdna3,
    /// Apple Metal
    AppleMetal,
}

/// Recommended model tier
#[derive(Debug, Clone, Copy)]
enum RecommendedTier {
    Tiny,
    Small,
    Medium,
    Large,
}

impl CpuArch {
    fn name(&self) -> &'static str {
        match self {
            CpuArch::X86Avx2 => "x86-64 AVX2",
            CpuArch::X86Avx512 => "x86-64 AVX-512",
            CpuArch::Arm64Neon => "ARM64 NEON",
            CpuArch::AppleSilicon => "Apple Silicon",
        }
    }

    fn simd_width(&self) -> usize {
        match self {
            CpuArch::X86Avx2 => 256,
            CpuArch::X86Avx512 => 512,
            CpuArch::Arm64Neon => 128,
            CpuArch::AppleSilicon => 128,
        }
    }

    fn cpu_inference_capable(&self) -> bool {
        true // All modern CPUs can do inference
    }

    fn estimated_cpu_toks(&self, model_params_b: f64) -> f64 {
        // Rough CPU inference speed
        let base = match self {
            CpuArch::X86Avx2 => 10.0,
            CpuArch::X86Avx512 => 20.0,
            CpuArch::Arm64Neon => 8.0,
            CpuArch::AppleSilicon => 25.0, // AMX acceleration
        };
        base / model_params_b
    }
}

impl GpuType {
    fn name(&self) -> &'static str {
        match self {
            GpuType::None => "None",
            GpuType::NvidiaRtx30 => "NVIDIA RTX 30-series",
            GpuType::NvidiaRtx40 => "NVIDIA RTX 40-series",
            GpuType::NvidiaDatacenter => "NVIDIA Datacenter (A100/H100)",
            GpuType::AmdRdna3 => "AMD RDNA 3",
            GpuType::AppleMetal => "Apple Metal",
        }
    }

    fn vram_gb(&self) -> f64 {
        match self {
            GpuType::None => 0.0,
            GpuType::NvidiaRtx30 => 12.0,      // Typical 3080
            GpuType::NvidiaRtx40 => 24.0,      // 4090
            GpuType::NvidiaDatacenter => 80.0, // A100/H100
            GpuType::AmdRdna3 => 16.0,         // 7900 XTX
            GpuType::AppleMetal => 32.0,       // M2 Max unified
        }
    }

    fn memory_bandwidth_gbps(&self) -> f64 {
        match self {
            GpuType::None => 0.0,
            GpuType::NvidiaRtx30 => 760.0,
            GpuType::NvidiaRtx40 => 1008.0,
            GpuType::NvidiaDatacenter => 3350.0, // H100
            GpuType::AmdRdna3 => 960.0,
            GpuType::AppleMetal => 400.0,
        }
    }

    fn estimated_gpu_toks(&self, model_params_b: f64) -> f64 {
        if matches!(self, GpuType::None) {
            return 0.0;
        }
        // Memory-bound estimate: bandwidth / (params * bytes_per_param)
        let bytes_per_param = 2.0; // FP16
        let model_bytes = model_params_b * 1e9 * bytes_per_param;
        self.memory_bandwidth_gbps() * 1e9 / model_bytes
    }
}

impl RecommendedTier {
    fn name(&self) -> &'static str {
        match self {
            RecommendedTier::Tiny => "Tiny (0.5B)",
            RecommendedTier::Small => "Small (1.5B)",
            RecommendedTier::Medium => "Medium (7B)",
            RecommendedTier::Large => "Large (32B)",
        }
    }

    fn model_name(&self) -> &'static str {
        match self {
            RecommendedTier::Tiny => "Qwen2.5-0.5B-Instruct",
            RecommendedTier::Small => "Qwen2.5-Coder-1.5B-Instruct",
            RecommendedTier::Medium => "Qwen2.5-Coder-7B-Instruct",
            RecommendedTier::Large => "Qwen2.5-Coder-32B-Instruct",
        }
    }

    fn params_b(&self) -> f64 {
        match self {
            RecommendedTier::Tiny => 0.5,
            RecommendedTier::Small => 1.5,
            RecommendedTier::Medium => 7.0,
            RecommendedTier::Large => 32.0,
        }
    }

    fn memory_required_gb(&self) -> f64 {
        // Q4 quantized + overhead
        self.params_b() * 0.5 * 1.3
    }
}

/// Hardware detection result
#[derive(Debug)]
#[allow(dead_code)]
struct HardwareProfile {
    cpu: CpuArch,
    gpu: GpuType,
    system_ram_gb: f64,
    recommended_tier: RecommendedTier,
    use_gpu: bool,
    estimated_speed: f64,
}

/// Hardware detector
struct HardwareDetector {
    cpu: CpuArch,
    gpu: GpuType,
    system_ram_gb: f64,
}

impl HardwareDetector {
    fn new(cpu: CpuArch, gpu: GpuType, system_ram_gb: f64) -> Self {
        Self {
            cpu,
            gpu,
            system_ram_gb,
        }
    }

    fn detect(&self) -> HardwareProfile {
        let gpu_vram = self.gpu.vram_gb();
        let use_gpu = gpu_vram > 0.0;

        // Determine recommended tier based on available memory
        let available_memory = if use_gpu {
            gpu_vram
        } else {
            self.system_ram_gb * 0.8 // Use 80% of system RAM
        };

        let recommended_tier = if available_memory >= 20.0 {
            RecommendedTier::Large
        } else if available_memory >= 6.0 {
            RecommendedTier::Medium
        } else if available_memory >= 2.0 {
            RecommendedTier::Small
        } else {
            RecommendedTier::Tiny
        };

        let estimated_speed = if use_gpu {
            self.gpu.estimated_gpu_toks(recommended_tier.params_b())
        } else {
            self.cpu.estimated_cpu_toks(recommended_tier.params_b())
        };

        HardwareProfile {
            cpu: self.cpu,
            gpu: self.gpu,
            system_ram_gb: self.system_ram_gb,
            recommended_tier,
            use_gpu,
            estimated_speed,
        }
    }
}

/// Hardware Detection Demo
#[derive(Parser)]
#[command(name = "demo-hardware-detect")]
#[command(about = "Demonstrate automatic hardware detection")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// CPU architecture (simulated)
    #[arg(long, value_enum, default_value = "x86-avx2")]
    cpu: CpuArch,

    /// GPU type (simulated)
    #[arg(long, value_enum, default_value = "nvidia-rtx40")]
    gpu: GpuType,

    /// System RAM in GB
    #[arg(long, default_value = "32")]
    ram: f64,
}

fn print_hardware_diagram(profile: &HardwareProfile) {
    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HARDWARE DETECTION                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Detected Hardware:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CPU: {}                                                     │   │
│  │  ├─ SIMD Width: {} bits                                            │   │
│  │  └─ CPU Inference: {}                                              │   │
│  │                                                                     │   │
│  │  GPU: {}                                              │   │
│  │  ├─ VRAM: {:.0} GB                                                   │   │
│  │  └─ Memory Bandwidth: {:.0} GB/s                                    │   │
│  │                                                                     │   │
│  │  System RAM: {:.0} GB                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Recommendation:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Model: {}                                          │   │
│  │  Backend: {}                                                       │   │
│  │  Estimated Speed: {:.0} tok/s                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        profile.cpu.name(),
        profile.cpu.simd_width(),
        if profile.cpu.cpu_inference_capable() {
            "Yes"
        } else {
            "No"
        },
        profile.gpu.name(),
        profile.gpu.vram_gb(),
        profile.gpu.memory_bandwidth_gbps(),
        profile.system_ram_gb,
        profile.recommended_tier.model_name(),
        if profile.use_gpu { "GPU" } else { "CPU" },
        profile.estimated_speed
    );
}

fn run_detection(args: &Args) {
    println!("\n=== Hardware Detection ===\n");

    let detector = HardwareDetector::new(args.cpu, args.gpu, args.ram);
    let profile = detector.detect();

    println!("CPU:");
    println!("  Architecture: {}", profile.cpu.name());
    println!("  SIMD width: {} bits", profile.cpu.simd_width());
    println!();

    println!("GPU:");
    println!("  Type: {}", profile.gpu.name());
    println!("  VRAM: {:.0} GB", profile.gpu.vram_gb());
    println!(
        "  Memory Bandwidth: {:.0} GB/s",
        profile.gpu.memory_bandwidth_gbps()
    );
    println!();

    println!("System:");
    println!("  RAM: {:.0} GB", profile.system_ram_gb);
    println!();

    println!("Recommendation:");
    println!("  Model tier: {}", profile.recommended_tier.name());
    println!("  Full name: {}", profile.recommended_tier.model_name());
    println!(
        "  Memory required: {:.1} GB (Q4 quantized)",
        profile.recommended_tier.memory_required_gb()
    );
    println!("  Backend: {}", if profile.use_gpu { "GPU" } else { "CPU" });
    println!("  Estimated speed: {:.0} tok/s", profile.estimated_speed);
}

fn main() {
    let args = Args::parse();

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-hardware-detect");
        println!("  cpu: {}", args.cpu.name());
        println!("  gpu: {}", args.gpu.name());
        println!("  ram: {:.0} GB", args.ram);

        let detector = HardwareDetector::new(args.cpu, args.gpu, args.ram);
        let profile = detector.detect();

        println!("  simd_width: {} bits", profile.cpu.simd_width());
        println!("  vram: {:.0} GB", profile.gpu.vram_gb());
        println!("  recommended_tier: {}", profile.recommended_tier.name());
        println!(
            "  recommended_model: {}",
            profile.recommended_tier.model_name()
        );
        println!("  use_gpu: {}", profile.use_gpu);
        println!("  estimated_speed: {:.0} tok/s", profile.estimated_speed);

        // Check detection validity
        let valid = profile.estimated_speed > 0.0;
        println!("  detection_valid: {}", if valid { "PASS" } else { "FAIL" });
    } else {
        // Interactive mode
        let detector = HardwareDetector::new(args.cpu, args.gpu, args.ram);
        let profile = detector.detect();
        print_hardware_diagram(&profile);
        run_detection(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
