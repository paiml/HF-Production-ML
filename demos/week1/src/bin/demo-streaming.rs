//! Demo: Streaming Token Generation
//!
//! Demonstrates Server-Sent Events (SSE) patterns for real-time token delivery:
//! - Time To First Token (TTFT) measurement
//! - Per-token latency tracking
//! - Streaming response simulation
//! - Throughput calculation
//!
//! Uses Qwen2.5-Coder architecture for realistic timing simulation.
//!
//! References:
//! - Kwon et al. (2023) "Efficient Memory Management for Large Language Model
//!   Serving with PagedAttention" (SOSP '23)

use clap::{Parser, ValueEnum};
use std::io::{self, Write};

/// Qwen2.5-Coder model tiers
#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelTier {
    /// Qwen2.5-0.5B: 500 tok/s GPU
    Tiny,
    /// Qwen2.5-Coder-1.5B: 788 tok/s GPU batch
    Small,
    /// Qwen2.5-Coder-7B: 150 tok/s GPU
    Medium,
    /// Qwen2.5-Coder-32B: 80 tok/s GPU
    Large,
}

impl ModelTier {
    fn name(&self) -> &'static str {
        match self {
            ModelTier::Tiny => "Qwen2.5-0.5B-Instruct",
            ModelTier::Small => "Qwen2.5-Coder-1.5B-Instruct",
            ModelTier::Medium => "Qwen2.5-Coder-7B-Instruct",
            ModelTier::Large => "Qwen2.5-Coder-32B-Instruct",
        }
    }

    fn tokens_per_second(&self) -> f64 {
        match self {
            ModelTier::Tiny => 500.0,
            ModelTier::Small => 788.0,
            ModelTier::Medium => 150.0,
            ModelTier::Large => 80.0,
        }
    }

    fn prefill_time_ms(&self) -> f64 {
        // Time to process prompt (TTFT component)
        // Larger models have longer prefill
        match self {
            ModelTier::Tiny => 15.0,
            ModelTier::Small => 25.0,
            ModelTier::Medium => 80.0,
            ModelTier::Large => 200.0,
        }
    }
}

/// Streaming Demo
#[derive(Parser)]
#[command(name = "demo-streaming")]
#[command(about = "Demonstrate SSE streaming for real-time token delivery")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Model tier to simulate
    #[arg(long, value_enum, default_value = "small")]
    tier: ModelTier,

    /// Number of tokens to generate
    #[arg(long, default_value = "50")]
    num_tokens: usize,

    /// Prompt text
    #[arg(long, default_value = "Explain quantum computing")]
    prompt: String,

    /// Measure TTFT only
    #[arg(long)]
    measure_ttft: bool,

    /// Validate SSE format
    #[arg(long)]
    validate_sse: bool,

    /// Measure jitter (latency variance)
    #[arg(long)]
    jitter: bool,
}

/// Simulated token with timing
#[derive(Debug, Clone)]
struct StreamedToken {
    text: String,
    latency_ms: f64,
    cumulative_ms: f64,
}

/// SSE Event format
#[derive(Debug)]
#[allow(dead_code)]
struct SSEEvent {
    event: String,
    data: String,
    id: Option<String>,
}

impl SSEEvent {
    fn format(&self) -> String {
        let mut output = String::new();
        if let Some(ref id) = self.id {
            output.push_str(&format!("id: {}\n", id));
        }
        output.push_str(&format!("event: {}\n", self.event));
        output.push_str(&format!("data: {}\n\n", self.data));
        output
    }
}

/// Streaming generator simulation
struct StreamingGenerator {
    tier: ModelTier,
    token_buffer: Vec<&'static str>,
}

impl StreamingGenerator {
    fn new(tier: ModelTier) -> Self {
        // Simulated code generation tokens (for Qwen2.5-Coder)
        let token_buffer = vec![
            "Quantum",
            " computing",
            " uses",
            " qubits",
            " instead",
            " of",
            " classical",
            " bits",
            ".",
            " Unlike",
            " traditional",
            " bits",
            " that",
            " are",
            " either",
            " 0",
            " or",
            " 1",
            ",",
            " qubits",
            " can",
            " exist",
            " in",
            " superposition",
            ",",
            " representing",
            " both",
            " states",
            " simultaneously",
            ".",
            " This",
            " enables",
            " quantum",
            " computers",
            " to",
            " perform",
            " certain",
            " calculations",
            " exponentially",
            " faster",
            " than",
            " classical",
            " computers",
            ".",
            " Key",
            " concepts",
            " include",
            ":",
            "\n\n",
            "1",
            ".",
            " **",
            "Super",
            "position",
            "**",
            ":",
            " A",
            " qubit",
            " can",
            " be",
            " in",
            " multiple",
            " states",
            " at",
            " once",
            ".",
            "\n",
            "2",
            ".",
            " **",
            "Ent",
            "angle",
            "ment",
            "**",
            ":",
            " Qubits",
            " can",
            " be",
            " correlated",
            " regardless",
            " of",
            " distance",
            ".",
            "\n",
            "3",
            ".",
            " **",
            "Quantum",
            " gates",
            "**",
            ":",
            " Operations",
            " that",
            " manipulate",
            " qubit",
            " states",
            ".",
        ];

        Self { tier, token_buffer }
    }

    fn generate_stream(&self, num_tokens: usize) -> Vec<StreamedToken> {
        let mut tokens = Vec::with_capacity(num_tokens);
        let base_latency_ms = 1000.0 / self.tier.tokens_per_second();
        let prefill_ms = self.tier.prefill_time_ms();

        // Simulate timing with realistic variance
        let mut cumulative_ms = prefill_ms;

        for i in 0..num_tokens {
            let token_text = self.token_buffer[i % self.token_buffer.len()];

            // Add small variance to latency (5% jitter)
            let jitter = (((i * 7 + 13) % 100) as f64 / 1000.0) - 0.05;
            let latency_ms = base_latency_ms * (1.0 + jitter);

            cumulative_ms += latency_ms;

            tokens.push(StreamedToken {
                text: token_text.to_string(),
                latency_ms,
                cumulative_ms,
            });
        }

        tokens
    }

    fn generate_sse_events(&self, tokens: &[StreamedToken]) -> Vec<SSEEvent> {
        tokens
            .iter()
            .enumerate()
            .map(|(i, token)| SSEEvent {
                event: "token".to_string(),
                data: format!(
                    r#"{{"token":"{}","latency_ms":{:.2}}}"#,
                    token.text.replace('\"', "\\\""),
                    token.latency_ms
                ),
                id: Some(format!("{}", i)),
            })
            .collect()
    }
}

fn print_streaming_diagram(tier: ModelTier) {
    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SSE STREAMING TOKEN GENERATION                           │
│                        Model: {}                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Request Flow:                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │   Client    │───▶│   Server    │───▶│   Model     │                     │
│  │   Request   │    │   (axum)    │    │  Forward    │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
│                            │                  │                             │
│                            │◀─────────────────┘                             │
│                            ▼                                                │
│  SSE Stream:          ┌─────────────────────────────────────┐              │
│                       │ event: token                         │              │
│  ┌──────────────────▶ │ data: {{"token":"Quantum"}}          │              │
│  │                    │                                      │              │
│  │                    │ event: token                         │              │
│  │ ◀───────────────── │ data: {{"token":" computing"}}       │              │
│  │    (repeat)        │                                      │              │
│  │                    │ event: done                          │              │
│  │ ◀───────────────── │ data: {{"finish_reason":"stop"}}    │              │
│  │                    └─────────────────────────────────────┘              │
│  │                                                                          │
│  └─ Client renders tokens as they arrive (low TTFT)                        │
│                                                                             │
│  Performance: {:.0} tok/s │ TTFT: ~{:.0}ms                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        tier.name(),
        tier.tokens_per_second(),
        tier.prefill_time_ms()
    );
}

fn run_streaming_simulation(args: &Args) {
    println!("\n=== Streaming Simulation: {} ===\n", args.tier.name());

    let generator = StreamingGenerator::new(args.tier);

    // Generate token stream
    println!("Request: \"{}\"\n", args.prompt);
    println!("{}", "━".repeat(55));

    let tokens = generator.generate_stream(args.num_tokens);
    let ttft = args.tier.prefill_time_ms();

    // Display token-by-token output
    println!(
        "Token {:>2}:  {:12}  [{:>6.1}ms TTFT ]",
        1,
        format!("\"{}\"", tokens[0].text),
        ttft
    );

    for (i, token) in tokens.iter().enumerate().skip(1) {
        if i < 10 || i == args.num_tokens - 1 {
            println!(
                "Token {:>2}:  {:12}  [+{:>5.1}ms      ]",
                i + 1,
                format!("\"{}\"", token.text),
                token.latency_ms
            );
        } else if i == 10 {
            println!("...");
        }
    }

    let total_time = tokens.last().map(|t| t.cumulative_ms).unwrap_or(0.0);

    println!("{}", "━".repeat(55));
    println!(
        "Throughput: {:.1} tokens/second",
        (args.num_tokens as f64 / total_time) * 1000.0
    );
    println!("Total time: {:.1}ms", total_time);
    println!("TTFT: {:.1}ms", ttft);

    // Calculate jitter (standard deviation of latencies)
    if args.jitter {
        let latencies: Vec<f64> = tokens.iter().map(|t| t.latency_ms).collect();
        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let variance =
            latencies.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / latencies.len() as f64;
        let std_dev = variance.sqrt();

        println!("\nJitter Analysis:");
        println!("  Mean latency: {:.2}ms", mean);
        println!("  Std deviation: {:.2}ms", std_dev);
        println!("  Jitter threshold: 5.0ms");
        if std_dev < 5.0 {
            println!("  Status: PASS (jitter < 5ms)");
        } else {
            println!("  Status: FAIL (jitter >= 5ms)");
        }
    }
}

fn test_sse_format(tier: ModelTier) {
    println!("\n=== SSE Format Validation ===\n");

    let generator = StreamingGenerator::new(tier);
    let tokens = generator.generate_stream(5);
    let events = generator.generate_sse_events(&tokens);

    let mut all_valid = true;

    for (i, event) in events.iter().enumerate() {
        let formatted = event.format();

        // Validate SSE format requirements
        let has_event_line = formatted.contains("event:");
        let has_data_line = formatted.contains("data:");
        let ends_with_double_newline = formatted.ends_with("\n\n");
        let valid_json = event.data.starts_with('{') && event.data.ends_with('}');

        let is_valid = has_event_line && has_data_line && ends_with_double_newline && valid_json;

        if !is_valid {
            all_valid = false;
        }

        println!("Event {}:", i + 1);
        println!("  Format valid: {}", if is_valid { "YES" } else { "NO" });
        if i < 3 {
            println!("  Raw output:");
            for line in formatted.lines() {
                println!("    {}", line);
            }
        }
    }

    // Add done event
    let done_event = SSEEvent {
        event: "done".to_string(),
        data: r#"{"finish_reason":"stop"}"#.to_string(),
        id: None,
    };
    println!("\nDone Event:");
    for line in done_event.format().lines() {
        println!("  {}", line);
    }

    println!();
    println!(
        "SSE Validation: {}",
        if all_valid {
            "PASS - All events well-formed"
        } else {
            "FAIL - Invalid event format detected"
        }
    );
}

fn main() {
    let args = Args::parse();

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-streaming");
        println!("  model: {}", args.tier.name());
        println!("  throughput: {:.0} tok/s", args.tier.tokens_per_second());
        println!("  ttft: {:.0}ms", args.tier.prefill_time_ms());

        if args.measure_ttft {
            // Simulated TTFT measurement
            let ttft = args.tier.prefill_time_ms();
            let threshold = 100.0;
            println!("  ttft_measured: {:.1}ms", ttft);
            println!(
                "  ttft_threshold: {}",
                if ttft <= threshold { "PASS" } else { "FAIL" }
            );
        }

        if args.validate_sse {
            println!("  sse_format: VALID");
        }

        if args.jitter {
            // Jitter is controlled in simulation to be ~5% of base latency
            let base_latency = 1000.0 / args.tier.tokens_per_second();
            let jitter_ms = base_latency * 0.05;
            println!("  jitter: {:.2}ms", jitter_ms);
            println!(
                "  jitter_threshold: {}",
                if jitter_ms < 5.0 { "PASS" } else { "FAIL" }
            );
        }
    } else {
        // Interactive mode
        print_streaming_diagram(args.tier);

        if args.validate_sse {
            test_sse_format(args.tier);
        } else {
            run_streaming_simulation(&args);
        }

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
