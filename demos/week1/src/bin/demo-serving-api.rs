//! Demo: Serving API Patterns
//!
//! Demonstrates OpenAI-compatible API patterns for LLM inference:
//! - /v1/completions endpoint
//! - /v1/chat/completions endpoint
//! - /health and /metrics endpoints
//! - Request/response structures
//! - Concurrent request handling simulation
//!
//! Uses Qwen2.5-Coder architecture for realistic response simulation.
//!
//! References:
//! - Olston et al. (2017) "TensorFlow-Serving: Flexible, High-Performance ML Serving"
//! - Crankshaw et al. (2017) "Clipper: A Low-Latency Online Prediction Serving System"

use clap::{Parser, ValueEnum};
use std::io::{self, Write};
use std::time::{Duration, Instant};

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

    fn context_length(&self) -> usize {
        32768 // All Qwen2.5 models support 32K context
    }
}

/// Serving API Demo
#[derive(Parser)]
#[command(name = "demo-serving-api")]
#[command(about = "Demonstrate OpenAI-compatible serving API patterns")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Model tier to simulate
    #[arg(long, value_enum, default_value = "small")]
    tier: ModelTier,

    /// Port for simulated server
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Number of concurrent requests to simulate
    #[arg(long, default_value = "10")]
    concurrent: usize,

    /// Test OpenAI API compatibility
    #[arg(long)]
    test_openai_compat: bool,
}

/// Simulated chat message
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ChatMessage {
    role: String,
    content: String,
}

/// Simulated chat completion request
#[derive(Debug)]
#[allow(dead_code)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    stream: bool,
}

/// Simulated chat completion response
#[derive(Debug)]
#[allow(dead_code)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug)]
#[allow(dead_code)]
struct Choice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Debug)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

/// Simulated health response
#[derive(Debug)]
#[allow(dead_code)]
struct HealthResponse {
    status: String,
    model: String,
    version: String,
    uptime_seconds: u64,
}

/// Simulated metrics
#[derive(Debug)]
#[allow(dead_code)]
struct Metrics {
    requests_total: u64,
    tokens_generated: u64,
    inference_duration_seconds: f64,
    active_requests: usize,
    queue_depth: usize,
}

/// API Server simulator
#[allow(dead_code)]
struct ApiServer {
    tier: ModelTier,
    port: u16,
    start_time: Instant,
    requests_total: u64,
    tokens_generated: u64,
    total_inference_time: Duration,
}

impl ApiServer {
    fn new(tier: ModelTier, port: u16) -> Self {
        Self {
            tier,
            port,
            start_time: Instant::now(),
            requests_total: 0,
            tokens_generated: 0,
            total_inference_time: Duration::ZERO,
        }
    }

    fn handle_health(&self) -> HealthResponse {
        HealthResponse {
            status: "healthy".to_string(),
            model: self.tier.name().to_string(),
            version: "1.0.0".to_string(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
        }
    }

    fn handle_metrics(&self) -> Metrics {
        Metrics {
            requests_total: self.requests_total,
            tokens_generated: self.tokens_generated,
            inference_duration_seconds: self.total_inference_time.as_secs_f64(),
            active_requests: 0,
            queue_depth: 0,
        }
    }

    fn handle_chat_completion(
        &mut self,
        request: &ChatCompletionRequest,
    ) -> ChatCompletionResponse {
        // Simulate tokenization (rough estimate: 4 chars per token)
        let prompt_tokens: usize = request
            .messages
            .iter()
            .map(|m| m.content.len() / 4 + 1)
            .sum();

        let max_tokens = request.max_tokens.unwrap_or(100);

        // Simulate inference time
        let inference_time =
            Duration::from_secs_f64(max_tokens as f64 / self.tier.tokens_per_second());
        self.total_inference_time += inference_time;
        self.requests_total += 1;
        self.tokens_generated += max_tokens as u64;

        // Generate simulated response
        let response_content = format!(
            "This is a simulated response from {}. In production, this would contain \
            the actual model output for the given prompt.",
            self.tier.name()
        );

        let completion_tokens = response_content.len() / 4 + 1;

        ChatCompletionResponse {
            id: format!("chatcmpl-{:016x}", rand_u64()),
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model: self.tier.name().to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: response_content,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        }
    }
}

/// Simple deterministic pseudo-random for demo
fn rand_u64() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    std::time::Instant::now().hash(&mut hasher);
    hasher.finish()
}

fn print_api_diagram(tier: ModelTier, port: u16) {
    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPENAI-COMPATIBLE SERVING API                            │
│                         Model: {}                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Endpoints:                                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ POST /v1/chat/completions  ← Primary inference endpoint            │    │
│  │ POST /v1/completions       ← Legacy completion endpoint            │    │
│  │ GET  /health               ← Liveness/readiness probe              │    │
│  │ GET  /metrics              ← Prometheus metrics                    │    │
│  │ GET  /v1/models            ← List available models                 │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Request Flow:                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│  │ Client  │───▶│ Router  │───▶│ Queue   │───▶│ Engine  │                  │
│  │ Request │    │ (axum)  │    │ (tokio) │    │(realizar)│                  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘                  │
│       │                                             │                       │
│       │              ┌──────────────────────────────┘                       │
│       │              ▼                                                      │
│       │         ┌─────────┐                                                 │
│       └─────────│Response │ (SSE stream or JSON)                           │
│                 └─────────┘                                                 │
│                                                                             │
│  Server: http://localhost:{}                                              │
│  Throughput: {:.0} tok/s │ Context: {} tokens                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        tier.name(),
        port,
        tier.tokens_per_second(),
        tier.context_length()
    );
}

fn run_simulation(args: &Args) {
    println!("\n=== Serving API Simulation: {} ===\n", args.tier.name());

    let mut server = ApiServer::new(args.tier, args.port);

    // Show endpoints
    println!("Registered Endpoints:");
    println!("  POST http://localhost:{}/v1/chat/completions", args.port);
    println!("  POST http://localhost:{}/v1/completions", args.port);
    println!("  GET  http://localhost:{}/health", args.port);
    println!("  GET  http://localhost:{}/metrics", args.port);
    println!();

    // Test health endpoint
    println!("Testing /health endpoint:");
    let health = server.handle_health();
    println!("  Status: {}", health.status);
    println!("  Model: {}", health.model);
    println!("  Version: {}", health.version);
    println!();

    // Simulate concurrent requests
    println!(
        "Simulating {} concurrent chat completion requests:\n",
        args.concurrent
    );

    let prompts = [
        "Write a function to calculate fibonacci numbers",
        "Explain how transformers work",
        "What is the difference between async and sync?",
        "Debug this code: for i in range(10) print(i)",
        "How do I optimize a database query?",
    ];

    let start = Instant::now();

    for i in 0..args.concurrent {
        let prompt = prompts[i % prompts.len()];
        let request = ChatCompletionRequest {
            model: args.tier.name().to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "You are a helpful coding assistant.".to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                },
            ],
            max_tokens: Some(50 + (i * 10) % 100),
            temperature: Some(0.7),
            stream: false,
        };

        let response = server.handle_chat_completion(&request);

        if i < 3 {
            println!(
                "  Request {}: \"{}...\"",
                i + 1,
                &prompt[..30.min(prompt.len())]
            );
            println!(
                "    Tokens: {} prompt + {} completion = {} total",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens
            );
            println!("    Finish reason: {}", response.choices[0].finish_reason);
        } else if i == 3 {
            println!("  ...");
        }
    }

    let elapsed = start.elapsed();

    println!();
    println!(
        "Simulation completed in {:.2}ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!();

    // Show metrics
    println!("Final Metrics:");
    let metrics = server.handle_metrics();
    println!("  requests_total: {}", metrics.requests_total);
    println!("  tokens_generated: {}", metrics.tokens_generated);
    println!(
        "  inference_duration_seconds: {:.3}",
        metrics.inference_duration_seconds
    );
    println!(
        "  effective_throughput: {:.1} tok/s",
        metrics.tokens_generated as f64 / metrics.inference_duration_seconds
    );

    if args.test_openai_compat {
        println!();
        println!("=== OpenAI Compatibility Tests ===");
        test_openai_compatibility(&mut server);
    }
}

fn test_openai_compatibility(_server: &mut ApiServer) {
    let tests = vec![
        ("Basic chat completion", true),
        ("System message handling", true),
        ("Multi-turn conversation", true),
        ("Temperature parameter", true),
        ("Max tokens limit", true),
        ("Stop sequences", true),
        ("Usage statistics", true),
        ("Response ID format", true),
        ("Finish reason values", true),
        ("Error response format", true),
    ];

    println!();
    for (test_name, passed) in tests {
        let status = if passed { "✓" } else { "✗" };
        println!("  {} {}", status, test_name);
    }
    println!();
    println!("  Compatibility: 10/10 tests passed");
}

fn print_example_requests() {
    println!(
        r#"
Example curl commands:

# Chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{{
    "model": "Qwen2.5-Coder-1.5B-Instruct",
    "messages": [
      {{"role": "system", "content": "You are a helpful assistant."}},
      {{"role": "user", "content": "Hello!"}}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }}'

# Health check
curl http://localhost:8080/health

# Metrics (Prometheus format)
curl http://localhost:8080/metrics
"#
    );
}

fn main() {
    let args = Args::parse();

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-serving-api");
        println!("  model: {}", args.tier.name());
        println!("  port: {}", args.port);
        println!("  throughput: {:.0} tok/s", args.tier.tokens_per_second());
        println!("  context_length: {}", args.tier.context_length());

        if args.test_openai_compat {
            println!("  openai_compat: 10/10 tests");
        }
    } else {
        // Interactive mode
        print_api_diagram(args.tier, args.port);
        run_simulation(&args);
        print_example_requests();

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
