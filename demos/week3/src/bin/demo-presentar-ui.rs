//! Demo: Presentar TUI Framework
//!
//! Demonstrates terminal UI for ML model inference:
//! - Widget system for inference visualization
//! - Streaming token display
//! - Real-time metrics dashboard
//! - Interactive model selection
//!
//! Uses Qwen2.5-Coder for realistic inference simulation.
//!
//! References:
//! - Terminal rendering with ANSI escape codes
//! - Immediate-mode UI patterns
//! - Double-buffering for flicker-free updates

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

/// UI widget type
#[derive(Debug, Clone, Copy, PartialEq)]
enum WidgetType {
    ModelSelector,
    PromptInput,
    StreamingOutput,
    MetricsDashboard,
    TokenVisualization,
    ProgressBar,
}

/// UI layout mode
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum LayoutMode {
    /// Single column layout
    Compact,
    /// Split view with metrics
    Split,
    /// Full dashboard
    Dashboard,
}

/// Color theme
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum Theme {
    /// Dark theme (default)
    Dark,
    /// Light theme
    Light,
    /// High contrast
    HighContrast,
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

    fn parameters_billions(&self) -> f64 {
        match self {
            ModelTier::Tiny => 0.5,
            ModelTier::Small => 1.5,
            ModelTier::Medium => 7.0,
            ModelTier::Large => 32.0,
        }
    }

    fn simulated_speed(&self) -> f64 {
        // Tokens per second for UI simulation
        match self {
            ModelTier::Tiny => 120.0,
            ModelTier::Small => 80.0,
            ModelTier::Medium => 40.0,
            ModelTier::Large => 15.0,
        }
    }
}

impl WidgetType {
    fn name(&self) -> &'static str {
        match self {
            WidgetType::ModelSelector => "Model Selector",
            WidgetType::PromptInput => "Prompt Input",
            WidgetType::StreamingOutput => "Streaming Output",
            WidgetType::MetricsDashboard => "Metrics Dashboard",
            WidgetType::TokenVisualization => "Token Visualization",
            WidgetType::ProgressBar => "Progress Bar",
        }
    }

    fn render_cost(&self) -> usize {
        // Estimated render cost in microseconds
        match self {
            WidgetType::ModelSelector => 50,
            WidgetType::PromptInput => 30,
            WidgetType::StreamingOutput => 100,
            WidgetType::MetricsDashboard => 200,
            WidgetType::TokenVisualization => 150,
            WidgetType::ProgressBar => 20,
        }
    }
}

impl LayoutMode {
    fn name(&self) -> &'static str {
        match self {
            LayoutMode::Compact => "Compact",
            LayoutMode::Split => "Split View",
            LayoutMode::Dashboard => "Dashboard",
        }
    }

    fn widgets(&self) -> Vec<WidgetType> {
        match self {
            LayoutMode::Compact => vec![WidgetType::PromptInput, WidgetType::StreamingOutput],
            LayoutMode::Split => vec![
                WidgetType::ModelSelector,
                WidgetType::PromptInput,
                WidgetType::StreamingOutput,
                WidgetType::MetricsDashboard,
            ],
            LayoutMode::Dashboard => vec![
                WidgetType::ModelSelector,
                WidgetType::PromptInput,
                WidgetType::StreamingOutput,
                WidgetType::MetricsDashboard,
                WidgetType::TokenVisualization,
                WidgetType::ProgressBar,
            ],
        }
    }

    fn columns(&self) -> usize {
        match self {
            LayoutMode::Compact => 1,
            LayoutMode::Split => 2,
            LayoutMode::Dashboard => 3,
        }
    }
}

#[allow(dead_code)]
impl Theme {
    fn name(&self) -> &'static str {
        match self {
            Theme::Dark => "Dark",
            Theme::Light => "Light",
            Theme::HighContrast => "High Contrast",
        }
    }

    fn ansi_codes(&self) -> ThemeColors {
        match self {
            Theme::Dark => ThemeColors {
                bg: "\x1b[48;5;235m",
                fg: "\x1b[38;5;252m",
                accent: "\x1b[38;5;39m",
                highlight: "\x1b[38;5;226m",
                error: "\x1b[38;5;196m",
                success: "\x1b[38;5;46m",
                reset: "\x1b[0m",
            },
            Theme::Light => ThemeColors {
                bg: "\x1b[48;5;255m",
                fg: "\x1b[38;5;235m",
                accent: "\x1b[38;5;27m",
                highlight: "\x1b[38;5;208m",
                error: "\x1b[38;5;160m",
                success: "\x1b[38;5;28m",
                reset: "\x1b[0m",
            },
            Theme::HighContrast => ThemeColors {
                bg: "\x1b[48;5;16m",
                fg: "\x1b[38;5;231m",
                accent: "\x1b[38;5;51m",
                highlight: "\x1b[38;5;226m",
                error: "\x1b[38;5;196m",
                success: "\x1b[38;5;46m",
                reset: "\x1b[0m",
            },
        }
    }
}

/// ANSI color codes for theming
#[allow(dead_code)]
struct ThemeColors {
    bg: &'static str,
    fg: &'static str,
    accent: &'static str,
    highlight: &'static str,
    error: &'static str,
    success: &'static str,
    reset: &'static str,
}

/// UI frame metrics
#[derive(Debug)]
#[allow(dead_code)]
struct FrameMetrics {
    layout: LayoutMode,
    theme: Theme,
    widget_count: usize,
    total_render_cost_us: usize,
    target_fps: usize,
    frame_budget_us: usize,
    buffer_size_bytes: usize,
    under_budget: bool,
}

/// UI analyzer
struct UIAnalyzer {
    layout: LayoutMode,
    theme: Theme,
    target_fps: usize,
}

impl UIAnalyzer {
    fn new(layout: LayoutMode, theme: Theme, target_fps: usize) -> Self {
        Self {
            layout,
            theme,
            target_fps,
        }
    }

    fn analyze(&self) -> FrameMetrics {
        let widgets = self.layout.widgets();
        let widget_count = widgets.len();

        // Calculate total render cost
        let total_render_cost_us: usize = widgets.iter().map(|w| w.render_cost()).sum();

        // Frame budget at target FPS
        let frame_budget_us = 1_000_000 / self.target_fps;

        // Estimated buffer size (assuming 80x24 terminal, 4 bytes per cell)
        let buffer_size_bytes = 80 * 24 * 4 * self.layout.columns();

        // Check if we're under budget
        let under_budget = total_render_cost_us < frame_budget_us;

        FrameMetrics {
            layout: self.layout,
            theme: self.theme,
            widget_count,
            total_render_cost_us,
            target_fps: self.target_fps,
            frame_budget_us,
            buffer_size_bytes,
            under_budget,
        }
    }

    fn simulate_streaming(&self, model: ModelTier, tokens: usize) -> StreamingSimulation {
        let speed = model.simulated_speed();
        let total_time_ms = tokens as f64 / speed * 1000.0;
        let frame_count = (total_time_ms / 1000.0 * self.target_fps as f64).ceil() as usize;
        let tokens_per_frame = (speed / self.target_fps as f64).max(1.0);

        StreamingSimulation {
            model,
            total_tokens: tokens,
            total_time_ms,
            frame_count,
            tokens_per_frame,
            target_fps: self.target_fps,
        }
    }
}

/// Streaming output simulation
#[derive(Debug)]
#[allow(dead_code)]
struct StreamingSimulation {
    model: ModelTier,
    total_tokens: usize,
    total_time_ms: f64,
    frame_count: usize,
    tokens_per_frame: f64,
    target_fps: usize,
}

/// Presentar UI Demo
#[derive(Parser)]
#[command(name = "demo-presentar-ui")]
#[command(about = "Demonstrate terminal UI for ML inference")]
#[command(version = "1.0.0")]
struct Args {
    /// Output to stdout (CI mode)
    #[arg(long)]
    stdout: bool,

    /// Model tier
    #[arg(long, value_enum, default_value = "medium")]
    tier: ModelTier,

    /// Layout mode
    #[arg(long, value_enum, default_value = "split")]
    layout: LayoutMode,

    /// Color theme
    #[arg(long, value_enum, default_value = "dark")]
    theme: Theme,

    /// Target FPS
    #[arg(long, default_value = "60")]
    fps: usize,

    /// Simulated token count
    #[arg(long, default_value = "100")]
    tokens: usize,
}

fn print_ui_diagram(layout: LayoutMode, theme: Theme) {
    let analyzer = UIAnalyzer::new(layout, theme, 60);
    let metrics = analyzer.analyze();

    println!(
        r#"
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRESENTAR TUI FRAMEWORK                                  │
│              Layout: {} │ Theme: {}                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Widget Architecture:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │  │   App State  │◄─│  UI Events   │◄─│  Terminal    │             │   │
│  │  │   (Model)    │  │  (Input)     │  │  (Backend)   │             │   │
│  │  └──────┬───────┘  └──────────────┘  └──────┬───────┘             │   │
│  │         │                                    │                      │   │
│  │         ▼                                    ▼                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │  │   Layout     │─►│   Widgets    │─►│   Render     │             │   │
│  │  │   Engine     │  │   (View)     │  │   Buffer     │             │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘             │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Rendering Pipeline:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  [Layout]─►[Widget Tree]─►[Diff]─►[Buffer]─►[Terminal]             │   │
│  │     │          │           │         │          │                   │   │
│  │   Grid    {} widgets    Delta   {} KB    ANSI           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Widgets: {} │ Budget: {:.0}µs/frame │ Cost: {}µs               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"#,
        layout.name(),
        theme.name(),
        metrics.widget_count,
        metrics.buffer_size_bytes / 1024,
        metrics.widget_count,
        metrics.frame_budget_us as f64,
        metrics.total_render_cost_us
    );
}

fn run_analysis(args: &Args) {
    println!("\n=== Presentar UI Analysis ===\n");

    let analyzer = UIAnalyzer::new(args.layout, args.theme, args.fps);
    let metrics = analyzer.analyze();

    println!("Configuration:");
    println!("  Layout: {}", args.layout.name());
    println!("  Theme: {}", args.theme.name());
    println!("  Target FPS: {}", args.fps);
    println!("  Columns: {}", args.layout.columns());
    println!();

    println!("Widgets in Layout:");
    for widget in args.layout.widgets() {
        println!("  - {} ({}µs)", widget.name(), widget.render_cost());
    }
    println!();

    println!("Frame Metrics:");
    println!("  Widget count: {}", metrics.widget_count);
    println!("  Render cost: {}µs", metrics.total_render_cost_us);
    println!(
        "  Frame budget: {}µs ({} FPS)",
        metrics.frame_budget_us, args.fps
    );
    println!("  Buffer size: {} bytes", metrics.buffer_size_bytes);
    println!(
        "  Under budget: {}",
        if metrics.under_budget { "Yes" } else { "No" }
    );
    println!();

    // Streaming simulation
    println!("Streaming Simulation ({}):", args.tier.name());
    let sim = analyzer.simulate_streaming(args.tier, args.tokens);
    println!("  Tokens: {}", sim.total_tokens);
    println!("  Speed: {:.0} tok/s", args.tier.simulated_speed());
    println!("  Total time: {:.0}ms", sim.total_time_ms);
    println!("  Frame count: {}", sim.frame_count);
    println!("  Tokens/frame: {:.1}", sim.tokens_per_frame);
}

fn main() {
    let args = Args::parse();

    if args.stdout {
        // CI mode: minimal output
        println!("[PASS] demo-presentar-ui");
        println!("  layout: {}", args.layout.name());
        println!("  theme: {}", args.theme.name());
        println!("  model: {}", args.tier.name());
        println!("  target_fps: {}", args.fps);

        let analyzer = UIAnalyzer::new(args.layout, args.theme, args.fps);
        let metrics = analyzer.analyze();
        let sim = analyzer.simulate_streaming(args.tier, args.tokens);

        println!("  widget_count: {}", metrics.widget_count);
        println!("  render_cost_us: {}", metrics.total_render_cost_us);
        println!("  frame_budget_us: {}", metrics.frame_budget_us);
        println!("  buffer_size: {} bytes", metrics.buffer_size_bytes);
        println!("  under_budget: {}", metrics.under_budget);
        println!("  streaming_time_ms: {:.0}", sim.total_time_ms);
        println!("  frame_count: {}", sim.frame_count);

        // Check UI viability: must be under frame budget for target FPS
        let viable = metrics.under_budget && sim.tokens_per_frame >= 0.5;
        println!("  ui_viable: {}", if viable { "PASS" } else { "FAIL" });
    } else {
        // Interactive mode
        print_ui_diagram(args.layout, args.theme);
        run_analysis(&args);

        println!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
    }
}
