//! Demo: demo-model-registry
use clap::Parser;

#[derive(Parser)]
#[command(name = "demo-model-registry")]
struct Args {
    #[arg(long)]
    stdout: bool,
}

fn main() {
    let args = Args::parse();
    if args.stdout {
        println!("[PASS] demo-model-registry");
    } else {
        println!("demo-model-registry - TUI mode not yet implemented");
    }
}
