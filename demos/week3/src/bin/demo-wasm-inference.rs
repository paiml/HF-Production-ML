//! Demo: demo-wasm-inference
use clap::Parser;

#[derive(Parser)]
#[command(name = "demo-wasm-inference")]
struct Args {
    #[arg(long)]
    stdout: bool,
}

fn main() {
    let args = Args::parse();
    if args.stdout {
        println!("[PASS] demo-wasm-inference");
    } else {
        println!("demo-wasm-inference - TUI mode not yet implemented");
    }
}
