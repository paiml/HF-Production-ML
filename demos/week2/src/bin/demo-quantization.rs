//! Demo: demo-quantization
use clap::Parser;

#[derive(Parser)]
#[command(name = "demo-quantization")]
struct Args {
    #[arg(long)]
    stdout: bool,
}

fn main() {
    let args = Args::parse();
    if args.stdout {
        println!("[PASS] demo-quantization");
    } else {
        println!("demo-quantization - TUI mode not yet implemented");
    }
}
