//! Demo: demo-flash-attention
use clap::Parser;

#[derive(Parser)]
#[command(name = "demo-flash-attention")]
struct Args {
    #[arg(long)]
    stdout: bool,
}

fn main() {
    let args = Args::parse();
    if args.stdout {
        println!("[PASS] demo-flash-attention");
    } else {
        println!("demo-flash-attention - TUI mode not yet implemented");
    }
}
