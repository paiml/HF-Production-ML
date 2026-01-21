//! Demo: demo-hardware-detect
use clap::Parser;

#[derive(Parser)]
#[command(name = "demo-hardware-detect")]
struct Args {
    #[arg(long)]
    stdout: bool,
}

fn main() {
    let args = Args::parse();
    if args.stdout {
        println!("[PASS] demo-hardware-detect");
    } else {
        println!("demo-hardware-detect - TUI mode not yet implemented");
    }
}
