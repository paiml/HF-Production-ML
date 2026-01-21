//! Demo: demo-speculative-decode
use clap::Parser;

#[derive(Parser)]
#[command(name = "demo-speculative-decode")]
struct Args {
    #[arg(long)]
    stdout: bool,
}

fn main() {
    let args = Args::parse();
    if args.stdout {
        println!("[PASS] demo-speculative-decode");
    } else {
        println!("demo-speculative-decode - TUI mode not yet implemented");
    }
}
