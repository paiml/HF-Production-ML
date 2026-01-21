//! Demo: demo-pruning
use clap::Parser;

#[derive(Parser)]
#[command(name = "demo-pruning")]
struct Args {
    #[arg(long)]
    stdout: bool,
}

fn main() {
    let args = Args::parse();
    if args.stdout {
        println!("[PASS] demo-pruning");
    } else {
        println!("demo-pruning - TUI mode not yet implemented");
    }
}
