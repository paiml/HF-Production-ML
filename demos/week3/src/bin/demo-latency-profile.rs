//! Demo: demo-latency-profile
use clap::Parser;

#[derive(Parser)]
#[command(name = "demo-latency-profile")]
struct Args {
    #[arg(long)]
    stdout: bool,
}

fn main() {
    let args = Args::parse();
    if args.stdout {
        println!("[PASS] demo-latency-profile");
    } else {
        println!("demo-latency-profile - TUI mode not yet implemented");
    }
}
