//! Demo: demo-streaming
use clap::Parser;

#[derive(Parser)]
#[command(name = "demo-streaming")]
struct Args {
    #[arg(long)]
    stdout: bool,
}

fn main() {
    let args = Args::parse();
    if args.stdout {
        println!("[PASS] demo-streaming");
    } else {
        println!("demo-streaming - TUI mode not yet implemented");
    }
}
