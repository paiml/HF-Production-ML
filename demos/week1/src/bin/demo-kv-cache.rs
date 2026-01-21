//! Demo: demo-kv-cache
use clap::Parser;

#[derive(Parser)]
#[command(name = "demo-kv-cache")]
struct Args {
    #[arg(long)]
    stdout: bool,
}

fn main() {
    let args = Args::parse();
    if args.stdout {
        println!("[PASS] demo-kv-cache");
    } else {
        println!("demo-kv-cache - TUI mode not yet implemented");
    }
}
