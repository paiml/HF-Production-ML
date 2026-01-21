//! Demo: demo-presentar-ui
use clap::Parser;

#[derive(Parser)]
#[command(name = "demo-presentar-ui")]
struct Args {
    #[arg(long)]
    stdout: bool,
}

fn main() {
    let args = Args::parse();
    if args.stdout {
        println!("[PASS] demo-presentar-ui");
    } else {
        println!("demo-presentar-ui - TUI mode not yet implemented");
    }
}
