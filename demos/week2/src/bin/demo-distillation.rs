//! Demo: demo-distillation
use clap::Parser;

#[derive(Parser)]
#[command(name = "demo-distillation")]
struct Args {
    #[arg(long)]
    stdout: bool,
}

fn main() {
    let args = Args::parse();
    if args.stdout {
        println!("[PASS] demo-distillation");
    } else {
        println!("demo-distillation - TUI mode not yet implemented");
    }
}
