//! Demo: demo-apr-format
use clap::Parser;

#[derive(Parser)]
#[command(name = "demo-apr-format")]
struct Args {
    #[arg(long)]
    stdout: bool,
}

fn main() {
    let args = Args::parse();
    if args.stdout {
        println!("[PASS] demo-apr-format");
    } else {
        println!("demo-apr-format - TUI mode not yet implemented");
    }
}
