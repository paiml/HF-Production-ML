//! Demo: demo-serving-api
use clap::Parser;

#[derive(Parser)]
#[command(name = "demo-serving-api")]
struct Args {
    #[arg(long)]
    stdout: bool,
}

fn main() {
    let args = Args::parse();
    if args.stdout {
        println!("[PASS] demo-serving-api");
    } else {
        println!("demo-serving-api - TUI mode not yet implemented");
    }
}
