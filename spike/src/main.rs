mod bench;
mod embed_check;
mod sqlite_check;
mod yaml_check;

use std::process::ExitCode;

pub type CheckResult = anyhow::Result<bool>;

const USAGE: &str = "usage: notrieve-spike <embed-check|bench|sqlite-check|yaml-check> ...";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("embed-check") => embed_check::run(&args[1..]),
        Some("bench") => bench::run(&args[1..]),
        Some("sqlite-check") => sqlite_check::run(&args[1..]),
        Some("yaml-check") => yaml_check::run(&args[1..]),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
