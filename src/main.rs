use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use notrieve::app::{self, Event, IndexRequest, Paths};
use notrieve::embed::download::DownloadEvent;
use notrieve::indexer::IndexReport;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "notrieve",
    version,
    about = "Local hybrid search over Markdown notes"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create .notrieve/ here, register the vault, download the model and build the index
    Init {
        /// Name in the vault registry (default: the directory name)
        #[arg(long)]
        name: Option<String>,
        /// Make this the default vault (the first registered vault always is)
        #[arg(long)]
        default: bool,
    },
    /// Update the index
    Index {
        /// Registered vault to use (default: the vault containing the current directory, then the default vault)
        #[arg(long)]
        vault: Option<String>,
        /// Drop and rebuild the index (also recovers an unreadable database)
        #[arg(long)]
        rebuild: bool,
        /// Exit with code 3 instead of waiting when another process holds the lock
        #[arg(long)]
        no_wait: bool,
    },
    /// Manage the embedding model
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Manage registered vaults
    Vault {
        #[command(subcommand)]
        command: VaultCommand,
    },
}

#[derive(Subcommand)]
enum ModelCommand {
    /// Download the model files into the cache
    Pull {
        /// Registered vault to use (default: the vault containing the current directory, then the default vault)
        #[arg(long)]
        vault: Option<String>,
        /// Download again even when the cached files are complete
        #[arg(long)]
        force: bool,
    },
}

#[derive(Subcommand)]
enum VaultCommand {
    /// Register a directory that already has .notrieve/
    Add {
        path: PathBuf,
        /// Name in the vault registry (default: the directory name)
        #[arg(long)]
        name: Option<String>,
        /// Make this the default vault (the first registered vault always is)
        #[arg(long)]
        default: bool,
    },
    /// List registered vaults
    List,
    /// Unregister a vault (files are kept)
    Remove { name: String },
}

fn main() -> ExitCode {
    let filter = EnvFilter::try_from_env("NOTRIEVE_LOG").unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .init();
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            // Exit code 2 is reserved for "vault not identified"; clap would use it for usage errors.
            let _ = e.print();
            return if e.use_stderr() {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            };
        }
    };
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::from(exit_code(&err))
        }
    }
}

fn exit_code(err: &anyhow::Error) -> u8 {
    use notrieve::Error;
    match err.downcast_ref::<Error>() {
        Some(
            Error::VaultNotFound
            | Error::VaultNotRegistered { .. }
            | Error::VaultPathMissing { .. },
        ) => 2,
        Some(Error::LockBusy) => 3,
        _ => 1,
    }
}

fn run(cli: Cli) -> Result<()> {
    let paths = Paths::from_env()?;
    let cwd = std::env::current_dir().context("cannot read the current directory")?;
    match cli.command {
        Command::Init { name, default } => {
            let vault = app::init(&cwd, name.as_deref(), default, &paths)?;
            println!(
                "initialized vault `{}` at {}",
                vault.name.as_deref().unwrap_or_default(),
                vault.root.display()
            );
            let request = IndexRequest {
                rebuild: false,
                wait: true,
                budget: None,
            };
            print_report(&app::index(&vault, &paths, &request, &mut print_event)?);
        }
        Command::Index {
            vault,
            rebuild,
            no_wait,
        } => {
            let vault = app::open_vault(vault.as_deref(), &cwd, &paths)?;
            let request = IndexRequest {
                rebuild,
                wait: !no_wait,
                budget: None,
            };
            print_report(&app::index(&vault, &paths, &request, &mut print_event)?);
        }
        Command::Model {
            command: ModelCommand::Pull { vault, force },
        } => {
            let vault = app::open_vault(vault.as_deref(), &cwd, &paths)?;
            let dir = app::pull_model(&vault, &paths, force, &mut print_event)?;
            println!("model files are in {}", dir.display());
        }
        Command::Vault { command } => match command {
            VaultCommand::Add {
                path,
                name,
                default,
            } => {
                let name = app::add_vault(&path, name.as_deref(), default, &paths)?;
                println!("registered vault `{name}`");
            }
            VaultCommand::List => {
                let registry = app::list_vaults(&paths)?;
                if registry.vaults.is_empty() {
                    println!("no vaults registered");
                }
                for (name, root) in &registry.vaults {
                    let mark = if registry.default.as_deref() == Some(name.as_str()) {
                        "\tdefault"
                    } else {
                        ""
                    };
                    println!("{name}\t{}{mark}", root.display());
                }
            }
            VaultCommand::Remove { name } => {
                let root = app::remove_vault(&name, &paths)?;
                println!(
                    "unregistered vault `{name}` ({}); files were kept",
                    root.display()
                );
            }
        },
    }
    Ok(())
}

fn print_event(event: Event<'_>) {
    match event {
        Event::WaitingForLock => {
            eprintln!("waiting for another notrieve process to finish updating the index...")
        }
        Event::Download(DownloadEvent::Started {
            file,
            total: Some(bytes),
        }) => {
            eprintln!("downloading {file} ({:.1} MB)", bytes as f64 / 1_000_000.0)
        }
        Event::Download(DownloadEvent::Started { file, total: None }) => {
            eprintln!("downloading {file}")
        }
        Event::Download(DownloadEvent::Finished { .. }) => {}
        Event::Rebuilding(reason) => eprintln!("{reason}"),
        Event::Indexing(p) => eprintln!("[{}/{}] {}", p.done + 1, p.total, p.path),
        Event::CorruptDbMoved(path) => {
            eprintln!("moved the unreadable database to {}", path.display())
        }
    }
}

fn print_report(report: &IndexReport) {
    println!(
        "indexed {}, timestamp-only {}, renamed {}, deleted {}, skipped {}",
        report.indexed,
        report.touched,
        report.renamed,
        report.deleted,
        report.skipped.len()
    );
    for skipped in &report.skipped {
        println!("skipped {}: {}", skipped.path, skipped.reason);
    }
    if report.remaining > 0 {
        println!("{} file(s) left for the next run", report.remaining);
    }
    if !report.scan_complete {
        println!("warning: some directories could not be read; deletions were not applied");
    }
}
