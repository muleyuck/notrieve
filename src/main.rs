use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use notrieve::app::{self, Event, IndexRequest, Paths, PullOutcome};
use notrieve::embed::download::DownloadEvent;
use notrieve::indexer::IndexReport;
use notrieve::mcp::{self, NotrieveServer};
use notrieve::search::{DEFAULT_K, MAX_K, SearchQuery, SearchResponse};
use notrieve::session::Session;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::prelude::*;

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
        /// Stop after writing the settings: no model download and no index
        #[arg(long)]
        no_index: bool,
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
    /// Search the vault (changed notes are indexed first, within search_budget_ms)
    Search {
        query: String,
        /// Registered vault to use (default: the vault containing the current directory, then the default vault)
        #[arg(long)]
        vault: Option<String>,
        /// Number of results (1-50)
        #[arg(short, default_value_t = DEFAULT_K as u8, value_parser = clap::value_parser!(u8).range(1..=MAX_K as i64))]
        k: u8,
        /// Only notes with this tag or one of its child tags; repeat to require several
        #[arg(long = "tag")]
        tags: Vec<String>,
        /// Only notes whose vault-relative path starts with this
        #[arg(long)]
        path: Option<String>,
        /// Print the results as JSON
        #[arg(long)]
        json: bool,
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
    /// Serve one vault over MCP (stdio) to AI agents such as Claude Code and Codex
    Mcp {
        /// Registered vault to use (default: the vault containing the current directory, then the default vault)
        #[arg(long)]
        vault: Option<String>,
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
    /// Register a directory that already has .notrieve/, or approve its changed config again
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
    let user_filter =
        EnvFilter::try_from_env("NOTRIEVE_LOG").unwrap_or_else(|_| EnvFilter::new("warn"));
    // ureq prints some request header values at debug level and ureq-proto the whole request at
    // trace level, so an API key can reach either. A directive cannot cap them (a longer target in
    // NOTRIEVE_LOG wins), so a second filter keeps only their info and above whatever the first
    // allows.
    let cap_ureq = filter_fn(|meta| {
        !(meta.target().starts_with("ureq") && *meta.level() > tracing::Level::INFO)
    });
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_filter(user_filter)
                .with_filter(cap_ureq),
        )
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
        Command::Init {
            name,
            default,
            no_index,
        } => {
            let vault = app::init(&cwd, name.as_deref(), default, &paths)?;
            println!(
                "initialized vault `{}` at {}",
                vault.name.as_deref().unwrap_or_default(),
                vault.root.display()
            );
            if no_index {
                println!(
                    "edit {} and then run `notrieve index`",
                    vault.config_dir().join("config.toml").display()
                );
            } else {
                let request = IndexRequest {
                    rebuild: false,
                    wait: true,
                    budget: None,
                };
                print_report(&app::index(&vault, &paths, &request, &mut print_event)?);
            }
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
        Command::Search {
            query,
            vault,
            k,
            tags,
            path,
            json,
        } => {
            let vault = app::open_vault(vault.as_deref(), &cwd, &paths)?;
            let query = SearchQuery {
                text: query,
                k: k as usize,
                tags,
                path_prefix: path,
            };
            let response = app::search(&vault, &paths, &query, &mut |event| {
                if !matches!(event, Event::SendingToRemote(_)) {
                    print_event(event)
                }
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&response)?);
            } else {
                print_results(&response);
                // `--json` carries both in the JSON (remote spec §5.2).
                for notice in search_notices(&response) {
                    eprintln!("{notice}");
                }
            }
            // On stderr in both formats, so stdout stays the results alone (spec §5.3).
            if let Some(notice) = stale_notice(&response) {
                eprintln!("{notice}");
            }
        }
        Command::Mcp { vault } => {
            // stdout carries the protocol; progress and errors go to stderr (spec §6).
            let server = match app::open_vault(vault.as_deref(), &cwd, &paths)
                .and_then(|vault| Session::start(vault, &paths, &mut print_event))
            {
                Ok(session) => NotrieveServer::new(Arc::new(session)),
                Err(e) => {
                    eprintln!("error: {e}");
                    NotrieveServer::unavailable(e.to_string())
                }
            };
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("cannot start the async runtime")?;
            let served = runtime.block_on(mcp::serve_stdio(server));
            // Dropping a runtime waits for blocking work; a request still running when stdin
            // closed must not keep the process alive.
            runtime.shutdown_background();
            served.map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        Command::Model {
            command: ModelCommand::Pull { vault, force },
        } => {
            let vault = app::open_vault(vault.as_deref(), &cwd, &paths)?;
            match app::pull_model(&vault, &paths, force, &mut print_event)? {
                PullOutcome::Downloaded(dir) => println!("model files are in {}", dir.display()),
                PullOutcome::Remote { host } => {
                    println!("model is remote ({host}); nothing to download")
                }
            }
        }
        Command::Vault { command } => match command {
            VaultCommand::Add {
                path,
                name,
                default,
            } => {
                let (name, settings) = app::add_vault(&path, name.as_deref(), default, &paths)?;
                println!("registered vault `{name}`");
                for setting in settings {
                    println!("  its config {setting}");
                }
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
        Event::SendingToRemote(host) => eprintln!("sending note text to {host} for embedding"),
        Event::CorruptDbMoved(path) => {
            eprintln!("moved the unreadable database to {}", path.display())
        }
    }
}

fn stopped_notice(report: &IndexReport) -> Option<String> {
    report.interrupted.as_ref().map(|reason| {
        format!(
            "stopped: {reason}; {} file(s) left for the next run",
            report.remaining
        )
    })
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
    if let Some(notice) = stopped_notice(report) {
        eprintln!("{notice}");
    } else if report.remaining > 0 {
        println!("{} file(s) left for the next run", report.remaining);
    }
    if !report.scan_complete {
        println!("warning: some directories could not be read; deletions were not applied");
    }
}

const BODY_PREVIEW_LINES: usize = 3;

fn print_results(response: &SearchResponse) {
    if response.results.is_empty() {
        println!("no results");
    }
    for (i, hit) in response.results.iter().enumerate() {
        let place = if hit.heading_path.is_empty() {
            hit.title.clone()
        } else {
            format!("{} > {}", hit.title, hit.heading_path)
        };
        println!(
            "{}. {}:{}-{}  {place}",
            i + 1,
            hit.path,
            hit.start_line,
            hit.end_line
        );
        for line in hit.body.lines().take(BODY_PREVIEW_LINES) {
            println!("   {line}");
        }
    }
}

fn search_notices(response: &SearchResponse) -> Vec<String> {
    let mut notices = Vec::new();
    if let Some(error) = &response.indexer_error {
        notices.push(format!("the index was not updated: {error}"));
    }
    if let Some(reason) = &response.vector_error {
        notices.push(format!(
            "semantic search is unavailable ({reason}); showing full-text matches only"
        ));
    }
    notices
}

fn stale_notice(response: &SearchResponse) -> Option<String> {
    (response.stale_files > 0).then(|| {
        format!(
            "{} file(s) changed since the last index ({}); run `notrieve index` to bring it up to date",
            response.stale_files,
            response.index_state.label()
        )
    })
}

#[cfg(test)]
mod search_output_tests {
    use super::*;
    use notrieve::search::{IndexState, VectorSearch};

    #[test]
    fn stopped_notice_names_the_reason_and_the_files_left() {
        let report = IndexReport {
            remaining: 4,
            interrupted: Some("HTTP 503 from https://api.example.com/v1/embeddings".into()),
            ..IndexReport::default()
        };
        assert_eq!(
            stopped_notice(&report).as_deref(),
            Some(
                "stopped: HTTP 503 from https://api.example.com/v1/embeddings; 4 file(s) left for the next run"
            )
        );
        assert_eq!(stopped_notice(&IndexReport::default()), None);
    }

    fn response(stale_files: usize, index_state: IndexState) -> SearchResponse {
        SearchResponse {
            results: Vec::new(),
            stale_files,
            index_state,
            indexer_error: None,
            vector_search: VectorSearch::Ok,
            vector_error: None,
        }
    }

    #[test]
    fn a_failed_update_is_noticed() {
        let response = SearchResponse {
            indexer_error: Some("embedding API error: HTTP 401".into()),
            ..response(0, IndexState::Fresh)
        };
        assert_eq!(
            search_notices(&response),
            ["the index was not updated: embedding API error: HTTP 401"]
        );
    }

    #[test]
    fn a_missing_vector_side_is_noticed() {
        let response = SearchResponse {
            vector_search: VectorSearch::Unavailable,
            vector_error: Some("HTTP 503".into()),
            ..response(0, IndexState::Fresh)
        };
        assert_eq!(
            search_notices(&response),
            ["semantic search is unavailable (HTTP 503); showing full-text matches only"]
        );
    }

    #[test]
    fn a_complete_search_has_no_notice() {
        assert!(search_notices(&response(0, IndexState::Fresh)).is_empty());
    }

    #[test]
    fn stale_notice_names_the_count_the_state_and_the_index_command() {
        let notice = stale_notice(&response(3, IndexState::IdleStale)).unwrap();
        assert!(notice.starts_with("3 file(s) changed"), "{notice}");
        assert!(notice.contains("idle_stale"), "{notice}");
        assert!(notice.contains("notrieve index"), "{notice}");
    }

    #[test]
    fn nothing_stale_gives_no_notice() {
        assert_eq!(
            stale_notice(&response(0, IndexState::UpdatingElsewhere)),
            None
        );
    }
}
