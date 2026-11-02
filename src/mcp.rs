//! MCP server over stdio (spec §4.6, §5.4). It calls only the `session` API; stdout carries the
//! protocol and nothing else.
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::search::{DEFAULT_K, SearchQuery};
use crate::session::Session;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    /// What to look for, in natural language or keywords.
    pub query: String,
    /// Number of results, 1 to 50 (default 8).
    pub k: Option<usize>,
    /// Only notes that have all of these tags; a tag also matches its child tags, ignoring case.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Only notes whose vault-relative path starts with this.
    pub path_prefix: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetNoteParams {
    /// Vault-relative path, as returned by `search`.
    pub path: String,
    /// First line to return, 1-based (default 1).
    pub start_line: Option<usize>,
    /// Last line to return, inclusive (default: the end of the note).
    pub end_line: Option<usize>,
}

type Started = std::result::Result<Arc<Session>, String>;

/// How long `notrieve mcp` lets a tool call wait for the session to start. Loading a downloaded
/// model takes about a second; a download takes longer and answers "still starting" (a guess,
/// below the per-call timeouts of MCP clients; not measured).
pub const START_WAIT: Duration = Duration::from_secs(30);

/// Clones share the session.
#[derive(Clone)]
pub struct NotrieveServer {
    start: Arc<Start>,
}

struct Start {
    /// Unset while the session starts. `Err` holds why it could not start; every tool then
    /// returns it (spec §6).
    session: Mutex<Option<Started>>,
    ready: Condvar,
    wait: Duration,
}

impl Start {
    /// The session, once started within `wait`.
    fn wait(&self) -> Started {
        let guard = self.session.lock().unwrap_or_else(|e| e.into_inner());
        let (guard, _) = self
            .ready
            .wait_timeout_while(guard, self.wait, |session| session.is_none())
            .unwrap_or_else(|e| e.into_inner());
        guard.clone().unwrap_or_else(|| {
            Err(
                "notrieve is still starting (loading the embedding model); try again shortly"
                    .to_string(),
            )
        })
    }
}

#[tool_router]
impl NotrieveServer {
    pub fn new(session: Arc<Session>) -> Self {
        Self::with(Ok(session))
    }

    pub fn unavailable(reason: String) -> Self {
        Self::with(Err(reason))
    }

    /// A server that can answer the MCP handshake before `started` is called. Loading the model
    /// can mean a download, longer than a client waits for the handshake. A tool call waits up to
    /// `wait` for the session.
    pub fn starting(wait: Duration) -> Self {
        Self {
            start: Arc::new(Start {
                session: Mutex::new(None),
                ready: Condvar::new(),
                wait,
            }),
        }
    }

    /// Ends the start of a server made by `starting`; later calls are ignored.
    pub fn started(&self, session: Started) {
        let mut current = self.start.session.lock().unwrap_or_else(|e| e.into_inner());
        if current.is_none() {
            *current = Some(session);
            self.start.ready.notify_all();
        }
    }

    fn with(session: Started) -> Self {
        let server = Self::starting(Duration::ZERO);
        server.started(session);
        server
    }

    #[tool(
        description = "Search the user's Markdown notes (full-text and semantic). Returns chunks with path, start_line, end_line, heading_path and body, plus stale_files and index_state."
    )]
    async fn search(&self, Parameters(p): Parameters<SearchParams>) -> CallToolResult {
        let query = SearchQuery {
            text: p.query,
            k: p.k.unwrap_or(DEFAULT_K),
            tags: p.tags,
            path_prefix: p.path_prefix,
        };
        self.run(move |session| session.search(&query)).await
    }

    #[tool(
        description = "Read a note that search can return, whole or a line range, to see the context around a search hit. `path` is a path from the search results."
    )]
    async fn get_note(&self, Parameters(p): Parameters<GetNoteParams>) -> CallToolResult {
        self.run(move |session| session.get_note(&p.path, p.start_line, p.end_line))
            .await
    }
}

impl NotrieveServer {
    /// Runs blocking session work off the async runtime. Failures, including a panic in the work,
    /// become tool errors; the server keeps running (spec §6).
    async fn run<T, F>(&self, work: F) -> CallToolResult
    where
        T: Serialize + Send + 'static,
        F: FnOnce(&Session) -> crate::Result<T> + Send + 'static,
    {
        let start = Arc::clone(&self.start);
        // Waiting for the start blocks too, so it shares the blocking task with the work.
        let outcome =
            tokio::task::spawn_blocking(move || start.wait().map(|session| work(&session)));
        match outcome.await {
            Ok(Ok(Ok(value))) => match serde_json::to_string_pretty(&value) {
                Ok(json) => CallToolResult::success(vec![ContentBlock::text(json)]),
                Err(e) => tool_error(format!("internal error: {e}")),
            },
            Ok(Ok(Err(e))) => tool_error(error_text(&e)),
            Ok(Err(reason)) => tool_error(reason),
            Err(e) => tool_error(format!("internal error: {e}")),
        }
    }
}

fn tool_error(text: String) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(text)])
}

fn error_text(err: &Error) -> String {
    match err {
        Error::FingerprintChanged => "the index settings changed after this MCP server started; \
             restart the MCP server (start a new agent session)"
            .to_string(),
        other => other.to_string(),
    }
}

#[tool_handler(
    name = "notrieve",
    instructions = "These tools search the user's Markdown notes. Answer only from the chunks that `search` returns. Cite each claim as `path:start_line`. When a chunk lacks context, read the lines around it with `get_note`. If `stale_files` is not 0, tell the user that recent edits may be missing from the results."
)]
impl ServerHandler for NotrieveServer {}

/// Serves until stdin closes.
pub async fn serve_stdio(
    server: NotrieveServer,
) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let running = server
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await?;
    running.waiting().await?;
    Ok(())
}
