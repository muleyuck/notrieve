//! MCP server over stdio (spec §4.6, §5.4). It calls only the `session` API; stdout carries the
//! protocol and nothing else.
use std::sync::Arc;

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

#[derive(Clone)]
pub struct NotrieveServer {
    /// `Err` holds why the session could not start; every tool then returns it (spec §6).
    session: std::result::Result<Arc<Session>, String>,
}

#[tool_router]
impl NotrieveServer {
    pub fn new(session: Arc<Session>) -> Self {
        Self {
            session: Ok(session),
        }
    }

    pub fn unavailable(reason: String) -> Self {
        Self {
            session: Err(reason),
        }
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
        let session = match &self.session {
            Ok(session) => Arc::clone(session),
            Err(reason) => return tool_error(reason.clone()),
        };
        match tokio::task::spawn_blocking(move || work(&session)).await {
            Ok(Ok(value)) => match serde_json::to_string_pretty(&value) {
                Ok(json) => CallToolResult::success(vec![ContentBlock::text(json)]),
                Err(e) => tool_error(format!("internal error: {e}")),
            },
            Ok(Err(e)) => tool_error(error_text(&e)),
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
