//! Helpers shared by the background-update and MCP tests. Each test crate uses a subset.
#![allow(dead_code)]
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use notrieve::cache::{EmbeddingCache, cache_path};
use notrieve::config::Vault;
use notrieve::embed::fake::FakeEmbedder;
use notrieve::embed::{Embedder, lock_unpoisoned};
use notrieve::fingerprint::Fingerprints;
use notrieve::indexer::{IndexOptions, index_vault};
use notrieve::lock;
use notrieve::store::Store;
use serde_json::{Value, json};
use tempfile::TempDir;

/// Fingerprints that differ per `tag`, as two processes with different settings would have.
pub fn fps(tag: &str) -> Fingerprints {
    Fingerprints {
        model: format!("model-{tag}"),
        index: format!("index-{tag}"),
    }
}

/// An empty vault at `<tmp>/notes` with the default config.
pub fn vault() -> (TempDir, Vault) {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("notes");
    fs::create_dir_all(root.join(".notrieve")).unwrap();
    fs::write(root.join(".notrieve/config.toml"), "").unwrap();
    let vault = Vault::load(&root, None).unwrap();
    (tmp, vault)
}

pub fn write(vault: &Vault, rel: &str, text: &str) {
    let path = vault.root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, text).unwrap();
}

/// Polls `probe` every 20 ms until it returns `Some`, for at most 10 s.
pub fn wait_for<T>(mut probe: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(value) = probe() {
            return value;
        }
        assert!(Instant::now() < deadline, "timed out waiting");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Indexed paths, sorted.
pub fn indexed_paths(vault: &Vault) -> Vec<String> {
    let mut paths: Vec<String> = Store::open(&vault.db_path())
        .unwrap()
        .file_states()
        .unwrap()
        .into_keys()
        .collect();
    paths.sort();
    paths
}

/// Indexes the vault as another process with fingerprints `fps` would (rebuilding on a mismatch).
/// Waits for the write lock, so a background run still in progress finishes first.
pub fn index_as_another_process(vault: &Vault, fps: &Fingerprints) {
    let _lock = lock::lock_blocking(&vault.lock_path()).unwrap();
    index_with(vault, fps);
}

/// Indexes the vault with `fps`; the caller holds the write lock.
pub fn index_with(vault: &Vault, fps: &Fingerprints) {
    let db = vault.db_path();
    let mut store = Store::open(&db).unwrap();
    let mut cache = EmbeddingCache::open(&cache_path(&db)).unwrap();
    let opts = IndexOptions {
        exclude: &[],
        chunk_max_tokens: 450,
        budget: None,
        rebuild: false,
    };
    index_vault(
        &vault.root,
        &mut store,
        &mut cache,
        &FakeEmbedder::new(16),
        fps,
        &opts,
        &mut |_| {},
    )
    .unwrap();
}

/// `FakeEmbedder` whose `embed_documents` blocks while the gate is closed.
pub struct GateEmbedder {
    inner: FakeEmbedder,
    open: Mutex<bool>,
    changed: Condvar,
    entered: AtomicUsize,
}

impl GateEmbedder {
    pub fn new(open: bool) -> Self {
        Self {
            inner: FakeEmbedder::new(16),
            open: Mutex::new(open),
            changed: Condvar::new(),
            entered: AtomicUsize::new(0),
        }
    }

    pub fn set_open(&self, open: bool) {
        *self.open.lock().unwrap() = open;
        self.changed.notify_all();
    }

    /// Calls of `embed_documents` so far, including one blocked at the gate.
    pub fn entered(&self) -> usize {
        self.entered.load(Ordering::SeqCst)
    }
}

impl Embedder for GateEmbedder {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn embed_documents(&self, texts: &[String]) -> notrieve::Result<Vec<Vec<f32>>> {
        self.entered.fetch_add(1, Ordering::SeqCst);
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.changed.wait(open).unwrap();
        }
        drop(open);
        self.inner.embed_documents(texts)
    }

    fn embed_query(&self, text: &str) -> notrieve::Result<Vec<f32>> {
        self.inner.embed_query(text)
    }

    fn count_tokens(&self, text: &str) -> notrieve::Result<usize> {
        self.inner.count_tokens(text)
    }
}

/// Like `OnnxEmbedder`, every call holds one model lock; a text containing `PANIC` panics while
/// the lock is held, poisoning it.
pub struct PanicEmbedder {
    inner: FakeEmbedder,
    model: Mutex<()>,
}

impl PanicEmbedder {
    pub fn new() -> Self {
        Self {
            inner: FakeEmbedder::new(16),
            model: Mutex::new(()),
        }
    }
}

impl Embedder for PanicEmbedder {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn embed_documents(&self, texts: &[String]) -> notrieve::Result<Vec<Vec<f32>>> {
        let _model = lock_unpoisoned(&self.model);
        if texts.iter().any(|t| t.contains("PANIC")) {
            panic!("test embedder hit PANIC");
        }
        self.inner.embed_documents(texts)
    }

    fn embed_query(&self, text: &str) -> notrieve::Result<Vec<f32>> {
        let _model = lock_unpoisoned(&self.model);
        if text.contains("PANIC") {
            panic!("test embedder hit PANIC in a query");
        }
        self.inner.embed_query(text)
    }

    fn count_tokens(&self, text: &str) -> notrieve::Result<usize> {
        self.inner.count_tokens(text)
    }
}

/// Drives `notrieve mcp` over stdio: `initialize`, then one `search` call, then closes stdin.
/// Returns every stdout line up to EOF parsed as JSON (a non-JSON line fails the test) and the exit
/// status. The process is killed, failing the test, if it gives no answer within 60 s or has not
/// closed stdout and exited 10 s after stdin was closed.
pub fn mcp_search_session(mut command: Command, query: &str) -> (Vec<Value>, ExitStatus) {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut stdin = Some(child.stdin.take().unwrap());
    let requests = [
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}}}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "search", "arguments": {"query": query}}}),
    ];
    let writer = stdin.as_mut().unwrap();
    for request in &requests {
        writeln!(writer, "{request}").unwrap();
    }
    writer.flush().unwrap();
    // A reader thread lets the deadlines below hold even while the server keeps stdout open.
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut deadline = Instant::now() + Duration::from_secs(60);
    let mut lines = Vec::new();
    loop {
        match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(line) => {
                let line = line.unwrap();
                let value: Value = serde_json::from_str(&line)
                    .unwrap_or_else(|e| panic!("stdout line is not JSON ({e}): {line}"));
                if value["id"] == 2 && stdin.is_some() {
                    stdin = None;
                    deadline = Instant::now() + Duration::from_secs(10);
                }
                lines.push(value);
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    panic!(
                        "notrieve mcp did not answer and exit in time; stdout so far: {lines:?}"
                    );
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {
                let _ = child.kill();
                panic!("notrieve mcp did not answer and exit in time; stdout so far: {lines:?}");
            }
        }
    }
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            panic!("notrieve mcp closed stdout but did not exit");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    assert!(
        stdin.is_none(),
        "stdout closed before the search answer: {lines:?}"
    );
    (lines, status)
}
