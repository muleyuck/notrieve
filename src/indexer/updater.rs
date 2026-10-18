//! Background index updates for a long-running process (MCP; spec §4.6, §5.4).
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::cache::{EmbeddingCache, cache_path};
use crate::config::Vault;
use crate::embed::Embedder;
use crate::fingerprint::Fingerprints;
use crate::indexer::{self, IndexEvent, IndexOptions, IndexReport};
use crate::lock;
use crate::store::{RebuildReason, Store};
use crate::{Error, Result};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpdaterStatus {
    /// A run is in progress in this process (it holds or is taking the write lock).
    pub running: bool,
    /// The current run rebuilds the index from scratch.
    pub rebuilding: bool,
    /// Rebuilds started so far; a reader compares it before and after a search (see `Session`).
    pub rebuild_generation: u64,
    /// Files left in the current run, once the first file has started.
    pub remaining: Option<usize>,
    /// The index is known to match this process's fingerprints: at startup, or after a run.
    pub synced: bool,
    /// The index settings changed after `synced`; no further runs (spec §4.6).
    pub stopped: bool,
    /// The last run found the write lock held by another process.
    pub lock_busy: bool,
    /// Files the last completed run could not index (spec §6).
    pub skipped_files: usize,
    pub last_error: Option<String>,
}

/// The status and whether another process holds the write lock, read together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    pub status: UpdaterStatus,
    pub locked_elsewhere: bool,
}

#[derive(Default)]
struct State {
    status: UpdaterStatus,
    pending: bool,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

pub struct Updater {
    shared: Arc<Shared>,
    lock_path: PathBuf,
}

impl Updater {
    /// Starts the thread; it waits for `request` before its first run. `synced` says the index
    /// already matches `fps`, so a mismatch found later needs a restart instead of a rebuild.
    pub fn start(
        vault: Vault,
        embedder: Arc<dyn Embedder>,
        fps: Fingerprints,
        synced: bool,
    ) -> Result<Self> {
        let mut state = State::default();
        state.status.synced = synced;
        let shared = Arc::new(Shared {
            state: Mutex::new(state),
            wake: Condvar::new(),
        });
        let lock_path = vault.lock_path();
        let thread_shared = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("notrieve-indexer".into())
            .spawn(move || run_loop(&thread_shared, &vault, embedder.as_ref(), &fps))
            .map_err(|e| {
                Error::IndexerFailed(format!("cannot start the background indexer: {e}"))
            })?;
        Ok(Self { shared, lock_path })
    }

    /// Asks for a run. Requests made while a run is in progress coalesce into one more run.
    pub fn request(&self) {
        self.shared.lock().pending = true;
        self.shared.wake.notify_one();
    }

    /// Records that a search found the index matching this process's settings, so a later
    /// mismatch is another process's change and needs a restart.
    pub fn mark_synced(&self) {
        self.shared.lock().status.synced = true;
    }

    pub fn status(&self) -> UpdaterStatus {
        self.shared.lock().status.clone()
    }

    /// Probes the write lock while the state is locked. A run marks itself `running` under the same
    /// lock before taking the write lock, so a held lock seen here belongs to another process.
    pub fn observe(&self) -> Result<Observation> {
        let state = self.shared.lock();
        let locked_elsewhere = !state.status.running && lock::try_lock(&self.lock_path)?.is_none();
        Ok(Observation {
            status: state.status.clone(),
            locked_elsewhere,
        })
    }
}

impl Drop for Updater {
    /// The thread ends after its current run. It is not joined: a run may take minutes, and a
    /// process that exits mid-run leaves at most the file being written uncommitted.
    fn drop(&mut self) {
        self.shared.lock().shutdown = true;
        self.shared.wake.notify_one();
    }
}

fn run_loop(shared: &Shared, vault: &Vault, embedder: &dyn Embedder, fps: &Fingerprints) {
    lower_priority();
    loop {
        let synced = {
            let mut state = shared.lock();
            while !state.pending && !state.shutdown {
                state = shared.wake.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            if state.shutdown || state.status.stopped {
                return;
            }
            state.pending = false;
            state.status.running = true;
            state.status.synced
        };
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            run_once(shared, vault, embedder, fps, synced)
        }));
        let mut state = shared.lock();
        let status = &mut state.status;
        status.running = false;
        status.rebuilding = false;
        status.remaining = None;
        status.lock_busy = false;
        match outcome {
            Ok(Ok(Some(report))) => {
                status.synced = true;
                status.skipped_files = report.skipped.len();
                status.last_error = None;
            }
            Ok(Ok(None)) => status.lock_busy = true,
            Ok(Err(Error::FingerprintChanged)) => {
                status.stopped = true;
                status.last_error = Some(format!(
                    "the background indexer stopped: {}",
                    Error::FingerprintChanged
                ));
            }
            Ok(Err(e)) => status.last_error = Some(format!("the background indexer failed: {e}")),
            Err(panic) => {
                status.last_error = Some(format!(
                    "the background indexer panicked: {}",
                    panic_message(panic.as_ref())
                ))
            }
        }
    }
}

/// `Ok(None)` when another process holds the write lock.
fn run_once(
    shared: &Shared,
    vault: &Vault,
    embedder: &dyn Embedder,
    fps: &Fingerprints,
    synced: bool,
) -> Result<Option<IndexReport>> {
    let Some(_lock) = lock::try_lock(&vault.lock_path())? else {
        return Ok(None);
    };
    let db = vault.db_path();
    // After the sync, a newer schema means another process rebuilt the index with a newer notrieve:
    // stop like any other settings change. Before it, normal operation refuses the index (spec §4.1).
    let mut store = match Store::open(&db) {
        Err(Error::SchemaTooNew { .. }) if synced => return Err(Error::FingerprintChanged),
        opened => opened?,
    };
    // Once the index matched, settings changed by another process must not be overwritten; that
    // includes the cache, which `EmbeddingCache::open` may recreate (spec §4.6). The lock is held,
    // so the check cannot go stale.
    if synced && changed_elsewhere(store.rebuild_reason(fps)?) {
        return Err(Error::FingerprintChanged);
    }
    let mut cache = EmbeddingCache::open(&cache_path(&db))?;
    let opts = IndexOptions {
        exclude: &vault.config.exclude,
        chunk_max_tokens: vault.config.chunk_max_tokens(),
        budget: None,
        deadline: None,
        rebuild: false,
    };
    let report = indexer::index_vault(
        &vault.root,
        &mut store,
        &mut cache,
        embedder,
        fps,
        &opts,
        &mut |event| {
            let mut state = shared.lock();
            match event {
                IndexEvent::Rebuilding(_) => {
                    state.status.rebuilding = true;
                    state.status.rebuild_generation += 1;
                }
                IndexEvent::File(p) => state.status.remaining = Some(p.total - p.done),
                // The MCP process has no terminal to show it on.
                IndexEvent::SendingToRemote { .. } => {}
            }
        },
    )?;
    // Spec §6: skipped files are reported. The MCP process has no terminal, so they go to the log.
    for skipped in &report.skipped {
        tracing::warn!(path = %skipped.path, "skipped: {}", skipped.reason);
    }
    if !report.skipped.is_empty() {
        tracing::warn!("{} file(s) could not be indexed", report.skipped.len());
    }
    Ok(Some(report))
}

/// Whether `reason` means another process rebuilt the index with other settings (spec §4.6). A
/// missing index (`NewIndex`) or a replaced cache is this process's to rebuild, even after a sync.
pub fn changed_elsewhere(reason: Option<RebuildReason>) -> bool {
    matches!(
        reason,
        Some(
            RebuildReason::ModelChanged
                | RebuildReason::SchemaChanged
                | RebuildReason::SettingsChanged
        )
    )
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Spec §5.4: the update thread runs at a lower OS priority. A failure keeps the default priority.
fn lower_priority() {
    #[cfg(target_os = "macos")]
    // SAFETY: sets the calling thread's QoS class; no pointers are passed.
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0);
    }
    #[cfg(target_os = "linux")]
    // SAFETY: on Linux the nice value is per thread, so this lowers only the calling thread.
    unsafe {
        libc::setpriority(libc::PRIO_PROCESS, libc::gettid() as libc::id_t, 10);
    }
}
