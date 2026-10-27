//! Background index updates for a long-running process (MCP; spec §4.6, §5.4).
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

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
    /// A remote API error that needs a settings change (auth, protocol): no further runs until
    /// the process restarts (remote spec §5.3).
    pub halted: bool,
    /// The wait before the next run, after the remote API was unavailable; `None` after a run
    /// that finished.
    pub retry_delay: Option<Duration>,
    pub last_error: Option<String>,
}

/// The waits after runs stopped by an unavailable remote API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Backoff {
    pub first: Duration,
    pub max: Duration,
}

impl Backoff {
    /// Remote spec §5.3: 30 s, doubling up to 10 min (guesses, not measured).
    pub const DEFAULT: Backoff = Backoff {
        first: Duration::from_secs(30),
        max: Duration::from_secs(600),
    };

    fn next(&self, current: Option<Duration>) -> Duration {
        current.map_or(self.first, |delay| (delay * 2).min(self.max))
    }
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
    /// No run starts before this; requests made meanwhile wait for it.
    retry_at: Option<Instant>,
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
    vault: Vault,
    /// Where the index was at startup; see `Vault::ensure_index_location`.
    location: PathBuf,
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
        Self::start_with_backoff(vault, embedder, fps, synced, Backoff::DEFAULT)
    }

    /// `start` with other waits after an unavailable remote API.
    pub fn start_with_backoff(
        vault: Vault,
        embedder: Arc<dyn Embedder>,
        fps: Fingerprints,
        synced: bool,
        backoff: Backoff,
    ) -> Result<Self> {
        let mut state = State::default();
        state.status.synced = synced;
        let shared = Arc::new(Shared {
            state: Mutex::new(state),
            wake: Condvar::new(),
        });
        let location = vault.index_location();
        let (thread_vault, thread_location) = (vault.clone(), location.clone());
        let thread_shared = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("notrieve-indexer".into())
            .spawn(move || {
                run_loop(
                    &thread_shared,
                    &thread_vault,
                    &thread_location,
                    embedder.as_ref(),
                    &fps,
                    backoff,
                )
            })
            .map_err(|e| {
                Error::IndexerFailed(format!("cannot start the background indexer: {e}"))
            })?;
        Ok(Self {
            shared,
            vault,
            location,
        })
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
        // The probe creates the lock file next to the index.
        self.vault.ensure_index_location(&self.location)?;
        let state = self.shared.lock();
        let locked_elsewhere =
            !state.status.running && lock::try_lock(&self.vault.lock_path())?.is_none();
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

/// `location` is where the index was when the updater started (`Vault::ensure_index_location`).
fn run_loop(
    shared: &Shared,
    vault: &Vault,
    location: &Path,
    embedder: &dyn Embedder,
    fps: &Fingerprints,
    backoff: Backoff,
) {
    lower_priority();
    while let Some(synced) = wait_for_run(shared) {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            run_once(shared, vault, location, embedder, fps, synced)
        }));
        record_outcome(&mut shared.lock(), backoff, outcome);
    }
}

/// Waits until a run is due and marks it started. Returns whether the index was synced before
/// it, or `None` once the updater shuts down, stops or halts.
fn wait_for_run(shared: &Shared) -> Option<bool> {
    let mut state = shared.lock();
    loop {
        if state.shutdown || state.status.stopped || state.status.halted {
            return None;
        }
        // Remote spec §5.3: during the wait, requests only coalesce; one run follows it.
        if let Some(at) = state.retry_at {
            let now = Instant::now();
            if now < at {
                state = shared
                    .wake
                    .wait_timeout(state, at - now)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
                continue;
            }
            state.retry_at = None;
            state.pending = true;
        }
        if state.pending {
            break;
        }
        state = shared.wake.wait(state).unwrap_or_else(|e| e.into_inner());
    }
    state.pending = false;
    state.status.running = true;
    Some(state.status.synced)
}

/// Records how a run ended in the status the session reads.
fn record_outcome(
    state: &mut State,
    backoff: Backoff,
    outcome: std::thread::Result<Result<Option<IndexReport>>>,
) {
    let State {
        status, retry_at, ..
    } = state;
    status.running = false;
    status.rebuilding = false;
    status.remaining = None;
    status.lock_busy = false;
    match outcome {
        Ok(Ok(Some(report))) => {
            status.synced = true;
            status.skipped_files = report.skipped.len();
            match report.interrupted {
                Some(reason) => {
                    let delay = backoff.next(status.retry_delay);
                    status.retry_delay = Some(delay);
                    *retry_at = Some(Instant::now() + delay);
                    status.last_error = Some(format!(
                        "the background indexer paused for {delay:?}: {reason}"
                    ));
                }
                None => {
                    status.retry_delay = None;
                    status.last_error = None;
                }
            }
        }
        Ok(Ok(None)) => status.lock_busy = true,
        Ok(Err(Error::FingerprintChanged)) => {
            status.stopped = true;
            status.last_error = Some(format!(
                "the background indexer stopped: {}",
                Error::FingerprintChanged
            ));
        }
        // Only auth and protocol errors end a run: transient ones become `interrupted`.
        Ok(Err(e @ Error::Remote { .. })) => {
            status.halted = true;
            status.last_error = Some(format!(
                "the background indexer stopped: {e}; fix the [model] settings and restart the MCP server"
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

/// `Ok(None)` when another process holds the write lock.
fn run_once(
    shared: &Shared,
    vault: &Vault,
    location: &Path,
    embedder: &dyn Embedder,
    fps: &Fingerprints,
    synced: bool,
) -> Result<Option<IndexReport>> {
    // Before the lock, which creates its file next to the index.
    vault.ensure_index_location(location)?;
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
    let opts = IndexOptions::for_vault(&vault.config);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_next_delay_doubles_up_to_the_maximum() {
        let backoff = Backoff {
            first: Duration::from_millis(100),
            max: Duration::from_millis(500),
        };
        let ms = Duration::from_millis;
        assert_eq!(backoff.next(None), ms(100));
        assert_eq!(backoff.next(Some(ms(100))), ms(200));
        assert_eq!(backoff.next(Some(ms(200))), ms(400));
        assert_eq!(backoff.next(Some(ms(400))), ms(500));
    }
}
