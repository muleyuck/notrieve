use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::cache::{EmbeddingCache, cache_path};
use crate::config::{self, ModelContract, Registry, Vault};
use crate::embed::download::{DownloadEvent, ensure_model, with_model};
use crate::embed::onnx::resolve_threads;
use crate::embed::{Embedder, OnnxEmbedder, RemoteEmbedder};
use crate::fingerprint::{Fingerprints, fingerprints, remote_fingerprints};
use crate::indexer::scan::{count_stale, scan_vault};
use crate::indexer::{self, IndexEvent, IndexOptions, IndexReport, Progress};
use crate::lock;
use crate::search::{self, IndexState, SearchQuery, SearchResponse};
use crate::store::{RebuildReason, Store};
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    pub registry: PathBuf,
    pub model_cache: PathBuf,
}

impl Paths {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            registry: config::registry_path()?,
            model_cache: config::model_cache_dir()?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Event<'a> {
    WaitingForLock,
    Download(DownloadEvent<'a>),
    Rebuilding(RebuildReason),
    Indexing(Progress<'a>),
    CorruptDbMoved(&'a Path),
    /// Note text is about to go to an embedding API on this host (remote spec §5.1).
    SendingToRemote(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexRequest {
    pub rebuild: bool,
    pub wait: bool,
    pub budget: Option<Duration>,
}

pub fn open_vault(explicit: Option<&str>, cwd: &Path, paths: &Paths) -> Result<Vault> {
    let registry = Registry::load(&paths.registry)?;
    config::resolve_vault(explicit, cwd, &registry)
}

pub fn init(cwd: &Path, name: Option<&str>, make_default: bool, paths: &Paths) -> Result<Vault> {
    let root = cwd.canonicalize().map_err(|e| Error::io(cwd, e))?;
    let name = vault_name(name, &root)?;
    Registry::update(&paths.registry, |registry| {
        if registry.vaults.contains_key(&name) {
            return Err(Error::DuplicateVaultName(name.clone()));
        }
        config::init_vault(&root)?;
        registry.add(&name, &root, make_default)
    })?;
    Vault::load(&root, Some(name))
}

/// Registers the vault, or approves its current config again when `name` already names it, and
/// returns the name and the settings now trusted (`TrustedSettings`) for the user to review.
pub fn add_vault(
    path: &Path,
    name: Option<&str>,
    make_default: bool,
    paths: &Paths,
) -> Result<(String, Vec<String>)> {
    let root = path.canonicalize().map_err(|e| Error::io(path, e))?;
    let settings = Vault::load(&root, None)?.trusted_settings();
    let name = vault_name(name, &root)?;
    Registry::update(&paths.registry, |registry| {
        let registered = registry
            .vaults
            .get(&name)
            .and_then(|p| p.canonicalize().ok());
        if registered.as_deref() == Some(root.as_path()) {
            if make_default {
                registry.default = Some(name.clone());
            }
        } else {
            registry.add(&name, &root, make_default)?;
        }
        registry.trust(&name, settings.clone());
        Ok(())
    })?;
    Ok((name, settings.describe()))
}

pub fn remove_vault(name: &str, paths: &Paths) -> Result<PathBuf> {
    Registry::update(&paths.registry, |registry| registry.remove(name))
}

pub fn list_vaults(paths: &Paths) -> Result<Registry> {
    Registry::load(&paths.registry)
}

/// What `notrieve model pull` did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullOutcome {
    Downloaded(PathBuf),
    /// A remote model has nothing to download.
    Remote {
        host: String,
    },
}

pub fn pull_model(
    vault: &Vault,
    paths: &Paths,
    force: bool,
    on_event: &mut dyn FnMut(Event<'_>),
) -> Result<PullOutcome> {
    match vault.config.model.resolve(&vault.root)? {
        ModelContract::Local(contract) => {
            ensure_model(&contract, &paths.model_cache, force, &mut |e| {
                on_event(Event::Download(e))
            })
            .map(PullOutcome::Downloaded)
        }
        ModelContract::Remote(contract) => Ok(PullOutcome::Remote {
            host: contract.host,
        }),
    }
}

pub fn index(
    vault: &Vault,
    paths: &Paths,
    request: &IndexRequest,
    on_event: &mut dyn FnMut(Event<'_>),
) -> Result<IndexReport> {
    let _lock = acquire(vault, request.wait, on_event)?;
    let (embedder, fps) = load_model(vault, paths, on_event)?;
    let db = vault.db_path();
    let cache_file = cache_path(&db);
    let (mut store, mut cache) = if request.rebuild {
        let (store, moved) = Store::open_for_rebuild(&db)?;
        if let Some(moved) = &moved {
            on_event(Event::CorruptDbMoved(moved));
        }
        (store, EmbeddingCache::open_for_rebuild(&cache_file)?)
    } else {
        (Store::open(&db)?, EmbeddingCache::open(&cache_file)?)
    };
    let opts = IndexOptions {
        budget: request.budget,
        rebuild: request.rebuild,
        ..IndexOptions::for_vault(&vault.config)
    };
    indexer::index_vault(
        &vault.root,
        &mut store,
        &mut cache,
        embedder.as_ref(),
        &fps,
        &opts,
        &mut |e| forward(e, on_event),
    )
}

/// `notrieve search` (spec §5.3). The query is checked before the model is loaded.
pub fn search(
    vault: &Vault,
    paths: &Paths,
    query: &SearchQuery,
    on_event: &mut dyn FnMut(Event<'_>),
) -> Result<SearchResponse> {
    query.validate()?;
    let (embedder, fps) = load_model(vault, paths, on_event)?;
    search_vault(vault, embedder.as_ref(), &fps, query, on_event)
}

/// Remote spec §5.2: the end of `search_budget_ms` is kept for embedding the query (a guess at
/// one round trip with a TLS handshake; not measured).
pub const QUERY_RESERVE: Duration = Duration::from_millis(1000);

/// Updates within `search_budget_ms` when no other process holds the lock, then light-scans and
/// searches (spec §5.3). A remote model gets deadlines instead (remote spec §5.2).
pub fn search_vault<E: Embedder + ?Sized>(
    vault: &Vault,
    embedder: &E,
    fps: &Fingerprints,
    query: &SearchQuery,
    on_event: &mut dyn FnMut(Event<'_>),
) -> Result<SearchResponse> {
    query.validate()?;
    let started = Instant::now();
    let budget = Duration::from_millis(vault.config.search_budget_ms);
    let remote = embedder.is_remote();
    // A remote update ends `QUERY_RESERVE` early; with no more budget than that, it is skipped.
    let update_deadline = budget
        .checked_sub(QUERY_RESERVE)
        .filter(|left| !left.is_zero())
        .map(|left| started + left);
    let mut indexer_error = None;
    let db = vault.db_path();
    if (!remote || update_deadline.is_some())
        && let Some(_lock) = lock::try_lock(&vault.lock_path())?
    {
        let mut store = Store::open(&db)?;
        let mut cache = EmbeddingCache::open(&cache_path(&db))?;
        let opts = IndexOptions {
            budget: (!remote).then_some(budget),
            deadline: if remote { update_deadline } else { None },
            ..IndexOptions::for_vault(&vault.config)
        };
        match indexer::index_vault(
            &vault.root,
            &mut store,
            &mut cache,
            embedder,
            fps,
            &opts,
            &mut |e| forward(e, on_event),
        ) {
            Ok(_) => {}
            // An API setting that is wrong (auth, protocol) still leaves the index searchable.
            Err(e @ Error::Remote { .. }) => indexer_error = Some(e.to_string()),
            Err(e) => return Err(e),
        }
    }
    let reader = Store::open_reader(&db)?;
    let scan = scan_vault(&vault.root, &vault.config.exclude)?;
    let stale_files = count_stale(&scan, &reader)?;
    // Our own lock was released above, so a held lock belongs to another process (spec §4.5).
    let index_state = IndexState::classify(
        scan.complete,
        false,
        || Ok(lock::try_lock(&vault.lock_path())?.is_none()),
        stale_files,
    )?;
    let query_deadline = remote.then(|| (started + budget).max(Instant::now() + QUERY_RESERVE));
    let found = search::search(&reader, embedder, &fps.index, query, query_deadline)?;
    Ok(SearchResponse::new(
        found,
        stale_files,
        index_state,
        indexer_error,
    ))
}

/// Loads the vault's model and computes its fingerprints. A local model is read while the model
/// cache lock is held; a remote one downloads nothing (its API key is read per request).
pub fn load_model(
    vault: &Vault,
    paths: &Paths,
    on_event: &mut dyn FnMut(Event<'_>),
) -> Result<(Box<dyn Embedder>, Fingerprints)> {
    let max = vault.config.chunk_max_tokens();
    let contract = match vault.config.model.resolve(&vault.root)? {
        ModelContract::Remote(contract) => {
            let fps = remote_fingerprints(&contract, max);
            return Ok((Box::new(RemoteEmbedder::new(contract)), fps));
        }
        ModelContract::Local(contract) => contract,
    };
    with_model(
        &contract,
        &paths.model_cache,
        false,
        &mut |e| on_event(Event::Download(e)),
        |dir| {
            let embedder =
                OnnxEmbedder::load(&contract, dir, resolve_threads(vault.config.index_threads))?;
            // Spec §4.3; a remote model has no token limit to check against (remote spec §2).
            check_chunk_limit(max, embedder.effective_max_tokens())?;
            let fps = fingerprints(&contract, dir, max)?;
            Ok((Box::new(embedder) as Box<dyn Embedder>, fps))
        },
    )
}

fn forward(event: IndexEvent<'_>, on_event: &mut dyn FnMut(Event<'_>)) {
    on_event(match event {
        IndexEvent::Rebuilding(reason) => Event::Rebuilding(reason),
        IndexEvent::File(progress) => Event::Indexing(progress),
        IndexEvent::SendingToRemote { host } => Event::SendingToRemote(host),
    });
}

/// Spec §4.3: `chunk_max_tokens` above the model's effective limit is a configuration error.
pub fn check_chunk_limit(chunk_max_tokens: usize, effective_max_tokens: usize) -> Result<()> {
    if chunk_max_tokens > effective_max_tokens {
        return Err(Error::Config(format!(
            "chunk_max_tokens ({chunk_max_tokens}) exceeds the model limit ({effective_max_tokens})"
        )));
    }
    Ok(())
}

fn acquire(
    vault: &Vault,
    wait: bool,
    on_event: &mut dyn FnMut(Event<'_>),
) -> Result<lock::WriteLock> {
    let path = vault.lock_path();
    if let Some(lock) = lock::try_lock(&path)? {
        return Ok(lock);
    }
    if !wait {
        return Err(Error::LockBusy);
    }
    on_event(Event::WaitingForLock);
    lock::lock_blocking(&path)
}

/// `name`, or the directory name when none is given.
fn vault_name(name: Option<&str>, root: &Path) -> Result<String> {
    match name {
        Some(name) => Ok(name.to_string()),
        None => dir_name(root),
    }
}

fn dir_name(root: &Path) -> Result<String> {
    root.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .ok_or_else(|| {
            Error::Config("cannot derive a vault name from the path; pass --name".into())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_limit_equal_to_the_model_limit_is_accepted() {
        assert!(check_chunk_limit(512, 512).is_ok());
    }

    #[test]
    fn chunk_limit_above_the_model_limit_is_a_config_error() {
        let err = check_chunk_limit(513, 512).unwrap_err();
        assert!(
            matches!(&err, Error::Config(m) if m.contains("513") && m.contains("512")),
            "{err}"
        );
    }
}
