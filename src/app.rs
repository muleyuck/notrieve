use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::config::{self, Registry, Vault};
use crate::embed::OnnxEmbedder;
use crate::embed::download::{DownloadEvent, ensure_model, with_model};
use crate::embed::onnx::resolve_threads;
use crate::fingerprint::index_fingerprint;
use crate::indexer::{self, IndexOptions, IndexReport, Progress};
use crate::lock;
use crate::store::Store;
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
    Indexing(Progress<'a>),
    CorruptDbMoved(&'a Path),
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
    let name = match name {
        Some(name) => name.to_string(),
        None => dir_name(&root)?,
    };
    Registry::update(&paths.registry, |registry| {
        if registry.vaults.contains_key(&name) {
            return Err(Error::DuplicateVaultName(name.clone()));
        }
        config::init_vault(&root)?;
        registry.add(&name, &root, make_default)
    })?;
    Vault::load(&root, Some(name))
}

pub fn add_vault(
    path: &Path,
    name: Option<&str>,
    make_default: bool,
    paths: &Paths,
) -> Result<String> {
    let root = path.canonicalize().map_err(|e| Error::io(path, e))?;
    Vault::load(&root, None)?;
    let name = match name {
        Some(name) => name.to_string(),
        None => dir_name(&root)?,
    };
    Registry::update(&paths.registry, |registry| {
        registry.add(&name, &root, make_default)
    })?;
    Ok(name)
}

pub fn remove_vault(name: &str, paths: &Paths) -> Result<PathBuf> {
    Registry::update(&paths.registry, |registry| registry.remove(name))
}

pub fn list_vaults(paths: &Paths) -> Result<Registry> {
    Registry::load(&paths.registry)
}

pub fn pull_model(
    vault: &Vault,
    paths: &Paths,
    force: bool,
    on_event: &mut dyn FnMut(Event<'_>),
) -> Result<PathBuf> {
    let contract = vault.config.model.resolve(&vault.root)?;
    ensure_model(&contract, &paths.model_cache, force, &mut |e| {
        on_event(Event::Download(e))
    })
}

pub fn index(
    vault: &Vault,
    paths: &Paths,
    request: &IndexRequest,
    on_event: &mut dyn FnMut(Event<'_>),
) -> Result<IndexReport> {
    let _lock = acquire(vault, request.wait, on_event)?;
    let contract = vault.config.model.resolve(&vault.root)?;
    let max = vault.config.chunk_max_tokens;
    let (embedder, fingerprint) = with_model(
        &contract,
        &paths.model_cache,
        false,
        &mut |e| on_event(Event::Download(e)),
        |dir| {
            let embedder =
                OnnxEmbedder::load(&contract, dir, resolve_threads(vault.config.index_threads))?;
            check_chunk_limit(max, embedder.effective_max_tokens())?;
            let fingerprint = index_fingerprint(&contract, dir, max)?;
            Ok((embedder, fingerprint))
        },
    )?;
    let db = vault.db_path();
    let mut store = if request.rebuild {
        let (mut store, moved) = Store::open_for_rebuild(&db)?;
        if let Some(moved) = &moved {
            on_event(Event::CorruptDbMoved(moved));
        }
        store.rebuild(&fingerprint, &contract.id, contract.dim)?;
        store
    } else {
        Store::open(&db)?
    };
    let opts = IndexOptions {
        exclude: &vault.config.exclude,
        chunk_max_tokens: max,
        budget: request.budget,
    };
    let mut report = indexer::index_vault(
        &vault.root,
        &mut store,
        &embedder,
        &fingerprint,
        &opts,
        &mut |p| on_event(Event::Indexing(*p)),
    )?;
    report.rebuilt |= request.rebuild;
    Ok(report)
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
