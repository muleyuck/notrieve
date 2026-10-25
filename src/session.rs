//! A long-running session over one vault: the model stays loaded and a background thread keeps the
//! index current (spec §4.6, §5.4). The MCP server calls only this API.
use std::sync::Arc;

use crate::app::{self, Event, Paths};
use crate::config::Vault;
use crate::embed::Embedder;
use crate::fingerprint::Fingerprints;
use crate::indexer::scan::{plan_changes, scan_vault};
use crate::indexer::updater::{Updater, changed_elsewhere};
use crate::note::{NoteText, read_note};
use crate::search::{self, IndexState, SearchQuery, SearchResponse};
use crate::store::Store;
use crate::{Error, Result};
use rusqlite::OptionalExtension;

pub struct Session {
    vault: Vault,
    /// The vault root resolved at the start; `get_note` reads below it only.
    root: std::path::PathBuf,
    embedder: Arc<dyn Embedder>,
    fps: Fingerprints,
    updater: Updater,
}

impl Session {
    /// Loads the vault's model (downloading it if needed), then starts as `with_embedder`.
    pub fn start(vault: Vault, paths: &Paths, on_event: &mut dyn FnMut(Event<'_>)) -> Result<Self> {
        let (embedder, fps) = app::load_model(&vault, paths, on_event)?;
        Self::with_embedder(vault, Arc::new(embedder), fps)
    }

    /// Starts the background updater and asks for a first run, which rebuilds on a mismatch.
    pub fn with_embedder(
        vault: Vault,
        embedder: Arc<dyn Embedder>,
        fps: Fingerprints,
    ) -> Result<Self> {
        // Spec §4.6: a mismatch at startup is rebuilt. An index that already matches is in sync
        // from the start, so a later mismatch needs a restart even before the updater first runs.
        let db = vault.db_path();
        let synced = db.is_file()
            && matches!(
                Store::open(&db).and_then(|store| store.rebuild_reason(&fps)),
                Ok(None)
            );
        let root = vault
            .root
            .canonicalize()
            .map_err(|e| Error::io(&vault.root, e))?;
        let updater = Updater::start(vault.clone(), Arc::clone(&embedder), fps.clone(), synced)?;
        updater.request();
        Ok(Self {
            vault,
            root,
            embedder,
            fps,
            updater,
        })
    }

    pub fn vault(&self) -> &Vault {
        &self.vault
    }

    /// Spec §5.4: light scan, signal the updater when there are changes, then search (§4.5).
    pub fn search(&self, query: &SearchQuery) -> Result<SearchResponse> {
        query.validate()?;
        let seen = self.updater.observe()?;
        // A stopped updater indexes nothing more, even if the settings later change back.
        if seen.status.stopped {
            return Err(Error::FingerprintChanged);
        }
        // A rebuild rewrites the meta first, so its partial index would pass the fingerprint check.
        if seen.status.running && seen.status.rebuilding {
            return Err(Error::Rebuilding {
                remaining: seen.status.remaining,
            });
        }
        let reader = match search::open_reader(&self.vault.db_path()) {
            Err(Error::SchemaTooNew { .. }) if seen.status.synced => {
                return Err(Error::FingerprintChanged);
            }
            Err(Error::IndexNotReady) => return Err(self.not_ready()?),
            // A cache replaced by an unreadable file fails the ATTACH; the updater discards it.
            Err(Error::Sqlite(_) | Error::DbCorrupt(_)) => return Err(self.not_ready()?),
            other => other?,
        };
        // `index_vault` rebuilds when the schema or the cache is not the one the index was built
        // against (spec §4.1). The search fingerprint does not cover the schema, and a replaced
        // cache leaves no stale file, so the search path has to notice both and ask for that
        // rebuild (`not_ready` requests a run). A cache that cannot be queried is replaced too.
        if reader.rebuild_reason(&self.fps)?.is_some() || !cache_matches(&reader).unwrap_or(false) {
            return Err(self.not_ready()?);
        }
        self.updater.mark_synced();
        let scan = scan_vault(&self.vault.root, &self.vault.config.exclude)?;
        // Spec §6: a listing error is warned about. It may produce no change, so the updater would
        // never run to report it.
        for err in &scan.errors {
            tracing::warn!("{err}");
        }
        let stale_files =
            plan_changes(&scan, &reader.file_states()?, &reader.skipped_states()?).stale_files();
        let index_state = if !scan.complete {
            IndexState::ScanError
        } else if seen.status.running {
            IndexState::Updating
        } else if seen.locked_elsewhere {
            IndexState::UpdatingElsewhere
        } else if stale_files == 0 {
            IndexState::Fresh
        } else {
            IndexState::IdleStale
        };
        // Not gated on `seen`: the run may have ended during the scan, and requests coalesce.
        if stale_files > 0 {
            self.updater.request();
        }
        let results = match search::search(&reader, self.embedder.as_ref(), &self.fps.index, query)
        {
            Err(Error::IndexNotReady) => return Err(self.not_ready()?),
            other => other?,
        };
        // A rebuild that started after `observe` may have committed its meta before the search
        // took its snapshot; the generation shows any rebuild that began in between.
        let after = self.updater.status();
        if after.rebuild_generation != seen.status.rebuild_generation
            || (after.running && after.rebuilding)
        {
            return Err(Error::Rebuilding {
                remaining: after.remaining,
            });
        }
        Ok(SearchResponse {
            results,
            stale_files,
            index_state,
            indexer_error: seen.status.last_error,
        })
    }

    pub fn get_note(
        &self,
        path: &str,
        start_line: Option<usize>,
        end_line: Option<usize>,
    ) -> Result<NoteText> {
        // Only indexed notes: files the index leaves out (`.env`, `.git/*`, excluded or ignored
        // ones) stay unreadable. A rebuild refills the index file by file.
        let status = self.updater.status();
        if status.stopped {
            return Err(Error::FingerprintChanged);
        }
        if status.running && status.rebuilding {
            return Err(Error::Rebuilding {
                remaining: status.remaining,
            });
        }
        let db = self.vault.db_path();
        // `Store::open` would create a missing index.
        if !db.is_file() {
            return Err(self.not_ready()?);
        }
        // The same index `search` accepts: its rows were listed under the current settings.
        let store = match Store::open(&db) {
            Err(Error::SchemaTooNew { .. }) if status.synced => {
                return Err(Error::FingerprintChanged);
            }
            Err(Error::Sqlite(_) | Error::DbCorrupt(_)) => return Err(self.not_ready()?),
            other => other?,
        };
        // One read snapshot: a rebuild committed between the two checks would otherwise let rows
        // the fingerprint check never saw authorize the path.
        let indexed = {
            let _snapshot = store.conn().unchecked_transaction()?;
            match store.rebuild_reason(&self.fps)? {
                Some(_) => None,
                None => Some(store.has_file(path)?),
            }
        };
        match indexed {
            None => return Err(self.not_ready()?),
            Some(false) => return Err(Error::NoteNotIndexed(path.to_string())),
            Some(true) => {}
        }
        read_note(&self.root, path, start_line, end_line)
    }

    /// Why the index cannot be searched (spec §4.6): settings changed by another process after the
    /// sync need a restart; a missing index or cache is rebuilt by the updater.
    fn not_ready(&self) -> Result<Error> {
        let seen = self.updater.observe()?;
        let status = seen.status;
        if status.stopped || (status.synced && self.changed_elsewhere()?) {
            return Ok(Error::FingerprintChanged);
        }
        if status.running {
            return Ok(Error::Rebuilding {
                remaining: status.remaining,
            });
        }
        self.updater.request();
        Ok(if let Some(message) = status.last_error {
            Error::IndexerFailed(message)
        } else if seen.locked_elsewhere {
            Error::LockBusy
        } else {
            Error::Rebuilding { remaining: None }
        })
    }

    fn changed_elsewhere(&self) -> Result<bool> {
        let db = self.vault.db_path();
        if !db.is_file() {
            return Ok(false);
        }
        match Store::open(&db) {
            Ok(store) => Ok(changed_elsewhere(store.rebuild_reason(&self.fps)?)),
            Err(Error::SchemaTooNew { .. }) => Ok(true),
            Err(e) => Err(e),
        }
    }
}

/// Whether the attached cache (`open_reader`) is the one the index was built against.
fn cache_matches(reader: &Store) -> Result<bool> {
    let attached: Option<String> = reader
        .conn()
        .query_row(
            "SELECT value FROM cache.meta WHERE key = 'cache_id'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    Ok(attached.is_some() && attached == reader.cache_id()?)
}
