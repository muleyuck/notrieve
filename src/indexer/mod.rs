pub mod scan;
pub mod updater;

use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::cache::{EmbeddingCache, EmbeddingKey, embedding_key};
use crate::config::{DEFAULT_CHUNK_MAX_TOKENS, VaultConfig};
use crate::embed::Embedder;
use crate::error::RemoteErrorKind;
use crate::fingerprint::Fingerprints;
use crate::markdown::chunk::{MERGE_BELOW_TOKENS, chunk_note};
use crate::markdown::{parse_note, title_from_path};
use crate::store::{ChunkRecord, FileRecord, RebuildReason, Store};
use crate::{Error, Result};
use scan::{Scan, ScanEntry, plan_changes, scan_vault};

/// Remote spec §6: this many files in a row refused by the API means the settings are wrong.
pub const INPUT_STREAK_LIMIT: usize = 5;

pub struct IndexOptions<'a> {
    pub exclude: &'a [String],
    pub chunk_max_tokens: usize,
    /// Start new files only while this much time has not elapsed since the run began; a started
    /// file always finishes. Local models only: a remote run uses `deadline`.
    /// When the budget stops a run, deletions are deferred to a later run.
    /// `IndexReport::remaining` counts unprocessed files plus deferred deletions.
    pub budget: Option<Duration>,
    /// Replaces `budget` when set, and also cuts off the embedding call of the file in progress
    /// (remote spec §5.1); that file is left for a later run.
    pub deadline: Option<Instant>,
    /// Rebuild even when the stored fingerprints match (`index --rebuild`).
    pub rebuild: bool,
}

impl Default for IndexOptions<'_> {
    /// Nothing excluded, the default chunk size, no time limit, no forced rebuild.
    fn default() -> Self {
        Self {
            exclude: &[],
            chunk_max_tokens: DEFAULT_CHUNK_MAX_TOKENS,
            budget: None,
            deadline: None,
            rebuild: false,
        }
    }
}

impl<'a> IndexOptions<'a> {
    /// The vault's `exclude` and chunk size; callers set the rest with struct update syntax.
    pub fn for_vault(config: &'a VaultConfig) -> Self {
        Self {
            exclude: &config.exclude,
            chunk_max_tokens: config.chunk_max_tokens(),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedFile {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct IndexReport {
    pub indexed: usize,
    pub touched: usize,
    pub renamed: usize,
    pub deleted: usize,
    pub remaining: usize,
    pub skipped: Vec<SkippedFile>,
    pub scan_complete: bool,
    pub rebuilt: bool,
    pub rebuild_reason: Option<RebuildReason>,
    /// Why the run stopped with files left: the embedding API was unavailable or out of time.
    pub interrupted: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress<'a> {
    pub done: usize,
    pub total: usize,
    pub path: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexEvent<'a> {
    /// Sent before the index is dropped, so the user sees why the run takes long.
    Rebuilding(RebuildReason),
    File(Progress<'a>),
    /// Sent once per run, right before the first note text goes to an embedding API off this
    /// machine (remote spec §5.1).
    SendingToRemote {
        host: &'a str,
    },
}

/// Remote spec §4: every vector in a remote model's index has the dimension of the first one,
/// read from the index or, without one, taken from the first vector this run sees.
struct DimCheck {
    remote: bool,
    base: Option<usize>,
}

impl DimCheck {
    fn check(&mut self, dim: usize) -> Result<()> {
        if !self.remote {
            return Ok(());
        }
        match self.base {
            None => {
                self.base = Some(dim);
                Ok(())
            }
            Some(base) if base == dim => Ok(()),
            Some(base) => Err(Error::Remote {
                kind: RemoteErrorKind::Protocol,
                message: format!(
                    "the API returned {dim}-dimensional vectors but the index holds {base}-dimensional ones; set a new model.id to rebuild the index with this model"
                ),
            }),
        }
    }
}

pub fn index_vault<E: Embedder + ?Sized>(
    root: &Path,
    store: &mut Store,
    cache: &mut EmbeddingCache,
    embedder: &E,
    fps: &Fingerprints,
    opts: &IndexOptions<'_>,
    on_event: &mut dyn FnMut(IndexEvent<'_>),
) -> Result<IndexReport> {
    let started = Instant::now();
    // A deadline, when given, decides instead of the budget (remote spec §5.1).
    let out_of_time = || match opts.deadline {
        Some(deadline) => Instant::now() >= deadline,
        None => opts
            .budget
            .is_some_and(|budget| started.elapsed() >= budget),
    };
    let mut report = IndexReport::default();
    // Scan first: invalid exclude or ignore rules must fail before anything is dropped.
    let scan = scan_vault(root, opts.exclude)?;
    let mut run = Run {
        store,
        cache,
        embedder,
        fps,
        opts,
        dims: DimCheck {
            remote: embedder.is_remote(),
            base: None,
        },
    };
    run.begin(&mut report, on_event)?;
    for err in &scan.errors {
        tracing::warn!("{err}");
    }
    report.scan_complete = scan.complete;
    if !scan.complete {
        tracing::warn!(
            "the vault could not be scanned completely; deletions and renames are skipped this time"
        );
    }
    let states = run.store.file_states()?;
    let skipped = run.store.skipped_states()?;
    let plan = plan_changes(&scan, &states, &skipped);

    let mut deleted = plan.deleted;
    let total = plan.candidates.len();
    let mut finished = true;
    let mut announced = false;
    let mut input_streak = 0;
    for (done, entry) in plan.candidates.iter().enumerate() {
        // The budget decides whether a new file may start; a started file always finishes.
        if out_of_time() {
            report.remaining = total - done + deleted.len();
            finished = false;
            break;
        }
        on_event(IndexEvent::File(Progress {
            done,
            total,
            path: &entry.rel,
        }));
        let bytes = match std::fs::read(&entry.abs) {
            Ok(bytes) => bytes,
            Err(e) => {
                run.skip(entry, &e)?;
                continue;
            }
        };
        let hash = blake3::hash(&bytes).to_hex().to_string();
        if states
            .get(&entry.rel)
            .is_some_and(|s| s.content_hash == hash)
        {
            run.store
                .touch_file(&fps.index, &entry.rel, entry.mtime_ns, entry.size)?;
            report.touched += 1;
            continue;
        }
        if !states.contains_key(&entry.rel) {
            let title = title_from_path(&entry.rel);
            let source = deleted.iter().position(|old| {
                states
                    .get(old)
                    .is_some_and(|s| s.content_hash == hash && s.title == title)
            });
            if let Some(index) = source {
                let old = deleted.remove(index);
                run.store
                    .rename_file(&fps.index, &old, &entry.rel, entry.mtime_ns, entry.size)?;
                report.renamed += 1;
                continue;
            }
        }
        let outcome = run.index_file(entry, bytes, hash, &mut || {
            if !announced {
                announced = true;
                if let Some(host) = embedder.remote_host() {
                    on_event(IndexEvent::SendingToRemote { host });
                }
            }
        });
        match outcome {
            Ok(requested) => {
                report.indexed += 1;
                if requested {
                    input_streak = 0;
                }
            }
            // Like the budget: the rest waits for a later run, and nothing is recorded as skipped.
            Err(Error::Remote {
                kind: RemoteErrorKind::Transient,
                message,
            }) => {
                report.remaining = total - done + deleted.len();
                report.interrupted = Some(message);
                finished = false;
                break;
            }
            Err(
                e @ Error::Remote {
                    kind: RemoteErrorKind::Input,
                    ..
                },
            ) => {
                tracing::warn!(path = %entry.rel, "skipped: {e}");
                run.skip(entry, &e)?;
                // Only a request draws an `Input` error, so every one counts.
                input_streak += 1;
                if input_streak == INPUT_STREAK_LIMIT {
                    return Err(Error::Remote {
                        kind: RemoteErrorKind::Protocol,
                        message: format!(
                            "the API refused {INPUT_STREAK_LIMIT} files in a row (last: {e}); check the [model] settings"
                        ),
                    });
                }
            }
            Err(e @ (Error::NotUtf8(_) | Error::Embedding(_) | Error::ChunkTooLarge { .. })) => {
                tracing::warn!(path = %entry.rel, "skipped: {e}");
                run.skip(entry, &e)?;
            }
            Err(e) => return Err(e),
        }
    }
    // Deletions wait until every candidate was seen (an unprocessed candidate may be the moved
    // file) and, like files, start only while budget remains.
    if finished {
        if out_of_time() {
            report.remaining += deleted.len();
        } else {
            for path in &deleted {
                run.store.delete_file(&fps.index, path)?;
                report.deleted += 1;
            }
        }
    }
    run.finish(&scan, finished, &mut report)?;
    Ok(report)
}

/// What every step of one `index_vault` run works with.
struct Run<'a, E: ?Sized> {
    store: &'a mut Store,
    cache: &'a mut EmbeddingCache,
    embedder: &'a E,
    fps: &'a Fingerprints,
    opts: &'a IndexOptions<'a>,
    dims: DimCheck,
}

impl<E: Embedder + ?Sized> Run<'_, E> {
    /// Rebuilds the index when it has to be, then reads the dimension remote vectors must keep.
    fn begin(
        &mut self,
        report: &mut IndexReport,
        on_event: &mut dyn FnMut(IndexEvent<'_>),
    ) -> Result<()> {
        let reason = match self.store.rebuild_reason(self.fps)? {
            Some(reason @ (RebuildReason::NewIndex | RebuildReason::ModelChanged)) => Some(reason),
            // Rows written against another cache point to vectors this one does not hold (spec
            // §4.1). The id is persistent, so a run that fails before rebuilding is caught by the
            // next one.
            _ if self.store.cache_id()?.as_deref() != Some(self.cache.id()) => {
                Some(RebuildReason::CacheReset)
            }
            None if self.opts.rebuild => Some(RebuildReason::Requested),
            other => other,
        };
        if let Some(reason) = reason {
            on_event(IndexEvent::Rebuilding(reason));
            self.store
                .rebuild(self.fps, self.embedder.model_id(), self.embedder.dim())?;
            self.store.bind_cache(self.cache.id())?;
            report.rebuilt = true;
            report.rebuild_reason = Some(reason);
        }
        // Read after any rebuild, which drops the chunks the dimension was read from.
        if self.dims.remote
            && let Some(key) = self.store.any_embedding_key()?
        {
            self.dims.base = self.cache.vector_dim(&key)?;
        }
        Ok(())
    }

    fn skip(&mut self, entry: &ScanEntry, reason: &dyn std::fmt::Display) -> Result<()> {
        self.store.skip_file(
            &self.fps.index,
            &entry.rel,
            entry.mtime_ns,
            entry.size,
            &reason.to_string(),
        )
    }

    /// Returns whether texts were sent to the embedder; files served from the cache were not.
    fn index_file(
        &mut self,
        entry: &ScanEntry,
        bytes: Vec<u8>,
        hash: String,
        before_request: &mut dyn FnMut(),
    ) -> Result<bool> {
        let text = String::from_utf8(bytes).map_err(|_| Error::NotUtf8(entry.rel.clone()))?;
        let note = parse_note(&entry.rel, &text);
        for warning in &note.meta.warnings {
            tracing::warn!(path = %entry.rel, "{warning}");
        }
        let chunks = chunk_note(
            note.body,
            note.body_start_line,
            &note.meta.title,
            self.opts.chunk_max_tokens,
            MERGE_BELOW_TOKENS,
            self.embedder,
        )?;
        let inputs: Vec<String> = chunks
            .iter()
            .map(|c| c.embedding_input(&note.meta.title))
            .collect();
        let keys: Vec<EmbeddingKey> = inputs
            .iter()
            .map(|input| embedding_key(&self.fps.model, input))
            .collect();
        let mut missing: Vec<usize> = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            match self.cache.vector_dim(key)? {
                Some(dim) => self.dims.check(dim)?,
                None if !missing.iter().any(|&j| keys[j] == *key) => missing.push(i),
                None => {}
            }
        }
        let requested = !missing.is_empty();
        if requested {
            let texts: Vec<String> = missing.iter().map(|&i| inputs[i].clone()).collect();
            before_request();
            let vectors = self
                .embedder
                .embed_documents_within(&texts, self.opts.deadline)?;
            // A short answer would leave chunks pointing to keys that were never stored (spec
            // §4.1).
            if vectors.len() != texts.len() {
                return Err(Error::Embedding(format!(
                    "the model returned {} vectors for {} texts",
                    vectors.len(),
                    texts.len()
                )));
            }
            // Checked before `put`, so the cache never holds a vector of another dimension.
            for vector in &vectors {
                self.dims.check(vector.len())?;
            }
            let entries: Vec<(EmbeddingKey, Vec<f32>)> =
                missing.iter().map(|&i| keys[i]).zip(vectors).collect();
            // Committed before the index rows, so a failure below leaves only unreferenced entries.
            self.cache.put(&entries)?;
        }
        let records: Vec<ChunkRecord> = chunks
            .into_iter()
            .zip(keys)
            .enumerate()
            .map(|(ord, (chunk, embedding_key))| ChunkRecord {
                ord,
                heading_path: chunk.heading_path_text(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                body: chunk.body,
                embedding_key,
            })
            .collect();
        let meta = note.meta;
        let file = FileRecord {
            path: entry.rel.clone(),
            mtime_ns: entry.mtime_ns,
            size: entry.size,
            content_hash: hash,
            title: meta.title,
            aliases: meta.aliases,
            tags: meta.tags,
            frontmatter: meta.frontmatter,
        };
        self.store.write_file(&self.fps.index, &file, &records)?;
        Ok(requested)
    }

    /// Lists the skipped files and, after a complete run, prunes skipped rows of files that are
    /// gone and cache entries no chunk uses. A run stopped by the budget or an incomplete scan
    /// still needs the vectors of files it has not re-indexed yet.
    fn finish(&mut self, scan: &Scan, finished: bool, report: &mut IndexReport) -> Result<()> {
        let complete = finished && report.remaining == 0 && scan.complete;
        if complete {
            let present: HashSet<&str> = scan.entries.iter().map(|e| e.rel.as_str()).collect();
            self.store.retain_skipped(&self.fps.index, &present)?;
        }
        let mut recorded: Vec<SkippedFile> = self
            .store
            .skipped_states()?
            .into_iter()
            .map(|(path, s)| SkippedFile {
                path,
                reason: s.reason,
            })
            .collect();
        recorded.sort_by(|a, b| a.path.cmp(&b.path));
        report.skipped = recorded;
        if complete {
            let changed = report.rebuilt || report.indexed > 0 || report.deleted > 0;
            // The count check also collects entries left by an earlier run that failed after its
            // cache write, without reading every key on each search.
            if changed || self.cache.entry_count()? > self.store.distinct_key_count()? {
                self.cache.retain(&self.store.embedding_keys()?)?;
            }
        }
        Ok(())
    }
}
