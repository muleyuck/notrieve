pub mod scan;

use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::cache::{EmbeddingCache, EmbeddingKey, embedding_key};
use crate::embed::Embedder;
use crate::fingerprint::Fingerprints;
use crate::markdown::chunk::{MERGE_BELOW_TOKENS, chunk_note};
use crate::markdown::{parse_note, title_from_path};
use crate::store::{ChunkRecord, FileRecord, RebuildReason, Store};
use crate::{Error, Result};
use scan::{ScanEntry, plan_changes, scan_vault};

pub struct IndexOptions<'a> {
    pub exclude: &'a [String],
    pub chunk_max_tokens: usize,
    /// Start new files only while this much time has not elapsed; a started file always finishes.
    /// When the budget stops a run, deletions are deferred to a later run.
    /// `IndexReport::remaining` counts unprocessed files plus deferred deletions.
    pub budget: Option<Duration>,
    /// Rebuild even when the stored fingerprints match (`index --rebuild`).
    pub rebuild: bool,
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
    let mut report = IndexReport::default();
    // Scan first: invalid exclude or ignore rules must fail before anything is dropped.
    let scan = scan_vault(root, opts.exclude)?;
    let reason = match store.rebuild_reason(fps)? {
        Some(reason @ (RebuildReason::NewIndex | RebuildReason::ModelChanged)) => Some(reason),
        // Rows written against another cache point to vectors this one does not hold (spec §4.1).
        // The id is persistent, so a run that fails before rebuilding is caught by the next one.
        _ if store.cache_id()?.as_deref() != Some(cache.id()) => Some(RebuildReason::CacheReset),
        None if opts.rebuild => Some(RebuildReason::Requested),
        other => other,
    };
    if let Some(reason) = reason {
        on_event(IndexEvent::Rebuilding(reason));
        store.rebuild(fps, embedder.model_id(), embedder.dim())?;
        store.bind_cache(cache.id())?;
        report.rebuilt = true;
        report.rebuild_reason = Some(reason);
    }
    for err in &scan.errors {
        tracing::warn!("{err}");
    }
    report.scan_complete = scan.complete;
    if !scan.complete {
        tracing::warn!(
            "the vault could not be scanned completely; deletions and renames are skipped this time"
        );
    }
    let states = store.file_states()?;
    let skipped = store.skipped_states()?;
    let plan = plan_changes(&scan, &states, &skipped);

    let mut deleted = plan.deleted;
    let total = plan.candidates.len();
    let mut finished = true;
    for (done, entry) in plan.candidates.iter().enumerate() {
        // The budget decides whether a new file may start; a started file always finishes.
        if opts
            .budget
            .is_some_and(|budget| started.elapsed() >= budget)
        {
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
                store.skip_file(
                    &fps.index,
                    &entry.rel,
                    entry.mtime_ns,
                    entry.size,
                    &e.to_string(),
                )?;
                continue;
            }
        };
        let hash = blake3::hash(&bytes).to_hex().to_string();
        if states
            .get(&entry.rel)
            .is_some_and(|s| s.content_hash == hash)
        {
            store.touch_file(&fps.index, &entry.rel, entry.mtime_ns, entry.size)?;
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
                store.rename_file(&fps.index, &old, &entry.rel, entry.mtime_ns, entry.size)?;
                report.renamed += 1;
                continue;
            }
        }
        match index_file(
            store,
            cache,
            embedder,
            fps,
            opts.chunk_max_tokens,
            entry,
            bytes,
            hash,
        ) {
            Ok(()) => report.indexed += 1,
            Err(e @ (Error::NotUtf8(_) | Error::Embedding(_) | Error::ChunkTooLarge { .. })) => {
                tracing::warn!(path = %entry.rel, "skipped: {e}");
                store.skip_file(
                    &fps.index,
                    &entry.rel,
                    entry.mtime_ns,
                    entry.size,
                    &e.to_string(),
                )?;
            }
            Err(e) => return Err(e),
        }
    }
    // Deletions wait until every candidate was seen (an unprocessed candidate may be the moved
    // file) and, like files, start only while budget remains.
    if finished {
        if opts
            .budget
            .is_some_and(|budget| started.elapsed() >= budget)
        {
            report.remaining += deleted.len();
        } else {
            for path in &deleted {
                store.delete_file(&fps.index, path)?;
                report.deleted += 1;
            }
        }
    }
    let present: HashSet<&str> = scan.entries.iter().map(|e| e.rel.as_str()).collect();
    if finished && report.remaining == 0 && scan.complete {
        store.retain_skipped(&fps.index, &present)?;
    }
    let mut recorded: Vec<SkippedFile> = store
        .skipped_states()?
        .into_iter()
        .map(|(path, s)| SkippedFile {
            path,
            reason: s.reason,
        })
        .collect();
    recorded.sort_by(|a, b| a.path.cmp(&b.path));
    report.skipped = recorded;
    // Only after a complete run: a run stopped by the budget (or an incomplete scan) still needs
    // the vectors of files it has not re-indexed yet.
    if finished && report.remaining == 0 && scan.complete {
        let changed = report.rebuilt || report.indexed > 0 || report.deleted > 0;
        // The count check also collects entries left by an earlier run that failed after its
        // cache write, without reading every key on each search.
        if changed || cache.entry_count()? > store.distinct_key_count()? {
            cache.retain(&store.embedding_keys()?)?;
        }
    }
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn index_file<E: Embedder + ?Sized>(
    store: &mut Store,
    cache: &mut EmbeddingCache,
    embedder: &E,
    fps: &Fingerprints,
    chunk_max_tokens: usize,
    entry: &ScanEntry,
    bytes: Vec<u8>,
    hash: String,
) -> Result<()> {
    let text = String::from_utf8(bytes).map_err(|_| Error::NotUtf8(entry.rel.clone()))?;
    let note = parse_note(&entry.rel, &text);
    for warning in &note.meta.warnings {
        tracing::warn!(path = %entry.rel, "{warning}");
    }
    let chunks = chunk_note(
        note.body,
        note.body_start_line,
        &note.meta.title,
        chunk_max_tokens,
        MERGE_BELOW_TOKENS,
        embedder,
    )?;
    let inputs: Vec<String> = chunks
        .iter()
        .map(|c| c.embedding_input(&note.meta.title))
        .collect();
    let keys: Vec<EmbeddingKey> = inputs
        .iter()
        .map(|input| embedding_key(&fps.model, input))
        .collect();
    let mut missing: Vec<usize> = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        if !cache.contains(key)? && !missing.iter().any(|&j| keys[j] == *key) {
            missing.push(i);
        }
    }
    if !missing.is_empty() {
        let texts: Vec<String> = missing.iter().map(|&i| inputs[i].clone()).collect();
        let vectors = embedder.embed_documents(&texts)?;
        // A short answer would leave chunks pointing to keys that were never stored (spec §4.1).
        if vectors.len() != texts.len() {
            return Err(Error::Embedding(format!(
                "the model returned {} vectors for {} texts",
                vectors.len(),
                texts.len()
            )));
        }
        let entries: Vec<(EmbeddingKey, Vec<f32>)> =
            missing.iter().map(|&i| keys[i]).zip(vectors).collect();
        // Committed before the index rows, so a failure below leaves only unreferenced entries.
        cache.put(&entries)?;
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
    store.write_file(&fps.index, &file, &records)
}
