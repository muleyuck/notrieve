use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use std::fmt;

use rusqlite::{Connection, ErrorCode, OptionalExtension, Transaction, params};

use crate::cache::EmbeddingKey;
use crate::fingerprint::Fingerprints;
use crate::{Error, Result};

pub const SCHEMA_VERSION: i64 = 3;

const SCHEMA_SQL: &str = include_str!("schema.sql");

#[derive(Debug, Clone, PartialEq)]
pub struct FileRecord {
    pub path: String,
    pub mtime_ns: i64,
    pub size: i64,
    pub content_hash: String,
    pub title: String,
    pub aliases: Vec<String>,
    pub tags: Vec<String>,
    pub frontmatter: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkRecord {
    pub ord: usize,
    pub heading_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub body: String,
    pub embedding_key: EmbeddingKey,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileState {
    pub mtime_ns: i64,
    pub size: i64,
    pub content_hash: String,
    pub title: String,
}

/// A file `index_vault` could not index, with the mtime and size it was seen at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedState {
    pub mtime_ns: i64,
    pub size: i64,
    pub reason: String,
}

/// Why `index_vault` rebuilds the index; shown to the user (spec §4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildReason {
    NewIndex,
    Requested,
    SchemaChanged,
    ModelChanged,
    SettingsChanged,
    /// The embedding cache was missing, unreadable, of another version, or not the one the index
    /// was built against.
    CacheReset,
}

impl fmt::Display for RebuildReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RebuildReason::NewIndex => "building a new index",
            RebuildReason::Requested => {
                "rebuilding the index as requested (unchanged chunks reuse their embeddings)"
            }
            RebuildReason::SchemaChanged => {
                "rebuilding the index because notrieve was updated (unchanged chunks reuse their embeddings)"
            }
            RebuildReason::ModelChanged => {
                "rebuilding the index because the model changed; every chunk is embedded again"
            }
            RebuildReason::SettingsChanged => {
                "rebuilding the index because notrieve or the chunk settings changed (unchanged chunks reuse their embeddings)"
            }
            RebuildReason::CacheReset => {
                "rebuilding the index because the embedding cache was missing or unreadable; every chunk is embedded again"
            }
        })
    }
}

pub struct Store {
    conn: Connection,
}

static REGISTER_VEC: Once = Once::new();

fn register_sqlite_vec() {
    // Official sqlite-vec registration; `sqlite3_vec_init` is declared without arguments.
    REGISTER_VEC.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
            *const (),
            unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *mut std::ffi::c_char,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> std::ffi::c_int,
        >(
            sqlite_vec::sqlite3_vec_init as *const ()
        )));
    });
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        let store = Self::connect(path)?;
        if let Some(found) = store.schema_version()?
            && found > SCHEMA_VERSION
        {
            return Err(Error::SchemaTooNew {
                found,
                supported: SCHEMA_VERSION,
            });
        }
        Ok(store)
    }

    /// Opens for an explicit rebuild: a newer schema is accepted, and a file that is not a database
    /// or fails `PRAGMA quick_check` is moved aside (never deleted).
    pub fn open_for_rebuild(path: &Path) -> Result<(Self, Option<PathBuf>)> {
        match Self::connect(path) {
            Ok(store) if store.is_intact() => Ok((store, None)),
            Ok(store) => {
                drop(store);
                let moved = quarantine(path)?;
                Ok((Self::connect(path)?, Some(moved)))
            }
            Err(Error::DbCorrupt(_)) => {
                let moved = quarantine(path)?;
                Ok((Self::connect(path)?, Some(moved)))
            }
            Err(e) => Err(e),
        }
    }

    fn is_intact(&self) -> bool {
        matches!(self.conn.query_row("PRAGMA quick_check", [], |r| r.get::<_, String>(0)), Ok(ref s) if s == "ok")
    }

    fn connect(path: &Path) -> Result<Self> {
        register_sqlite_vec();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let conn = Connection::open(path).map_err(|e| classify(e, path))?;
        conn.query_row("SELECT count(*) FROM sqlite_master", [], |r| {
            r.get::<_, i64>(0)
        })
        .map_err(|e| classify(e, path))?;
        conn.query_row("PRAGMA journal_mode=WAL", [], |r| r.get::<_, String>(0))
            .map_err(|e| classify(e, path))?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.busy_timeout(Duration::from_secs(5))?;
        Ok(Self { conn })
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    fn has_table(&self, name: &str) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
            [name],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    fn meta(&self, key: &str) -> Result<Option<String>> {
        if !self.has_table("meta")? {
            return Ok(None);
        }
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
            .optional()?)
    }

    fn schema_version(&self) -> Result<Option<i64>> {
        Ok(self.meta("schema_version")?.and_then(|v| v.parse().ok()))
    }

    pub fn fingerprint(&self) -> Result<Option<String>> {
        self.meta("index_fingerprint")
    }

    pub fn needs_rebuild(&self, fingerprint: &str) -> Result<bool> {
        Ok(self.schema_version()? != Some(SCHEMA_VERSION)
            || self.fingerprint()?.as_deref() != Some(fingerprint))
    }

    /// `None` when the index matches the schema and both fingerprints.
    pub fn rebuild_reason(&self, fps: &Fingerprints) -> Result<Option<RebuildReason>> {
        let Some(version) = self.schema_version()? else {
            return Ok(Some(RebuildReason::NewIndex));
        };
        if self
            .meta("model_fingerprint")?
            .is_some_and(|stored| stored != fps.model)
        {
            return Ok(Some(RebuildReason::ModelChanged));
        }
        if version != SCHEMA_VERSION {
            return Ok(Some(RebuildReason::SchemaChanged));
        }
        if self.fingerprint()?.as_deref() != Some(fps.index.as_str()) {
            return Ok(Some(RebuildReason::SettingsChanged));
        }
        Ok(None)
    }

    pub fn rebuild(&mut self, fps: &Fingerprints, model_id: &str, dim: usize) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute_batch(SCHEMA_SQL)?;
        let entries = [
            ("schema_version", SCHEMA_VERSION.to_string()),
            ("index_fingerprint", fps.index.clone()),
            ("model_fingerprint", fps.model.clone()),
            ("model_id", model_id.to_string()),
            ("dim", dim.to_string()),
        ];
        for (key, value) in entries {
            tx.execute(
                "INSERT INTO meta(key, value) VALUES (?1, ?2)",
                params![key, value],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn write_file(
        &mut self,
        fingerprint: &str,
        file: &FileRecord,
        chunks: &[ChunkRecord],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        check_fingerprint(&tx, fingerprint)?;
        delete_rows(&tx, &file.path)?;
        tx.execute(
            "INSERT INTO files(path, mtime_ns, size, content_hash, title, aliases, frontmatter)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                file.path,
                file.mtime_ns,
                file.size,
                file.content_hash,
                file.title,
                file.aliases.join("\n"),
                file.frontmatter
            ],
        )?;
        let file_id = tx.last_insert_rowid();
        for tag in &file.tags {
            tx.execute(
                "INSERT INTO file_tags(file_id, tag, tag_key) VALUES (?1, ?2, ?3)",
                params![file_id, tag, tag.to_lowercase()],
            )?;
        }
        let fts_title = std::iter::once(file.title.as_str())
            .chain(file.aliases.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        for c in chunks {
            tx.execute(
                "INSERT INTO chunks(file_id, ord, heading_path, start_line, end_line, body, embedding_key)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    file_id,
                    c.ord as i64,
                    c.heading_path,
                    c.start_line as i64,
                    c.end_line as i64,
                    c.body,
                    c.embedding_key
                ],
            )?;
            let id = tx.last_insert_rowid();
            tx.execute(
                "INSERT INTO chunks_fts(rowid, title, heading_path, body) VALUES (?1, ?2, ?3, ?4)",
                params![id, fts_title, c.heading_path, c.body],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn touch_file(
        &mut self,
        fingerprint: &str,
        path: &str,
        mtime_ns: i64,
        size: i64,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        check_fingerprint(&tx, fingerprint)?;
        tx.execute(
            "UPDATE files SET mtime_ns = ?2, size = ?3 WHERE path = ?1",
            params![path, mtime_ns, size],
        )?;
        tx.execute("DELETE FROM skipped_files WHERE path = ?1", [path])?;
        tx.commit()?;
        Ok(())
    }

    pub fn rename_file(
        &mut self,
        fingerprint: &str,
        from: &str,
        to: &str,
        mtime_ns: i64,
        size: i64,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        check_fingerprint(&tx, fingerprint)?;
        tx.execute(
            "UPDATE files SET path = ?2, mtime_ns = ?3, size = ?4 WHERE path = ?1",
            params![from, to, mtime_ns, size],
        )?;
        tx.execute("DELETE FROM skipped_files WHERE path = ?1", [to])?;
        tx.commit()?;
        Ok(())
    }

    pub fn delete_file(&mut self, fingerprint: &str, path: &str) -> Result<()> {
        let tx = self.conn.transaction()?;
        check_fingerprint(&tx, fingerprint)?;
        delete_rows(&tx, path)?;
        tx.commit()?;
        Ok(())
    }

    /// Records a file that could not be indexed, so it is not retried until its mtime or size changes.
    pub fn skip_file(
        &mut self,
        fingerprint: &str,
        path: &str,
        mtime_ns: i64,
        size: i64,
        reason: &str,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        check_fingerprint(&tx, fingerprint)?;
        // The old content must not stay searchable once the file cannot be indexed.
        delete_rows(&tx, path)?;
        tx.execute(
            "INSERT OR REPLACE INTO skipped_files(path, mtime_ns, size, reason)
             VALUES (?1, ?2, ?3, ?4)",
            params![path, mtime_ns, size, reason],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Removes the skipped rows of paths that are no longer in the vault.
    pub fn retain_skipped(&mut self, fingerprint: &str, present: &HashSet<&str>) -> Result<()> {
        let tx = self.conn.transaction()?;
        check_fingerprint(&tx, fingerprint)?;
        let stale: Vec<String> = {
            let mut stmt = tx.prepare("SELECT path FROM skipped_files")?;
            stmt.query_map([], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?
                .into_iter()
                .filter(|p| !present.contains(p.as_str()))
                .collect()
        };
        for path in stale {
            tx.execute("DELETE FROM skipped_files WHERE path = ?1", [path])?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn skipped_states(&self) -> Result<HashMap<String, SkippedState>> {
        if !self.has_table("skipped_files")? {
            return Ok(HashMap::new());
        }
        let mut stmt = self
            .conn
            .prepare("SELECT path, mtime_ns, size, reason FROM skipped_files")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                SkippedState {
                    mtime_ns: r.get(1)?,
                    size: r.get(2)?,
                    reason: r.get(3)?,
                },
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn has_file(&self, path: &str) -> Result<bool> {
        if !self.has_table("files")? {
            return Ok(false);
        }
        Ok(self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM files WHERE path = ?1)",
            [path],
            |r| r.get(0),
        )?)
    }

    pub fn file_states(&self) -> Result<HashMap<String, FileState>> {
        if !self.has_table("files")? {
            return Ok(HashMap::new());
        }
        let mut stmt = self
            .conn
            .prepare("SELECT path, mtime_ns, size, content_hash, title FROM files")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                FileState {
                    mtime_ns: r.get(1)?,
                    size: r.get(2)?,
                    content_hash: r.get(3)?,
                    title: r.get(4)?,
                },
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Keys of every chunk's vector in the embedding cache.
    /// The embedding key of one chunk, to read the index's vector dimension from the cache.
    pub fn any_embedding_key(&self) -> Result<Option<EmbeddingKey>> {
        if !self.has_table("chunks")? {
            return Ok(None);
        }
        Ok(self
            .conn
            .query_row("SELECT embedding_key FROM chunks LIMIT 1", [], |r| r.get(0))
            .optional()?)
    }

    pub fn embedding_keys(&self) -> Result<HashSet<EmbeddingKey>> {
        if !self.has_table("chunks")? {
            return Ok(HashSet::new());
        }
        let mut stmt = self.conn.prepare("SELECT embedding_key FROM chunks")?;
        let keys = stmt
            .query_map([], |r| r.get::<_, EmbeddingKey>(0))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(keys)
    }

    /// `embedding_keys().len()` without loading the keys.
    pub fn distinct_key_count(&self) -> Result<usize> {
        if !self.has_table("chunks")? {
            return Ok(0);
        }
        let n: i64 = self.conn.query_row(
            "SELECT count(DISTINCT embedding_key) FROM chunks",
            [],
            |r| r.get(0),
        )?;
        Ok(n as usize)
    }

    /// The embedding cache the index rows were written against (`EmbeddingCache::id`).
    pub fn cache_id(&self) -> Result<Option<String>> {
        self.meta("cache_id")
    }

    /// Records the cache after a rebuild; a rebuild drops the meta table, so it is recorded again.
    pub fn bind_cache(&mut self, cache_id: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('cache_id', ?1)",
            [cache_id],
        )?;
        Ok(())
    }

    /// (files, chunks)
    pub fn stats(&self) -> Result<(usize, usize)> {
        let files: i64 = self
            .conn
            .query_row("SELECT count(*) FROM files", [], |r| r.get(0))?;
        let chunks: i64 = self
            .conn
            .query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))?;
        Ok((files as usize, chunks as usize))
    }
}

fn check_fingerprint(tx: &Transaction, fingerprint: &str) -> Result<()> {
    let stored: Option<String> = tx
        .query_row(
            "SELECT value FROM meta WHERE key = 'index_fingerprint'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    if stored.as_deref() == Some(fingerprint) {
        Ok(())
    } else {
        Err(Error::FingerprintChanged)
    }
}

fn delete_rows(tx: &Transaction, path: &str) -> Result<()> {
    let ids: Vec<i64> = {
        let mut stmt = tx.prepare(
            "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path = ?1",
        )?;
        stmt.query_map([path], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?
    };
    // chunks_fts is not covered by the foreign-key cascade.
    for id in ids {
        tx.execute("DELETE FROM chunks_fts WHERE rowid = ?1", [id])?;
    }
    tx.execute("DELETE FROM files WHERE path = ?1", [path])?;
    tx.execute("DELETE FROM skipped_files WHERE path = ?1", [path])?;
    Ok(())
}

pub(crate) fn classify(error: rusqlite::Error, path: &Path) -> Error {
    match error.sqlite_error_code() {
        Some(ErrorCode::NotADatabase | ErrorCode::DatabaseCorrupt) => {
            Error::DbCorrupt(path.to_path_buf())
        }
        _ => Error::Sqlite(error),
    }
}

pub(crate) fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(suffix);
    PathBuf::from(s)
}

fn quarantine(path: &Path) -> Result<PathBuf> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    quarantine_at(path, ts)
}

fn quarantine_at(path: &Path, ts: u64) -> Result<PathBuf> {
    let taken = |dest: &Path| {
        ["", "-wal", "-shm"]
            .iter()
            .any(|side| with_suffix(dest, side).symlink_metadata().is_ok())
    };
    let mut dest = with_suffix(path, &format!(".corrupt-{ts}"));
    let mut n = 0u32;
    while taken(&dest) {
        n += 1;
        dest = with_suffix(path, &format!(".corrupt-{ts}-{n}"));
    }
    std::fs::rename(path, &dest).map_err(|e| Error::io(path, e))?;
    for side in ["-wal", "-shm"] {
        let src = with_suffix(path, side);
        if src.exists() {
            let to = with_suffix(&dest, side);
            std::fs::rename(&src, &to).map_err(|e| Error::io(&src, e))?;
        }
    }
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn file(path: &str) -> FileRecord {
        FileRecord {
            path: path.to_string(),
            mtime_ns: 1_700_000_000_123,
            size: 321,
            content_hash: format!("hash-of-{path}"),
            title: "税金メモ".into(),
            aliases: vec!["tax".into()],
            tags: vec!["税金".into(), "inbox/todo".into()],
            frontmatter: Some("tags: [税金]\n".into()),
        }
    }

    fn chunk(ord: usize, body: &str) -> ChunkRecord {
        ChunkRecord {
            ord,
            heading_path: "確定申告 > 医療費".into(),
            start_line: 3 + ord,
            end_line: 7 + ord,
            body: body.into(),
            embedding_key: *blake3::hash(body.as_bytes()).as_bytes(),
        }
    }

    fn count(store: &Store, table: &str) -> i64 {
        store
            .conn()
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    }

    fn fps(index: &str) -> Fingerprints {
        Fingerprints {
            model: "model-a".into(),
            index: index.into(),
        }
    }

    fn rebuilt(dir: &TempDir, fp: &str, dim: usize) -> Store {
        let mut s = Store::open(&dir.path().join("index.db")).unwrap();
        s.rebuild(&fps(fp), "fake", dim).unwrap();
        s
    }

    #[test]
    fn fresh_database_needs_a_rebuild() {
        let dir = TempDir::new().unwrap();
        let s = Store::open(&dir.path().join("index.db")).unwrap();
        assert!(s.needs_rebuild("fp-a").unwrap());
        assert_eq!(s.fingerprint().unwrap(), None);
    }

    #[test]
    fn rebuild_records_the_fingerprint() {
        let dir = TempDir::new().unwrap();
        let s = rebuilt(&dir, "fp-a", 4);
        assert!(!s.needs_rebuild("fp-a").unwrap());
        assert!(s.needs_rebuild("fp-b").unwrap());
        assert_eq!(s.fingerprint().unwrap().as_deref(), Some("fp-a"));
    }

    #[test]
    fn database_uses_wal() {
        let dir = TempDir::new().unwrap();
        let s = rebuilt(&dir, "fp-a", 4);
        let mode: String = s
            .conn()
            .query_row("PRAGMA journal_mode", [], |r| r.get(0))
            .unwrap();
        assert_eq!(mode, "wal");
    }

    #[test]
    fn write_file_fills_every_table() {
        let dir = TempDir::new().unwrap();
        let mut s = rebuilt(&dir, "fp-a", 4);
        s.write_file(
            "fp-a",
            &file("a.md"),
            &[chunk(0, "医療費控除"), chunk(1, "領収書")],
        )
        .unwrap();
        assert_eq!(s.stats().unwrap(), (1, 2));
        assert_eq!((count(&s, "chunks_fts"), count(&s, "file_tags")), (2, 2));
        let keys: Vec<EmbeddingKey> = s
            .conn()
            .prepare("SELECT embedding_key FROM chunks ORDER BY ord")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .map(|k| k.unwrap())
            .collect();
        assert_eq!(
            keys,
            [
                chunk(0, "医療費控除").embedding_key,
                chunk(1, "領収書").embedding_key
            ]
        );
    }

    #[test]
    fn rewriting_a_file_leaves_no_orphan_rows() {
        let dir = TempDir::new().unwrap();
        let mut s = rebuilt(&dir, "fp-a", 4);
        let three = [chunk(0, "a"), chunk(1, "b"), chunk(2, "c")];
        s.write_file("fp-a", &file("a.md"), &three).unwrap();
        s.write_file("fp-a", &file("a.md"), &[chunk(0, "d")])
            .unwrap();
        assert_eq!((count(&s, "chunks"), count(&s, "chunks_fts")), (1, 1));
    }

    #[test]
    fn delete_removes_search_rows() {
        let dir = TempDir::new().unwrap();
        let mut s = rebuilt(&dir, "fp-a", 4);
        s.write_file("fp-a", &file("a.md"), &[chunk(0, "a")])
            .unwrap();
        s.delete_file("fp-a", "a.md").unwrap();
        assert_eq!((count(&s, "files"), count(&s, "chunks_fts")), (0, 0));
    }

    #[test]
    fn write_with_a_stale_fingerprint_is_rejected_without_rows() {
        let dir = TempDir::new().unwrap();
        let mut s = rebuilt(&dir, "fp-a", 4);
        let err = s
            .write_file("fp-b", &file("a.md"), &[chunk(0, "a")])
            .unwrap_err();
        assert!(matches!(err, Error::FingerprintChanged));
        assert_eq!(count(&s, "files"), 0);
    }

    #[test]
    fn rename_keeps_chunk_rows() {
        let dir = TempDir::new().unwrap();
        let mut s = rebuilt(&dir, "fp-a", 4);
        s.write_file("fp-a", &file("a.md"), &[chunk(0, "a")])
            .unwrap();
        let before: i64 = s
            .conn()
            .query_row("SELECT id FROM chunks", [], |r| r.get(0))
            .unwrap();
        s.rename_file("fp-a", "a.md", "sub/a.md", 42, 321).unwrap();
        let states = s.file_states().unwrap();
        assert_eq!(states["sub/a.md"].mtime_ns, 42);
        assert!(!states.contains_key("a.md"));
        let after: i64 = s
            .conn()
            .query_row("SELECT id FROM chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(after, before);
    }

    #[test]
    fn touch_updates_only_mtime_and_size() {
        let dir = TempDir::new().unwrap();
        let mut s = rebuilt(&dir, "fp-a", 4);
        s.write_file("fp-a", &file("a.md"), &[chunk(0, "a")])
            .unwrap();
        s.touch_file("fp-a", "a.md", 99, 400).unwrap();
        let st = &s.file_states().unwrap()["a.md"];
        assert_eq!(
            (st.mtime_ns, st.size, st.content_hash.as_str()),
            (99, 400, "hash-of-a.md")
        );
    }

    #[test]
    fn rebuild_drops_the_vector_table_of_schema_1() {
        let dir = TempDir::new().unwrap();
        let mut s = rebuilt(&dir, "fp-a", 4);
        s.conn()
            .execute_batch("CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[4]);")
            .unwrap();
        s.rebuild(&fps("fp-b"), "fake", 4).unwrap();
        let n: i64 = s
            .conn()
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name = 'chunks_vec'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn embedding_keys_lists_the_key_of_every_chunk() {
        let dir = TempDir::new().unwrap();
        let mut s = rebuilt(&dir, "fp-a", 4);
        s.write_file("fp-a", &file("a.md"), &[chunk(0, "a"), chunk(1, "b")])
            .unwrap();
        s.write_file("fp-a", &file("b.md"), &[chunk(0, "c")])
            .unwrap();
        let expected: HashSet<EmbeddingKey> = ["a", "b", "c"]
            .iter()
            .map(|body| chunk(0, body).embedding_key)
            .collect();
        assert_eq!(s.embedding_keys().unwrap(), expected);
    }

    #[test]
    fn newer_schema_blocks_open_but_not_rebuild() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index.db");
        let s = rebuilt(&dir, "fp-a", 4);
        s.conn()
            .execute(
                "UPDATE meta SET value = '99' WHERE key = 'schema_version'",
                [],
            )
            .unwrap();
        drop(s);
        assert!(matches!(
            Store::open(&path),
            Err(Error::SchemaTooNew { found: 99, .. })
        ));
        let (mut s, moved) = Store::open_for_rebuild(&path).unwrap();
        assert_eq!(moved, None);
        s.rebuild(&fps("fp-a"), "fake", 4).unwrap();
        assert!(!s.needs_rebuild("fp-a").unwrap());
    }

    #[test]
    fn rebuild_clears_skipped_rows() {
        let dir = TempDir::new().unwrap();
        let mut s = rebuilt(&dir, "fp-a", 4);
        s.skip_file("fp-a", "bad.md", 1, 3, "not UTF-8").unwrap();
        assert_eq!(s.skipped_states().unwrap().len(), 1);
        s.rebuild(&fps("fp-a"), "fake", 4).unwrap();
        assert!(s.skipped_states().unwrap().is_empty());
    }

    #[test]
    fn damaged_pages_are_detected_and_moved_aside_on_rebuild() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index.db");
        {
            let s = rebuilt(&dir, "fp-a", 4);
            s.conn()
                .execute_batch("CREATE TABLE filler(x TEXT);")
                .unwrap();
            for i in 0..300 {
                s.conn()
                    .execute("INSERT INTO filler(x) VALUES (?1)", [format!("{i:0>200}")])
                    .unwrap();
            }
            s.conn()
                .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()))
                .unwrap();
        }
        let mut bytes = std::fs::read(&path).unwrap();
        let page = 4096;
        assert!(
            bytes.len() > page * 6,
            "database too small to damage: {}",
            bytes.len()
        );
        bytes[page * 4..page * 6].fill(0xA5);
        std::fs::write(&path, &bytes).unwrap();
        let (mut s, moved) = Store::open_for_rebuild(&path).unwrap();
        assert!(
            moved
                .unwrap()
                .to_string_lossy()
                .contains("index.db.corrupt-")
        );
        s.rebuild(&fps("fp-a"), "fake", 4).unwrap();
        assert!(!s.needs_rebuild("fp-a").unwrap());
    }

    #[test]
    fn healthy_database_is_not_moved_on_rebuild() {
        let dir = TempDir::new().unwrap();
        drop(rebuilt(&dir, "fp-a", 4));
        let (_, moved) = Store::open_for_rebuild(&dir.path().join("index.db")).unwrap();
        assert_eq!(moved, None);
    }

    #[test]
    fn failure_midway_through_a_file_keeps_its_previous_rows_in_every_table() {
        let dir = TempDir::new().unwrap();
        let mut s = rebuilt(&dir, "fp-a", 4);
        s.write_file("fp-a", &file("a.md"), &[chunk(0, "old body")])
            .unwrap();
        // Rejects the second chunk, so the insert fails after earlier rows of the file were written.
        s.conn()
            .execute_batch(
                "CREATE TRIGGER reject_bad BEFORE INSERT ON chunks WHEN NEW.body = 'bad'
                 BEGIN SELECT RAISE(ABORT, 'rejected'); END;",
            )
            .unwrap();
        let mut newer = file("a.md");
        newer.content_hash = "new-hash".into();
        newer.tags = vec!["changed".into()];
        let err = s
            .write_file("fp-a", &newer, &[chunk(5, "new body"), chunk(6, "bad")])
            .unwrap_err();
        assert!(matches!(err, Error::Sqlite(_)), "{err}");
        assert_eq!(
            s.file_states().unwrap()["a.md"].content_hash,
            "hash-of-a.md"
        );
        let tags: Vec<String> = s
            .conn()
            .prepare("SELECT tag FROM file_tags ORDER BY tag")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .map(|row| row.unwrap())
            .collect();
        assert_eq!(tags, ["inbox/todo", "税金"]);
        let (body, key): (String, EmbeddingKey) = s
            .conn()
            .query_row("SELECT body, embedding_key FROM chunks", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(body, "old body");
        assert_eq!(key, chunk(0, "old body").embedding_key);
        let fts: String = s
            .conn()
            .query_row("SELECT body FROM chunks_fts", [], |r| r.get(0))
            .unwrap();
        assert_eq!(fts, "old body");
        assert_eq!((count(&s, "chunks"), count(&s, "chunks_fts")), (1, 1));
    }

    #[test]
    fn corrupt_file_is_reported_and_moved_aside_on_rebuild() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index.db");
        std::fs::write(&path, vec![b'x'; 4096]).unwrap();
        assert!(matches!(Store::open(&path), Err(Error::DbCorrupt(_))));
        let (mut s, moved) = Store::open_for_rebuild(&path).unwrap();
        let moved = moved.unwrap();
        assert!(moved.to_string_lossy().contains("index.db.corrupt-"));
        assert_eq!(std::fs::read(&moved).unwrap(), vec![b'x'; 4096]);
        s.rebuild(&fps("fp-a"), "fake", 4).unwrap();
        assert!(!s.needs_rebuild("fp-a").unwrap());
    }

    #[test]
    fn move_aside_never_replaces_an_earlier_corrupt_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index.db");
        std::fs::write(&path, b"first").unwrap();
        let first = quarantine_at(&path, 1000).unwrap();
        assert_eq!(first, dir.path().join("index.db.corrupt-1000"));
        std::fs::write(&path, b"second").unwrap();
        std::fs::write(dir.path().join("index.db-wal"), b"second-wal").unwrap();
        let second = quarantine_at(&path, 1000).unwrap();
        assert_ne!(second, first);
        assert_eq!(std::fs::read(&first).unwrap(), b"first");
        assert_eq!(std::fs::read(&second).unwrap(), b"second");
        assert_eq!(
            std::fs::read(with_suffix(&second, "-wal")).unwrap(),
            b"second-wal"
        );
        std::fs::write(&path, b"third").unwrap();
        std::fs::write(
            with_suffix(&dir.path().join("index.db.corrupt-1000-2"), "-shm"),
            b"x",
        )
        .unwrap();
        let third = quarantine_at(&path, 1000).unwrap();
        assert_eq!(third, dir.path().join("index.db.corrupt-1000-3"));
    }

    #[test]
    fn fresh_database_is_a_new_index() {
        let dir = TempDir::new().unwrap();
        let s = Store::open(&dir.path().join("index.db")).unwrap();
        assert_eq!(
            s.rebuild_reason(&fps("fp-a")).unwrap(),
            Some(RebuildReason::NewIndex)
        );
    }

    #[test]
    fn matching_fingerprints_need_no_rebuild() {
        let dir = TempDir::new().unwrap();
        let s = rebuilt(&dir, "fp-a", 4);
        assert_eq!(s.rebuild_reason(&fps("fp-a")).unwrap(), None);
    }

    #[test]
    fn other_index_fingerprint_with_the_same_model_is_a_settings_change() {
        let dir = TempDir::new().unwrap();
        let s = rebuilt(&dir, "fp-a", 4);
        assert_eq!(
            s.rebuild_reason(&fps("fp-b")).unwrap(),
            Some(RebuildReason::SettingsChanged)
        );
    }

    fn with_an_old_schema(dir: &TempDir) -> Store {
        let s = rebuilt(dir, "fp-a", 4);
        s.conn()
            .execute(
                "UPDATE meta SET value = '0' WHERE key = 'schema_version'",
                [],
            )
            .unwrap();
        s
    }

    #[test]
    fn model_change_wins_over_an_old_schema() {
        let dir = TempDir::new().unwrap();
        let s = with_an_old_schema(&dir);
        let other_model = Fingerprints {
            model: "model-b".into(),
            index: "fp-a".into(),
        };
        assert_eq!(
            s.rebuild_reason(&other_model).unwrap(),
            Some(RebuildReason::ModelChanged)
        );
    }

    #[test]
    fn an_old_schema_alone_is_a_schema_change() {
        let dir = TempDir::new().unwrap();
        let s = with_an_old_schema(&dir);
        assert_eq!(
            s.rebuild_reason(&fps("fp-a")).unwrap(),
            Some(RebuildReason::SchemaChanged)
        );
    }

    #[test]
    fn index_without_a_recorded_model_fingerprint_reports_a_settings_change() {
        // Indexes written by plan 1 have no model_fingerprint row.
        let dir = TempDir::new().unwrap();
        let s = rebuilt(&dir, "fp-a", 4);
        s.conn()
            .execute("DELETE FROM meta WHERE key = 'model_fingerprint'", [])
            .unwrap();
        let other_model = Fingerprints {
            model: "model-b".into(),
            index: "fp-b".into(),
        };
        assert_eq!(
            s.rebuild_reason(&other_model).unwrap(),
            Some(RebuildReason::SettingsChanged)
        );
    }
}
