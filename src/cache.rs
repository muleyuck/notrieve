//! Embedding cache: vectors keyed by model and model input, reused across index rebuilds.
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, params};

use crate::store::{classify, with_suffix};
use crate::{Error, Result};

pub const CACHE_SCHEMA_VERSION: i64 = 1;

const CACHE_SCHEMA_SQL: &str = include_str!("cache_schema.sql");

pub type EmbeddingKey = [u8; 32];

/// The model fingerprint covers the whole contract, document prefix included, so hashing it with
/// the prefix-less input identifies the exact string the model embeds.
pub fn embedding_key(model_fingerprint: &str, input: &str) -> EmbeddingKey {
    let mut hasher = blake3::Hasher::new();
    hasher.update(model_fingerprint.as_bytes());
    hasher.update(&[0]);
    hasher.update(input.as_bytes());
    *hasher.finalize().as_bytes()
}

/// `<dir>/<index file name>.embeddings.db`. Naming it after the full index file name (extension
/// included) keeps vaults whose `db_path` share a directory, even ones differing only in
/// extension, from sharing (and garbage-collecting) one cache.
pub fn cache_path(db_path: &Path) -> PathBuf {
    let mut name = db_path
        .file_name()
        .map_or_else(|| "index".into(), |name| name.to_os_string());
    name.push(".embeddings.db");
    db_path.with_file_name(name)
}

pub fn f32_blob(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub struct EmbeddingCache {
    conn: Connection,
    created: bool,
    id: String,
}

/// Unique per created cache; index.db records it to notice a cache that was replaced.
fn new_cache_id(path: &Path) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut hasher = blake3::Hasher::new();
    hasher.update(path.as_os_str().as_encoded_bytes());
    hasher.update(&now.to_le_bytes());
    hasher.update(&std::process::id().to_le_bytes());
    hasher.finalize().to_hex()[..32].to_string()
}

impl EmbeddingCache {
    /// Opens the cache, creating it when missing. An unreadable cache or one with another schema
    /// version is discarded and recreated: it is derived data.
    pub fn open(path: &Path) -> Result<Self> {
        match Self::connect(path) {
            Err(Error::DbCorrupt(_)) => {
                discard(path)?;
                Self::connect(path)
            }
            other => other,
        }
    }

    /// Like `open`, and also discards a cache that fails `PRAGMA quick_check`.
    pub fn open_for_rebuild(path: &Path) -> Result<Self> {
        let cache = Self::open(path)?;
        if cache.is_intact() {
            return Ok(cache);
        }
        drop(cache);
        discard(path)?;
        Self::connect(path)
    }

    fn connect(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let conn = Connection::open(path).map_err(|e| classify(e, path))?;
        let version: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .map_err(|e| classify(e, path))?;
        if version != 0 && version != CACHE_SCHEMA_VERSION {
            drop(conn);
            discard(path)?;
            return Self::connect(path);
        }
        conn.query_row("PRAGMA journal_mode=WAL", [], |r| r.get::<_, String>(0))
            .map_err(|e| classify(e, path))?;
        conn.busy_timeout(Duration::from_secs(5))?;
        let created = version == 0;
        if created {
            let tx = conn.unchecked_transaction()?;
            tx.execute_batch(CACHE_SCHEMA_SQL)?;
            tx.execute(
                "INSERT INTO meta(key, value) VALUES ('cache_id', ?1)",
                [new_cache_id(path)],
            )?;
            tx.execute_batch(&format!("PRAGMA user_version = {CACHE_SCHEMA_VERSION};"))?;
            tx.commit()?;
        }
        let id: String = conn
            .query_row("SELECT value FROM meta WHERE key = 'cache_id'", [], |r| {
                r.get(0)
            })
            .map_err(|e| classify(e, path))?;
        Ok(Self { conn, created, id })
    }

    /// Recorded in index.db by every rebuild; a different value means the index's rows point to
    /// vectors of another cache.
    pub fn id(&self) -> &str {
        &self.id
    }

    fn is_intact(&self) -> bool {
        matches!(self.conn.query_row("PRAGMA quick_check", [], |r| r.get::<_, String>(0)), Ok(ref s) if s == "ok")
    }

    /// True when this open created the cache (it was missing, unreadable or of another version).
    pub fn is_new(&self) -> bool {
        self.created
    }

    pub fn contains(&self, key: &EmbeddingKey) -> Result<bool> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT 1 FROM embeddings WHERE key = ?1")?;
        Ok(stmt.exists([key])?)
    }

    /// One transaction; it commits before the index rows that reference these keys (spec §4.1).
    pub fn put(&mut self, entries: &[(EmbeddingKey, Vec<f32>)]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx
                .prepare_cached("INSERT OR REPLACE INTO embeddings(key, vector) VALUES (?1, ?2)")?;
            for (key, vector) in entries {
                stmt.execute(params![key, f32_blob(vector)])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Deletes every entry whose key is not in `keep`; returns how many were deleted.
    pub fn retain(&mut self, keep: &HashSet<EmbeddingKey>) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let unused: Vec<EmbeddingKey> = {
            let mut stmt = tx.prepare("SELECT key FROM embeddings")?;
            let keys = stmt
                .query_map([], |r| r.get::<_, EmbeddingKey>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            keys.into_iter().filter(|k| !keep.contains(k)).collect()
        };
        for key in &unused {
            tx.execute("DELETE FROM embeddings WHERE key = ?1", [key])?;
        }
        tx.commit()?;
        Ok(unused.len())
    }

    pub fn entry_count(&self) -> Result<usize> {
        let n: i64 = self
            .conn
            .query_row("SELECT count(*) FROM embeddings", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

/// Removes the cache and its WAL files; the cache is derived data, so nothing is kept.
fn discard(path: &Path) -> Result<()> {
    for suffix in ["", "-wal", "-shm"] {
        let file = with_suffix(path, suffix);
        match std::fs::remove_file(&file) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(Error::io(&file, e));
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn key(n: u8) -> EmbeddingKey {
        [n; 32]
    }

    #[test]
    fn key_depends_on_the_model_and_the_input() {
        let base = embedding_key("model-a", "税金\n医療費");
        assert_eq!(base, embedding_key("model-a", "税金\n医療費"));
        assert_ne!(base, embedding_key("model-b", "税金\n医療費"));
        assert_ne!(base, embedding_key("model-a", "税金\n医療"));
        // The separator keeps "ab" + "c" apart from "a" + "bc".
        assert_ne!(embedding_key("ab", "c"), embedding_key("a", "bc"));
    }

    #[test]
    fn cache_file_is_named_after_the_index_file() {
        assert_ne!(
            cache_path(Path::new("/c/work.db")),
            cache_path(Path::new("/c/work.sqlite"))
        );
        assert_ne!(
            cache_path(Path::new("/c/work")),
            cache_path(Path::new("/c/work.db"))
        );
        assert_eq!(
            cache_path(Path::new("/v/.notrieve/index.db")),
            PathBuf::from("/v/.notrieve/index.db.embeddings.db")
        );
        assert_eq!(
            cache_path(Path::new("/c/work.sqlite")),
            PathBuf::from("/c/work.sqlite.embeddings.db")
        );
        assert_eq!(
            cache_path(Path::new("/c/work")),
            PathBuf::from("/c/work.embeddings.db")
        );
        assert_ne!(
            cache_path(Path::new("/cache/work.db")),
            cache_path(Path::new("/cache/personal.db"))
        );
    }

    #[test]
    fn cache_id_survives_a_reopen_and_changes_when_the_cache_is_recreated() {
        let dir = TempDir::new().unwrap();
        let path = cache_path(&dir.path().join("index.db"));
        let first = EmbeddingCache::open(&path).unwrap().id().to_string();
        assert_eq!(EmbeddingCache::open(&path).unwrap().id(), first);
        std::fs::write(&path, vec![b'x'; 4096]).unwrap();
        let recreated = EmbeddingCache::open(&path).unwrap();
        assert!(recreated.is_new());
        assert_ne!(recreated.id(), first);
    }

    #[test]
    fn first_open_creates_the_cache_and_a_reopen_does_not() {
        let dir = TempDir::new().unwrap();
        let path = cache_path(&dir.path().join("index.db"));
        assert!(EmbeddingCache::open(&path).unwrap().is_new());
        assert!(!EmbeddingCache::open(&path).unwrap().is_new());
    }

    #[test]
    fn stored_vectors_are_little_endian_f32_and_survive_a_reopen() {
        let dir = TempDir::new().unwrap();
        let path = cache_path(&dir.path().join("index.db"));
        let mut cache = EmbeddingCache::open(&path).unwrap();
        cache.put(&[(key(7), vec![1.5, -2.0])]).unwrap();
        drop(cache);
        let cache = EmbeddingCache::open(&path).unwrap();
        assert!(cache.contains(&key(7)).unwrap());
        assert!(!cache.contains(&key(8)).unwrap());
        let blob: Vec<u8> = cache
            .conn()
            .query_row("SELECT vector FROM embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(blob, f32_blob(&[1.5, -2.0]));
    }

    #[test]
    fn retain_deletes_only_unlisted_entries() {
        let dir = TempDir::new().unwrap();
        let mut cache = EmbeddingCache::open(&dir.path().join("c.embeddings.db")).unwrap();
        cache
            .put(&[
                (key(1), vec![0.1]),
                (key(2), vec![0.2]),
                (key(3), vec![0.3]),
            ])
            .unwrap();
        let keep: HashSet<EmbeddingKey> = [key(2)].into_iter().collect();
        assert_eq!(cache.retain(&keep).unwrap(), 2);
        assert_eq!(cache.entry_count().unwrap(), 1);
        assert!(cache.contains(&key(2)).unwrap());
        assert!(!cache.contains(&key(1)).unwrap());
    }

    #[test]
    fn unreadable_file_is_discarded_and_recreated() {
        let dir = TempDir::new().unwrap();
        let path = cache_path(&dir.path().join("index.db"));
        std::fs::write(&path, vec![b'x'; 4096]).unwrap();
        let cache = EmbeddingCache::open(&path).unwrap();
        assert!(cache.is_new());
        assert_eq!(cache.entry_count().unwrap(), 0);
    }

    #[test]
    fn cache_of_another_schema_version_is_discarded() {
        let dir = TempDir::new().unwrap();
        let path = cache_path(&dir.path().join("index.db"));
        let mut cache = EmbeddingCache::open(&path).unwrap();
        cache.put(&[(key(1), vec![0.1])]).unwrap();
        cache
            .conn()
            .execute_batch("PRAGMA user_version = 99;")
            .unwrap();
        drop(cache);
        let cache = EmbeddingCache::open(&path).unwrap();
        assert!(cache.is_new());
        assert_eq!(cache.entry_count().unwrap(), 0);
    }

    #[test]
    fn damaged_pages_pass_a_normal_open_but_are_discarded_on_rebuild() {
        let dir = TempDir::new().unwrap();
        let path = cache_path(&dir.path().join("index.db"));
        {
            let mut cache = EmbeddingCache::open(&path).unwrap();
            let entries: Vec<(EmbeddingKey, Vec<f32>)> = (0..300u32)
                .map(|i| {
                    (
                        *blake3::hash(&i.to_le_bytes()).as_bytes(),
                        vec![i as f32; 512],
                    )
                })
                .collect();
            cache.put(&entries).unwrap();
            cache
                .conn()
                .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()))
                .unwrap();
        }
        let mut bytes = std::fs::read(&path).unwrap();
        let page = 4096;
        assert!(
            bytes.len() > page * 6,
            "cache too small to damage: {}",
            bytes.len()
        );
        bytes[page * 4..page * 6].fill(0xA5);
        std::fs::write(&path, &bytes).unwrap();
        assert!(!EmbeddingCache::open(&path).unwrap().is_new());
        let cache = EmbeddingCache::open_for_rebuild(&path).unwrap();
        assert!(cache.is_new());
        assert_eq!(cache.entry_count().unwrap(), 0);
    }
}
