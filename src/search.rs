//! Hybrid search (spec §4.5): FTS5 trigram and LIKE for text, cosine distance over the embedding
//! cache for vectors, fused with reciprocal rank fusion.
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use rusqlite::types::Value;
use rusqlite::{Connection, OptionalExtension, params_from_iter};
use serde::Serialize;

use crate::cache::{cache_path, f32_blob};
use crate::embed::Embedder;
use crate::error::RemoteErrorKind;
use crate::store::Store;
use crate::{Error, Result};

pub const DEFAULT_K: usize = 8;
pub const MAX_K: usize = 50;
/// Candidates taken from each ranking before fusion.
const CANDIDATES: usize = 50;
const RRF_K: f64 = 60.0;
/// The trigram tokenizer cannot match shorter terms; they go through LIKE.
const MIN_FTS_CHARS: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchQuery {
    pub text: String,
    pub k: usize,
    /// AND; each matches the tag or a child tag (`work` matches `work/proj`), ignoring case.
    pub tags: Vec<String>,
    /// Vault-relative path prefix, compared exactly.
    pub path_prefix: Option<String>,
}

impl SearchQuery {
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            k: DEFAULT_K,
            tags: Vec::new(),
            path_prefix: None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.text.trim().is_empty() {
            return Err(Error::InvalidQuery("the query is empty".into()));
        }
        if !(1..=MAX_K).contains(&self.k) {
            return Err(Error::InvalidQuery(format!(
                "k must be between 1 and {MAX_K} (got {})",
                self.k
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Hit {
    pub path: String,
    pub title: String,
    pub heading_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub body: String,
    pub score: f64,
    pub fts_rank: Option<usize>,
    pub vec_rank: Option<usize>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexState {
    Fresh,
    /// This process's own background indexer is running (MCP).
    Updating,
    UpdatingElsewhere,
    IdleStale,
    ScanError,
}

impl IndexState {
    /// Spec §4.5, in this order. `updating_here` is this process's own background indexer (MCP);
    /// `locked_elsewhere` is asked only when nothing before it decided.
    pub fn classify(
        scan_complete: bool,
        updating_here: bool,
        locked_elsewhere: impl FnOnce() -> Result<bool>,
        stale_files: usize,
    ) -> Result<Self> {
        Ok(if !scan_complete {
            IndexState::ScanError
        } else if updating_here {
            IndexState::Updating
        } else if locked_elsewhere()? {
            IndexState::UpdatingElsewhere
        } else if stale_files == 0 {
            IndexState::Fresh
        } else {
            IndexState::IdleStale
        })
    }

    pub fn label(self) -> &'static str {
        match self {
            IndexState::Fresh => "fresh",
            IndexState::Updating => "updating",
            IndexState::UpdatingElsewhere => "updating_elsewhere",
            IndexState::IdleStale => "idle_stale",
            IndexState::ScanError => "scan_error",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorSearch {
    Ok,
    /// The query could not be embedded; the results come from full-text search alone.
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SearchResponse {
    pub results: Vec<Hit>,
    /// Changes found by the light scan just before the search (spec §4.2).
    pub stale_files: usize,
    pub index_state: IndexState,
    /// The last failure of a background indexer (MCP), or of the update before a CLI search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexer_error: Option<String>,
    /// Remote spec §5.2: a remote API that is down, slow or without its key leaves full-text
    /// search.
    pub vector_search: VectorSearch,
    /// Why `vector_search` is `unavailable`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_error: Option<String>,
}

impl SearchResponse {
    pub fn new(
        found: SearchResults,
        stale_files: usize,
        index_state: IndexState,
        indexer_error: Option<String>,
    ) -> Self {
        Self {
            vector_search: found.vector_search(),
            vector_error: found.vector_error,
            results: found.hits,
            stale_files,
            index_state,
            indexer_error,
        }
    }
}

/// What `search` found, and why the vector side was left out when it was.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResults {
    pub hits: Vec<Hit>,
    pub vector_error: Option<String>,
}

impl SearchResults {
    pub fn vector_search(&self) -> VectorSearch {
        if self.vector_error.is_some() {
            VectorSearch::Unavailable
        } else {
            VectorSearch::Ok
        }
    }
}

/// Opens the index for searching, with its embedding cache attached as `cache`.
pub fn open_reader(db_path: &Path) -> Result<Store> {
    let cache = cache_path(db_path);
    if !cache.is_file() {
        return Err(Error::IndexNotReady);
    }
    let store = Store::open(db_path)?;
    let cache = cache
        .to_str()
        .ok_or_else(|| Error::Config(format!("{} is not valid UTF-8", cache.display())))?;
    store
        .conn()
        .execute("ATTACH DATABASE ?1 AS cache", [cache])?;
    Ok(store)
}

/// `store` must come from `open_reader`; `fingerprint` is this process's index fingerprint.
/// `deadline` bounds embedding the query with a remote model.
pub fn search<E: Embedder + ?Sized>(
    store: &Store,
    embedder: &E,
    fingerprint: &str,
    query: &SearchQuery,
    deadline: Option<Instant>,
) -> Result<SearchResults> {
    query.validate()?;
    let text = query.text.trim();
    // One read transaction over both files (spec §4.5). SQLite pins each WAL file's snapshot at its
    // first read, so the cache goes first: a writer between the two reads can then only add chunks
    // whose vectors this snapshot lacks (they drop out of the vector side), never leave old chunks
    // whose vectors a garbage collection already removed.
    let tx = store.conn().unchecked_transaction()?;
    tx.query_row("SELECT 1 FROM cache.embeddings LIMIT 1", [], |_| Ok(()))
        .optional()?;
    if stored_fingerprint(&tx)?.as_deref() != Some(fingerprint) {
        return Err(Error::IndexNotReady);
    }
    let (vector, vector_error) = match embedder.embed_query_within(text, deadline) {
        Ok(vector) => (Some(vector), None),
        Err(Error::Remote {
            kind: RemoteErrorKind::Transient | RemoteErrorKind::Auth,
            message,
        }) => (None, Some(message)),
        Err(e) => return Err(e),
    };
    if let Some(vector) = &vector
        && embedder.is_remote()
    {
        check_query_dim(&tx, vector.len())?;
    }
    let filter = Filter::new(query);
    let fts = fts_ranking(&tx, text, &filter)?;
    let vec = match &vector {
        Some(vector) => vector_ranking(&tx, vector, &filter)?,
        None => Vec::new(),
    };
    let hits = rrf(&fts, &vec)
        .into_iter()
        .take(query.k)
        .map(|fused| load_hit(&tx, &fused))
        .collect::<Result<Vec<_>>>()?;
    tx.commit()?;
    Ok(SearchResults { hits, vector_error })
}

/// Remote spec §4: the query vector must have the index's dimension. Checked here, so the
/// mismatch is a `Protocol` error rather than one from `vec_distance_cosine`.
fn check_query_dim(conn: &Connection, dim: usize) -> Result<()> {
    let stored: Option<i64> = conn
        .query_row(
            "SELECT length(e.vector) / 4 FROM chunks c
             JOIN cache.embeddings e ON e.key = c.embedding_key LIMIT 1",
            [],
            |r| r.get(0),
        )
        .optional()?;
    match stored {
        Some(stored) if stored as usize != dim => Err(Error::Remote {
            kind: RemoteErrorKind::Protocol,
            message: format!(
                "the API returned a {dim}-dimensional query vector but the index holds {stored}-dimensional ones; set a new model.id to rebuild the index with this model"
            ),
        }),
        _ => Ok(()),
    }
}

fn stored_fingerprint(conn: &Connection) -> Result<Option<String>> {
    let has_meta: i64 = conn.query_row(
        "SELECT count(*) FROM main.sqlite_master WHERE type = 'table' AND name = 'meta'",
        [],
        |r| r.get(0),
    )?;
    if has_meta == 0 {
        return Ok(None);
    }
    Ok(conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'index_fingerprint'",
            [],
            |r| r.get(0),
        )
        .optional()?)
}

/// A WHERE fragment over `chunks c JOIN files f`, applied before candidates are cut (spec §4.5).
struct Filter {
    sql: String,
    params: Vec<Value>,
}

impl Filter {
    fn new(query: &SearchQuery) -> Self {
        let mut sql = String::from("1");
        let mut params = Vec::new();
        for tag in &query.tags {
            // Same folding as the stored tag_key (and as tag deduplication in markdown::tags).
            let key = tag.trim_start_matches('#').to_lowercase();
            let child = format!("{key}/");
            sql.push_str(
                " AND EXISTS (SELECT 1 FROM file_tags t WHERE t.file_id = c.file_id
                  AND (t.tag_key = ? OR substr(t.tag_key, 1, length(?)) = ?))",
            );
            params.push(Value::Text(key));
            params.push(Value::Text(child.clone()));
            params.push(Value::Text(child));
        }
        if let Some(prefix) = &query.path_prefix {
            sql.push_str(" AND substr(f.path, 1, length(?)) = ?");
            params.push(Value::Text(prefix.clone()));
            params.push(Value::Text(prefix.clone()));
        }
        Self { sql, params }
    }
}

fn split_terms(text: &str) -> (Vec<&str>, Vec<&str>) {
    text.split_whitespace()
        .partition(|term| term.chars().count() >= MIN_FTS_CHARS)
}

fn fts_expression(terms: &[&str]) -> String {
    terms
        .iter()
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn escape_like(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if matches!(c, '\\' | '%' | '_') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

fn query_ids(conn: &Connection, sql: &str, params: Vec<Value>) -> Result<Vec<i64>> {
    let mut stmt = conn.prepare(sql)?;
    let ids = stmt
        .query_map(params_from_iter(params), |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(ids)
}

/// BM25 order for long terms, then LIKE matches of short terms not already listed (spec §4.5 step 2).
fn fts_ranking(conn: &Connection, text: &str, filter: &Filter) -> Result<Vec<i64>> {
    let (long, short) = split_terms(text);
    let mut ranked: Vec<i64> = Vec::new();
    if !long.is_empty() {
        let sql = format!(
            "SELECT c.id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
             JOIN files f ON f.id = c.file_id
             WHERE chunks_fts MATCH ? AND {}
             ORDER BY bm25(chunks_fts), c.id LIMIT {CANDIDATES}",
            filter.sql
        );
        let mut params = vec![Value::Text(fts_expression(&long))];
        params.extend(filter.params.iter().cloned());
        ranked = query_ids(conn, &sql, params)?;
    }
    if !short.is_empty() {
        let mut conditions = Vec::new();
        let mut params = Vec::new();
        for term in &short {
            conditions.push(
                "(chunks_fts.title LIKE ? ESCAPE '\\' OR chunks_fts.heading_path LIKE ? ESCAPE '\\'
                  OR chunks_fts.body LIKE ? ESCAPE '\\')",
            );
            let pattern = format!("%{}%", escape_like(term));
            for _ in 0..3 {
                params.push(Value::Text(pattern.clone()));
            }
        }
        let sql = format!(
            "SELECT c.id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
             JOIN files f ON f.id = c.file_id
             WHERE ({}) AND {}
             ORDER BY c.id LIMIT {CANDIDATES}",
            conditions.join(" OR "),
            filter.sql
        );
        params.extend(filter.params.iter().cloned());
        for id in query_ids(conn, &sql, params)? {
            if !ranked.contains(&id) {
                ranked.push(id);
            }
        }
    }
    Ok(ranked)
}

/// Brute-force cosine distance over the attached cache (spec §4.5 step 3).
fn vector_ranking(conn: &Connection, vector: &[f32], filter: &Filter) -> Result<Vec<i64>> {
    let sql = format!(
        "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id
         JOIN cache.embeddings e ON e.key = c.embedding_key
         WHERE {}
         ORDER BY vec_distance_cosine(e.vector, ?), c.id LIMIT {CANDIDATES}",
        filter.sql
    );
    let mut params = filter.params.clone();
    params.push(Value::Blob(f32_blob(vector)));
    query_ids(conn, &sql, params)
}

#[derive(Debug, Clone, PartialEq)]
struct Fused {
    id: i64,
    score: f64,
    fts_rank: Option<usize>,
    vec_rank: Option<usize>,
}

/// `score = Σ 1 / (60 + rank)` over the rankings a chunk appears in; ties go to the lower chunk id.
fn rrf(fts: &[i64], vec: &[i64]) -> Vec<Fused> {
    let mut by_id: HashMap<i64, Fused> = HashMap::new();
    for (list, is_fts) in [(fts, true), (vec, false)] {
        for (i, &id) in list.iter().enumerate() {
            let rank = i + 1;
            let entry = by_id.entry(id).or_insert(Fused {
                id,
                score: 0.0,
                fts_rank: None,
                vec_rank: None,
            });
            entry.score += 1.0 / (RRF_K + rank as f64);
            if is_fts {
                entry.fts_rank = Some(rank);
            } else {
                entry.vec_rank = Some(rank);
            }
        }
    }
    let mut fused: Vec<Fused> = by_id.into_values().collect();
    fused.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
    fused
}

fn load_hit(conn: &Connection, fused: &Fused) -> Result<Hit> {
    let (file_id, mut hit) = conn.query_row(
        "SELECT c.file_id, f.path, f.title, c.heading_path, c.start_line, c.end_line, c.body
         FROM chunks c JOIN files f ON f.id = c.file_id WHERE c.id = ?1",
        [fused.id],
        |r| {
            Ok((
                r.get::<_, i64>(0)?,
                Hit {
                    path: r.get(1)?,
                    title: r.get(2)?,
                    heading_path: r.get(3)?,
                    start_line: r.get::<_, i64>(4)? as usize,
                    end_line: r.get::<_, i64>(5)? as usize,
                    body: r.get(6)?,
                    score: fused.score,
                    fts_rank: fused.fts_rank,
                    vec_rank: fused.vec_rank,
                    tags: Vec::new(),
                },
            ))
        },
    )?;
    let mut stmt =
        conn.prepare_cached("SELECT tag FROM file_tags WHERE file_id = ?1 ORDER BY tag")?;
    hit.tags = stmt
        .query_map([file_id], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(hit)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response() -> SearchResponse {
        let hit = Hit {
            path: "a.md".into(),
            title: "a".into(),
            heading_path: "".into(),
            start_line: 1,
            end_line: 2,
            body: "alpha".into(),
            score: 0.5,
            fts_rank: None,
            vec_rank: Some(1),
            tags: vec!["work".into()],
        };
        SearchResponse {
            results: vec![hit],
            stale_files: 3,
            index_state: IndexState::UpdatingElsewhere,
            indexer_error: None,
            vector_search: VectorSearch::Ok,
            vector_error: None,
        }
    }

    #[test]
    fn response_serializes_the_spec_field_names() {
        let v = serde_json::to_value(response()).unwrap();
        assert_eq!(v["index_state"], "updating_elsewhere");
        assert_eq!(v["stale_files"], 3);
        assert_eq!(v["vector_search"], "ok");
        let first = &v["results"][0];
        for key in [
            "path",
            "title",
            "heading_path",
            "start_line",
            "end_line",
            "body",
            "score",
            "fts_rank",
            "vec_rank",
            "tags",
        ] {
            assert!(first.get(key).is_some(), "{key}");
        }
        assert!(first["fts_rank"].is_null());
    }

    #[test]
    fn absent_errors_are_left_out_of_the_response() {
        let v = serde_json::to_value(response()).unwrap();
        assert!(v.get("indexer_error").is_none());
        assert!(v.get("vector_error").is_none());
    }

    #[test]
    fn errors_and_a_degraded_vector_search_are_serialized() {
        let degraded = SearchResponse {
            indexer_error: Some("boom".into()),
            vector_search: VectorSearch::Unavailable,
            vector_error: Some("down".into()),
            ..response()
        };
        let v = serde_json::to_value(&degraded).unwrap();
        assert_eq!(
            (&v["indexer_error"], &v["vector_search"], &v["vector_error"]),
            (
                &serde_json::json!("boom"),
                &serde_json::json!("unavailable"),
                &serde_json::json!("down")
            )
        );
    }

    #[test]
    fn index_state_asks_about_other_processes_only_when_nothing_before_decides() {
        let never = || -> Result<bool> { panic!("the lock was probed") };
        let state = |complete, here, elsewhere: bool, stale| {
            IndexState::classify(complete, here, || Ok(elsewhere), stale).unwrap()
        };
        assert_eq!(
            IndexState::classify(false, true, never, 0).unwrap(),
            IndexState::ScanError
        );
        assert_eq!(
            IndexState::classify(true, true, never, 0).unwrap(),
            IndexState::Updating
        );
        assert_eq!(state(true, false, true, 0), IndexState::UpdatingElsewhere);
        assert_eq!(state(true, false, false, 0), IndexState::Fresh);
        assert_eq!(state(true, false, false, 2), IndexState::IdleStale);
    }

    #[test]
    fn rrf_sums_both_rankings_and_breaks_ties_by_chunk_id() {
        let fused = rrf(&[10, 20, 30], &[30, 40]);
        let ids: Vec<i64> = fused.iter().map(|f| f.id).collect();
        assert_eq!(ids, [30, 10, 20, 40]);
        assert!((fused[0].score - (1.0 / 63.0 + 1.0 / 61.0)).abs() < 1e-12);
        assert_eq!((fused[0].fts_rank, fused[0].vec_rank), (Some(3), Some(1)));
        assert_eq!((fused[1].fts_rank, fused[1].vec_rank), (Some(1), None));
        assert_eq!((fused[3].fts_rank, fused[3].vec_rank), (None, Some(2)));
    }

    #[test]
    fn terms_of_three_or_more_characters_go_to_fts_and_shorter_ones_to_like() {
        assert_eq!(
            split_terms(" 医療費 税金  ab abc "),
            (vec!["医療費", "abc"], vec!["税金", "ab"])
        );
    }

    #[test]
    fn fts_phrases_are_quoted_with_inner_quotes_doubled() {
        assert_eq!(
            fts_expression(&["医療費", "a\"b"]),
            "\"医療費\" OR \"a\"\"b\""
        );
    }

    #[test]
    fn like_wildcards_and_the_escape_character_are_escaped() {
        assert_eq!(escape_like(r"50%_off\x"), r"50\%\_off\\x");
    }

    #[test]
    fn blank_text_is_rejected() {
        assert!(matches!(
            SearchQuery::new(" \t").validate(),
            Err(Error::InvalidQuery(_))
        ));
    }

    #[test]
    fn k_must_be_from_1_to_50() {
        let ok = SearchQuery::new("税金");
        for k in [0, MAX_K + 1] {
            let q = SearchQuery { k, ..ok.clone() };
            assert!(matches!(q.validate(), Err(Error::InvalidQuery(_))), "{k}");
        }
        for k in [1, MAX_K] {
            assert!(SearchQuery { k, ..ok.clone() }.validate().is_ok(), "{k}");
        }
    }
}
