use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use notrieve::Error;
use notrieve::cache::EmbeddingCache;
use notrieve::embed::Embedder;
use notrieve::embed::fake::FakeEmbedder;
use notrieve::fingerprint::Fingerprints;
use notrieve::indexer::{IndexEvent, IndexOptions, IndexReport, index_vault};
use notrieve::store::{RebuildReason, Store};
use tempfile::TempDir;

fn fps(model: &str, index: &str) -> Fingerprints {
    Fingerprints {
        model: model.into(),
        index: index.into(),
    }
}

struct Fixture {
    vault: TempDir,
    data: TempDir,
}

impl Fixture {
    fn new() -> Self {
        Self {
            vault: TempDir::new().unwrap(),
            data: TempDir::new().unwrap(),
        }
    }

    fn write(&self, rel: &str, text: &str) {
        let path = self.vault.path().join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    fn path(&self, rel: &str) -> std::path::PathBuf {
        self.vault.path().join(rel)
    }

    fn store(&self) -> Store {
        Store::open(&self.data.path().join("index.db")).unwrap()
    }

    fn cache_path(&self) -> PathBuf {
        notrieve::cache::cache_path(&self.data.path().join("index.db"))
    }

    fn cache(&self) -> EmbeddingCache {
        EmbeddingCache::open(&self.cache_path()).unwrap()
    }

    /// Chunks whose embedding_key has no vector in the cache.
    fn dangling_keys(&self, store: &Store) -> i64 {
        let conn = store.conn();
        conn.execute(
            "ATTACH DATABASE ?1 AS cache",
            [self.cache_path().to_str().unwrap()],
        )
        .unwrap();
        let n = conn
            .query_row(
                "SELECT count(*) FROM chunks c LEFT JOIN cache.embeddings e
                 ON e.key = c.embedding_key WHERE e.key IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        conn.execute("DETACH DATABASE cache", []).unwrap();
        n
    }

    fn run(
        &self,
        store: &mut Store,
        e: &FakeEmbedder,
        fps: &Fingerprints,
        exclude: &[&str],
        budget: Option<Duration>,
        rebuild: bool,
    ) -> IndexReport {
        let exclude: Vec<String> = exclude.iter().map(|s| s.to_string()).collect();
        let opts = IndexOptions {
            exclude: &exclude,
            chunk_max_tokens: 450,
            budget,
            deadline: None,
            rebuild,
        };
        let mut cache = self.cache();
        index_vault(
            self.vault.path(),
            store,
            &mut cache,
            e,
            fps,
            &opts,
            &mut |_| {},
        )
        .unwrap()
    }

    fn index_with(
        &self,
        store: &mut Store,
        e: &FakeEmbedder,
        fp: &str,
        exclude: &[&str],
        budget: Option<Duration>,
    ) -> IndexReport {
        self.run(store, e, &fps("model-a", fp), exclude, budget, false)
    }

    fn index(&self, store: &mut Store, e: &FakeEmbedder) -> IndexReport {
        self.index_with(store, e, "fp-a", &[], None)
    }
}

fn paths(store: &Store) -> Vec<String> {
    let mut v: Vec<String> = store.file_states().unwrap().into_keys().collect();
    v.sort();
    v
}

#[test]
fn unchanged_files_are_not_embedded_again() {
    let f = Fixture::new();
    f.write("a.md", "# A\nalpha\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    let embedded = e.embedded_count();
    let r = f.index(&mut s, &e);
    assert_eq!((r.indexed, r.touched, r.rebuilt), (0, 0, false));
    assert_eq!(e.embedded_count(), embedded);
}

#[test]
fn touched_file_with_same_content_only_updates_its_timestamp() {
    let f = Fixture::new();
    f.write("a.md", "# A\nalpha\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    let later = SystemTime::now() + Duration::from_secs(3600);
    fs::File::options()
        .write(true)
        .open(f.path("a.md"))
        .unwrap()
        .set_modified(later)
        .unwrap();
    let r = f.index(&mut s, &e);
    assert_eq!((r.touched, r.indexed), (1, 0));
}

#[test]
fn changed_file_is_reindexed() {
    let f = Fixture::new();
    f.write("a.md", "# A\nalpha\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    f.write("a.md", "# A\ngamma delta\n");
    let r = f.index(&mut s, &e);
    assert_eq!(r.indexed, 1);
    let body: String = s
        .conn()
        .query_row("SELECT body FROM chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(body, "# A\ngamma delta");
}

#[test]
fn deleted_file_is_removed() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    f.write("b.md", "beta\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    fs::remove_file(f.path("a.md")).unwrap();
    let r = f.index(&mut s, &e);
    assert_eq!(r.deleted, 1);
    assert_eq!(paths(&s), ["b.md"]);
}

#[test]
fn newly_excluded_file_is_removed() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    f.write("drafts/d.md", "draft\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    let r = f.index_with(&mut s, &e, "fp-a", &["drafts/**"], None);
    assert_eq!(r.deleted, 1);
    assert_eq!(paths(&s), ["a.md"]);
}

#[test]
fn moving_a_file_without_renaming_it_keeps_its_embeddings() {
    let f = Fixture::new();
    f.write("a.md", "# A\nalpha\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    let embedded = e.embedded_count();
    fs::create_dir_all(f.path("archive")).unwrap();
    fs::rename(f.path("a.md"), f.path("archive/a.md")).unwrap();
    let r = f.index(&mut s, &e);
    assert_eq!((r.renamed, r.indexed, r.deleted), (1, 0, 0));
    assert_eq!(e.embedded_count(), embedded);
    assert_eq!(paths(&s), ["archive/a.md"]);
}

#[test]
fn renaming_a_file_reembeds_it_because_the_title_changes() {
    let f = Fixture::new();
    f.write("a.md", "# A\nalpha\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    fs::rename(f.path("a.md"), f.path("b.md")).unwrap();
    let r = f.index(&mut s, &e);
    assert_eq!((r.renamed, r.indexed, r.deleted), (0, 1, 1));
    let title: String = s
        .conn()
        .query_row("SELECT title FROM files", [], |r| r.get(0))
        .unwrap();
    assert_eq!(title, "b");
}

#[cfg(unix)]
#[test]
fn unreadable_directory_prevents_deletions() {
    use std::os::unix::fs::PermissionsExt;
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    f.write("locked/x.md", "x\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    fs::remove_file(f.path("a.md")).unwrap();
    let locked = f.path("locked");
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
    if fs::read_dir(&locked).is_ok() {
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
        eprintln!("skipped: permissions are not enforced for this user");
        return;
    }
    let r = f.index(&mut s, &e);
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(!r.scan_complete);
    assert_eq!(r.deleted, 0);
    assert_eq!(paths(&s), ["a.md", "locked/x.md"]);
}

#[test]
fn exhausted_budget_starts_no_file_work_and_defers_deletions() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    f.write("b.md", "beta\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    f.write("a.md", "alpha changed\n");
    fs::remove_file(f.path("b.md")).unwrap();
    f.write("c.md", "gamma\n");
    let embedded = e.embedded_count();
    let r = f.index_with(&mut s, &e, "fp-a", &[], Some(Duration::ZERO));
    assert_eq!(
        (r.indexed, r.touched, r.renamed, r.deleted, r.remaining),
        (0, 0, 0, 0, 3)
    );
    assert_eq!(e.embedded_count(), embedded);
    assert_eq!(paths(&s), ["a.md", "b.md"]);
    let r = f.index_with(&mut s, &e, "fp-a", &[], None);
    assert_eq!((r.indexed, r.deleted, r.remaining), (2, 1, 0));
    assert_eq!(paths(&s), ["a.md", "c.md"]);
}

#[test]
fn deletions_alone_also_wait_for_budget() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    f.write("b.md", "beta\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    fs::remove_file(f.path("b.md")).unwrap();
    let r = f.index_with(&mut s, &e, "fp-a", &[], Some(Duration::ZERO));
    assert_eq!((r.deleted, r.remaining), (0, 1));
    assert_eq!(paths(&s), ["a.md", "b.md"]);
    let r = f.index_with(&mut s, &e, "fp-a", &[], None);
    assert_eq!((r.deleted, r.remaining), (1, 0));
}

#[test]
fn embedding_failure_skips_only_that_file() {
    let f = Fixture::new();
    f.write("ok.md", "fine\n");
    f.write("bad.md", "contains BOOM\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8).failing_on("BOOM"));
    let r = f.index(&mut s, &e);
    assert_eq!(r.indexed, 1);
    assert_eq!(
        r.skipped
            .iter()
            .map(|s| s.path.as_str())
            .collect::<Vec<_>>(),
        ["bad.md"]
    );
    assert_eq!(paths(&s), ["ok.md"]);
}

#[test]
fn non_utf8_file_is_skipped() {
    let f = Fixture::new();
    fs::write(f.path("latin1.md"), [0x63, 0x61, 0x66, 0xe9, 0x0a]).unwrap();
    f.write("ok.md", "fine\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    let r = f.index(&mut s, &e);
    assert_eq!((r.indexed, r.skipped.len()), (1, 1));
    assert_eq!(r.skipped[0].path, "latin1.md");
}

fn indexed_tax_note() -> (Fixture, Store) {
    let f = Fixture::new();
    f.write(
        "税金メモ.md",
        "---\ntags: [税金]\naliases: [tax]\n---\n# 医療費\n領収書 #確定申告\n",
    );
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    (f, s)
}

#[test]
fn frontmatter_and_inline_tags_are_stored() {
    let (_f, s) = indexed_tax_note();
    let mut tags: Vec<String> = s
        .conn()
        .prepare("SELECT tag FROM file_tags")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    tags.sort();
    assert_eq!(tags, ["確定申告", "税金"]);
}

#[test]
fn title_and_aliases_are_stored() {
    let (_f, s) = indexed_tax_note();
    let (title, aliases): (String, String) = s
        .conn()
        .query_row("SELECT title, aliases FROM files", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!((title.as_str(), aliases.as_str()), ("税金メモ", "tax"));
}

#[test]
fn chunk_line_numbers_count_the_frontmatter() {
    let (_f, s) = indexed_tax_note();
    let lines: (i64, i64) = s
        .conn()
        .query_row("SELECT start_line, end_line FROM chunks", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(lines, (5, 6));
}

#[test]
fn invalid_exclude_fails_before_a_fingerprint_rebuild_wipes_the_index() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    let bad = vec!["[".to_string()];
    let opts = IndexOptions {
        exclude: &bad,
        chunk_max_tokens: 450,
        budget: None,
        deadline: None,
        rebuild: false,
    };
    let result = index_vault(
        f.vault.path(),
        &mut s,
        &mut f.cache(),
        &e,
        &fps("model-a", "fp-b"),
        &opts,
        &mut |_| {},
    );
    assert!(result.is_err());
    assert_eq!(paths(&s), ["a.md"]);
    let r = f.index_with(&mut s, &e, "fp-b", &["drafts/**"], None);
    assert_eq!((r.rebuilt, r.indexed), (true, 1));
}

#[test]
fn malformed_gitignore_line_is_an_error_and_well_formed_one_is_complete() {
    let f = Fixture::new();
    f.write(".gitignore", "private/\n{unclosed\n");
    f.write("a.md", "alpha\n");
    f.write("private/p.md", "secret\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    let opts = IndexOptions {
        exclude: &[],
        chunk_max_tokens: 450,
        budget: None,
        deadline: None,
        rebuild: false,
    };
    let result = index_vault(
        f.vault.path(),
        &mut s,
        &mut f.cache(),
        &e,
        &fps("model-a", "fp-a"),
        &opts,
        &mut |_| {},
    );
    assert!(result.is_err());
    assert!(paths(&s).is_empty());
    f.write(".gitignore", "private/\n");
    let r = f.index(&mut s, &e);
    assert!(r.scan_complete);
    assert_eq!(paths(&s), ["a.md"]);
}

#[test]
fn rebuild_reason_tells_a_model_change_from_a_settings_change() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    let first = f.run(&mut s, &e, &fps("model-a", "fp-a"), &[], None, false);
    assert_eq!(first.rebuild_reason, Some(RebuildReason::NewIndex));
    let settings = f.run(&mut s, &e, &fps("model-a", "fp-b"), &[], None, false);
    assert_eq!(
        settings.rebuild_reason,
        Some(RebuildReason::SettingsChanged)
    );
    let model = f.run(&mut s, &e, &fps("model-b", "fp-c"), &[], None, false);
    assert_eq!(model.rebuild_reason, Some(RebuildReason::ModelChanged));
    let unchanged = f.run(&mut s, &e, &fps("model-b", "fp-c"), &[], None, false);
    assert_eq!((unchanged.rebuilt, unchanged.rebuild_reason), (false, None));
}

#[test]
fn explicit_rebuild_with_matching_fingerprints_is_reported_as_requested() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    let r = f.run(&mut s, &e, &fps("model-a", "fp-a"), &[], None, true);
    assert_eq!(
        (r.rebuilt, r.rebuild_reason, r.indexed),
        (true, Some(RebuildReason::Requested), 1)
    );
}

#[test]
fn rebuilding_is_announced_before_any_file_starts() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    f.write("b.md", "beta\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    let opts = IndexOptions {
        exclude: &[],
        chunk_max_tokens: 450,
        budget: None,
        deadline: None,
        rebuild: false,
    };
    let mut events: Vec<String> = Vec::new();
    index_vault(
        f.vault.path(),
        &mut s,
        &mut f.cache(),
        &e,
        &fps("model-a", "fp-a"),
        &opts,
        &mut |event| {
            events.push(match event {
                IndexEvent::Rebuilding(reason) => format!("rebuild:{reason:?}"),
                IndexEvent::File(p) => format!("file:{}", p.path),
                IndexEvent::SendingToRemote { host } => format!("send:{host}"),
            })
        },
    )
    .unwrap();
    assert_eq!(events, ["rebuild:NewIndex", "file:a.md", "file:b.md"]);
}

#[test]
fn invalid_exclude_with_an_explicit_rebuild_keeps_the_index() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    let bad = vec!["[".to_string()];
    let opts = IndexOptions {
        exclude: &bad,
        chunk_max_tokens: 450,
        budget: None,
        deadline: None,
        rebuild: true,
    };
    let result = index_vault(
        f.vault.path(),
        &mut s,
        &mut f.cache(),
        &e,
        &fps("model-a", "fp-a"),
        &opts,
        &mut |_| {},
    );
    assert!(result.is_err());
    assert_eq!(paths(&s), ["a.md"]);
}

/// Two sections long enough not to be merged (the fake embedder counts characters).
const TWO_SECTIONS: &str = "# 医療費\n医療費控除の対象になる費用と領収書の保管方法をまとめておく。\n# 年金\n年金の受給額と手続きの時期を確認して、必要な書類をそろえる。\n";

#[test]
fn settings_change_rebuild_embeds_nothing_again() {
    let f = Fixture::new();
    f.write("a.md", TWO_SECTIONS);
    f.write(
        "b.md",
        "# B\nbeta gamma delta epsilon zeta eta theta iota\n",
    );
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.run(&mut s, &e, &fps("model-a", "fp-a"), &[], None, false);
    assert_eq!(s.stats().unwrap(), (2, 3));
    let embedded = e.embedded_count();
    let r = f.run(&mut s, &e, &fps("model-a", "fp-b"), &[], None, false);
    assert_eq!(
        (r.rebuild_reason, r.indexed),
        (Some(RebuildReason::SettingsChanged), 2)
    );
    assert_eq!(e.embedded_count(), embedded);
    assert_eq!(f.dangling_keys(&s), 0);
}

#[test]
fn schema_change_rebuild_embeds_nothing_again() {
    let f = Fixture::new();
    f.write("a.md", TWO_SECTIONS);
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    s.conn()
        .execute(
            "UPDATE meta SET value = '1' WHERE key = 'schema_version'",
            [],
        )
        .unwrap();
    let embedded = e.embedded_count();
    let r = f.index(&mut s, &e);
    assert_eq!(
        (r.rebuild_reason, r.indexed),
        (Some(RebuildReason::SchemaChanged), 1)
    );
    assert_eq!(e.embedded_count(), embedded);
}

#[test]
fn only_the_changed_chunk_is_embedded_again() {
    let f = Fixture::new();
    f.write("a.md", TWO_SECTIONS);
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    let before = e.embedded_count();
    f.write("a.md", &TWO_SECTIONS.replace("必要な書類", "必要な証明書"));
    let r = f.index(&mut s, &e);
    assert_eq!(r.indexed, 1);
    assert_eq!(e.embedded_count() - before, 1);
}

#[test]
fn model_change_embeds_every_chunk_again() {
    let f = Fixture::new();
    f.write("a.md", TWO_SECTIONS);
    f.write(
        "b.md",
        "# B\nbeta gamma delta epsilon zeta eta theta iota\n",
    );
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    let before = e.embedded_count();
    let r = f.run(&mut s, &e, &fps("model-b", "fp-b"), &[], None, false);
    assert_eq!(r.rebuild_reason, Some(RebuildReason::ModelChanged));
    assert_eq!(e.embedded_count() - before, 3);
}

#[test]
fn unreferenced_vectors_are_removed_after_a_complete_run() {
    let f = Fixture::new();
    f.write("a.md", TWO_SECTIONS);
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    let old_keys = s.embedding_keys().unwrap();
    f.write("a.md", &TWO_SECTIONS.replace("必要な書類", "必要な証明書"));
    f.index(&mut s, &e);
    assert_eq!(f.cache().entry_count().unwrap(), 2);
    f.run(&mut s, &e, &fps("model-b", "fp-b"), &[], None, false);
    let cache = f.cache();
    assert_eq!(
        cache.entry_count().unwrap(),
        s.embedding_keys().unwrap().len()
    );
    for key in &old_keys {
        assert_eq!(cache.vector_dim(key).unwrap(), None);
    }
}

#[test]
fn budget_stopped_rebuild_keeps_the_vectors_of_files_not_yet_reindexed() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    f.write("b.md", "beta\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    let stored = f.cache().entry_count().unwrap();
    let r = f.run(
        &mut s,
        &e,
        &fps("model-a", "fp-b"),
        &[],
        Some(Duration::ZERO),
        false,
    );
    assert_eq!((r.rebuilt, r.indexed, r.remaining), (true, 0, 2));
    assert_eq!(f.cache().entry_count().unwrap(), stored);
    let embedded = e.embedded_count();
    f.run(&mut s, &e, &fps("model-a", "fp-b"), &[], None, false);
    assert_eq!(e.embedded_count(), embedded);
}

/// Returns one vector fewer than it is asked for.
struct ShortEmbedder(FakeEmbedder);

impl Embedder for ShortEmbedder {
    fn model_id(&self) -> &str {
        self.0.model_id()
    }

    fn dim(&self) -> usize {
        self.0.dim()
    }

    fn embed_documents(&self, texts: &[String]) -> notrieve::Result<Vec<Vec<f32>>> {
        let mut vectors = self.0.embed_documents(texts)?;
        vectors.pop();
        Ok(vectors)
    }

    fn embed_query(&self, text: &str) -> notrieve::Result<Vec<f32>> {
        self.0.embed_query(text)
    }

    fn count_tokens(&self, text: &str) -> notrieve::Result<usize> {
        self.0.count_tokens(text)
    }
}

#[test]
fn embedder_returning_too_few_vectors_skips_the_file_and_writes_no_index_rows() {
    let f = Fixture::new();
    f.write("a.md", TWO_SECTIONS);
    let mut s = f.store();
    let e = ShortEmbedder(FakeEmbedder::new(8));
    let opts = IndexOptions {
        exclude: &[],
        chunk_max_tokens: 450,
        budget: None,
        deadline: None,
        rebuild: false,
    };
    let r = index_vault(
        f.vault.path(),
        &mut s,
        &mut f.cache(),
        &e,
        &fps("model-a", "fp-a"),
        &opts,
        &mut |_| {},
    )
    .unwrap();
    assert_eq!((r.indexed, r.skipped.len()), (0, 1));
    assert!(
        r.skipped[0].reason.contains("1 vectors for 2 texts"),
        "{}",
        r.skipped[0].reason
    );
    assert_eq!(s.stats().unwrap(), (0, 0));
    assert_eq!(f.dangling_keys(&s), 0);
}

fn remove_cache_files(f: &Fixture) {
    for suffix in ["", "-wal", "-shm"] {
        let mut path = f.cache_path().into_os_string();
        path.push(suffix);
        let _ = fs::remove_file(path);
    }
}

#[test]
fn deleted_cache_is_recreated_and_every_chunk_embedded_again() {
    let f = Fixture::new();
    f.write("a.md", TWO_SECTIONS);
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    remove_cache_files(&f);
    let before = e.embedded_count();
    let r = f.index(&mut s, &e);
    assert_eq!(r.rebuild_reason, Some(RebuildReason::CacheReset));
    assert_eq!(e.embedded_count() - before, 2);
    assert_eq!(f.dangling_keys(&s), 0);
}

#[test]
fn failure_after_the_cache_write_leaves_no_dangling_key() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    // index.db rejects the new chunk after its vector has been committed to the cache.
    s.conn()
        .execute_batch(
            "CREATE TRIGGER reject BEFORE INSERT ON chunks WHEN NEW.body LIKE '%EXPLODE%'
             BEGIN SELECT RAISE(ABORT, 'rejected'); END;",
        )
        .unwrap();
    f.write("a.md", "alpha EXPLODE\n");
    let opts = IndexOptions {
        exclude: &[],
        chunk_max_tokens: 450,
        budget: None,
        deadline: None,
        rebuild: false,
    };
    let err = index_vault(
        f.vault.path(),
        &mut s,
        &mut f.cache(),
        &e,
        &fps("model-a", "fp-a"),
        &opts,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(matches!(err, Error::Sqlite(_)), "{err}");
    assert_eq!(f.dangling_keys(&s), 0);
    assert_eq!(f.cache().entry_count().unwrap(), 2);
    s.conn().execute_batch("DROP TRIGGER reject;").unwrap();
    let embedded = e.embedded_count();
    f.index(&mut s, &e);
    assert_eq!(e.embedded_count(), embedded);
    assert_eq!(f.cache().entry_count().unwrap(), 1);
}

#[test]
fn vectors_orphaned_by_a_failed_run_are_removed_even_when_nothing_is_reindexed() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    s.conn()
        .execute_batch(
            "CREATE TRIGGER reject BEFORE INSERT ON chunks WHEN NEW.body LIKE '%EXPLODE%'
             BEGIN SELECT RAISE(ABORT, 'rejected'); END;",
        )
        .unwrap();
    f.write("a.md", "alpha EXPLODE\n");
    let opts = IndexOptions {
        exclude: &[],
        chunk_max_tokens: 450,
        budget: None,
        deadline: None,
        rebuild: false,
    };
    let failed = index_vault(
        f.vault.path(),
        &mut s,
        &mut f.cache(),
        &e,
        &fps("model-a", "fp-a"),
        &opts,
        &mut |_| {},
    );
    assert!(failed.is_err());
    assert_eq!(f.cache().entry_count().unwrap(), 2);
    // The note goes back to the indexed content, so the next run re-indexes nothing.
    f.write("a.md", "alpha\n");
    let r = f.index(&mut s, &e);
    assert_eq!((r.indexed, r.rebuilt), (0, false));
    assert_eq!(f.cache().entry_count().unwrap(), 1);
    assert_eq!(f.dangling_keys(&s), 0);
}

#[test]
fn cache_recreated_before_a_failed_run_is_still_rebuilt_by_the_next_run() {
    let f = Fixture::new();
    f.write("a.md", TWO_SECTIONS);
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    remove_cache_files(&f);
    // This run creates the new cache, then fails on the exclude rule before it can rebuild.
    let bad = vec!["[".to_string()];
    let opts = IndexOptions {
        exclude: &bad,
        chunk_max_tokens: 450,
        budget: None,
        deadline: None,
        rebuild: false,
    };
    let failed = index_vault(
        f.vault.path(),
        &mut s,
        &mut f.cache(),
        &e,
        &fps("model-a", "fp-a"),
        &opts,
        &mut |_| {},
    );
    assert!(failed.is_err());
    assert!(f.cache_path().exists());
    let before = e.embedded_count();
    let r = f.index(&mut s, &e);
    assert_eq!(r.rebuild_reason, Some(RebuildReason::CacheReset));
    assert_eq!(e.embedded_count() - before, 2);
    assert_eq!(f.dangling_keys(&s), 0);
}

#[test]
fn dimension_change_stores_vectors_of_the_new_dimension() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    let mut s = f.store();
    f.run(
        &mut s,
        &FakeEmbedder::new(8),
        &fps("model-a", "fp-a"),
        &[],
        None,
        false,
    );
    f.run(
        &mut s,
        &FakeEmbedder::new(16),
        &fps("model-b", "fp-b"),
        &[],
        None,
        false,
    );
    let lengths: Vec<i64> = f
        .cache()
        .conn()
        .prepare("SELECT length(vector) FROM embeddings")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(lengths, [64]);
}

fn skipped_paths(r: &IndexReport) -> Vec<(&str, &str)> {
    r.skipped
        .iter()
        .map(|s| (s.path.as_str(), s.reason.as_str()))
        .collect()
}

#[test]
fn unchanged_skipped_file_is_not_retried_but_stays_in_the_report() {
    let f = Fixture::new();
    f.write("ok.md", "fine\n");
    f.write("bad.md", "contains BOOM\n");
    fs::write(f.path("latin1.md"), [0xff, 0xfe, 0x00]).unwrap();
    let (mut s, e) = (f.store(), FakeEmbedder::new(8).failing_on("BOOM"));
    let first = f.index(&mut s, &e);
    assert_eq!(first.skipped.len(), 2);
    let opts = IndexOptions {
        exclude: &[],
        chunk_max_tokens: 450,
        budget: None,
        deadline: None,
        rebuild: false,
    };
    let mut attempted: Vec<String> = Vec::new();
    let second = index_vault(
        f.vault.path(),
        &mut s,
        &mut f.cache(),
        &e,
        &fps("model-a", "fp-a"),
        &opts,
        &mut |ev| {
            if let IndexEvent::File(p) = ev {
                attempted.push(p.path.to_string());
            }
        },
    )
    .unwrap();
    assert_eq!(attempted, Vec::<String>::new());
    assert_eq!(second.indexed, 0);
    assert_eq!(second.skipped, first.skipped);
    assert_eq!(
        second
            .skipped
            .iter()
            .map(|s| s.path.as_str())
            .collect::<Vec<_>>(),
        ["bad.md", "latin1.md"]
    );
    assert!(second.skipped[0].reason.contains("BOOM"));
}

#[test]
fn deleted_skipped_file_leaves_the_report_and_the_table() {
    let f = Fixture::new();
    f.write("ok.md", "fine\n");
    fs::write(f.path("latin1.md"), [0xff, 0xfe, 0x00]).unwrap();
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    assert_eq!(f.index(&mut s, &e).skipped.len(), 1);
    fs::remove_file(f.path("latin1.md")).unwrap();
    let r = f.index(&mut s, &e);
    assert_eq!(skipped_paths(&r), []);
    assert!(s.skipped_states().unwrap().is_empty());
}

#[test]
fn rebuild_retries_skipped_files() {
    let f = Fixture::new();
    f.write("bad.md", "contains BOOM\n");
    let mut s = f.store();
    let failing = FakeEmbedder::new(8).failing_on("BOOM");
    assert_eq!(f.index(&mut s, &failing).skipped.len(), 1);
    // Only a rebuild may re-attempt it: the file is unchanged on disk.
    let working = FakeEmbedder::new(8);
    let r = f.run(&mut s, &working, &fps("model-a", "fp-a"), &[], None, true);
    assert_eq!((r.indexed, r.skipped.len()), (1, 0));
    assert_eq!(paths(&s), ["bad.md"]);
    assert!(s.skipped_states().unwrap().is_empty());
}

#[test]
fn changed_skipped_file_is_retried_and_its_row_removed() {
    let f = Fixture::new();
    f.write("ok.md", "fine\n");
    fs::write(f.path("bad.md"), [0xff, 0xfe, 0x00]).unwrap();
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    f.write("bad.md", "now valid text\n");
    let r = f.index(&mut s, &e);
    assert_eq!((r.indexed, r.skipped.len()), (1, 0));
    assert!(s.skipped_states().unwrap().is_empty());
}

#[test]
fn previously_indexed_file_that_becomes_unindexable_loses_its_index_rows() {
    let f = Fixture::new();
    f.write("a.md", "uniqueoldword here\n");
    f.write("ok.md", "fine\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    fs::write(f.path("a.md"), [0xff, 0xfe, 0x00]).unwrap();
    let r = f.index(&mut s, &e);
    assert_eq!(
        skipped_paths(&r).iter().map(|p| p.0).collect::<Vec<_>>(),
        ["a.md"]
    );
    assert_eq!(paths(&s), ["ok.md"]);
    let hits: i64 = s
        .conn()
        .query_row(
            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH '\"uniqueoldword\"'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(hits, 0);
    assert_eq!(s.stats().unwrap().0, 1);
}

#[test]
fn stored_skips_stay_in_the_report_when_the_run_is_stopped_by_the_budget() {
    let f = Fixture::new();
    f.write("ok.md", "fine\n");
    fs::write(f.path("bad.md"), [0xff, 0xfe, 0x00]).unwrap();
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    fs::remove_file(f.path("bad.md")).unwrap();
    f.write("new.md", "gamma\n");
    let r = f.index_with(&mut s, &e, "fp-a", &[], Some(Duration::ZERO));
    assert_eq!(r.remaining, 1);
    assert_eq!(
        skipped_paths(&r).iter().map(|p| p.0).collect::<Vec<_>>(),
        ["bad.md"]
    );
}

#[test]
fn refailure_of_a_changed_skipped_file_overwrites_the_stored_state() {
    let f = Fixture::new();
    f.write("bad.md", "contains BOOM\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8).failing_on("BOOM"));
    f.index(&mut s, &e);
    let before = s.skipped_states().unwrap()["bad.md"].clone();
    fs::write(f.path("bad.md"), [0xff, 0xfe, 0x00, 0x01]).unwrap();
    f.index(&mut s, &e);
    let after = s.skipped_states().unwrap()["bad.md"].clone();
    assert_eq!(after.size, 4);
    assert_ne!(after.size, before.size);
    assert_ne!(after.reason, before.reason);
}
