use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

use notrieve::embed::fake::FakeEmbedder;
use notrieve::indexer::{IndexOptions, IndexReport, index_vault};
use notrieve::store::Store;
use tempfile::TempDir;

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

    fn index_with(
        &self,
        store: &mut Store,
        e: &FakeEmbedder,
        fp: &str,
        exclude: &[&str],
        budget: Option<Duration>,
    ) -> IndexReport {
        let exclude: Vec<String> = exclude.iter().map(|s| s.to_string()).collect();
        let opts = IndexOptions {
            exclude: &exclude,
            chunk_max_tokens: 450,
            budget,
        };
        index_vault(self.vault.path(), store, e, fp, &opts, &mut |_| {}).unwrap()
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
fn indexes_markdown_files_only() {
    let f = Fixture::new();
    f.write("a.md", "# A\nalpha\n");
    f.write("sub/b.md", "# B\nbeta\n");
    f.write("c.txt", "not markdown");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    let r = f.index(&mut s, &e);
    assert_eq!((r.indexed, r.rebuilt, r.scan_complete), (2, true, true));
    assert_eq!(paths(&s), ["a.md", "sub/b.md"]);
}

#[test]
fn notrieve_directory_and_hidden_directories_are_not_indexed() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    f.write(".notrieve/x.md", "internal\n");
    f.write(".obsidian/y.md", "app data\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    assert_eq!(paths(&s), ["a.md"]);
}

#[test]
fn gitignore_is_respected_without_a_git_repository() {
    let f = Fixture::new();
    f.write(".gitignore", "private/\n");
    f.write("a.md", "alpha\n");
    f.write("private/p.md", "secret\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    assert_eq!(paths(&s), ["a.md"]);
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

#[test]
fn fingerprint_change_rebuilds_everything() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    f.write("b.md", "beta\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    let r = f.index_with(&mut s, &e, "fp-b", &[], None);
    assert_eq!((r.rebuilt, r.indexed), (true, 2));
}

#[test]
fn dimension_change_recreates_the_vector_table() {
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    let mut s = f.store();
    f.index_with(&mut s, &FakeEmbedder::new(8), "fp-a", &[], None);
    f.index_with(&mut s, &FakeEmbedder::new(16), "fp-b", &[], None);
    let len: i64 = s
        .conn()
        .query_row("SELECT vec_length(embedding) FROM chunks_vec", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(len, 16);
}

#[test]
fn tags_aliases_and_line_numbers_are_stored() {
    let f = Fixture::new();
    f.write(
        "税金メモ.md",
        "---\ntags: [税金]\naliases: [tax]\n---\n# 医療費\n領収書 #確定申告\n",
    );
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
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
    let (title, aliases): (String, String) = s
        .conn()
        .query_row("SELECT title, aliases FROM files", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!((title.as_str(), aliases.as_str()), ("税金メモ", "tax"));
    let lines: (i64, i64) = s
        .conn()
        .query_row("SELECT start_line, end_line FROM chunks", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(lines, (5, 6));
}

#[test]
fn stale_files_counts_changes_without_hashing() {
    use notrieve::indexer::scan::{plan_changes, scan_vault};
    let f = Fixture::new();
    f.write("a.md", "alpha\n");
    f.write("b.md", "beta\n");
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    f.write("c.md", "new\n");
    fs::remove_file(f.path("b.md")).unwrap();
    let scan = scan_vault(Path::new(f.vault.path()), &[]).unwrap();
    assert_eq!(
        plan_changes(&scan, &s.file_states().unwrap()).stale_files(),
        2
    );
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
    };
    let result = index_vault(f.vault.path(), &mut s, &e, "fp-b", &opts, &mut |_| {});
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
    };
    let result = index_vault(f.vault.path(), &mut s, &e, "fp-a", &opts, &mut |_| {});
    assert!(result.is_err());
    assert!(paths(&s).is_empty());
    f.write(".gitignore", "private/\n");
    let r = f.index(&mut s, &e);
    assert!(r.scan_complete);
    assert_eq!(paths(&s), ["a.md"]);
}

#[cfg(unix)]
#[test]
fn non_utf8_file_names_are_skipped_instead_of_colliding() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    let f = Fixture::new();
    f.write("ok.md", "fine\n");
    for name in [&b"a\x80.md"[..], &b"a\x81.md"[..]] {
        if fs::write(f.vault.path().join(OsStr::from_bytes(name)), "x\n").is_err() {
            eprintln!("skipped: filesystem refuses non-UTF-8 names");
            return;
        }
    }
    let (mut s, e) = (f.store(), FakeEmbedder::new(8));
    f.index(&mut s, &e);
    assert_eq!(paths(&s), ["ok.md"]);
}

#[test]
fn malformed_parent_gitignore_is_an_error() {
    let parent = TempDir::new().unwrap();
    fs::write(parent.path().join(".gitignore"), "{unclosed\n").unwrap();
    let vault = parent.path().join("vault");
    fs::create_dir(&vault).unwrap();
    fs::write(vault.join("a.md"), "alpha\n").unwrap();
    let data = TempDir::new().unwrap();
    let mut s = Store::open(&data.path().join("index.db")).unwrap();
    let e = FakeEmbedder::new(8);
    let opts = IndexOptions {
        exclude: &[],
        chunk_max_tokens: 450,
        budget: None,
    };
    let result = index_vault(&vault, &mut s, &e, "fp-a", &opts, &mut |_| {});
    assert!(result.is_err());
    assert!(paths(&s).is_empty());
}
