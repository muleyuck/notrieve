mod support;

use std::sync::Arc;

use notrieve::cache::cache_path;
use notrieve::embed::fake::FakeEmbedder;
use notrieve::lock;
use notrieve::search::{IndexState, SearchQuery};
use notrieve::session::Session;
use notrieve::{Error, RemoteErrorKind};
use support::{
    FlakyRemote, GateEmbedder, PanicEmbedder, fps, index_as_another_process, index_with,
    indexed_paths, vault, wait_for, write,
};

fn q(text: &str) -> SearchQuery {
    SearchQuery::new(text)
}

fn fake() -> Arc<FakeEmbedder> {
    Arc::new(FakeEmbedder::new(16))
}

#[test]
fn search_after_the_first_sync_finds_notes_in_a_fresh_index() {
    let (_tmp, vault) = vault();
    write(
        &vault,
        "税金.md",
        "# 医療費\n医療費控除の領収書をまとめる。\n",
    );
    write(&vault, "旅行.md", "# 京都\n紅葉の季節に嵐山を歩いた。\n");
    let session = Session::with_embedder(vault, fake(), fps("a")).unwrap();
    let r = wait_for(|| {
        session
            .search(&q("医療費控除"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    assert_eq!(r.stale_files, 0);
    assert_eq!(r.results[0].path, "税金.md");
    assert_eq!(r.indexer_error, None);
}

#[test]
fn search_during_the_startup_rebuild_is_a_rebuilding_error_with_the_files_left() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    write(&vault, "b.md", "# B\nbeta\n");
    let gate = Arc::new(GateEmbedder::new(false));
    let session = Session::with_embedder(vault, gate.clone(), fps("a")).unwrap();
    wait_for(|| (gate.entered() >= 1).then_some(()));
    let err = session.search(&q("alpha")).unwrap_err();
    assert!(
        matches!(err, Error::Rebuilding { remaining: Some(2) }),
        "{err}"
    );
    assert!(err.to_string().contains("2 files left"), "{err}");
    gate.set_open(true);
    let r = wait_for(|| session.search(&q("alpha")).ok());
    assert_eq!(r.results[0].path, "a.md");
}

#[test]
fn startup_with_the_lock_held_elsewhere_is_a_lock_busy_error_until_it_is_released() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let session = Session::with_embedder(vault, fake(), fps("a")).unwrap();
    wait_for(|| match session.search(&q("alpha")) {
        Err(Error::LockBusy) => Some(()),
        _ => None,
    });
    drop(held);
    let r = wait_for(|| session.search(&q("alpha")).ok());
    assert_eq!(r.results[0].path, "a.md");
}

#[test]
fn a_new_note_is_reported_stale_and_then_indexed_in_the_background() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let session = Session::with_embedder(vault.clone(), fake(), fps("a")).unwrap();
    wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    write(&vault, "b.md", "# B\nbeta gamma\n");
    let r = session.search(&q("beta")).unwrap();
    assert_eq!((r.stale_files, r.index_state), (1, IndexState::IdleStale));
    let r = wait_for(|| {
        session
            .search(&q("beta"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    assert_eq!(r.results[0].path, "b.md");
}

#[test]
fn own_background_run_is_reported_as_updating() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let gate = Arc::new(GateEmbedder::new(true));
    let session = Session::with_embedder(vault.clone(), gate.clone(), fps("a")).unwrap();
    wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    gate.set_open(false);
    let before = gate.entered();
    write(&vault, "b.md", "# B\nbeta\n");
    session.search(&q("beta")).unwrap();
    wait_for(|| (gate.entered() > before).then_some(()));
    let r = session.search(&q("alpha")).unwrap();
    assert_eq!((r.index_state, r.stale_files), (IndexState::Updating, 1));
    gate.set_open(true);
    wait_for(|| {
        session
            .search(&q("beta"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
}

#[test]
fn a_panicked_background_run_is_attached_and_search_keeps_working() {
    let (_tmp, vault) = vault();
    write(&vault, "ok.md", "# OK\nfine words here\n");
    let session =
        Session::with_embedder(vault.clone(), Arc::new(PanicEmbedder::new()), fps("a")).unwrap();
    wait_for(|| {
        session
            .search(&q("fine"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    write(&vault, "bad.md", "# Bad\nPANIC here\n");
    let r = wait_for(|| {
        session
            .search(&q("fine"))
            .ok()
            .filter(|r| r.indexer_error.is_some())
    });
    let error = r.indexer_error.unwrap();
    assert!(error.contains("panicked"), "{error}");
    assert_eq!(r.results[0].path, "ok.md");
}

#[test]
fn another_process_holding_the_lock_is_reported_as_updating_elsewhere() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let session = Session::with_embedder(vault.clone(), fake(), fps("a")).unwrap();
    wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    let held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let r = session.search(&q("alpha")).unwrap();
    drop(held);
    assert_eq!(r.index_state, IndexState::UpdatingElsewhere);
    assert_eq!(r.results[0].path, "a.md");
}

#[cfg(unix)]
#[test]
fn an_unreadable_directory_is_reported_as_a_scan_error() {
    use std::os::unix::fs::PermissionsExt;
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    write(&vault, "locked/x.md", "# X\nx\n");
    let session = Session::with_embedder(vault.clone(), fake(), fps("a")).unwrap();
    wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    let locked = vault.root.join("locked");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    if std::fs::read_dir(&locked).is_ok() {
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        eprintln!("skipped: permissions are not enforced for this user");
        return;
    }
    let r = session.search(&q("alpha"));
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(r.unwrap().index_state, IndexState::ScanError);
}

#[test]
fn an_index_matching_at_startup_asks_for_a_restart_when_another_process_changes_it() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    index_as_another_process(&vault, &fps("a"));
    // The test holds the write lock throughout, as the other process, so this session's updater
    // never completes a run: `synced` can only come from the startup check.
    let held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let session = Session::with_embedder(vault.clone(), fake(), fps("a")).unwrap();
    let r = session.search(&q("alpha")).unwrap();
    assert_eq!(r.results[0].path, "a.md");
    index_with(&vault, &fps("b"));
    let err = session.search(&q("alpha")).unwrap_err();
    assert!(matches!(err, Error::FingerprintChanged), "{err}");
    drop(held);
    let store = notrieve::store::Store::open(&vault.db_path()).unwrap();
    assert_eq!(
        store.rebuild_reason(&fps("b")).unwrap(),
        None,
        "the other index stays"
    );
}

#[test]
fn a_deleted_embedding_cache_is_rebuilt_instead_of_asking_for_a_restart() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let session = Session::with_embedder(vault.clone(), fake(), fps("a")).unwrap();
    wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    let cache = cache_path(&vault.db_path());
    for suffix in ["", "-wal", "-shm"] {
        let mut path = cache.clone().into_os_string();
        path.push(suffix);
        let _ = std::fs::remove_file(path);
    }
    if let Err(Error::FingerprintChanged) = session.search(&q("alpha")) {
        panic!("a missing cache must not ask for a restart");
    }
    let r = wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    assert_eq!(r.results[0].path, "a.md");
}

#[test]
fn a_replaced_embedding_cache_is_noticed_and_rebuilt() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let session = Session::with_embedder(vault.clone(), fake(), fps("a")).unwrap();
    wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    let cache = cache_path(&vault.db_path());
    for suffix in ["", "-wal", "-shm"] {
        let mut path = cache.clone().into_os_string();
        path.push(suffix);
        let _ = std::fs::remove_file(path);
    }
    // A valid but empty cache with a new id, as another tool or a restore might leave behind.
    drop(notrieve::cache::EmbeddingCache::open(&cache).unwrap());
    let r = wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh && r.results[0].vec_rank.is_some())
    });
    assert_eq!(r.results[0].path, "a.md");
}

#[test]
fn get_note_reads_lines_of_an_indexed_note() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\nbeta\n");
    let session = Session::with_embedder(vault, fake(), fps("a")).unwrap();
    let n = wait_for(|| session.get_note("a.md", Some(2), None).ok());
    assert_eq!(
        (n.start_line, n.end_line, n.text.as_str()),
        (2, 3, "alpha\nbeta\n")
    );
}

#[test]
fn get_note_refuses_a_vault_file_the_index_leaves_out() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    write(&vault, ".env", "SECRET=1\n");
    let session = Session::with_embedder(vault, fake(), fps("a")).unwrap();
    wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    let err = session.get_note(".env", None, None).unwrap_err();
    assert!(
        matches!(err, Error::NoteNotIndexed(ref p) if p == ".env"),
        "{err}"
    );
}

#[test]
fn get_note_before_any_index_exists_reports_why_without_creating_one() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let session = Session::with_embedder(vault.clone(), fake(), fps("a")).unwrap();
    // Until the updater's first run has failed to take the lock, the answer is Rebuilding.
    wait_for(|| match session.get_note("a.md", None, None) {
        Err(Error::LockBusy) => Some(()),
        _ => None,
    });
    assert!(!vault.db_path().exists());
    drop(held);
}

#[test]
fn get_note_refuses_an_index_built_with_other_settings() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    index_as_another_process(&vault, &fps("b"));
    let held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let session = Session::with_embedder(vault, fake(), fps("a")).unwrap();
    wait_for(|| match session.get_note("a.md", None, None) {
        Err(Error::LockBusy) => Some(()),
        _ => None,
    });
    drop(held);
}

#[test]
fn get_note_on_a_garbage_index_reports_why_like_search() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    std::fs::create_dir_all(vault.db_path().parent().unwrap()).unwrap();
    std::fs::write(vault.db_path(), "not a database at all, just text").unwrap();
    let held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let session = Session::with_embedder(vault, fake(), fps("a")).unwrap();
    wait_for(|| match session.get_note("a.md", None, None) {
        Err(Error::LockBusy) => Some(()),
        _ => None,
    });
    drop(held);
}

#[test]
fn get_note_on_a_newer_schema_after_the_sync_asks_for_a_restart() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    index_as_another_process(&vault, &fps("a"));
    let held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let session = Session::with_embedder(vault.clone(), fake(), fps("a")).unwrap();
    session.search(&q("alpha")).unwrap();
    let newer = (notrieve::store::SCHEMA_VERSION + 1).to_string();
    rusqlite::Connection::open(vault.db_path())
        .unwrap()
        .execute(
            "UPDATE meta SET value = ?1 WHERE key = 'schema_version'",
            [&newer],
        )
        .unwrap();
    let err = session.get_note("a.md", None, None).unwrap_err();
    assert!(matches!(err, Error::FingerprintChanged), "{err}");
    drop(held);
}

#[test]
fn get_note_keeps_to_the_vault_root_resolved_at_the_start() {
    let (tmp, mut vault) = vault();
    vault.config.db_path = tmp.path().join("index.db");
    write(&vault, "a.md", "# A\nalpha\n");
    let session = Session::with_embedder(vault.clone(), fake(), fps("a")).unwrap();
    wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    let moved = tmp.path().join("moved");
    std::fs::rename(&vault.root, &moved).unwrap();
    std::fs::create_dir_all(moved.join("private")).unwrap();
    std::fs::write(moved.join("private/a.md"), "SECRET\n").unwrap();
    std::os::unix::fs::symlink(moved.join("private"), &vault.root).unwrap();
    let err = session.get_note("a.md", None, None).unwrap_err();
    assert!(matches!(err, Error::OutsideVault(_)), "{err}");
}

#[test]
fn a_stopped_updater_asks_for_a_restart_even_after_the_settings_change_back() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let gate = Arc::new(GateEmbedder::new(true));
    let session = Session::with_embedder(vault.clone(), gate.clone(), fps("a")).unwrap();
    wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    let set_index_fingerprint = |fp: &str| {
        rusqlite::Connection::open(vault.db_path())
            .unwrap()
            .execute(
                "UPDATE meta SET value = ?1 WHERE key = 'index_fingerprint'",
                [fp],
            )
            .unwrap();
    };
    gate.set_open(false);
    let entered = gate.entered();
    write(&vault, "b.md", "# B\nbeta\n");
    session.search(&q("alpha")).unwrap();
    wait_for(|| (gate.entered() > entered).then_some(()));
    // Another process's settings land while the run embeds b.md: its write fails and it stops.
    set_index_fingerprint(&fps("b").index);
    gate.set_open(true);
    let held = wait_for(|| lock::try_lock(&vault.lock_path()).unwrap());
    set_index_fingerprint(&fps("a").index);
    drop(held);
    wait_for(|| {
        matches!(session.search(&q("alpha")), Err(Error::FingerprintChanged)).then_some(())
    });
    let err = session.get_note("a.md", None, None).unwrap_err();
    assert!(matches!(err, Error::FingerprintChanged), "{err}");
}

#[test]
fn get_note_during_the_startup_rebuild_is_a_rebuilding_error() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    write(&vault, "b.md", "# B\nbeta\n");
    let gate = Arc::new(GateEmbedder::new(false));
    let session = Session::with_embedder(vault, gate.clone(), fps("a")).unwrap();
    wait_for(|| (gate.entered() >= 1).then_some(()));
    let err = session.get_note("b.md", None, None).unwrap_err();
    assert!(matches!(err, Error::Rebuilding { .. }), "{err}");
    gate.set_open(true);
    let n = wait_for(|| session.get_note("b.md", None, None).ok());
    assert_eq!(n.text, "# B\nbeta\n");
}

#[test]
fn an_older_schema_with_matching_fingerprints_is_not_searchable_while_the_lock_is_held() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    index_as_another_process(&vault, &fps("a"));
    let older = (notrieve::store::SCHEMA_VERSION - 1).to_string();
    rusqlite::Connection::open(vault.db_path())
        .unwrap()
        .execute(
            "UPDATE meta SET value = ?1 WHERE key = 'schema_version'",
            [&older],
        )
        .unwrap();
    let _held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let session = Session::with_embedder(vault, fake(), fps("a")).unwrap();
    wait_for(|| match session.search(&q("alpha")) {
        Err(Error::LockBusy) => Some(()),
        _ => None,
    });
}

#[test]
fn a_garbage_embedding_cache_is_rebuilt() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let session = Session::with_embedder(vault.clone(), fake(), fps("a")).unwrap();
    wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    let cache = cache_path(&vault.db_path());
    for suffix in ["-wal", "-shm"] {
        let mut path = cache.clone().into_os_string();
        path.push(suffix);
        let _ = std::fs::remove_file(path);
    }
    std::fs::write(&cache, b"this is not a sqlite database, just garbage bytes").unwrap();
    let r = wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh && r.results[0].vec_rank.is_some())
    });
    assert_eq!(r.results[0].path, "a.md");
}

#[test]
fn a_note_written_while_a_run_is_in_progress_is_indexed_without_another_search() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let gate = Arc::new(GateEmbedder::new(true));
    let session = Session::with_embedder(vault.clone(), gate.clone(), fps("a")).unwrap();
    wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    gate.set_open(false);
    let before = gate.entered();
    write(&vault, "b.md", "# B\nbeta\n");
    session.search(&q("beta")).unwrap();
    wait_for(|| (gate.entered() > before).then_some(()));
    write(&vault, "c.md", "# C\ngamma\n");
    let r = session.search(&q("alpha")).unwrap();
    assert_eq!((r.index_state, r.stale_files), (IndexState::Updating, 2));
    gate.set_open(true);
    wait_for(|| {
        indexed_paths(&vault)
            .contains(&"c.md".to_string())
            .then_some(())
    });
}

#[test]
fn a_stopped_updater_keeps_asking_for_a_restart() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let session = Session::with_embedder(vault.clone(), fake(), fps("a")).unwrap();
    wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    index_as_another_process(&vault, &fps("b"));
    write(&vault, "b.md", "# B\nbeta\n");
    let _ = session.search(&q("alpha"));
    for _ in 0..20 {
        let err = session.search(&q("alpha")).unwrap_err();
        assert!(matches!(err, Error::FingerprintChanged), "{err}");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[test]
fn a_change_after_a_search_found_the_index_matching_asks_for_a_restart() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    index_as_another_process(&vault, &fps("b"));
    let held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let session = Session::with_embedder(vault.clone(), fake(), fps("a")).unwrap();
    wait_for(|| match session.search(&q("alpha")) {
        Err(Error::LockBusy) => Some(()),
        _ => None,
    });
    index_with(&vault, &fps("a"));
    let r = session.search(&q("alpha")).unwrap();
    assert_eq!(r.results[0].path, "a.md");
    index_with(&vault, &fps("b"));
    let err = session.search(&q("alpha")).unwrap_err();
    assert!(matches!(err, Error::FingerprintChanged), "{err}");
    drop(held);
    for _ in 0..30 {
        let err = session.search(&q("alpha")).unwrap_err();
        assert!(matches!(err, Error::FingerprintChanged), "{err}");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let store = notrieve::store::Store::open(&vault.db_path()).unwrap();
    assert_eq!(
        store.rebuild_reason(&fps("b")).unwrap(),
        None,
        "the other index stays"
    );
}

#[test]
fn a_newer_schema_after_the_sync_asks_for_a_restart() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    index_as_another_process(&vault, &fps("a"));
    let held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let session = Session::with_embedder(vault.clone(), fake(), fps("a")).unwrap();
    session.search(&q("alpha")).unwrap();
    let newer = (notrieve::store::SCHEMA_VERSION + 1).to_string();
    rusqlite::Connection::open(vault.db_path())
        .unwrap()
        .execute(
            "UPDATE meta SET value = ?1 WHERE key = 'schema_version'",
            [&newer],
        )
        .unwrap();
    let err = session.search(&q("alpha")).unwrap_err();
    assert!(matches!(err, Error::FingerprintChanged), "{err}");
    drop(held);
}

#[test]
fn a_newer_schema_with_the_cache_missing_asks_for_a_restart() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    index_as_another_process(&vault, &fps("a"));
    let held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let session = Session::with_embedder(vault.clone(), fake(), fps("a")).unwrap();
    session.search(&q("alpha")).unwrap();
    let newer = (notrieve::store::SCHEMA_VERSION + 1).to_string();
    rusqlite::Connection::open(vault.db_path())
        .unwrap()
        .execute(
            "UPDATE meta SET value = ?1 WHERE key = 'schema_version'",
            [&newer],
        )
        .unwrap();
    let cache = cache_path(&vault.db_path());
    for suffix in ["", "-wal", "-shm"] {
        let mut path = cache.clone().into_os_string();
        path.push(suffix);
        let _ = std::fs::remove_file(path);
    }
    let err = session.search(&q("alpha")).unwrap_err();
    assert!(matches!(err, Error::FingerprintChanged), "{err}");
    drop(held);
}

/// Runs `hook` once, inside the first `embed_query`, which `Session::search` calls after it has
/// opened the index and before it reads the updater state again.
struct QueryHookEmbedder {
    inner: FakeEmbedder,
    hook: std::sync::Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl notrieve::embed::Embedder for QueryHookEmbedder {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn embed_documents(&self, texts: &[String]) -> notrieve::Result<Vec<Vec<f32>>> {
        self.inner.embed_documents(texts)
    }

    fn embed_query(&self, text: &str) -> notrieve::Result<Vec<f32>> {
        let hook = self.hook.lock().unwrap().take();
        if let Some(hook) = hook {
            hook();
        }
        self.inner.embed_query(text)
    }

    fn count_tokens(&self, text: &str) -> notrieve::Result<usize> {
        self.inner.count_tokens(text)
    }
}

#[test]
fn a_rebuild_that_starts_and_finishes_during_a_search_is_reported_as_rebuilding() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let embedder = Arc::new(QueryHookEmbedder {
        inner: FakeEmbedder::new(16),
        hook: std::sync::Mutex::new(None),
    });
    let session =
        Arc::new(Session::with_embedder(vault.clone(), embedder.clone(), fps("a")).unwrap());
    wait_for(|| {
        session
            .search(&q("alpha"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    let cache = cache_path(&vault.db_path());
    let hooked = Arc::clone(&session);
    *embedder.hook.lock().unwrap() = Some(Box::new(move || {
        // The cache is replaced after the outer search validated it; the nested searches ask the
        // updater for the rebuild and wait until it has finished.
        for suffix in ["", "-wal", "-shm"] {
            let mut path = cache.clone().into_os_string();
            path.push(suffix);
            let _ = std::fs::remove_file(path);
        }
        drop(notrieve::cache::EmbeddingCache::open(&cache).unwrap());
        wait_for(|| {
            hooked
                .search(&q("alpha"))
                .ok()
                .filter(|r| r.index_state == IndexState::Fresh)
        });
    }));
    let err = session.search(&q("alpha")).unwrap_err();
    assert!(matches!(err, Error::Rebuilding { .. }), "{err}");
}

#[test]
fn a_halted_updater_still_lets_search_use_the_index() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    index_as_another_process(&vault, &fps("a"));
    let e = Arc::new(FlakyRemote::new(&[RemoteErrorKind::Auth]));
    let session = Session::with_embedder(vault.clone(), e, fps("a")).unwrap();
    write(&vault, "b.md", "# B\nbeta\n");
    let r = wait_for(|| {
        session.search(&q("alpha")).ok().filter(|r| {
            r.indexer_error
                .as_deref()
                .is_some_and(|m| m.contains("restart"))
        })
    });
    assert_eq!(r.results[0].path, "a.md");
}
