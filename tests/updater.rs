mod support;

use std::sync::Arc;
use std::time::Duration;

use notrieve::RemoteErrorKind;
use notrieve::indexer::updater::{Backoff, Updater};
use notrieve::lock;
use notrieve::store::Store;
use support::{
    FlakyRemote, GateEmbedder, PanicEmbedder, fake_embedder, fps, index_as_another_process,
    indexed_paths, vault, vault_with, wait_for, write,
};

/// A vault whose index lives in `data/`, and a function that swaps `data/` for a link to a new
/// directory outside the vault, as a sync could while a process runs. The function returns that
/// outside directory.
#[cfg(unix)]
fn vault_with_a_data_dir() -> (
    tempfile::TempDir,
    notrieve::config::Vault,
    impl Fn() -> std::path::PathBuf,
) {
    let (tmp, vault) = vault_with("db_path = \"data/index.db\"\n");
    std::fs::create_dir_all(vault.root.join("data")).unwrap();
    let base = tmp.path().to_path_buf();
    let root = vault.root.clone();
    let relink = move || {
        let outside = base.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::rename(root.join("data"), base.join("old-data")).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("data")).unwrap();
        outside
    };
    (tmp, vault, relink)
}

#[cfg(unix)]
#[test]
fn observing_after_the_index_directory_was_relinked_creates_nothing_there() {
    let (_tmp, vault, relink) = vault_with_a_data_dir();
    let updater = Updater::start(vault, fake_embedder(), fps("a"), false).unwrap();
    let outside = relink();
    let err = updater.observe().unwrap_err();
    assert!(matches!(err, notrieve::Error::FingerprintChanged), "{err}");
    assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 0);
}

#[cfg(unix)]
#[test]
fn a_run_stops_when_the_index_directory_was_relinked_since_the_start() {
    let (_tmp, vault, relink) = vault_with_a_data_dir();
    write(&vault, "a.md", "# A\nalpha\n");
    let updater = Updater::start(vault.clone(), fake_embedder(), fps("a"), false).unwrap();
    updater.request();
    wait_for(|| Some(updater.status()).filter(|s| s.synced));
    let outside = relink();
    updater.request();
    let status = wait_for(|| Some(updater.status()).filter(|s| s.stopped));
    assert!(status.last_error.is_some());
    assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 0);
}

#[test]
fn requested_run_indexes_the_vault_and_marks_the_index_synced() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    write(&vault, "b.md", "# B\nbeta\n");
    let updater = Updater::start(vault.clone(), fake_embedder(), fps("a"), false).unwrap();
    assert!(!updater.status().synced, "nothing runs before a request");
    updater.request();
    let status = wait_for(|| Some(updater.status()).filter(|s| s.synced));
    assert_eq!((status.running, status.last_error), (false, None));
    assert_eq!(indexed_paths(&vault), ["a.md", "b.md"]);
}

#[test]
fn lock_held_elsewhere_skips_the_run_until_the_lock_is_free() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let updater = Updater::start(vault.clone(), fake_embedder(), fps("a"), false).unwrap();
    updater.request();
    let status = wait_for(|| Some(updater.status()).filter(|s| s.lock_busy && !s.running));
    assert!(!status.synced);
    assert!(
        !vault.db_path().exists(),
        "nothing was opened without the lock"
    );
    drop(held);
    updater.request();
    let status = wait_for(|| Some(updater.status()).filter(|s| s.synced));
    assert!(!status.lock_busy);
    assert_eq!(indexed_paths(&vault), ["a.md"]);
}

#[test]
fn mismatch_before_the_sync_is_rebuilt_with_this_process_settings() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    index_as_another_process(&vault, &fps("b"));
    let updater = Updater::start(vault.clone(), fake_embedder(), fps("a"), false).unwrap();
    updater.request();
    let status = wait_for(|| Some(updater.status()).filter(|s| s.synced));
    assert_eq!((status.stopped, status.rebuild_generation), (false, 1));
    let store = Store::open(&vault.db_path()).unwrap();
    assert_eq!(store.rebuild_reason(&fps("a")).unwrap(), None);
}

#[test]
fn settings_changed_after_a_run_stop_the_updater_without_writing() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let updater = Updater::start(vault.clone(), fake_embedder(), fps("a"), false).unwrap();
    updater.request();
    wait_for(|| Some(updater.status()).filter(|s| s.synced));
    index_as_another_process(&vault, &fps("b"));
    write(&vault, "c.md", "# C\ngamma\n");
    updater.request();
    let status = wait_for(|| Some(updater.status()).filter(|s| s.stopped));
    let error = status.last_error.unwrap();
    assert!(error.contains("settings changed"), "{error}");
    let store = Store::open(&vault.db_path()).unwrap();
    assert_eq!(store.rebuild_reason(&fps("b")).unwrap(), None);
    assert_eq!(indexed_paths(&vault), ["a.md"]);
}

#[test]
fn index_matching_at_startup_counts_as_synced_before_any_run() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    index_as_another_process(&vault, &fps("a"));
    let updater = Updater::start(vault.clone(), fake_embedder(), fps("a"), true).unwrap();
    assert!(updater.status().synced);
    index_as_another_process(&vault, &fps("b"));
    updater.request();
    let status = wait_for(|| Some(updater.status()).filter(|s| s.stopped));
    assert!(status.last_error.unwrap().contains("settings changed"));
    let store = Store::open(&vault.db_path()).unwrap();
    assert_eq!(store.rebuild_reason(&fps("b")).unwrap(), None);
}

#[test]
fn a_deleted_index_after_the_sync_is_rebuilt_not_treated_as_foreign() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let updater = Updater::start(vault.clone(), fake_embedder(), fps("a"), false).unwrap();
    updater.request();
    wait_for(|| Some(updater.status()).filter(|s| s.synced));
    for suffix in ["", "-wal", "-shm"] {
        let mut path = vault.db_path().into_os_string();
        path.push(suffix);
        let _ = std::fs::remove_file(path);
    }
    updater.request();
    let status =
        wait_for(|| Some(updater.status()).filter(|s| s.rebuild_generation == 2 && !s.running));
    assert_eq!((status.stopped, status.last_error), (false, None));
    assert_eq!(indexed_paths(&vault), ["a.md"]);
}

#[test]
fn a_panicking_run_is_recorded_and_a_later_run_recovers() {
    let (_tmp, vault) = vault();
    write(&vault, "ok.md", "# OK\nfine\n");
    write(&vault, "bad.md", "# Bad\nPANIC here\n");
    let updater = Updater::start(
        vault.clone(),
        Arc::new(PanicEmbedder::new()),
        fps("a"),
        false,
    )
    .unwrap();
    updater.request();
    let status =
        wait_for(|| Some(updater.status()).filter(|s| s.last_error.is_some() && !s.running));
    let error = status.last_error.unwrap();
    assert!(error.contains("panicked"), "{error}");
    assert!(!status.synced);
    write(&vault, "bad.md", "# Bad\nfixed now\n");
    updater.request();
    let status = wait_for(|| Some(updater.status()).filter(|s| s.synced));
    assert_eq!(status.last_error, None);
    assert_eq!(indexed_paths(&vault), ["bad.md", "ok.md"]);
}

#[test]
fn skipped_files_are_counted_and_a_clean_run_resets_the_count() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    std::fs::write(vault.root.join("bad.md"), [0xff, 0xfe, 0x00]).unwrap();
    let updater = Updater::start(vault.clone(), fake_embedder(), fps("a"), false).unwrap();
    updater.request();
    let status = wait_for(|| Some(updater.status()).filter(|s| s.synced));
    assert_eq!(status.skipped_files, 1);
    std::fs::remove_file(vault.root.join("bad.md")).unwrap();
    updater.request();
    let status = wait_for(|| Some(updater.status()).filter(|s| s.skipped_files == 0 && !s.running));
    assert!(status.synced);
}

#[test]
fn rebuild_progress_is_visible_while_the_run_is_in_progress() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    write(&vault, "b.md", "# B\nbeta\n");
    let gate = Arc::new(GateEmbedder::new(false));
    let updater = Updater::start(vault.clone(), gate.clone(), fps("a"), false).unwrap();
    updater.request();
    wait_for(|| (gate.entered() >= 1).then_some(()));
    let status = updater.status();
    assert_eq!(
        (
            status.running,
            status.rebuilding,
            status.rebuild_generation,
            status.remaining
        ),
        (true, true, 1, Some(2))
    );
    gate.set_open(true);
    let status = wait_for(|| Some(updater.status()).filter(|s| s.synced));
    assert_eq!(
        (
            status.running,
            status.rebuilding,
            status.rebuild_generation,
            status.remaining
        ),
        (false, false, 1, None)
    );
}

#[test]
fn observe_reports_another_process_lock_only_while_this_updater_is_idle() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let gate = Arc::new(GateEmbedder::new(false));
    let updater = Updater::start(vault.clone(), gate.clone(), fps("a"), false).unwrap();
    assert!(!updater.observe().unwrap().locked_elsewhere);
    let held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    assert!(updater.observe().unwrap().locked_elsewhere);
    drop(held);
    updater.request();
    wait_for(|| (gate.entered() >= 1).then_some(()));
    let seen = updater.observe().unwrap();
    assert!(seen.status.running);
    assert!(!seen.locked_elsewhere, "our own run holds the lock");
    gate.set_open(true);
    wait_for(|| Some(updater.status()).filter(|s| s.synced));
}

#[test]
fn schema_newer_than_supported_after_the_sync_stops_the_updater() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let updater = Updater::start(vault.clone(), fake_embedder(), fps("a"), false).unwrap();
    updater.request();
    wait_for(|| Some(updater.status()).filter(|s| s.synced));
    let newer = (notrieve::store::SCHEMA_VERSION + 1).to_string();
    rusqlite::Connection::open(vault.db_path())
        .unwrap()
        .execute(
            "UPDATE meta SET value = ?1 WHERE key = 'schema_version'",
            [&newer],
        )
        .unwrap();
    updater.request();
    let status = wait_for(|| Some(updater.status()).filter(|s| s.stopped));
    assert!(status.last_error.unwrap().contains("settings changed"));
}

#[test]
fn schema_newer_than_supported_before_the_sync_is_a_plain_error() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    index_as_another_process(&vault, &fps("a"));
    let newer = (notrieve::store::SCHEMA_VERSION + 1).to_string();
    rusqlite::Connection::open(vault.db_path())
        .unwrap()
        .execute(
            "UPDATE meta SET value = ?1 WHERE key = 'schema_version'",
            [&newer],
        )
        .unwrap();
    let updater = Updater::start(vault.clone(), fake_embedder(), fps("a"), false).unwrap();
    updater.request();
    let status =
        wait_for(|| Some(updater.status()).filter(|s| s.last_error.is_some() && !s.running));
    assert!(!status.stopped && !status.synced);
}

#[test]
fn lock_busy_is_cleared_by_a_later_run_that_fails() {
    let (_tmp, vault) = vault();
    write(&vault, "bad.md", "# Bad\nPANIC here\n");
    let held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let updater = Updater::start(
        vault.clone(),
        Arc::new(PanicEmbedder::new()),
        fps("a"),
        false,
    )
    .unwrap();
    updater.request();
    wait_for(|| Some(updater.status()).filter(|s| s.lock_busy && !s.running));
    drop(held);
    updater.request();
    let status =
        wait_for(|| Some(updater.status()).filter(|s| s.last_error.is_some() && !s.running));
    assert!(!status.lock_busy);
}

#[test]
fn a_change_of_the_index_settings_only_stops_the_updater() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let a = notrieve::fingerprint::Fingerprints {
        model: "model-x".into(),
        index: "index-a".into(),
    };
    let b = notrieve::fingerprint::Fingerprints {
        model: "model-x".into(),
        index: "index-b".into(),
    };
    let updater = Updater::start(vault.clone(), fake_embedder(), a, false).unwrap();
    updater.request();
    wait_for(|| Some(updater.status()).filter(|s| s.synced));
    index_as_another_process(&vault, &b);
    updater.request();
    let status = wait_for(|| Some(updater.status()).filter(|s| s.stopped));
    assert!(status.last_error.unwrap().contains("settings changed"));
}

#[test]
fn a_request_after_the_updater_stopped_performs_no_run() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let updater = Updater::start(vault.clone(), fake_embedder(), fps("a"), false).unwrap();
    updater.request();
    wait_for(|| Some(updater.status()).filter(|s| s.synced));
    index_as_another_process(&vault, &fps("b"));
    updater.request();
    wait_for(|| Some(updater.status()).filter(|s| s.stopped));
    write(&vault, "new.md", "# New\nfresh\n");
    updater.request();
    let until = std::time::Instant::now() + std::time::Duration::from_millis(200);
    while std::time::Instant::now() < until {
        assert!(!updater.status().running);
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(indexed_paths(&vault), ["a.md"]);
}

#[test]
fn a_request_during_a_run_coalesces_into_one_more_run() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let gate = Arc::new(GateEmbedder::new(false));
    let updater = Updater::start(vault.clone(), gate.clone(), fps("a"), false).unwrap();
    updater.request();
    wait_for(|| (gate.entered() >= 1).then_some(()));
    write(&vault, "late.md", "# Late\nomega\n");
    updater.request();
    updater.request();
    gate.set_open(true);
    wait_for(|| Some(indexed_paths(&vault)).filter(|p| p.contains(&"late.md".to_string())));
    wait_for(|| Some(updater.status()).filter(|s| s.synced && !s.running));
    assert_eq!(indexed_paths(&vault), ["a.md", "late.md"]);
}

fn backoff(first_ms: u64, max_ms: u64) -> Backoff {
    Backoff {
        first: Duration::from_millis(first_ms),
        max: Duration::from_millis(max_ms),
    }
}

#[test]
fn an_unavailable_api_pauses_the_updater_until_the_delay_ends() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let e = Arc::new(FlakyRemote::new(&[RemoteErrorKind::Transient]));
    let updater = Updater::start_with_backoff(
        vault.clone(),
        e.clone(),
        fps("a"),
        false,
        backoff(1000, 5000),
    )
    .unwrap();
    updater.request();
    let paused =
        wait_for(|| Some(updater.status()).filter(|s| s.retry_delay.is_some() && !s.running));
    assert_eq!(paused.retry_delay, Some(Duration::from_millis(1000)));
    assert!(paused.last_error.unwrap().contains("paused"));
    updater.request();
    wait_for(|| {
        Some(updater.status()).filter(|s| e.calls() == 2 && !s.running && s.last_error.is_none())
    });
    assert!(e.gaps()[0] >= Duration::from_millis(1000), "{:?}", e.gaps());
    assert_eq!(indexed_paths(&vault), ["a.md"]);
}

#[test]
fn a_success_resets_the_retry_delay() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    use RemoteErrorKind::Transient;
    let e = Arc::new(FlakyRemote::new(&[Transient, Transient]));
    let updater = Updater::start_with_backoff(
        vault.clone(),
        e.clone(),
        fps("a"),
        false,
        backoff(300, 5000),
    )
    .unwrap();
    updater.request();
    // The second failure doubles the first wait: 300 ms, then 600 ms.
    wait_for(|| {
        Some(updater.status()).filter(|s| {
            e.calls() == 2 && !s.running && s.retry_delay == Some(Duration::from_millis(600))
        })
    });
    wait_for(|| {
        Some(updater.status()).filter(|s| e.calls() == 3 && s.retry_delay.is_none() && !s.running)
    });
    write(&vault, "b.md", "# B\nbeta\n");
    e.push(Transient);
    updater.request();
    let paused = wait_for(|| {
        Some(updater.status()).filter(|s| e.calls() == 4 && !s.running && s.retry_delay.is_some())
    });
    // The wait after the success starts again at 300 ms; without the reset it would continue from 600 to 1200.
    assert_eq!(paused.retry_delay, Some(Duration::from_millis(300)));
}

#[test]
fn an_auth_error_halts_the_updater_until_restart() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let e = Arc::new(FlakyRemote::new(&[RemoteErrorKind::Auth]));
    let updater = Updater::start(vault.clone(), e.clone(), fps("a"), false).unwrap();
    updater.request();
    let halted = wait_for(|| Some(updater.status()).filter(|s| s.halted));
    assert!(halted.last_error.unwrap().contains("restart"));
    write(&vault, "b.md", "# B\nbeta\n");
    updater.request();
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(e.calls(), 1);
}
