use std::fs;

use notrieve::app::search_vault;
use notrieve::config::Vault;
use notrieve::embed::fake::FakeEmbedder;
use notrieve::fingerprint::Fingerprints;
use notrieve::lock;
use notrieve::search::{IndexState, SearchQuery};
use tempfile::TempDir;

fn fps() -> Fingerprints {
    Fingerprints {
        model: "model-a".into(),
        index: "fp-a".into(),
    }
}

fn vault_with(config: &str) -> (TempDir, Vault) {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("notes");
    fs::create_dir_all(root.join(".notrieve")).unwrap();
    fs::write(root.join(".notrieve/config.toml"), config).unwrap();
    let vault = Vault::load(&root, None).unwrap();
    (tmp, vault)
}

fn write(vault: &Vault, rel: &str, text: &str) {
    let path = vault.root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, text).unwrap();
}

#[test]
fn search_indexes_new_notes_within_the_budget_and_reports_a_fresh_index() {
    let (_tmp, vault) = vault_with("");
    write(
        &vault,
        "税金.md",
        "# 医療費\n医療費控除の領収書をまとめる。\n",
    );
    write(&vault, "旅行.md", "# 京都\n紅葉の季節に嵐山を歩いた。\n");
    let e = FakeEmbedder::new(16);
    let r = search_vault(
        &vault,
        &e,
        &fps(),
        &SearchQuery::new("医療費控除"),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!((r.stale_files, r.index_state), (0, IndexState::Fresh));
    assert_eq!(r.results[0].path, "税金.md");
    assert_eq!(r.indexer_error, None);
}

#[test]
fn exhausted_budget_leaves_the_changes_and_reports_idle_stale() {
    let (_tmp, vault) = vault_with("search_budget_ms = 0\n");
    write(&vault, "a.md", "alpha beta\n");
    write(&vault, "b.md", "gamma delta\n");
    let e = FakeEmbedder::new(16);
    let r = search_vault(&vault, &e, &fps(), &SearchQuery::new("alpha"), &mut |_| {}).unwrap();
    assert_eq!(
        (r.results.len(), r.stale_files, r.index_state),
        (0, 2, IndexState::IdleStale)
    );
    assert_eq!(e.embedded_count(), 0);
}

#[test]
fn lock_held_elsewhere_skips_the_update_and_reports_updating_elsewhere() {
    let (_tmp, vault) = vault_with("");
    write(&vault, "a.md", "alpha beta\n");
    let e = FakeEmbedder::new(16);
    search_vault(&vault, &e, &fps(), &SearchQuery::new("alpha"), &mut |_| {}).unwrap();
    write(&vault, "b.md", "gamma delta\n");
    let _held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let r = search_vault(&vault, &e, &fps(), &SearchQuery::new("gamma"), &mut |_| {}).unwrap();
    assert_eq!(
        (r.stale_files, r.index_state),
        (1, IndexState::UpdatingElsewhere)
    );
    assert!(r.results.iter().all(|h| h.path != "b.md"));
}

#[test]
fn lock_held_elsewhere_is_reported_even_when_nothing_is_stale() {
    let (_tmp, vault) = vault_with("");
    write(&vault, "a.md", "alpha beta\n");
    let e = FakeEmbedder::new(16);
    let free = search_vault(&vault, &e, &fps(), &SearchQuery::new("alpha"), &mut |_| {}).unwrap();
    assert_eq!((free.stale_files, free.index_state), (0, IndexState::Fresh));
    let _held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let r = search_vault(&vault, &e, &fps(), &SearchQuery::new("alpha"), &mut |_| {}).unwrap();
    assert_eq!(
        (r.stale_files, r.index_state),
        (0, IndexState::UpdatingElsewhere)
    );
}

#[test]
fn invalid_query_fails_before_anything_is_indexed() {
    let (_tmp, vault) = vault_with("");
    write(&vault, "a.md", "alpha beta\n");
    let e = FakeEmbedder::new(16);
    let err = search_vault(&vault, &e, &fps(), &SearchQuery::new("  "), &mut |_| {}).unwrap_err();
    assert!(matches!(err, notrieve::Error::InvalidQuery(_)), "{err}");
    assert!(!vault.db_path().exists());
}

#[cfg(unix)]
#[test]
fn unreadable_directory_reports_a_scan_error() {
    use std::os::unix::fs::PermissionsExt;
    let (_tmp, vault) = vault_with("");
    write(&vault, "a.md", "alpha beta\n");
    write(&vault, "locked/x.md", "x\n");
    let e = FakeEmbedder::new(16);
    search_vault(&vault, &e, &fps(), &SearchQuery::new("alpha"), &mut |_| {}).unwrap();
    let locked = vault.root.join("locked");
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
    if fs::read_dir(&locked).is_ok() {
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
        eprintln!("skipped: permissions are not enforced for this user");
        return;
    }
    let r = search_vault(&vault, &e, &fps(), &SearchQuery::new("alpha"), &mut |_| {});
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(r.unwrap().index_state, IndexState::ScanError);
}

#[cfg(unix)]
#[test]
fn scan_error_takes_precedence_over_a_lock_held_elsewhere() {
    use std::os::unix::fs::PermissionsExt;
    let (_tmp, vault) = vault_with("");
    write(&vault, "a.md", "alpha beta\n");
    write(&vault, "locked/x.md", "x\n");
    let e = FakeEmbedder::new(16);
    search_vault(&vault, &e, &fps(), &SearchQuery::new("alpha"), &mut |_| {}).unwrap();
    let locked = vault.root.join("locked");
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
    if fs::read_dir(&locked).is_ok() {
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
        eprintln!("skipped: permissions are not enforced for this user");
        return;
    }
    let held = lock::try_lock(&vault.lock_path()).unwrap().unwrap();
    let r = search_vault(&vault, &e, &fps(), &SearchQuery::new("alpha"), &mut |_| {});
    drop(held);
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(r.unwrap().index_state, IndexState::ScanError);
}
