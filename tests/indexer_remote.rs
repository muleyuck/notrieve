mod support;

use std::time::{Duration, Instant};

use notrieve::cache::{EmbeddingCache, cache_path};
use notrieve::config::Vault;
use notrieve::embed::Embedder;
use notrieve::embed::fake::FakeEmbedder;
use notrieve::fingerprint::Fingerprints;
use notrieve::indexer::{IndexEvent, IndexOptions, IndexReport, index_vault};
use notrieve::store::Store;
use notrieve::{Error, RemoteErrorKind, Result};
use support::fake_api::{FakeApi, Reply, inputs};
use support::{fps, indexed_paths, remote_embedder, remote_fps, remote_vault, vault, write};

#[derive(Default)]
struct Run {
    rebuild: bool,
    budget: Option<Duration>,
    deadline: Option<Instant>,
}

fn index(vault: &Vault, e: &dyn Embedder, fps: &Fingerprints, run: Run) -> Result<IndexReport> {
    index_with_events(vault, e, fps, run, &mut |_| {})
}

fn index_with_events(
    vault: &Vault,
    e: &dyn Embedder,
    fps: &Fingerprints,
    run: Run,
    on_event: &mut dyn FnMut(IndexEvent<'_>),
) -> Result<IndexReport> {
    let db = vault.db_path();
    let mut store = Store::open(&db)?;
    let mut cache = EmbeddingCache::open(&cache_path(&db))?;
    let opts = IndexOptions {
        exclude: &[],
        chunk_max_tokens: vault.config.chunk_max_tokens(),
        budget: run.budget,
        deadline: run.deadline,
        rebuild: run.rebuild,
    };
    index_vault(&vault.root, &mut store, &mut cache, e, fps, &opts, on_event)
}

fn kind(err: &Error) -> Option<RemoteErrorKind> {
    match err {
        Error::Remote { kind, .. } => Some(*kind),
        _ => None,
    }
}

fn skipped(vault: &Vault) -> Vec<String> {
    let mut paths: Vec<String> = Store::open(&vault.db_path())
        .unwrap()
        .skipped_states()
        .unwrap()
        .into_keys()
        .collect();
    paths.sort();
    paths
}

/// 400 for any request whose inputs contain `REJECT`, embeddings of `dim` otherwise.
fn reject_marked(dim: usize) -> Reply {
    Reply::by(move |body| {
        if inputs(body).iter().any(|i| i.contains("REJECT")) {
            Reply::status(400)
        } else {
            Reply::Embed { dim }
        }
    })
}

fn vectors_of_dim(vault: &Vault, dim: usize) -> i64 {
    let cache = EmbeddingCache::open(&cache_path(&vault.db_path())).unwrap();
    cache
        .conn()
        .query_row(
            "SELECT count(*) FROM embeddings WHERE length(vector) = ?1",
            [dim as i64 * 4],
            |r| r.get(0),
        )
        .unwrap()
}

#[test]
fn a_transient_failure_leaves_the_rest_for_the_next_run() {
    let api = FakeApi::start(Reply::status(503));
    api.push(Reply::Embed { dim: 4 });
    let (_tmp, vault) = remote_vault(&api.base, "", "");
    for name in ["a", "b", "c"] {
        write(
            &vault,
            &format!("{name}.md"),
            &format!("# {name}\n{name} text\n"),
        );
    }
    // Three retries without waits: b.md uses them up.
    let e = remote_embedder(&api.base, "");
    let fps = remote_fps(&vault);
    let first = index(&vault, &e, &fps, Run::default()).unwrap();
    assert_eq!((first.indexed, first.remaining), (1, 2));
    assert!(
        first.interrupted.as_deref().unwrap().contains("503"),
        "{first:?}"
    );
    assert!(first.skipped.is_empty());
    assert_eq!(indexed_paths(&vault), ["a.md"]);
    assert_eq!(api.requests().len(), 1 + 4);
    api.set_default(Reply::Embed { dim: 4 });
    let second = index(&vault, &e, &fps, Run::default()).unwrap();
    assert_eq!(
        (second.indexed, second.remaining, second.interrupted),
        (2, 0, None)
    );
    assert_eq!(indexed_paths(&vault), ["a.md", "b.md", "c.md"]);
}

#[test]
fn an_auth_error_ends_the_run_and_keeps_the_files_already_written() {
    let api = FakeApi::start(Reply::status(401));
    api.push(Reply::Embed { dim: 4 });
    let (_tmp, vault) = remote_vault(&api.base, "", "");
    write(&vault, "a.md", "# a\nalpha\n");
    write(&vault, "b.md", "# b\nbeta\n");
    let e = remote_embedder(&api.base, "");
    let err = index(&vault, &e, &remote_fps(&vault), Run::default()).unwrap_err();
    assert_eq!(kind(&err), Some(RemoteErrorKind::Auth), "{err}");
    assert_eq!(indexed_paths(&vault), ["a.md"]);
    assert!(skipped(&vault).is_empty());
}

#[test]
fn an_input_error_skips_only_that_file() {
    let api = FakeApi::start(reject_marked(4));
    let (_tmp, vault) = remote_vault(&api.base, "", "");
    write(&vault, "a.md", "# a\nalpha\n");
    write(&vault, "b.md", "# b\nREJECT\n");
    write(&vault, "c.md", "# c\ngamma\n");
    let e = remote_embedder(&api.base, "");
    let report = index(&vault, &e, &remote_fps(&vault), Run::default()).unwrap();
    assert_eq!(report.skipped.len(), 1);
    assert_eq!(report.skipped[0].path, "b.md");
    assert!(
        report.skipped[0].reason.contains("400"),
        "{:?}",
        report.skipped
    );
    assert_eq!(indexed_paths(&vault), ["a.md", "c.md"]);
}

#[test]
fn five_input_errors_in_a_row_stop_the_run() {
    let api = FakeApi::start(reject_marked(4));
    let (_tmp, vault) = remote_vault(&api.base, "", "");
    for i in 1..=7 {
        write(
            &vault,
            &format!("f{i}.md"),
            &format!("# f{i}\nREJECT {i}\n"),
        );
    }
    let e = remote_embedder(&api.base, "");
    let err = index(&vault, &e, &remote_fps(&vault), Run::default()).unwrap_err();
    assert_eq!(kind(&err), Some(RemoteErrorKind::Protocol), "{err}");
    assert_eq!(
        skipped(&vault),
        ["f1.md", "f2.md", "f3.md", "f4.md", "f5.md"]
    );
    let sent = api.inputs();
    assert!(
        !sent
            .iter()
            .any(|i| i.contains("REJECT 6") || i.contains("REJECT 7")),
        "{sent:?}"
    );
}

#[test]
fn the_input_streak_restarts_after_a_file_is_embedded() {
    let api = FakeApi::start(reject_marked(4));
    let (_tmp, vault) = remote_vault(&api.base, "", "");
    for i in 1..=4 {
        write(
            &vault,
            &format!("a{i}.md"),
            &format!("# a{i}\nREJECT {i}\n"),
        );
        write(
            &vault,
            &format!("c{i}.md"),
            &format!("# c{i}\nREJECT {i}\n"),
        );
    }
    write(&vault, "b.md", "# b\naccepted\n");
    let e = remote_embedder(&api.base, "");
    let report = index(&vault, &e, &remote_fps(&vault), Run::default()).unwrap();
    assert_eq!(report.skipped.len(), 8);
    assert_eq!(indexed_paths(&vault), ["b.md"]);
}

#[test]
fn a_file_served_from_the_cache_does_not_restart_the_streak() {
    let api = FakeApi::start(reject_marked(4));
    let (_tmp, vault) = remote_vault(&api.base, "", "");
    write(&vault, "b.md", "# b\ncached\n");
    let e = remote_embedder(&api.base, "");
    let fps = remote_fps(&vault);
    index(&vault, &e, &fps, Run::default()).unwrap();
    for i in 1..=4 {
        write(
            &vault,
            &format!("a{i}.md"),
            &format!("# a{i}\nREJECT {i}\n"),
        );
    }
    write(&vault, "c.md", "# c\nREJECT 5\n");
    // The rebuild re-indexes b.md from the cache, between the fourth and the fifth rejection.
    let rebuild = Run {
        rebuild: true,
        ..Run::default()
    };
    let err = index(&vault, &e, &fps, rebuild).unwrap_err();
    assert_eq!(kind(&err), Some(RemoteErrorKind::Protocol), "{err}");
}

#[test]
fn an_answer_of_another_dimension_than_the_index_stops_the_run() {
    // Both directions: a check that only rejects a longer (or a shorter) answer lets one through.
    for (indexed, answered) in [(4, 8), (8, 4)] {
        let api = FakeApi::start(Reply::Embed { dim: indexed });
        let (_tmp, vault) = remote_vault(&api.base, "", "");
        write(&vault, "a.md", "# a\nalpha\n");
        let e = remote_embedder(&api.base, "");
        let fps = remote_fps(&vault);
        index(&vault, &e, &fps, Run::default()).unwrap();
        write(&vault, "b.md", "# b\nbeta\n");
        api.set_default(Reply::Embed { dim: answered });
        let err = index(&vault, &e, &fps, Run::default()).unwrap_err();
        assert_eq!(
            kind(&err),
            Some(RemoteErrorKind::Protocol),
            "{indexed} then {answered}: {err}"
        );
        assert!(err.to_string().contains("model.id"), "{err}");
        assert_eq!(indexed_paths(&vault), ["a.md"]);
        assert_eq!(vectors_of_dim(&vault, answered), 0);
    }
}

#[test]
fn a_rebuild_or_a_new_index_does_not_mix_dimensions_either() {
    for drop_index in [false, true] {
        let api = FakeApi::start(Reply::Embed { dim: 4 });
        let (_tmp, vault) = remote_vault(&api.base, "", "");
        write(&vault, "a.md", "# a\nalpha\n");
        let e = remote_embedder(&api.base, "");
        let fps = remote_fps(&vault);
        index(&vault, &e, &fps, Run::default()).unwrap();
        write(&vault, "b.md", "# b\nbeta\n");
        api.set_default(Reply::Embed { dim: 8 });
        let run = if drop_index {
            // A new index.db next to the kept cache: a.md is a cache hit of the old dimension.
            for suffix in ["", "-wal", "-shm"] {
                let _ = std::fs::remove_file(format!("{}{suffix}", vault.db_path().display()));
            }
            Run::default()
        } else {
            Run {
                rebuild: true,
                ..Run::default()
            }
        };
        let err = index(&vault, &e, &fps, run).unwrap_err();
        assert_eq!(
            kind(&err),
            Some(RemoteErrorKind::Protocol),
            "drop_index={drop_index}: {err}"
        );
        assert_eq!(vectors_of_dim(&vault, 8), 0);
    }
}

#[test]
fn two_dimensions_in_one_new_index_stop_the_run() {
    let api = FakeApi::start(Reply::Embed { dim: 8 });
    api.push(Reply::Embed { dim: 4 });
    let (_tmp, vault) = remote_vault(&api.base, "", "");
    write(&vault, "a.md", "# a\nalpha\n");
    write(&vault, "b.md", "# b\nbeta\n");
    let e = remote_embedder(&api.base, "");
    let err = index(&vault, &e, &remote_fps(&vault), Run::default()).unwrap_err();
    assert_eq!(kind(&err), Some(RemoteErrorKind::Protocol), "{err}");
    assert_eq!(vectors_of_dim(&vault, 8), 0);
}

#[test]
fn a_rebuild_sends_no_request_for_cached_chunks() {
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    let (_tmp, vault) = remote_vault(&api.base, "", "");
    write(&vault, "a.md", "# a\nalpha\n");
    write(&vault, "b.md", "# b\nbeta\n");
    let e = remote_embedder(&api.base, "");
    let fps = remote_fps(&vault);
    index(&vault, &e, &fps, Run::default()).unwrap();
    let sent = api.requests().len();
    let rebuild = Run {
        rebuild: true,
        ..Run::default()
    };
    let report = index(&vault, &e, &fps, rebuild).unwrap();
    assert_eq!((report.rebuilt, report.indexed), (true, 2));
    assert_eq!(api.requests().len(), sent);
}

/// `FakeEmbedder` that reports a host off this machine.
struct OffMachine(FakeEmbedder);

impl Embedder for OffMachine {
    fn model_id(&self) -> &str {
        self.0.model_id()
    }
    fn dim(&self) -> usize {
        self.0.dim()
    }
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.0.embed_documents(texts)
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.0.embed_query(text)
    }
    fn count_tokens(&self, text: &str) -> Result<usize> {
        self.0.count_tokens(text)
    }
    fn remote_host(&self) -> Option<&str> {
        Some("api.example.com")
    }
}

/// The hosts announced, each with how many texts had been embedded when it was.
fn announcements(vault: &Vault, e: &OffMachine, run: Run) -> Vec<(String, usize)> {
    let mut seen = Vec::new();
    index_with_events(vault, e, &fps("a"), run, &mut |event| {
        if let IndexEvent::SendingToRemote { host } = event {
            seen.push((host.to_string(), e.0.embedded_count()));
        }
    })
    .unwrap();
    seen
}

#[test]
fn the_first_request_off_this_machine_is_announced_once() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# a\nalpha\n");
    write(&vault, "b.md", "# b\nbeta\n");
    let e = OffMachine(FakeEmbedder::new(8));
    assert_eq!(
        announcements(&vault, &e, Run::default()),
        [("api.example.com".to_string(), 0)]
    );
}

#[test]
fn a_run_served_from_the_cache_is_not_announced() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# a\nalpha\n");
    let e = OffMachine(FakeEmbedder::new(8));
    announcements(&vault, &e, Run::default());
    let rebuild = Run {
        rebuild: true,
        ..Run::default()
    };
    assert!(announcements(&vault, &e, rebuild).is_empty());
}

#[test]
fn the_deadline_cuts_off_a_slow_file() {
    let api = FakeApi::start(Reply::Delayed(
        Duration::from_secs(5),
        Box::new(Reply::Embed { dim: 4 }),
    ));
    let (_tmp, vault) = remote_vault(&api.base, "", "");
    write(&vault, "a.md", "# a\nalpha\n");
    write(&vault, "b.md", "# b\nbeta\n");
    let e = remote_embedder(&api.base, "");
    let started = Instant::now();
    // Long enough for the setup before the first file, far shorter than the 5 s reply.
    let run = Run {
        deadline: Some(started + Duration::from_secs(1)),
        ..Run::default()
    };
    let report = index(&vault, &e, &remote_fps(&vault), run).unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "{:?}",
        started.elapsed()
    );
    assert_eq!((report.indexed, report.remaining), (0, 2));
    assert!(report.interrupted.is_some());
    assert!(report.skipped.is_empty());
}

#[test]
fn a_passed_deadline_starts_no_file() {
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    let (_tmp, vault) = remote_vault(&api.base, "", "");
    write(&vault, "a.md", "# a\nalpha\n");
    write(&vault, "b.md", "# b\nbeta\n");
    let e = remote_embedder(&api.base, "");
    let run = Run {
        deadline: Some(Instant::now()),
        ..Run::default()
    };
    let report = index(&vault, &e, &remote_fps(&vault), run).unwrap();
    assert_eq!(
        (report.indexed, report.remaining, report.interrupted),
        (0, 2, None)
    );
    assert!(api.requests().is_empty());
}

#[test]
fn a_deadline_replaces_the_budget() {
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    let (_tmp, vault) = remote_vault(&api.base, "", "");
    write(&vault, "a.md", "# a\nalpha\n");
    write(&vault, "b.md", "# b\nbeta\n");
    let e = remote_embedder(&api.base, "");
    let run = Run {
        budget: Some(Duration::ZERO),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        ..Run::default()
    };
    let report = index(&vault, &e, &remote_fps(&vault), run).unwrap();
    assert_eq!((report.indexed, report.remaining), (2, 0));
}
/// a.md and b.md indexed through a healthy API, then b.md removed.
fn indexed_then_b_removed() -> (FakeApi, tempfile::TempDir, Vault) {
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    let (tmp, vault) = remote_vault(&api.base, "", "");
    write(&vault, "a.md", "# a\nalpha\n");
    write(&vault, "b.md", "# b\nbeta\n");
    let e = remote_embedder(&api.base, "");
    index(&vault, &e, &remote_fps(&vault), Run::default()).unwrap();
    std::fs::remove_file(vault.root.join("b.md")).unwrap();
    (api, tmp, vault)
}

#[test]
fn a_passed_deadline_defers_deletions() {
    let (api, _tmp, vault) = indexed_then_b_removed();
    let e = remote_embedder(&api.base, "");
    let run = Run {
        deadline: Some(Instant::now()),
        ..Run::default()
    };
    let report = index(&vault, &e, &remote_fps(&vault), run).unwrap();
    assert_eq!((report.deleted, report.remaining), (0, 1));
    assert_eq!(indexed_paths(&vault), ["a.md", "b.md"]);
}

#[test]
fn a_deadline_lets_deletions_run_despite_a_zero_budget() {
    let (api, _tmp, vault) = indexed_then_b_removed();
    let e = remote_embedder(&api.base, "");
    let run = Run {
        budget: Some(Duration::ZERO),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        ..Run::default()
    };
    let report = index(&vault, &e, &remote_fps(&vault), run).unwrap();
    assert_eq!(report.deleted, 1);
}

#[test]
fn vectors_of_two_dimensions_inside_one_file_stop_the_run() {
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    api.push(Reply::Embed { dim: 4 });
    api.set_default(Reply::Embed { dim: 8 });
    let (_tmp, vault) = remote_vault(&api.base, "", "batch_size = 1");
    let section = |name: &str| {
        format!(
            "# {name}\n{}\n",
            "text of this section, long enough to stay a chunk of its own. ".repeat(2)
        )
    };
    write(
        &vault,
        "a.md",
        &format!("{}{}", section("one"), section("two")),
    );
    let e = remote_embedder(&api.base, "batch_size = 1");
    let err = index(&vault, &e, &remote_fps(&vault), Run::default()).unwrap_err();
    assert_eq!(kind(&err), Some(RemoteErrorKind::Protocol), "{err}");
    assert_eq!(api.requests().len(), 2);
    assert_eq!(
        (vectors_of_dim(&vault, 4), vectors_of_dim(&vault, 8)),
        (0, 0)
    );
}
