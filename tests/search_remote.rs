mod support;

use std::sync::Arc;
use std::time::{Duration, Instant};

use notrieve::app::search_vault;
use notrieve::config::Vault;
use notrieve::embed::RemoteEmbedder;
use notrieve::search::{IndexState, SearchQuery, SearchResponse, VectorSearch};
use notrieve::session::Session;
use notrieve::{Error, RemoteErrorKind, Result};
use support::fake_api::{FakeApi, Reply, inputs};
use support::{
    TEST_KEY, TEST_KEY_ENV, remote_contract, remote_embedder, remote_fps, remote_vault, wait_for,
    write,
};

fn search(vault: &Vault, e: &RemoteEmbedder, text: &str) -> Result<SearchResponse> {
    search_vault(
        vault,
        e,
        &remote_fps(vault),
        &SearchQuery::new(text),
        &mut |_| {},
    )
}

/// A vault whose `alphabet.md` is indexed through `api` (dimension 4).
fn indexed(api: &FakeApi, top: &str, model: &str) -> (tempfile::TempDir, Vault) {
    let (tmp, vault) = remote_vault(&api.base, top, model);
    write(&vault, "alphabet.md", "# alphabet\nthe alphabet song\n");
    let r = search(&vault, &remote_embedder(&api.base, model), "alphabet").unwrap();
    assert_eq!(r.index_state, IndexState::Fresh);
    (tmp, vault)
}

#[test]
fn a_reachable_api_ranks_by_vectors_too() {
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    let (_tmp, vault) = indexed(&api, "", "");
    let r = search(&vault, &remote_embedder(&api.base, ""), "alphabet").unwrap();
    assert_eq!((r.vector_search, r.vector_error), (VectorSearch::Ok, None));
    assert_eq!(r.results[0].path, "alphabet.md");
    assert!(r.results[0].vec_rank.is_some());
}

#[test]
fn a_missing_api_key_falls_back_to_full_text() {
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    let model = format!("api_key_env = \"{TEST_KEY_ENV}\"");
    let (_tmp, vault) = indexed(&api, "", &model);
    let keyless = RemoteEmbedder::new(remote_contract(&api.base, &model)).with_env(|_| None);
    let r = search(&vault, &keyless, "alphabet").unwrap();
    assert_eq!(r.vector_search, VectorSearch::Unavailable);
    let reason = r.vector_error.unwrap();
    assert!(
        reason.contains(TEST_KEY_ENV) && !reason.contains(TEST_KEY),
        "{reason}"
    );
    assert_eq!(r.results[0].path, "alphabet.md");
    assert_eq!(
        (r.results[0].fts_rank, r.results[0].vec_rank),
        (Some(1), None)
    );
}

#[test]
fn an_api_slower_than_the_budget_falls_back_within_the_budget() {
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    let (_tmp, vault) = indexed(&api, "search_budget_ms = 1500", "");
    api.set_default(Reply::Delayed(
        Duration::from_secs(10),
        Box::new(Reply::Embed { dim: 4 }),
    ));
    let started = Instant::now();
    let r = search(&vault, &remote_embedder(&api.base, ""), "alphabet").unwrap();
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_millis(1400) && elapsed < Duration::from_millis(2500),
        "{elapsed:?}"
    );
    assert_eq!(r.vector_search, VectorSearch::Unavailable);
    assert_eq!(r.results[0].path, "alphabet.md");
}

#[test]
fn a_query_vector_of_another_dimension_is_an_error() {
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    let (_tmp, vault) = indexed(&api, "", "");
    api.set_default(Reply::Embed { dim: 8 });
    let err = search(&vault, &remote_embedder(&api.base, ""), "alphabet").unwrap_err();
    assert!(
        matches!(
            err,
            Error::Remote {
                kind: RemoteErrorKind::Protocol,
                ..
            }
        ),
        "{err}"
    );
}

/// 401 for document requests (inputs start with `doc: `), embeddings for queries.
fn documents_refused() -> Reply {
    Reply::by(|body| {
        if inputs(body).iter().any(|i| i.starts_with("doc: ")) {
            Reply::status(401)
        } else {
            Reply::Embed { dim: 4 }
        }
    })
}

#[test]
fn an_update_refused_for_its_key_is_reported_and_search_still_answers() {
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    let model = "document_prefix = \"doc: \"";
    let (_tmp, vault) = indexed(&api, "", model);
    write(&vault, "beta.md", "# beta\nbeta notes\n");
    api.set_default(documents_refused());
    let r = search(&vault, &remote_embedder(&api.base, model), "alphabet").unwrap();
    assert!(r.indexer_error.as_deref().unwrap().contains("401"), "{r:?}");
    assert_eq!(r.results[0].path, "alphabet.md");
    assert_eq!(r.vector_search, VectorSearch::Ok);
}

#[test]
fn the_update_stops_at_its_deadline_and_leaves_time_for_the_query() {
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    let model = "document_prefix = \"doc: \"";
    let (_tmp, vault) = indexed(&api, "search_budget_ms = 2000", model);
    write(&vault, "beta.md", "# beta\nbeta notes\n");
    api.set_default(Reply::by(|body| {
        if inputs(body).iter().any(|i| i.starts_with("doc: ")) {
            Reply::Delayed(Duration::from_secs(10), Box::new(Reply::Embed { dim: 4 }))
        } else {
            Reply::Embed { dim: 4 }
        }
    }));
    let started = Instant::now();
    let r = search(&vault, &remote_embedder(&api.base, model), "alphabet").unwrap();
    assert!(
        started.elapsed() < Duration::from_millis(1500),
        "{:?}",
        started.elapsed()
    );
    assert_eq!((r.stale_files, r.vector_search), (1, VectorSearch::Ok));
}

#[test]
fn a_budget_of_one_second_or_less_skips_the_update() {
    // Without an index, a skipped update leaves nothing to search; an update with no time left
    // would still create the index.
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    let (_tmp, vault) = remote_vault(&api.base, "search_budget_ms = 1000", "");
    write(&vault, "alphabet.md", "# alphabet\nthe alphabet song\n");
    let err = search(&vault, &remote_embedder(&api.base, ""), "alphabet").unwrap_err();
    assert!(matches!(err, Error::IndexNotReady), "{err}");
    assert!(api.requests().is_empty());
    assert!(!vault.db_path().exists());
}

#[test]
fn mcp_search_gives_a_remote_query_five_seconds() {
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    let (_tmp, vault) = remote_vault(&api.base, "", "");
    write(&vault, "alphabet.md", "# alphabet\nthe alphabet song\n");
    let fps = remote_fps(&vault);
    let session =
        Session::with_embedder(vault, Arc::new(remote_embedder(&api.base, "")), fps).unwrap();
    wait_for(|| {
        session
            .search(&SearchQuery::new("alphabet"))
            .ok()
            .filter(|r| r.index_state == IndexState::Fresh)
    });
    api.set_default(Reply::Delayed(
        Duration::from_secs(20),
        Box::new(Reply::Embed { dim: 4 }),
    ));
    let started = Instant::now();
    let r = session.search(&SearchQuery::new("alphabet")).unwrap();
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_millis(4500) && elapsed < Duration::from_millis(6500),
        "{elapsed:?}"
    );
    assert_eq!(r.vector_search, VectorSearch::Unavailable);
}
