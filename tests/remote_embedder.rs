mod support;

use std::time::{Duration, Instant};

use notrieve::app::{self, Paths};
use notrieve::embed::{Embedder, RemoteEmbedder};
use notrieve::fingerprint::remote_fingerprints;
use notrieve::{Error, RemoteErrorKind};
use serde_json::json;
use support::fake_api::{FakeApi, Reply, fake_vector};
use support::{TEST_KEY, TEST_KEY_ENV, remote_contract, remote_embedder, remote_vault};

fn texts(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn kind(err: &Error) -> Option<RemoteErrorKind> {
    match err {
        Error::Remote { kind, .. } => Some(*kind),
        _ => None,
    }
}

/// No retries: one request per call.
fn once(api: &FakeApi, model: &str) -> RemoteEmbedder {
    remote_embedder(&api.base, model).with_retry_delays(&[])
}

#[test]
fn documents_are_posted_with_the_prefix_model_params_and_float_encoding() {
    let api = FakeApi::start(Reply::Embed { dim: 3 });
    let e = remote_embedder(
        &api.base,
        "document_prefix = \"passage: \"\ndocument_params = { input_type = \"document\", dimensions = 3 }",
    );
    let vectors = e.embed_documents(&texts(&["alpha", "beta"])).unwrap();
    let request = &api.requests()[0];
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/embeddings");
    assert_eq!(
        request.body,
        json!({
            "model": "test-model",
            "input": ["passage: alpha", "passage: beta"],
            "encoding_format": "float",
            "input_type": "document",
            "dimensions": 3
        })
    );
    assert_eq!(
        vectors,
        vec![
            fake_vector("passage: alpha", 3),
            fake_vector("passage: beta", 3)
        ]
    );
}

#[test]
fn a_query_is_posted_alone_with_the_query_prefix_and_params() {
    let api = FakeApi::start(Reply::Embed { dim: 3 });
    let e = remote_embedder(
        &api.base,
        "query_prefix = \"query: \"\nquery_params = { input_type = \"query\" }\n\
         document_params = { input_type = \"document\" }",
    );
    let vector = e.embed_query("what").unwrap();
    let body = &api.requests()[0].body;
    assert_eq!(
        (&body["input"], &body["input_type"]),
        (&json!(["query: what"]), &json!("query"))
    );
    assert_eq!(vector, fake_vector("query: what", 3));
}

#[test]
fn inputs_are_sent_in_batches_of_batch_size() {
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    let e = remote_embedder(&api.base, "batch_size = 2");
    let input = texts(&["a", "b", "c", "d", "e"]);
    let vectors = e.embed_documents(&input).unwrap();
    let expected: Vec<Vec<f32>> = input.iter().map(|t| fake_vector(t, 4)).collect();
    assert_eq!(vectors, expected);
    let sizes: Vec<usize> = api.requests().iter().map(|r| r.inputs().len()).collect();
    assert_eq!(sizes, [2, 2, 1]);
}

#[test]
fn vectors_follow_the_input_order_whatever_the_response_order() {
    let api = FakeApi::start(Reply::EmbedReversed { dim: 4 });
    let input = texts(&["a", "b", "c"]);
    let vectors = remote_embedder(&api.base, "")
        .embed_documents(&input)
        .unwrap();
    let expected: Vec<Vec<f32>> = input.iter().map(|t| fake_vector(t, 4)).collect();
    assert_eq!(vectors, expected);
}

#[test]
fn no_texts_send_no_request() {
    let api = FakeApi::start(Reply::Embed { dim: 4 });
    let vectors = remote_embedder(&api.base, "").embed_documents(&[]).unwrap();
    assert!(vectors.is_empty());
    assert!(api.requests().is_empty());
}

#[test]
fn the_api_key_goes_in_a_bearer_authorization_header() {
    let api = FakeApi::start(Reply::Embed { dim: 2 });
    remote_embedder(&api.base, &format!("api_key_env = \"{TEST_KEY_ENV}\""))
        .embed_query("q")
        .unwrap();
    let headers = &api.requests()[0].headers;
    assert_eq!(
        headers.get("authorization"),
        Some(&format!("Bearer {TEST_KEY}"))
    );
}

#[test]
fn another_auth_header_carries_the_bare_key() {
    let api = FakeApi::start(Reply::Embed { dim: 2 });
    remote_embedder(
        &api.base,
        &format!("api_key_env = \"{TEST_KEY_ENV}\"\nauth_header = \"api-key\""),
    )
    .embed_query("q")
    .unwrap();
    let headers = &api.requests()[0].headers;
    assert_eq!(headers.get("api-key"), Some(&TEST_KEY.to_string()));
    assert_eq!(headers.get("authorization"), None);
}

#[test]
fn no_api_key_env_sends_no_credentials() {
    let api = FakeApi::start(Reply::Embed { dim: 2 });
    remote_embedder(&api.base, "").embed_query("q").unwrap();
    let headers = &api.requests()[0].headers;
    assert!(
        !headers.values().any(|v| v.contains(TEST_KEY)),
        "{headers:?}"
    );
    assert_eq!(headers.get("authorization"), None);
}

#[test]
fn a_missing_or_empty_api_key_is_an_auth_error_before_any_request() {
    let api = FakeApi::start(Reply::Embed { dim: 2 });
    let contract = remote_contract(&api.base, "api_key_env = \"NOTRIEVE_KEY\"");
    for value in [None, Some(String::new())] {
        let e = RemoteEmbedder::new(contract.clone()).with_env(move |_| value.clone());
        let err = e.embed_query("q").unwrap_err();
        assert_eq!(kind(&err), Some(RemoteErrorKind::Auth), "{err}");
        assert!(err.to_string().contains("NOTRIEVE_KEY"), "{err}");
    }
    assert!(api.requests().is_empty());
}

#[test]
fn a_429_is_retried_after_the_retry_after_seconds() {
    let api = FakeApi::start(Reply::Embed { dim: 2 });
    api.push(Reply::Status {
        code: 429,
        headers: vec![("Retry-After".into(), "1".into())],
        body: "{}".into(),
    });
    let started = Instant::now();
    let vector = remote_embedder(&api.base, "").embed_query("q").unwrap();
    assert_eq!(vector, fake_vector("q", 2));
    assert_eq!(api.requests().len(), 2);
    assert!(
        started.elapsed() >= Duration::from_secs(1),
        "{:?}",
        started.elapsed()
    );
}

#[test]
fn transient_failures_are_retried_three_times_then_reported() {
    let api = FakeApi::start(Reply::status(503));
    let err = remote_embedder(&api.base, "").embed_query("q").unwrap_err();
    assert_eq!(kind(&err), Some(RemoteErrorKind::Transient), "{err}");
    assert_eq!(api.requests().len(), 4);
}

#[test]
fn with_a_deadline_a_transient_failure_is_not_retried() {
    let api = FakeApi::start(Reply::status(503));
    let deadline = Instant::now() + Duration::from_secs(10);
    let err = remote_embedder(&api.base, "")
        .embed_query_within("q", Some(deadline))
        .unwrap_err();
    assert_eq!(kind(&err), Some(RemoteErrorKind::Transient), "{err}");
    assert_eq!(api.requests().len(), 1);
}

#[test]
fn a_reply_slower_than_the_deadline_fails_at_the_deadline() {
    let api = FakeApi::start(Reply::Delayed(
        Duration::from_secs(5),
        Box::new(Reply::Embed { dim: 2 }),
    ));
    let started = Instant::now();
    let err = remote_embedder(&api.base, "")
        .embed_documents_within(
            &texts(&["slow"]),
            Some(started + Duration::from_millis(300)),
        )
        .unwrap_err();
    assert_eq!(kind(&err), Some(RemoteErrorKind::Transient), "{err}");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "{:?}",
        started.elapsed()
    );
}

#[test]
fn a_passed_deadline_sends_nothing() {
    let api = FakeApi::start(Reply::Embed { dim: 2 });
    let err = remote_embedder(&api.base, "")
        .embed_query_within("q", Some(Instant::now()))
        .unwrap_err();
    assert_eq!(kind(&err), Some(RemoteErrorKind::Transient), "{err}");
    assert!(api.requests().is_empty());
}

#[test]
fn status_codes_map_to_error_kinds() {
    use RemoteErrorKind::*;
    for (code, expected) in [
        (300, Protocol),
        (399, Protocol),
        (400, Input),
        (401, Auth),
        (403, Auth),
        (404, Protocol),
        (405, Protocol),
        (408, Transient),
        (429, Transient),
        (499, Input),
        (500, Transient),
        (599, Transient),
    ] {
        let api = FakeApi::start(Reply::status(code));
        let err = once(&api, "").embed_query("q").unwrap_err();
        assert_eq!(kind(&err), Some(expected), "{code}: {err}");
    }
}

#[test]
fn the_last_2xx_status_still_carries_the_vectors() {
    let api = FakeApi::start(Reply::Status {
        code: 299,
        headers: Vec::new(),
        body: json!({"data": [{"index": 0, "embedding": [0.5, 1.5, 2.5]}]}).to_string(),
    });
    let vectors = once(&api, "").embed_documents(&texts(&["a"])).unwrap();
    assert_eq!(vectors, vec![vec![0.5, 1.5, 2.5]]);
}

#[test]
fn an_unreachable_server_is_a_transient_error() {
    // Bound and dropped: nothing listens on the port any more.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let e = remote_embedder(&format!("http://127.0.0.1:{port}/v1"), "").with_retry_delays(&[]);
    let err = e.embed_query("q").unwrap_err();
    assert_eq!(kind(&err), Some(RemoteErrorKind::Transient), "{err}");
}

#[test]
fn a_redirect_is_not_followed() {
    let api = FakeApi::start(Reply::Embed { dim: 2 });
    api.push(Reply::Status {
        code: 302,
        headers: vec![("Location".into(), "/elsewhere".into())],
        body: String::new(),
    });
    let err = once(&api, "").embed_query("q").unwrap_err();
    assert_eq!(kind(&err), Some(RemoteErrorKind::Protocol), "{err}");
    assert_eq!(api.requests().len(), 1);
}

#[test]
fn an_input_error_keeps_at_most_200_characters_of_the_server_message() {
    let api = FakeApi::start(Reply::Status {
        code: 400,
        headers: Vec::new(),
        body: "x".repeat(300),
    });
    let message = once(&api, "").embed_query("q").unwrap_err().to_string();
    assert!(message.contains(&"x".repeat(200)), "{message}");
    assert!(!message.contains(&"x".repeat(201)), "{message}");
}

#[test]
fn the_api_key_never_appears_in_an_error() {
    let with_key = format!("api_key_env = \"{TEST_KEY_ENV}\"");
    let input_error = |body: String| Reply::Status {
        code: 400,
        headers: Vec::new(),
        body,
    };
    for reply in [
        Reply::status(401),
        input_error(format!("invalid key {TEST_KEY}")),
        // The key sits at the 200-character cut of the server message.
        input_error(format!("{}{TEST_KEY}", "x".repeat(190))),
        // serde repeats the offending string in its error.
        Reply::Body(format!(
            "{{\"data\":[{{\"index\":0,\"embedding\":[\"{TEST_KEY}\"]}}]}}"
        )),
    ] {
        let api = FakeApi::start(reply);
        let message = once(&api, &with_key)
            .embed_query("q")
            .unwrap_err()
            .to_string();
        assert!(!message.contains(TEST_KEY), "{message}");
        assert!(!message.contains(&TEST_KEY[..10]), "{message}");
    }
}

#[test]
fn invalid_responses_are_protocol_errors() {
    let v =
        |index: usize, embedding: &str| format!("{{\"index\":{index},\"embedding\":{embedding}}}");
    let list = |items: Vec<String>| format!("{{\"data\":[{}]}}", items.join(","));
    for body in [
        list(vec![v(0, "[0.1,0.2]")]),
        list(vec![v(0, "[0.1,0.2]"), v(0, "[0.3,0.4]")]),
        list(vec![v(0, "[0.1,0.2]"), v(2, "[0.3,0.4]")]),
        list(vec![v(0, "[0.1,0.2]"), v(1, "[0.3]")]),
        list(vec![v(0, "[]"), v(1, "[]")]),
        list(vec![v(0, "[0.1,1e39]"), v(1, "[0.3,0.4]")]),
        "not json".to_string(),
    ] {
        let api = FakeApi::start(Reply::Body(body.clone()));
        let err = once(&api, "")
            .embed_documents(&texts(&["a", "b"]))
            .unwrap_err();
        assert_eq!(kind(&err), Some(RemoteErrorKind::Protocol), "{body}: {err}");
    }
}

#[test]
fn count_tokens_counts_the_characters_of_the_prefixed_input() {
    let e = RemoteEmbedder::new(remote_contract(
        "https://api.example.com/v1",
        "document_prefix = \"検索文書: \"",
    ));
    assert_eq!(e.count_tokens("あいう").unwrap(), 9);
}

#[test]
fn only_an_endpoint_off_this_machine_reports_its_host() {
    let off = RemoteEmbedder::new(remote_contract("https://api.example.com/v1", ""));
    assert_eq!(off.remote_host(), Some("api.example.com"));
    let local = RemoteEmbedder::new(remote_contract("http://127.0.0.1:8080/v1", ""));
    assert_eq!(local.remote_host(), None);
}

fn paths(tmp: &std::path::Path) -> Paths {
    Paths {
        registry: tmp.join("vaults.toml"),
        model_cache: tmp.join("models"),
    }
}

#[test]
fn a_remote_model_gets_the_remote_fingerprints() {
    let (tmp, vault) = remote_vault("https://api.example.com/v1", "", "");
    let (_, fps) = app::load_model(&vault, &paths(tmp.path()), &mut |_| {}).unwrap();
    let contract = remote_contract("https://api.example.com/v1", "");
    assert_eq!(fps, remote_fingerprints(&contract, 400));
}

#[test]
fn a_remote_model_has_no_token_limit() {
    // 5000 exceeds any local model limit; the local-only check must not apply.
    let (tmp, vault) = remote_vault("https://api.example.com/v1", "", "");
    let config = vault.root.join(".notrieve/config.toml");
    let text = std::fs::read_to_string(&config)
        .unwrap()
        .replace("chunk_max_tokens = 400", "chunk_max_tokens = 5000");
    std::fs::write(&config, text).unwrap();
    let vault = notrieve::config::Vault::load(&vault.root, None).unwrap();
    let (_, fps) = app::load_model(&vault, &paths(tmp.path()), &mut |_| {}).unwrap();
    let contract = remote_contract("https://api.example.com/v1", "");
    assert_eq!(fps.index, remote_fingerprints(&contract, 5000).index);
}

#[test]
fn loading_a_remote_model_downloads_nothing() {
    let (tmp, vault) = remote_vault("https://api.example.com/v1", "", "");
    let paths = paths(tmp.path());
    app::load_model(&vault, &paths, &mut |_| {}).unwrap();
    assert!(!paths.model_cache.exists());
}

#[test]
fn a_response_over_64_mib_is_a_protocol_error() {
    let api = FakeApi::start(Reply::Body("x".repeat((64 << 20) + 1)));
    let err = once(&api, "").embed_query("q").unwrap_err();
    assert_eq!(kind(&err), Some(RemoteErrorKind::Protocol), "{err}");
    assert!(err.to_string().contains("64 MiB"), "{err}");
}

#[test]
fn a_json_escaped_api_key_never_appears_in_an_error() {
    let key = "sk-demo-\"SE\\CRET";
    let escaped = serde_json::to_string(key).unwrap();
    let escaped = escaped.trim_matches('"');
    let api = FakeApi::start(Reply::status(400));
    api.set_default(Reply::Status {
        code: 400,
        headers: Vec::new(),
        body: serde_json::to_string(&format!("invalid key {key}")).unwrap(),
    });
    let e = RemoteEmbedder::new(remote_contract(&api.base, "api_key_env = \"K\""))
        .with_retry_delays(&[])
        .with_env(move |_| Some(key.to_string()));
    let message = e.embed_query("q").unwrap_err().to_string();
    assert!(message.contains("invalid key"), "{message}");
    assert!(!message.contains(key), "{message}");
    assert!(!message.contains(escaped), "{message}");
}

#[test]
fn a_status_decides_the_kind_even_when_its_body_is_cut_short() {
    for (code, expected) in [
        (401, RemoteErrorKind::Auth),
        (404, RemoteErrorKind::Protocol),
    ] {
        let api = FakeApi::start(Reply::Truncated {
            code,
            declared_length: 100,
            body: "short".into(),
        });
        let err = once(&api, "").embed_query("q").unwrap_err();
        assert_eq!(kind(&err), Some(expected), "{code}: {err}");
    }
}
