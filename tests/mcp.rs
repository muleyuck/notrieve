mod support;

use std::sync::Arc;

use notrieve::embed::Embedder;
use notrieve::embed::fake::FakeEmbedder;
use notrieve::mcp::NotrieveServer;
use notrieve::search::SearchQuery;
use notrieve::session::Session;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::RunningService;
use rmcp::{RoleClient, ServiceExt};
use serde_json::{Value, json};
use support::{PanicEmbedder, fps, index_as_another_process, vault, wait_for, write};
use tempfile::TempDir;

async fn connect(server: NotrieveServer) -> RunningService<RoleClient, ()> {
    let (a, b) = tokio::io::duplex(1 << 16);
    let (ar, aw) = tokio::io::split(a);
    tokio::spawn(async move {
        let running = server.serve((ar, aw)).await.unwrap();
        let _ = running.waiting().await;
    });
    let (br, bw) = tokio::io::split(b);
    ().serve((br, bw)).await.unwrap()
}

async fn call(
    client: &RunningService<RoleClient, ()>,
    name: &'static str,
    args: Value,
) -> CallToolResult {
    client
        .call_tool(
            CallToolRequestParams::new(name).with_arguments(args.as_object().unwrap().clone()),
        )
        .await
        .unwrap()
}

fn text(result: &CallToolResult) -> String {
    result.content[0].as_text().unwrap().text.clone()
}

/// A session over `notes` whose first background sync has finished.
fn synced_session(notes: &[(&str, &str)]) -> (TempDir, Arc<Session>) {
    let (tmp, vault) = vault();
    for (rel, body) in notes {
        write(&vault, rel, body);
    }
    let session =
        Arc::new(Session::with_embedder(vault, Arc::new(FakeEmbedder::new(16)), fps("a")).unwrap());
    wait_for(|| {
        session
            .search(&SearchQuery::new("alpha"))
            .ok()
            .filter(|r| r.index_state == notrieve::search::IndexState::Fresh)
    });
    (tmp, session)
}

#[tokio::test(flavor = "multi_thread")]
async fn tools_are_search_and_get_note_and_instructions_ask_for_sources() {
    let (_tmp, session) = synced_session(&[("a.md", "# A\nalpha\n")]);
    let client = connect(NotrieveServer::new(session)).await;
    let mut names: Vec<String> = client
        .list_tools(Default::default())
        .await
        .unwrap()
        .tools
        .iter()
        .map(|t| t.name.to_string())
        .collect();
    names.sort();
    assert_eq!(names, ["get_note", "search"]);
    let instructions = client.peer_info().unwrap().instructions.clone().unwrap();
    for phrase in ["path:start_line", "get_note", "stale_files"] {
        assert!(instructions.contains(phrase), "{phrase}: {instructions}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn search_tool_returns_hits_with_the_index_state() {
    let (_tmp, session) =
        synced_session(&[("a.md", "# A\nalpha beta\n"), ("b.md", "# B\ngamma\n")]);
    let client = connect(NotrieveServer::new(session)).await;
    let result = call(&client, "search", json!({"query": "alpha", "k": 1})).await;
    assert_eq!(result.is_error, Some(false), "{}", text(&result));
    let v: Value = serde_json::from_str(&text(&result)).unwrap();
    assert_eq!(v["results"].as_array().unwrap().len(), 1);
    assert_eq!(v["results"][0]["path"], "a.md");
    assert_eq!(
        (v["stale_files"].as_u64(), v["index_state"].as_str()),
        (Some(0), Some("fresh"))
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn invalid_query_is_a_tool_error_not_a_crash() {
    let (_tmp, session) = synced_session(&[("a.md", "# A\nalpha\n")]);
    let client = connect(NotrieveServer::new(session)).await;
    let result = call(&client, "search", json!({"query": "   "})).await;
    assert_eq!(result.is_error, Some(true));
    assert!(
        text(&result).contains("the query is empty"),
        "{}",
        text(&result)
    );
    let again = call(&client, "search", json!({"query": "alpha"})).await;
    assert_eq!(again.is_error, Some(false), "the server keeps serving");
}

#[tokio::test(flavor = "multi_thread")]
async fn get_note_returns_the_line_range_and_refuses_paths_outside_the_index() {
    let (tmp, session) = synced_session(&[("a.md", "# A\nalpha\nbeta\n")]);
    std::fs::write(tmp.path().join("outside.md"), "SECRET\n").unwrap();
    let client = connect(NotrieveServer::new(session)).await;
    let result = call(
        &client,
        "get_note",
        json!({"path": "a.md", "start_line": 2, "end_line": 3}),
    )
    .await;
    assert_eq!(result.is_error, Some(false), "{}", text(&result));
    let v: Value = serde_json::from_str(&text(&result)).unwrap();
    assert_eq!(
        (v["text"].as_str(), v["end_line"].as_u64()),
        (Some("alpha\nbeta\n"), Some(3))
    );
    let outside = call(&client, "get_note", json!({"path": "../outside.md"})).await;
    assert_eq!(outside.is_error, Some(true));
    assert!(
        text(&outside).contains("not an indexed note"),
        "{}",
        text(&outside)
    );
    assert!(!text(&outside).contains("SECRET"));
}

#[tokio::test(flavor = "multi_thread")]
async fn settings_changed_after_start_tell_the_agent_to_restart_the_server() {
    let (_tmp, session) = synced_session(&[("a.md", "# A\nalpha\n")]);
    index_as_another_process(session.vault(), &fps("b"));
    let client = connect(NotrieveServer::new(session)).await;
    let result = call(&client, "search", json!({"query": "alpha"})).await;
    assert_eq!(result.is_error, Some(true));
    assert!(
        text(&result).contains("restart the MCP server"),
        "{}",
        text(&result)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_server_that_could_not_start_answers_every_tool_with_the_reason() {
    let reason = "no vault found; run `notrieve init` in your notes directory".to_string();
    let client = connect(NotrieveServer::unavailable(reason.clone())).await;
    for (name, args) in [
        ("search", json!({"query": "x"})),
        ("get_note", json!({"path": "a.md"})),
    ] {
        let result = call(&client, name, args).await;
        assert_eq!(result.is_error, Some(true), "{name}");
        assert_eq!(text(&result), reason, "{name}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn omitted_k_returns_the_default_eight_results_and_a_given_k_is_honoured() {
    let notes: Vec<(String, String)> = (0..10)
        .map(|i| {
            (
                format!("n{i:02}.md"),
                format!("# N{i}\nalpha note number {i}\n"),
            )
        })
        .collect();
    let notes: Vec<(&str, &str)> = notes
        .iter()
        .map(|(p, b)| (p.as_str(), b.as_str()))
        .collect();
    let (_tmp, session) = synced_session(&notes);
    let client = connect(NotrieveServer::new(session)).await;
    let count = |result: &CallToolResult| {
        serde_json::from_str::<Value>(&text(result)).unwrap()["results"]
            .as_array()
            .unwrap()
            .len()
    };
    let default = call(&client, "search", json!({"query": "alpha"})).await;
    assert_eq!(count(&default), 8);
    let three = call(&client, "search", json!({"query": "alpha", "k": 3})).await;
    assert_eq!(count(&three), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_panic_while_handling_a_request_is_a_tool_error_and_the_server_keeps_serving() {
    let (_tmp, vault) = vault();
    write(&vault, "a.md", "# A\nalpha\n");
    let embedder: Arc<dyn Embedder> = Arc::new(PanicEmbedder::new());
    let session = Arc::new(Session::with_embedder(vault, embedder, fps("a")).unwrap());
    wait_for(|| {
        session
            .search(&SearchQuery::new("alpha"))
            .ok()
            .filter(|r| r.index_state == notrieve::search::IndexState::Fresh)
    });
    let client = connect(NotrieveServer::new(session)).await;
    let panicked = call(&client, "search", json!({"query": "PANIC alpha"})).await;
    assert_eq!(panicked.is_error, Some(true));
    assert!(
        text(&panicked).starts_with("internal error"),
        "{}",
        text(&panicked)
    );
    let next = call(&client, "search", json!({"query": "alpha"})).await;
    assert_eq!(next.is_error, Some(false), "{}", text(&next));
}
