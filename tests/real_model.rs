//! Runs only with NOTRIEVE_TEST_MODEL=1 (downloads ~150MB on first run).
use std::path::Path;

use notrieve::config::{ModelSpec, PRESET_RURI_V3_30M, model_cache_dir};
use notrieve::embed::download::ensure_model;
use notrieve::embed::{Embedder, OnnxEmbedder};

mod support;

use support::{cosine, enabled};

fn preset_embedder() -> OnnxEmbedder {
    let contract = ModelSpec::Preset(PRESET_RURI_V3_30M.into())
        .resolve(Path::new("."))
        .unwrap()
        .into_local()
        .unwrap();
    let dir = ensure_model(&contract, &model_cache_dir().unwrap(), false, &mut |_| {}).unwrap();
    OnnxEmbedder::load(&contract, &dir, 2).unwrap()
}

fn sample_texts() -> Vec<String> {
    vec![
        "瑠璃色は紫みを帯びた濃い青。".to_string(),
        "SQLite の FTS5 で全文検索する。".to_string(),
    ]
}

#[test]
fn preset_model_produces_normalised_256_dimension_vectors() {
    if !enabled("NOTRIEVE_TEST_MODEL") {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1");
        return;
    }
    let e = preset_embedder();
    assert_eq!((e.dim(), e.effective_max_tokens()), (256, 512));
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let vectors = e.embed_documents(&sample_texts()).unwrap();
    assert_eq!(vectors.len(), 2);
    assert!(
        vectors
            .iter()
            .all(|v| v.len() == 256 && (norm(v) - 1.0).abs() < 1e-4)
    );
}

#[test]
fn preset_model_vectors_do_not_depend_on_the_batch_or_the_run() {
    if !enabled("NOTRIEVE_TEST_MODEL") {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1");
        return;
    }
    let e = preset_embedder();
    let texts = sample_texts();
    let batch = e.embed_documents(&texts).unwrap();
    let single = e.embed_documents(&texts[..1]).unwrap();
    assert!(
        cosine(&single[0], &batch[0]) >= 0.99999,
        "batch dependent: {}",
        cosine(&single[0], &batch[0])
    );
    assert_eq!(e.embed_documents(&texts[..1]).unwrap(), single);
}

#[test]
fn preset_model_applies_the_query_prefix() {
    if !enabled("NOTRIEVE_TEST_MODEL") {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1");
        return;
    }
    let e = preset_embedder();
    let text = &sample_texts()[0];
    let document = e.embed_documents(std::slice::from_ref(text)).unwrap();
    let query = e.embed_query(text).unwrap();
    assert!(cosine(&query, &document[0]) < 0.9999, "prefix not applied");
}

#[test]
fn preset_model_token_ends_are_increasing_char_boundaries() {
    if !enabled("NOTRIEVE_TEST_MODEL") {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1");
        return;
    }
    let e = preset_embedder();
    assert!(e.count_tokens("").unwrap() >= 3);
    let text = "医療費控除の領収書を集める";
    let ends = e.token_ends(text).unwrap();
    assert!(
        ends.len() >= 2 && ends.windows(2).all(|w| w[0] < w[1]),
        "{ends:?}"
    );
    assert_eq!(ends.last().copied(), Some(text.len()));
    assert!(ends.iter().all(|&end| text.is_char_boundary(end)));
}

const POOLING_TEXTS: [&str; 3] = [
    "瑠璃色は紫みを帯びた濃い青。",
    "SQLite の FTS5 で全文検索する。",
    "医療費控除の領収書を集める",
];

/// The preset's contract (mean pooling) and the directory holding its files.
fn mean_contract_and_dir() -> (notrieve::config::OnnxContract, std::path::PathBuf) {
    let contract = ModelSpec::Preset(PRESET_RURI_V3_30M.into())
        .resolve(Path::new("."))
        .unwrap()
        .into_local()
        .unwrap();
    assert_eq!(contract.pooling, notrieve::config::Pooling::Mean);
    let dir = ensure_model(&contract, &model_cache_dir().unwrap(), false, &mut |_| {}).unwrap();
    (contract, dir)
}

#[test]
fn mean_pooling_matches_the_models_own_sentence_embedding() {
    if !enabled("NOTRIEVE_TEST_MODEL") {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1");
        return;
    }
    use fastembed::{
        InitOptionsUserDefined, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
    };
    let (contract, dir) = mean_contract_and_dir();
    let mean = OnnxEmbedder::load(&contract, &dir, 2).unwrap();

    // fastembed's default output selection: the model's own pooled `sentence_embedding`.
    let read = |name: &str| std::fs::read(dir.join(name)).unwrap();
    let files = TokenizerFiles {
        tokenizer_file: read("tokenizer.json"),
        config_file: read("config.json"),
        special_tokens_map_file: read("special_tokens_map.json"),
        tokenizer_config_file: read("tokenizer_config.json"),
    };
    let model = UserDefinedEmbeddingModel::new(read(&contract.onnx_file), files)
        .with_pooling(fastembed::Pooling::Mean);
    let mut reference = TextEmbedding::try_new_from_user_defined(
        model,
        InitOptionsUserDefined::new().with_max_length(contract.max_tokens),
    )
    .unwrap();

    for text in POOLING_TEXTS {
        let input = format!("{}{text}", contract.document_prefix);
        let expected = reference.embed([input], None).unwrap().remove(0);
        let by_mean = mean.embed_documents(&[text.to_string()]).unwrap().remove(0);
        let to_reference = cosine(&by_mean, &expected);
        assert!(to_reference >= 0.99999, "mean drifted: {to_reference}");
    }
}

#[test]
fn cls_pooling_gives_normalised_vectors_unlike_mean() {
    if !enabled("NOTRIEVE_TEST_MODEL") {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1");
        return;
    }
    let (mean_contract, dir) = mean_contract_and_dir();
    let mut cls_contract = mean_contract.clone();
    cls_contract.pooling = notrieve::config::Pooling::Cls;
    let mean = OnnxEmbedder::load(&mean_contract, &dir, 2).unwrap();
    let cls = OnnxEmbedder::load(&cls_contract, &dir, 2).unwrap();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for text in POOLING_TEXTS {
        let by_mean = mean.embed_documents(&[text.to_string()]).unwrap().remove(0);
        let by_cls = cls.embed_documents(&[text.to_string()]).unwrap().remove(0);
        assert!((norm(&by_cls) - 1.0).abs() < 1e-4);
        let mean_vs_cls = cosine(&by_mean, &by_cls);
        assert!(mean_vs_cls < 0.9999, "cls pooling ignored: {mean_vs_cls}");
    }
}

#[test]
fn index_command_indexes_a_small_vault_once() {
    if !enabled("NOTRIEVE_TEST_MODEL") {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1");
        return;
    }
    use notrieve::app::{self, IndexRequest, Paths};
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path().join("vault");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("税金.md"), "# 医療費控除\n領収書を集める。\n").unwrap();
    std::fs::write(
        root.join("検索.md"),
        "# 全文検索\nFTS5 の trigram を使う。\n",
    )
    .unwrap();
    let paths = Paths {
        registry: tmp.path().join("vaults.toml"),
        model_cache: model_cache_dir().unwrap(),
    };
    let vault = app::init(&root, Some("t"), false, &paths).unwrap();
    let request = IndexRequest {
        rebuild: false,
        wait: true,
        budget: None,
    };
    let first = app::index(&vault, &paths, &request, &mut |_| {}).unwrap();
    assert_eq!((first.indexed, first.rebuilt), (2, true));
    let second = app::index(&vault, &paths, &request, &mut |_| {}).unwrap();
    assert_eq!(
        (second.indexed, second.touched, second.rebuilt),
        (0, 0, false)
    );
    let forced = IndexRequest {
        rebuild: true,
        ..request
    };
    let third = app::index(&vault, &paths, &forced, &mut |_| {}).unwrap();
    assert_eq!((third.indexed, third.rebuilt), (2, true));
}

#[test]
fn cli_init_then_index_prints_counts_and_exits_zero() {
    if !enabled("NOTRIEVE_TEST_MODEL") {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1");
        return;
    }
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path().join("notes");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("税金.md"), "# 医療費控除\n領収書を集める。\n").unwrap();
    let run = |args: &[&str]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_notrieve"))
            .args(args)
            .current_dir(&root)
            .env("XDG_STATE_HOME", tmp.path().join("state"))
            .output()
            .unwrap()
    };
    let init = run(&["init"]);
    assert_eq!(
        init.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&init.stderr)
    );
    let out = String::from_utf8_lossy(&init.stdout);
    assert!(
        out.contains("initialized vault `notes`") && out.contains("indexed 1,"),
        "{out}"
    );
    let again = run(&["index"]);
    assert_eq!(again.status.code(), Some(0));
    assert!(
        String::from_utf8_lossy(&again.stdout)
            .starts_with("indexed 0, timestamp-only 0, renamed 0, deleted 0, skipped 0"),
        "{}",
        String::from_utf8_lossy(&again.stdout)
    );
}

#[test]
fn cli_search_json_finds_the_note_and_reports_a_fresh_index() {
    if !enabled("NOTRIEVE_TEST_MODEL") {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1");
        return;
    }
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path().join("notes");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("税金.md"), "# 医療費控除\n領収書を集める。\n").unwrap();
    std::fs::write(root.join("旅行.md"), "# 京都\n紅葉の季節に嵐山を歩いた。\n").unwrap();
    let run = |args: &[&str]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_notrieve"))
            .args(args)
            .current_dir(&root)
            .env("XDG_STATE_HOME", tmp.path().join("state"))
            .output()
            .unwrap()
    };
    assert_eq!(run(&["init"]).status.code(), Some(0));
    let out = run(&["search", "医療費の控除", "--json"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["index_state"], "fresh");
    assert_eq!(v["stale_files"], 0);
    assert_eq!(v["results"][0]["path"], "税金.md");
}

#[test]
fn mcp_search_over_stdio_finds_the_note_and_keeps_stdout_to_json_rpc() {
    if !enabled("NOTRIEVE_TEST_MODEL") {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1");
        return;
    }
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path().join("notes");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("税金.md"), "# 医療費控除\n領収書を集める。\n").unwrap();
    std::fs::write(root.join("旅行.md"), "# 京都\n紅葉の季節に嵐山を歩いた。\n").unwrap();
    let command = |args: &[&str]| {
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_notrieve"));
        command
            .args(args)
            .current_dir(&root)
            .env("XDG_STATE_HOME", tmp.path().join("state"));
        command
    };
    assert_eq!(command(&["init"]).output().unwrap().status.code(), Some(0));
    let (lines, status) = support::mcp_search_session(command(&["mcp"]), "医療費の控除");
    assert!(status.success(), "{status}");
    assert!(lines.iter().all(|l| l["jsonrpc"] == "2.0"), "{lines:?}");
    let result = &lines.last().unwrap()["result"];
    assert_eq!(result["isError"], false, "{result}");
    let v: serde_json::Value =
        serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(v["results"][0]["path"], "税金.md");
}
