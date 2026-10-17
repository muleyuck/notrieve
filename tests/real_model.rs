//! Runs only with NOTRIEVE_TEST_MODEL=1 (downloads ~530MB on first run).
use std::path::Path;

use notrieve::config::{ModelSpec, PRESET_RURI_V3_130M, model_cache_dir};
use notrieve::embed::download::ensure_model;
use notrieve::embed::{Embedder, OnnxEmbedder};

fn enabled() -> bool {
    std::env::var("NOTRIEVE_TEST_MODEL").as_deref() == Ok("1")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

fn preset_embedder() -> OnnxEmbedder {
    let contract = ModelSpec::Preset(PRESET_RURI_V3_130M.into())
        .resolve(Path::new("."))
        .unwrap();
    let dir = ensure_model(&contract, &model_cache_dir().unwrap(), false, &mut |_| {}).unwrap();
    OnnxEmbedder::load(&contract, &dir, 2).unwrap()
}

#[test]
fn preset_model_produces_normalised_batch_independent_prefix_aware_vectors() {
    if !enabled() {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1");
        return;
    }
    let e = preset_embedder();
    assert_eq!((e.dim(), e.effective_max_tokens()), (512, 512));
    let texts = vec![
        "瑠璃色は紫みを帯びた濃い青。".to_string(),
        "SQLite の FTS5 で全文検索する。".to_string(),
    ];
    let batch = e.embed_documents(&texts).unwrap();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        batch
            .iter()
            .all(|v| v.len() == 512 && (norm(v) - 1.0).abs() < 1e-4)
    );
    let single = e.embed_documents(&texts[..1]).unwrap();
    assert!(
        cosine(&single[0], &batch[0]) >= 0.99999,
        "batch dependent: {}",
        cosine(&single[0], &batch[0])
    );
    assert_eq!(e.embed_documents(&texts[..1]).unwrap(), single);
    let query = e.embed_query(&texts[0]).unwrap();
    assert!(cosine(&query, &batch[0]) < 0.9999, "prefix not applied");
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

#[test]
fn contract_pooling_takes_effect_and_mean_matches_the_models_own_sentence_embedding() {
    if !enabled() {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1");
        return;
    }
    use fastembed::{
        InitOptionsUserDefined, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
    };
    use notrieve::config::Pooling;
    let mean_contract = ModelSpec::Preset(PRESET_RURI_V3_130M.into())
        .resolve(Path::new("."))
        .unwrap();
    assert_eq!(mean_contract.pooling, Pooling::Mean);
    let dir = ensure_model(
        &mean_contract,
        &model_cache_dir().unwrap(),
        false,
        &mut |_| {},
    )
    .unwrap();
    let mut cls_contract = mean_contract.clone();
    cls_contract.pooling = Pooling::Cls;
    let mean = OnnxEmbedder::load(&mean_contract, &dir, 2).unwrap();
    let cls = OnnxEmbedder::load(&cls_contract, &dir, 2).unwrap();

    // fastembed's default output selection: the model's own pooled `sentence_embedding`.
    let read = |name: &str| std::fs::read(dir.join(name)).unwrap();
    let files = TokenizerFiles {
        tokenizer_file: read("tokenizer.json"),
        config_file: read("config.json"),
        special_tokens_map_file: read("special_tokens_map.json"),
        tokenizer_config_file: read("tokenizer_config.json"),
    };
    let model = UserDefinedEmbeddingModel::new(read(&mean_contract.onnx_file), files)
        .with_pooling(fastembed::Pooling::Mean);
    let mut reference = TextEmbedding::try_new_from_user_defined(
        model,
        InitOptionsUserDefined::new().with_max_length(mean_contract.max_tokens),
    )
    .unwrap();

    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for text in [
        "瑠璃色は紫みを帯びた濃い青。",
        "SQLite の FTS5 で全文検索する。",
        "医療費控除の領収書を集める",
    ] {
        let input = format!("{}{text}", mean_contract.document_prefix);
        let expected = reference.embed([input], None).unwrap().remove(0);
        let by_mean = mean.embed_documents(&[text.to_string()]).unwrap().remove(0);
        let by_cls = cls.embed_documents(&[text.to_string()]).unwrap().remove(0);
        let to_reference = cosine(&by_mean, &expected);
        assert!(to_reference >= 0.99999, "mean drifted: {to_reference}");
        assert!((norm(&by_cls) - 1.0).abs() < 1e-4);
        let mean_vs_cls = cosine(&by_mean, &by_cls);
        assert!(mean_vs_cls < 0.9999, "cls pooling ignored: {mean_vs_cls}");
    }
}

#[test]
fn index_command_indexes_a_small_vault_once() {
    if !enabled() {
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
    if !enabled() {
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
            .env("XDG_CONFIG_HOME", tmp.path().join("config"))
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
