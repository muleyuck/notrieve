//! Compares the Rust embeddings of the preset with vectors made by `scripts/reference.py`.
//! The model comparison runs only with NOTRIEVE_TEST_MODEL=1; the `stale_fields` tests always run.
mod support;

use std::path::Path;
use support::{cosine, enabled};

use notrieve::config::{ModelSource, ModelSpec, OnnxContract, PRESET_RURI_V3_30M, model_cache_dir};
use notrieve::embed::download::ensure_model;
use notrieve::embed::onnx::required_files;
use notrieve::embed::{Embedder, OnnxEmbedder};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/reference-ruri-v3-30m.json");
const MIN_COSINE: f32 = 0.9999;

#[derive(Debug, Clone, Deserialize)]
struct Fixture {
    repo: String,
    onnx_file: String,
    files: Vec<FileHash>,
    query_prefix: String,
    document_prefix: String,
    max_tokens: usize,
    cases: Vec<Case>,
}

/// The hash is written before the name so that no line carries both a hash and a name containing
/// "token" (gitleaks flags that as a generic API key).
#[derive(Debug, Clone, Deserialize)]
struct FileHash {
    blake3: String,
    name: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Case {
    kind: String,
    text: String,
    vector: Vec<f32>,
}

fn preset_contract() -> OnnxContract {
    ModelSpec::Preset(PRESET_RURI_V3_30M.into())
        .resolve(Path::new("."))
        .unwrap()
        .into_local()
        .unwrap()
}

/// Names of the fixture fields whose recorded inputs differ from the contract or the model files.
fn stale_fields(fixture: &Fixture, contract: &OnnxContract, model_dir: &Path) -> Vec<String> {
    let ModelSource::Hf { repo } = &contract.source else {
        panic!("the preset is an HF model");
    };
    let mut stale = Vec::new();
    let mut check = |name: &str, same: bool| {
        if !same {
            stale.push(name.to_string());
        }
    };
    check("repo", fixture.repo == *repo);
    check("onnx_file", fixture.onnx_file == contract.onnx_file);
    check(
        "query_prefix",
        fixture.query_prefix == contract.query_prefix,
    );
    check(
        "document_prefix",
        fixture.document_prefix == contract.document_prefix,
    );
    check("max_tokens", fixture.max_tokens == contract.max_tokens);
    let required = required_files(contract);
    for name in &required {
        let recorded: Vec<&FileHash> = fixture.files.iter().filter(|f| f.name == *name).collect();
        let actual = std::fs::read(model_dir.join(name))
            .ok()
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string());
        check(
            &format!("files[{name}]"),
            recorded.len() == 1 && actual.as_deref() == Some(recorded[0].blake3.as_str()),
        );
    }
    let mut extra: Vec<&str> = fixture
        .files
        .iter()
        .map(|f| f.name.as_str())
        .filter(|name| !required.contains(name))
        .collect();
    extra.sort_unstable();
    extra.dedup();
    for name in extra {
        check(&format!("files[{name}]"), false);
    }
    stale
}

/// A fixture that matches the preset contract and a temp dir holding its 5 files.
fn matching_fixture() -> (Fixture, OnnxContract, tempfile::TempDir) {
    let contract = preset_contract();
    let dir = tempfile::tempdir().unwrap();
    let mut files = Vec::new();
    for (i, name) in required_files(&contract).into_iter().enumerate() {
        let path = dir.path().join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let content = format!("content {i} of {name}");
        std::fs::write(&path, &content).unwrap();
        files.push(FileHash {
            blake3: blake3::hash(content.as_bytes()).to_hex().to_string(),
            name: name.to_string(),
        });
    }
    let ModelSource::Hf { repo } = &contract.source else {
        panic!("the preset is an HF model");
    };
    let fixture = Fixture {
        repo: repo.clone(),
        onnx_file: contract.onnx_file.clone(),
        files,
        query_prefix: contract.query_prefix.clone(),
        document_prefix: contract.document_prefix.clone(),
        max_tokens: contract.max_tokens,
        cases: Vec::new(),
    };
    (fixture, contract, dir)
}

#[test]
fn stale_fields_is_empty_when_contract_and_files_match() {
    let (fixture, contract, dir) = matching_fixture();
    assert_eq!(
        stale_fields(&fixture, &contract, dir.path()),
        Vec::<String>::new()
    );
}

#[test]
fn stale_fields_names_a_changed_prefix() {
    let (mut fixture, contract, dir) = matching_fixture();
    fixture.document_prefix = "検索文書:".to_string();
    assert_eq!(
        stale_fields(&fixture, &contract, dir.path()),
        vec!["document_prefix"]
    );
}

#[test]
fn stale_fields_names_a_file_whose_hash_changed() {
    let (fixture, contract, dir) = matching_fixture();
    std::fs::write(dir.path().join("onnx/model.onnx"), "another model").unwrap();
    assert_eq!(
        stale_fields(&fixture, &contract, dir.path()),
        vec!["files[onnx/model.onnx]"]
    );
}

#[test]
fn stale_fields_names_missing_and_extra_files() {
    let (mut fixture, contract, dir) = matching_fixture();
    fixture.files.retain(|f| f.name != "config.json");
    fixture.files.push(FileHash {
        blake3: "0".repeat(64),
        name: "extra.bin".to_string(),
    });
    assert_eq!(
        stale_fields(&fixture, &contract, dir.path()),
        vec!["files[config.json]", "files[extra.bin]"]
    );
}

#[test]
fn stale_fields_names_a_file_recorded_twice() {
    let (mut fixture, contract, dir) = matching_fixture();
    fixture.files.push(fixture.files[0].clone());
    let name = fixture.files[0].name.clone();
    assert_eq!(
        stale_fields(&fixture, &contract, dir.path()),
        vec![format!("files[{name}]")]
    );
}

#[test]
fn preset_vectors_match_the_python_reference() {
    if !enabled("NOTRIEVE_TEST_MODEL") {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1");
        return;
    }
    let fixture: Fixture = serde_json::from_str(FIXTURE).unwrap();
    let contract = preset_contract();
    let cache = model_cache_dir().unwrap();
    let dir = ensure_model(&contract, &cache, false, &mut |_| {}).unwrap();
    let stale = stale_fields(&fixture, &contract, &dir);
    assert!(
        stale.is_empty(),
        "fixture is stale: {}; regenerate with `uv run --script scripts/reference.py`",
        stale.join(", ")
    );

    let embedder = OnnxEmbedder::load(&contract, &dir, 2).unwrap();
    let documents: Vec<&Case> = fixture
        .cases
        .iter()
        .filter(|c| c.kind == "document")
        .collect();
    let texts: Vec<String> = documents.iter().map(|c| c.text.clone()).collect();
    let mut document_vectors = embedder.embed_documents(&texts).unwrap().into_iter();

    let mut min = f32::MAX;
    for case in &fixture.cases {
        let actual = match case.kind.as_str() {
            "query" => embedder.embed_query(&case.text).unwrap(),
            "document" => document_vectors.next().unwrap(),
            other => panic!("unknown case kind `{other}`"),
        };
        let cos = cosine(&actual, &case.vector);
        assert!(
            cos >= MIN_COSINE,
            "vector differs from the reference for {:?}: cosine {cos}",
            case.text
        );
        min = min.min(cos);
    }
    eprintln!("minimum cosine against the Python reference: {min}");
}
