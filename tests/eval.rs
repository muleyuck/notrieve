//! Search quality (spec §7). Runs only with NOTRIEVE_TEST_MODEL=1; the comparison of candidate
//! models (spec §10.1) also needs NOTRIEVE_EVAL_COMPARE=1.
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use notrieve::app::{self, Paths};
use notrieve::config::{Vault, model_cache_dir};
use notrieve::embed::{Embedder, OnnxEmbedder};
use notrieve::fingerprint::Fingerprints;
use notrieve::search::SearchQuery;
use serde::Deserialize;
use tempfile::TempDir;

/// Measurement with ruri-v3-130m fp32 on macOS arm64 (2026-10-17, 35 queries over 25 notes):
/// Recall@5 1.000 (Recall@1 0.943, MRR@5 0.967). One 0.05 step lower for numeric differences on
/// other platforms (spec §10.5).
const RECALL_AT_5_THRESHOLD: f64 = 0.95;

#[derive(Deserialize)]
struct Queries {
    query: Vec<EvalQuery>,
}

#[derive(Deserialize)]
struct EvalQuery {
    text: String,
    path: String,
    heading: String,
}

fn enabled(var: &str) -> bool {
    std::env::var(var).as_deref() == Ok("1")
}

fn eval_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/eval")
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// A copy of the evaluation vault with `config` as its config.toml (the repository stays untouched).
fn eval_vault(tmp: &Path, config: &str) -> Vault {
    let root = tmp.join("vault");
    copy_dir(&eval_dir().join("vault"), &root);
    fs::create_dir_all(root.join(".notrieve")).unwrap();
    fs::write(root.join(".notrieve/config.toml"), config).unwrap();
    Vault::load(&root, None).unwrap()
}

fn paths(tmp: &Path) -> Paths {
    Paths {
        registry: tmp.join("vaults.toml"),
        model_cache: model_cache_dir().unwrap(),
    }
}

/// The budget covers indexing the whole evaluation vault on the first query.
const EVAL_CONFIG: &str = "search_budget_ms = 600000\n";

struct Outcome {
    recall: f64,
    recall_at_1: f64,
    /// Mean reciprocal rank within the top 5 (0 for a query with no hit).
    mrr_at_5: f64,
    /// Rank (1-based) of the first correct result per query, with the query text.
    ranks: Vec<(String, Option<usize>)>,
    misses: Vec<String>,
    /// The first query, which indexes the whole vault.
    index_time: Duration,
    /// Mean time of the queries after the first, when the index is warm.
    mean_query_time: Duration,
}

fn recall_at_5(vault: &Vault, paths: &Paths) -> Outcome {
    let (embedder, fps) = app::load_model(vault, paths, &mut |_| {}).unwrap();
    run_queries(vault, &embedder, &fps)
}

fn run_queries(vault: &Vault, embedder: &OnnxEmbedder, fps: &Fingerprints) -> Outcome {
    let text = fs::read_to_string(eval_dir().join("queries.toml")).unwrap();
    let queries: Queries = toml::from_str(&text).unwrap();
    let mut found = 0;
    let mut misses = Vec::new();
    let mut ranks = Vec::new();
    let mut index_time = Duration::ZERO;
    let mut rest_time = Duration::ZERO;
    for (i, q) in queries.query.iter().enumerate() {
        let started = Instant::now();
        let query = SearchQuery {
            k: 5,
            ..SearchQuery::new(&q.text)
        };
        let response = app::search_vault(vault, embedder, fps, &query, &mut |_| {}).unwrap();
        if i == 0 {
            index_time = started.elapsed();
        } else {
            rest_time += started.elapsed();
        }
        assert_eq!(
            response.stale_files, 0,
            "the evaluation vault was not fully indexed"
        );
        let rank = response
            .results
            .iter()
            .position(|h| h.path == q.path && h.heading_path.contains(&q.heading))
            .map(|p| p + 1);
        ranks.push((q.text.clone(), rank));
        if rank.is_some() {
            found += 1;
        } else {
            let got: Vec<String> = response
                .results
                .iter()
                .map(|h| format!("{} [{}]", h.path, h.heading_path))
                .collect();
            misses.push(format!("{} -> {got:?}", q.text));
        }
    }
    let n = queries.query.len() as f64;
    Outcome {
        recall: found as f64 / n,
        recall_at_1: ranks.iter().filter(|(_, r)| *r == Some(1)).count() as f64 / n,
        mrr_at_5: ranks
            .iter()
            .map(|(_, r)| r.map_or(0.0, |r| 1.0 / r as f64))
            .sum::<f64>()
            / n,
        ranks,
        misses,
        index_time,
        mean_query_time: rest_time / (queries.query.len() as u32 - 1),
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

fn note_texts(dir: &Path, out: &mut Vec<String>) {
    let mut entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        if entry.file_type().unwrap().is_dir() {
            note_texts(&entry.path(), out);
        } else {
            out.push(fs::read_to_string(entry.path()).unwrap());
        }
    }
}

/// Smallest cosine similarity between a text embedded alone and inside a batch of eight.
fn batch_independence(vault: &Vault, paths: &Paths) -> f32 {
    let (embedder, _) = app::load_model(vault, paths, &mut |_| {}).unwrap();
    let mut texts = Vec::new();
    note_texts(&eval_dir().join("vault"), &mut texts);
    let texts: Vec<String> = texts
        .into_iter()
        .take(8)
        .map(|t| t.chars().take(200).collect())
        .collect();
    let batch = embedder.embed_documents(&texts).unwrap();
    texts
        .iter()
        .zip(&batch)
        .map(|(text, in_batch)| {
            let alone = embedder
                .embed_documents(std::slice::from_ref(text))
                .unwrap()
                .remove(0);
            cosine(&alone, in_batch)
        })
        .fold(f32::INFINITY, f32::min)
}

#[test]
fn default_model_recall_at_5_meets_the_threshold() {
    if !enabled("NOTRIEVE_TEST_MODEL") {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1");
        return;
    }
    let tmp = TempDir::new().unwrap();
    let vault = eval_vault(tmp.path(), EVAL_CONFIG);
    let outcome = recall_at_5(&vault, &paths(tmp.path()));
    eprintln!(
        "Recall@5 = {:.3}, Recall@1 = {:.3}, MRR@5 = {:.3}",
        outcome.recall, outcome.recall_at_1, outcome.mrr_at_5
    );
    for miss in &outcome.misses {
        eprintln!("miss: {miss}");
    }
    assert!(
        outcome.recall >= RECALL_AT_5_THRESHOLD,
        "Recall@5 {:.3} is below {RECALL_AT_5_THRESHOLD}",
        outcome.recall
    );
}

#[test]
fn compare_candidate_models() {
    if !(enabled("NOTRIEVE_TEST_MODEL") && enabled("NOTRIEVE_EVAL_COMPARE")) {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1 and NOTRIEVE_EVAL_COMPARE=1");
        return;
    }
    let candidates = [
        (
            "ruri-v3-130m-fp32",
            "sirasagi62/ruri-v3-130m-ONNX",
            "onnx/model.onnx",
            512,
        ),
        (
            "ruri-v3-130m-int8",
            "sirasagi62/ruri-v3-130m-ONNX",
            "onnx/model_int8.onnx",
            512,
        ),
        (
            "ruri-v3-30m-fp32",
            "sirasagi62/ruri-v3-30m-ONNX",
            "onnx/model.onnx",
            256,
        ),
        (
            "ruri-v3-30m-int8",
            "sirasagi62/ruri-v3-30m-ONNX",
            "onnx/model_int8.onnx",
            256,
        ),
    ];
    for (id, repo, onnx_file, dim) in candidates {
        let tmp = TempDir::new().unwrap();
        let config = format!(
            "{EVAL_CONFIG}\n[model]\nid = \"{id}\"\nsource = \"hf:{repo}\"\nonnx_file = \"{onnx_file}\"\ndim = {dim}\nmax_tokens = 512\npooling = \"mean\"\nquantization = \"none\"\nquery_prefix = \"検索クエリ: \"\ndocument_prefix = \"検索文書: \"\n"
        );
        let vault = eval_vault(tmp.path(), &config);
        let paths = paths(tmp.path());
        // Model download and load stay outside the timed part.
        let (embedder, fps) = app::load_model(&vault, &paths, &mut |_| {}).unwrap();
        let outcome = run_queries(&vault, &embedder, &fps);
        let min_cos = batch_independence(&vault, &paths);
        println!(
            "{id}: Recall@5 {:.3} ({} misses), Recall@1 {:.3}, MRR@5 {:.3}, single-vs-batch min cos {min_cos:.7}, index+first search {:.1}s, mean query {:.0}ms",
            outcome.recall,
            outcome.misses.len(),
            outcome.recall_at_1,
            outcome.mrr_at_5,
            outcome.index_time.as_secs_f64(),
            outcome.mean_query_time.as_secs_f64() * 1000.0
        );
        for (i, (text, rank)) in outcome.ranks.iter().enumerate() {
            let rank = rank.map_or("-".to_string(), |r| r.to_string());
            println!("  {:>2} rank {rank} {text}", i + 1);
        }
        for miss in &outcome.misses {
            println!("  miss: {miss}");
        }
    }
}
