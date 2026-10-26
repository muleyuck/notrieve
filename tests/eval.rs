//! Search quality (spec §7). Runs only with NOTRIEVE_TEST_MODEL=1.
mod support;

use std::fs;
use std::path::{Path, PathBuf};

use notrieve::app::{self, Paths};
use notrieve::config::{Vault, model_cache_dir};
use notrieve::embed::Embedder;
use notrieve::fingerprint::Fingerprints;
use notrieve::search::SearchQuery;
use serde::Deserialize;
use support::{enabled, write_config};
use tempfile::TempDir;

/// The default model is ruri-v3-30m fp32. Measurement on macOS arm64 (2026-10-24, 35 queries over
/// 25 notes): Recall@5 1.000 (Recall@1 0.714, MRR@5 0.838). One 0.05 step lower for numeric
/// differences on other platforms (spec §10.5).
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
    write_config(&root, config);
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
    misses: Vec<String>,
}

fn recall_at_5(vault: &Vault, paths: &Paths) -> Outcome {
    let (embedder, fps) = app::load_model(vault, paths, &mut |_| {}).unwrap();
    run_queries(vault, embedder.as_ref(), &fps)
}

fn run_queries(vault: &Vault, embedder: &dyn Embedder, fps: &Fingerprints) -> Outcome {
    let text = fs::read_to_string(eval_dir().join("queries.toml")).unwrap();
    let queries: Queries = toml::from_str(&text).unwrap();
    let mut found = 0;
    let mut misses = Vec::new();
    let mut ranks = Vec::new();
    for q in &queries.query {
        let query = SearchQuery {
            k: 5,
            ..SearchQuery::new(&q.text)
        };
        let response = app::search_vault(vault, embedder, fps, &query, &mut |_| {}).unwrap();
        assert_eq!(
            response.stale_files, 0,
            "the evaluation vault was not fully indexed"
        );
        let rank = response
            .results
            .iter()
            .position(|h| h.path == q.path && h.heading_path.contains(&q.heading))
            .map(|p| p + 1);
        ranks.push(rank);
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
        recall_at_1: ranks.iter().filter(|r| **r == Some(1)).count() as f64 / n,
        mrr_at_5: ranks
            .iter()
            .map(|r| r.map_or(0.0, |r| 1.0 / r as f64))
            .sum::<f64>()
            / n,
        misses,
    }
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
