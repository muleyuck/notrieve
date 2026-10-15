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
