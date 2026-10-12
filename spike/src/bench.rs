use std::{path::Path, time::Instant};

use anyhow::bail;

use crate::{CheckResult, embed_check::load_model};

const PARAGRAPH: &str = "ハイブリッド検索では、全文検索の順位とベクトル検索の順位を Reciprocal Rank Fusion で統合する。";
const QUERY: &str = "検索クエリ: ハイブリッド検索の統合方法";
const QUERY_RUNS: u32 = 20;

pub fn run(args: &[String]) -> CheckResult {
    let [model_dir, onnx_file, batch, threads, n] = args else {
        bail!("usage: bench <model_dir> <onnx_file> <batch_size> <threads> <n_chunks>");
    };
    let (batch, threads, n): (usize, usize, usize) = (batch.parse()?, threads.parse()?, n.parse()?);

    let started = Instant::now();
    let mut model = load_model(Path::new(model_dir), onnx_file, threads)?;
    let load_ms = started.elapsed().as_millis();

    let docs: Vec<String> = (0..n)
        .map(|i| format!("検索文書: 第{i}節。{}", PARAGRAPH.repeat(8)))
        .collect();
    model.embed(&docs[..batch.min(n)], Some(batch))?;
    let started = Instant::now();
    model.embed(&docs, Some(batch))?;
    let secs = started.elapsed().as_secs_f64();

    let query = vec![QUERY.to_string()];
    model.embed(&query, Some(1))?;
    let started = Instant::now();
    for _ in 0..QUERY_RUNS {
        model.embed(&query, Some(1))?;
    }
    let query_ms = started.elapsed().as_secs_f64() * 1000.0 / f64::from(QUERY_RUNS);

    println!(
        "batch={batch} threads={threads} n={n} load_ms={load_ms} chunks_per_sec={:.2} batch_ms={:.0} query_ms={query_ms:.1}",
        n as f64 / secs,
        secs * 1000.0 / n.div_ceil(batch) as f64,
    );
    Ok(true)
}
