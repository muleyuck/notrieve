use std::{fs, path::Path};

use anyhow::{Context, bail};
use fastembed::{
    InitOptionsUserDefined, Pooling, QuantizationMode, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use serde_json::Value;

use crate::CheckResult;

const MAX_TOKENS: usize = 512;
const MIN_COS_VS_PYTHON: f32 = 0.9999;
const MIN_COS_SINGLE_VS_BATCH: f32 = 0.99999;
const MAX_NORM_DEVIATION: f32 = 1e-4;

pub fn load_model(model_dir: &Path, onnx_file: &str, threads: usize) -> anyhow::Result<TextEmbedding> {
    let read = |name: &str| fs::read(model_dir.join(name)).with_context(|| format!("read {name}"));
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: read("special_tokens_map.json")?,
        tokenizer_config_file: read("tokenizer_config.json")?,
    };
    let model = UserDefinedEmbeddingModel::new(read(onnx_file)?, tokenizer_files)
        .with_pooling(Pooling::Mean)
        .with_quantization(QuantizationMode::None);
    let options = InitOptionsUserDefined::new()
        .with_max_length(MAX_TOKENS)
        .with_intra_threads(threads);
    Ok(TextEmbedding::try_new_from_user_defined(model, options)?)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let denom = dot(a, a).sqrt() * dot(b, b).sqrt();
    if denom > 0.0 { dot(a, b) / denom } else { f32::NEG_INFINITY }
}

fn min_cos(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| cos(x, y))
        .map(|c| if c.is_nan() { f32::NEG_INFINITY } else { c })
        .fold(f32::INFINITY, f32::min)
}

fn report(ok: &mut bool, name: &str, pass: bool, detail: String) {
    println!("{} {name}: {detail}", if pass { "PASS" } else { "FAIL" });
    *ok &= pass;
}

pub fn run(args: &[String]) -> CheckResult {
    let [model_dir, onnx_file, reference] = args else {
        bail!("usage: embed-check <model_dir> <onnx_file> <reference.json>");
    };
    let reference: Value = serde_json::from_slice(&fs::read(reference)?)?;
    let texts: Vec<String> = serde_json::from_value(reference["texts"].clone())?;
    let expected: Vec<Vec<f32>> = serde_json::from_value(reference["onnx_vectors"].clone())?;

    if texts.is_empty() {
        println!("FAIL vector_counts: texts is empty");
        return Ok(false);
    }

    let mut model = load_model(Path::new(model_dir), onnx_file, 1)?;
    let batch = model.embed(&texts, Some(texts.len()))?;
    let single = texts
        .iter()
        .map(|t| model.embed(vec![t.clone()], Some(1)).map(|mut v| v.remove(0)))
        .collect::<Result<Vec<_>, _>>()?;

    let mut ok = true;
    let counts = (texts.len(), expected.len(), batch.len(), single.len());
    let counts_ok = counts.0 > 0 && counts.0 == counts.1 && counts.0 == counts.2 && counts.0 == counts.3;
    report(
        &mut ok,
        "vector_counts",
        counts_ok,
        format!("texts={} python={} rust_batch={} rust_single={}", counts.0, counts.1, counts.2, counts.3),
    );
    if !counts_ok {
        return Ok(false);
    }
    let dim = batch[0].len();
    let all = || expected.iter().chain(&batch).chain(&single);
    let dims_ok = all().all(|v| v.len() == dim);
    report(&mut ok, "vector_dims", dims_ok, format!("all vectors have dim {dim}"));
    let finite_ok = all().all(|v| v.iter().all(|x| x.is_finite()));
    report(&mut ok, "vectors_finite", finite_ok, "all components finite".to_string());
    if !dims_ok || !finite_ok {
        return Ok(false);
    }

    report(&mut ok, "dim", dim == expected[0].len(), format!("rust={dim} python={}", expected[0].len()));
    let vs_python = min_cos(&batch, &expected);
    report(&mut ok, "rust_vs_python_min_cos", vs_python >= MIN_COS_VS_PYTHON, format!("{vs_python:.7}"));
    let vs_batch = min_cos(&single, &batch);
    report(&mut ok, "rust_single_vs_batch_min_cos", vs_batch >= MIN_COS_SINGLE_VS_BATCH, format!("{vs_batch:.7}"));
    let norm_dev = all()
        .map(|v| (dot(v, v).sqrt() - 1.0).abs())
        .map(|d| if d.is_nan() { f32::INFINITY } else { d })
        .fold(0.0, f32::max);
    report(&mut ok, "l2_norm_max_deviation", norm_dev < MAX_NORM_DEVIATION, format!("{norm_dev:e}"));
    Ok(ok)
}
