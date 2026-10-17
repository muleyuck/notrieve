use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::config::ModelContract;
use crate::embed::onnx::required_files;
use crate::markdown::chunk::{CHUNKER_VERSION, MERGE_BELOW_TOKENS};
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fingerprints {
    /// Model contract and model files; also keys the embedding cache (spec §4.1).
    pub model: String,
    /// `model` plus the chunk settings, recorded as `index_fingerprint`.
    pub index: String,
}

pub fn fingerprints(
    contract: &ModelContract,
    model_dir: &Path,
    chunk_max_tokens: usize,
) -> Result<Fingerprints> {
    let model = model_fingerprint(contract, model_dir)?;
    let index = index_fingerprint(&model, chunk_max_tokens);
    Ok(Fingerprints { model, index })
}

/// Hash of the model contract and every model file.
pub fn model_fingerprint(contract: &ModelContract, model_dir: &Path) -> Result<String> {
    let mut hasher = blake3::Hasher::new();
    let contract_json = serde_json::to_string(contract).map_err(|e| Error::Model(e.to_string()))?;
    hasher.update(contract_json.as_bytes());
    for file in required_files(contract) {
        hasher.update(file.as_bytes());
        hasher.update(file_hash(&model_dir.join(file))?.as_bytes());
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// Hash of every input that changes index contents (spec §4.1).
pub fn index_fingerprint(model_fingerprint: &str, chunk_max_tokens: usize) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(model_fingerprint.as_bytes());
    hasher.update(
        format!(
            ";chunker={CHUNKER_VERSION};max={chunk_max_tokens};merge_below={MERGE_BELOW_TOKENS}"
        )
        .as_bytes(),
    );
    hasher.finalize().to_hex().to_string()
}

/// Content hash cached next to the file, keyed by size and mtime, so large ONNX files are hashed once.
fn file_hash(path: &Path) -> Result<String> {
    let meta = fs::metadata(path).map_err(|e| Error::io(path, e))?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    let stamp = format!("{} {mtime}", meta.len());
    let cache = sidecar(path);
    if let Ok(text) = fs::read_to_string(&cache)
        && let Some((cached_stamp, hash)) = text.trim_end().rsplit_once(' ')
        && cached_stamp == stamp
    {
        return Ok(hash.to_string());
    }
    let mut hasher = blake3::Hasher::new();
    hasher
        .update_reader(File::open(path).map_err(|e| Error::io(path, e))?)
        .map_err(|e| Error::io(path, e))?;
    let hash = hasher.finalize().to_hex().to_string();
    let _ = fs::write(&cache, format!("{stamp} {hash}\n"));
    Ok(hash)
}

fn sidecar(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!(".{name}.blake3"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelSource, Pooling};
    use std::fs;
    use tempfile::TempDir;

    fn model(dir: &Path) -> ModelContract {
        fs::create_dir_all(dir.join("onnx")).unwrap();
        for (f, body) in [
            ("tokenizer.json", "tok"),
            ("config.json", "cfg"),
            ("special_tokens_map.json", "special"),
            ("tokenizer_config.json", "tokcfg"),
            ("onnx/model.onnx", "weights"),
        ] {
            fs::write(dir.join(f), body).unwrap();
        }
        ModelContract {
            id: "m".into(),
            source: ModelSource::Path {
                dir: dir.to_path_buf(),
            },
            onnx_file: "onnx/model.onnx".into(),
            dim: 4,
            max_tokens: 64,
            pooling: Pooling::Mean,
            query_prefix: "q: ".into(),
            document_prefix: "d: ".into(),
        }
    }

    #[test]
    fn same_inputs_give_the_same_fingerprints() {
        let dir = TempDir::new().unwrap();
        let c = model(dir.path());
        assert_eq!(
            fingerprints(&c, dir.path(), 450).unwrap(),
            fingerprints(&c, dir.path(), 450).unwrap()
        );
    }

    #[test]
    fn changing_a_tokenizer_side_file_changes_both_fingerprints() {
        let dir = TempDir::new().unwrap();
        let c = model(dir.path());
        let before = fingerprints(&c, dir.path(), 450).unwrap();
        fs::write(
            dir.path().join("special_tokens_map.json"),
            "changed special tokens",
        )
        .unwrap();
        let after = fingerprints(&c, dir.path(), 450).unwrap();
        assert_ne!(after.model, before.model);
        assert_ne!(after.index, before.index);
    }

    #[test]
    fn chunk_max_tokens_changes_only_the_index_fingerprint() {
        let dir = TempDir::new().unwrap();
        let c = model(dir.path());
        let base = fingerprints(&c, dir.path(), 450).unwrap();
        let other = fingerprints(&c, dir.path(), 300).unwrap();
        assert_eq!(other.model, base.model);
        assert_ne!(other.index, base.index);
    }

    #[test]
    fn changing_the_contract_changes_the_model_fingerprint() {
        let dir = TempDir::new().unwrap();
        let c = model(dir.path());
        let base = fingerprints(&c, dir.path(), 450).unwrap();
        let other = ModelContract {
            document_prefix: "passage: ".into(),
            ..c
        };
        let changed = fingerprints(&other, dir.path(), 450).unwrap();
        assert_ne!(changed.model, base.model);
        assert_ne!(changed.index, base.index);
    }
}
