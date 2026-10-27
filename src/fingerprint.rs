use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::config::{OnnxContract, RemoteContract};
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
    contract: &OnnxContract,
    model_dir: &Path,
    chunk_max_tokens: usize,
) -> Result<Fingerprints> {
    let model = model_fingerprint(contract, model_dir)?;
    let index = index_fingerprint(&model, chunk_max_tokens);
    Ok(Fingerprints { model, index })
}

/// Hash of the model contract and every model file.
pub fn model_fingerprint(contract: &OnnxContract, model_dir: &Path) -> Result<String> {
    let mut hasher = blake3::Hasher::new();
    let contract_json = serde_json::to_string(contract).map_err(|e| Error::Model(e.to_string()))?;
    hasher.update(contract_json.as_bytes());
    for file in required_files(contract) {
        hasher.update(file.as_bytes());
        hasher.update(file_hash(&model_dir.join(file))?.as_bytes());
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// Remote spec §4. No model files: the fingerprint covers what decides the document vectors.
pub fn remote_fingerprints(contract: &RemoteContract, chunk_max_tokens: usize) -> Fingerprints {
    let model = remote_model_fingerprint(contract);
    let index = index_fingerprint(&model, chunk_max_tokens);
    Fingerprints { model, index }
}

/// The base URL, authentication, `batch_size` and the query settings are left out: they do not
/// change stored document vectors, so moving to another host does not re-embed every chunk.
pub fn remote_model_fingerprint(contract: &RemoteContract) -> String {
    // serde_json's map is ordered by key (its `preserve_order` feature is off), so the JSON does
    // not depend on the order the TOML lists the params in.
    let json = serde_json::json!({
        "kind": "remote",
        "id": contract.id,
        "remote_model": contract.remote_model,
        "document_prefix": contract.document_prefix,
        "document_params": contract.document_params,
    });
    blake3::hash(json.to_string().as_bytes())
        .to_hex()
        .to_string()
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
    // Only a regular file: a link or a FIFO planted in a model directory is not read.
    let regular = fs::symlink_metadata(&cache).is_ok_and(|m| m.is_file());
    if regular
        && let Ok(text) = fs::read_to_string(&cache)
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
    let _ = replace_file(&cache, &format!("{stamp} {hash}\n"));
    Ok(hash)
}

/// Writes a new file and renames it over `path`. A model directory can sit in a vault that came
/// from someone else, and a rename replaces a link at `path` instead of writing through it.
fn replace_file(path: &Path, text: &str) -> std::io::Result<()> {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{}.tmp", std::process::id()));
    let tmp = path.with_file_name(name);
    let mut file = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
    let written = file
        .write_all(text.as_bytes())
        .and_then(|()| fs::rename(&tmp, path));
    if written.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    written
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
    use crate::config::{ModelContract, OnnxContract, RemoteContract, VaultConfig};
    use crate::config::{ModelSource, Pooling};
    use serde_json::{Map, json};
    use std::fs;
    use tempfile::TempDir;

    fn model(dir: &Path) -> OnnxContract {
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
        OnnxContract {
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

    #[cfg(unix)]
    #[test]
    fn a_hash_cache_that_is_not_a_regular_file_is_not_read() {
        use std::os::unix::ffi::OsStrExt;
        let dir = TempDir::new().unwrap();
        let c = model(dir.path());
        // Reading a FIFO would block until a writer appears.
        let fifo = dir.path().join(".tokenizer.json.blake3");
        let path = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
        let (tx, rx) = std::sync::mpsc::channel();
        let root = dir.path().to_path_buf();
        std::thread::spawn(move || {
            let _ = tx.send(fingerprints(&c, &root, 450).is_ok());
        });
        let finished = rx.recv_timeout(std::time::Duration::from_secs(5));
        assert_eq!(finished, Ok(true));
    }

    #[cfg(unix)]
    #[test]
    fn a_hash_cache_reached_through_a_link_is_not_trusted() {
        let dir = TempDir::new().unwrap();
        model(dir.path());
        let file = dir.path().join("tokenizer.json");
        let real = file_hash(&file).unwrap();
        let cache = sidecar(&file);
        let stamp = fs::read_to_string(&cache).unwrap();
        let (stamp, _) = stamp.trim_end().rsplit_once(' ').unwrap();
        let forged = dir.path().join("forged");
        fs::write(&forged, format!("{stamp} forged-hash\n")).unwrap();
        fs::remove_file(&cache).unwrap();
        std::os::unix::fs::symlink(&forged, &cache).unwrap();
        assert_eq!(file_hash(&file).unwrap(), real);
    }

    #[cfg(unix)]
    #[test]
    fn a_link_at_a_hash_cache_file_is_replaced_not_written_through() {
        let dir = TempDir::new().unwrap();
        let victim = dir.path().join("victim.txt");
        fs::write(&victim, "keep").unwrap();
        let models = dir.path().join("models");
        let c = model(&models);
        let cache = models.join(".tokenizer.json.blake3");
        std::os::unix::fs::symlink(&victim, &cache).unwrap();
        fingerprints(&c, &models, 450).unwrap();
        assert_eq!(fs::read_to_string(&victim).unwrap(), "keep");
        assert!(!cache.symlink_metadata().unwrap().file_type().is_symlink());
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
        let other = OnnxContract {
            document_prefix: "passage: ".into(),
            ..c
        };
        let changed = fingerprints(&other, dir.path(), 450).unwrap();
        assert_ne!(changed.model, base.model);
        assert_ne!(changed.index, base.index);
    }

    #[test]
    fn a_local_contract_keeps_the_fingerprint_it_had_before_remote_models() {
        let dir = TempDir::new().unwrap();
        let c = OnnxContract {
            source: ModelSource::Hf {
                repo: "owner/repo".into(),
            },
            ..model(dir.path())
        };
        let fps = fingerprints(&c, dir.path(), 450).unwrap();
        assert_eq!(
            fps.model,
            "b0cac400047857c55d67422605ceca07d6be9ea81526222616989eca6837c1f7"
        );
        assert_eq!(
            fps.index,
            "e8dc47a17a5c2fababae02047e9770d91aa96d2bf11e22f8eea9b6ae5abe4aa0"
        );
    }

    fn remote() -> RemoteContract {
        RemoteContract {
            id: "embed-small".into(),
            base_url: "https://api.example.com/v1".into(),
            host: "api.example.com".into(),
            loopback: false,
            remote_model: "text-embedding-3-small".into(),
            api_key_env: Some("OPENAI_API_KEY".into()),
            auth_header: "Authorization".into(),
            query_prefix: String::new(),
            document_prefix: String::new(),
            query_params: Map::new(),
            document_params: Map::new(),
            batch_size: 32,
        }
    }

    fn params(key: &str, value: serde_json::Value) -> Map<String, serde_json::Value> {
        let mut map = Map::new();
        map.insert(key.into(), value);
        map
    }

    #[test]
    fn settings_that_decide_document_vectors_change_the_remote_fingerprint() {
        let base = remote_model_fingerprint(&remote());
        for changed in [
            RemoteContract {
                id: "embed-small-2".into(),
                ..remote()
            },
            RemoteContract {
                remote_model: "text-embedding-3-large".into(),
                ..remote()
            },
            RemoteContract {
                document_prefix: "passage: ".into(),
                ..remote()
            },
            RemoteContract {
                document_params: params("dimensions", json!(256)),
                ..remote()
            },
        ] {
            assert_ne!(remote_model_fingerprint(&changed), base, "{changed:?}");
        }
    }

    #[test]
    fn settings_that_leave_document_vectors_alone_keep_the_remote_fingerprint() {
        let base = remote_model_fingerprint(&remote());
        for same in [
            RemoteContract {
                base_url: "http://localhost:8080/v1".into(),
                host: "localhost".into(),
                loopback: true,
                ..remote()
            },
            RemoteContract {
                api_key_env: None,
                ..remote()
            },
            RemoteContract {
                auth_header: "api-key".into(),
                ..remote()
            },
            RemoteContract {
                query_prefix: "query: ".into(),
                ..remote()
            },
            RemoteContract {
                query_params: params("input_type", json!("query")),
                ..remote()
            },
            RemoteContract {
                batch_size: 8,
                ..remote()
            },
        ] {
            assert_eq!(remote_model_fingerprint(&same), base, "{same:?}");
        }
    }

    #[test]
    fn the_remote_fingerprint_ignores_the_order_of_document_params() {
        let contract = |params: &str| {
            let toml = format!(
                "chunk_max_tokens = 1500\n[model]\nsource = \"openai-compat:https://api.example.com/v1\"\n\
                 remote_model = \"m\"\ndocument_params = {params}\n"
            );
            match VaultConfig::parse(&toml)
                .unwrap()
                .model
                .resolve(Path::new("/"))
                .unwrap()
            {
                ModelContract::Remote(c) => c,
                other => panic!("{other:?}"),
            }
        };
        assert_eq!(
            remote_model_fingerprint(&contract("{ a = 1, b = { y = 2, x = 3 } }")),
            remote_model_fingerprint(&contract("{ b = { x = 3, y = 2 }, a = 1 }"))
        );
    }
}
