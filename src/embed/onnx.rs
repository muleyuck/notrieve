use std::fs;
use std::path::Path;
use std::sync::Mutex;

use fastembed::{
    InitOptionsUserDefined, OutputKey, Pooling as FastPooling, QuantizationMode, SingleBatchOutput,
    TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use tokenizers::{PostProcessor, Tokenizer};

use super::Embedder;
use crate::config::{ModelContract, Pooling};
use crate::{Error, Result};

pub const BATCH_SIZE: usize = 8;
pub const TOKENIZER_FILES: [&str; 4] = [
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

pub fn required_files(contract: &ModelContract) -> Vec<&str> {
    TOKENIZER_FILES
        .iter()
        .copied()
        .chain(std::iter::once(contract.onnx_file.as_str()))
        .collect()
}

/// Same rule as fastembed: `model_max_length` is read as a float because Hugging Face
/// uses 1e30-like sentinels for "no limit".
pub fn effective_max_tokens(max_tokens: usize, tokenizer_config: &[u8]) -> Result<usize> {
    let limit = serde_json::from_slice::<serde_json::Value>(tokenizer_config)
        .ok()
        .and_then(|v| {
            v.get("model_max_length")
                .and_then(serde_json::Value::as_f64)
        })
        .ok_or_else(|| {
            Error::Model("tokenizer_config.json has no numeric model_max_length".into())
        })?;
    Ok(max_tokens.min(limit as usize))
}

/// tokenizers computes `max_length - added_tokens` when truncating and underflows (a debug
/// panic) when the limit is smaller than the special tokens, so reject it before fastembed
/// configures truncation. At least one content token must remain.
pub fn check_room_for_special_tokens(effective_max: usize, tokenizer_json: &[u8]) -> Result<()> {
    let tokenizer =
        Tokenizer::from_bytes(tokenizer_json).map_err(|e| Error::Model(e.to_string()))?;
    let special = tokenizer
        .get_post_processor()
        .map_or(0, |p| p.added_tokens(false));
    if effective_max <= special {
        return Err(Error::Model(format!(
            "max_tokens {effective_max} leaves no room for the tokenizer's {special} special tokens; use at least {}",
            special + 1
        )));
    }
    Ok(())
}

/// Token-level outputs come first so the contract's pooling is applied to them. fastembed's own
/// default order would pick the model's built-in `sentence_embedding` (already pooled, so
/// `pooling = "cls"` would silently be ignored). The pooled outputs remain as a fallback.
const OUTPUT_PRECEDENCE: &[OutputKey] = &[
    OutputKey::ByName("last_hidden_state"),
    OutputKey::ByName("token_embeddings"),
    OutputKey::ByName("text_embeds"),
    OutputKey::ByName("sentence_embedding"),
    OutputKey::OnlyOne,
];

fn pool_batch(batch: &SingleBatchOutput, pooling: Pooling) -> Result<Vec<Vec<f32>>> {
    let embedding_err = |e: fastembed::Error| Error::Embedding(e.to_string());
    let precedence = OUTPUT_PRECEDENCE;
    let already_pooled = batch
        .select_output(&precedence)
        .map_err(embedding_err)?
        .ndim()
        == 2;
    if already_pooled && pooling == Pooling::Cls {
        return Err(Error::Model(
            "pooling = \"cls\" cannot be applied: the model only exposes an already pooled output (no token-level output)".into(),
        ));
    }
    let fast = match pooling {
        Pooling::Mean => FastPooling::Mean,
        Pooling::Cls => FastPooling::Cls,
    };
    let pooled = batch
        .select_and_pool_output(&precedence, Some(fast))
        .map_err(embedding_err)?;
    Ok(pooled
        .rows()
        .into_iter()
        .map(|row| {
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            row.iter().map(|x| x / norm.max(f32::EPSILON)).collect()
        })
        .collect())
}

pub fn resolve_threads(configured: usize) -> usize {
    if configured > 0 {
        configured
    } else {
        std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).max(1))
    }
}

pub struct OnnxEmbedder {
    contract: ModelContract,
    model: Mutex<TextEmbedding>,
    counter: Tokenizer,
    effective_max_tokens: usize,
}

impl OnnxEmbedder {
    pub fn load(contract: &ModelContract, dir: &Path, threads: usize) -> Result<Self> {
        for file in required_files(contract) {
            if !dir.join(file).is_file() {
                return Err(Error::ModelFileMissing {
                    dir: dir.to_path_buf(),
                    file: file.to_string(),
                });
            }
        }
        let read = |name: &str| fs::read(dir.join(name)).map_err(|e| Error::io(dir.join(name), e));
        let tokenizer_config = read("tokenizer_config.json")?;
        let effective_max_tokens = effective_max_tokens(contract.max_tokens, &tokenizer_config)?;
        let tokenizer_json = read("tokenizer.json")?;
        check_room_for_special_tokens(effective_max_tokens, &tokenizer_json)?;
        let files = TokenizerFiles {
            tokenizer_file: tokenizer_json,
            config_file: read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file: tokenizer_config,
        };
        let model = UserDefinedEmbeddingModel::new(read(&contract.onnx_file)?, files)
            .with_quantization(QuantizationMode::None);
        let options = InitOptionsUserDefined::new()
            .with_max_length(contract.max_tokens)
            .with_intra_threads(threads.max(1));
        let model = TextEmbedding::try_new_from_user_defined(model, options)
            .map_err(|e| Error::Model(e.to_string()))?;
        // Count with the tokenizer fastembed configured (special tokens included), minus truncation and padding.
        let mut counter = model.tokenizer.clone();
        counter
            .with_truncation(None)
            .map_err(|e| Error::Model(e.to_string()))?;
        counter.with_padding(None);
        Ok(Self {
            contract: contract.clone(),
            model: Mutex::new(model),
            counter,
            effective_max_tokens,
        })
    }

    /// `min(max_tokens, tokenizer_config.json model_max_length)`.
    pub fn effective_max_tokens(&self) -> usize {
        self.effective_max_tokens
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<tokenizers::Encoding> {
        self.counter
            .encode(text, add_special_tokens)
            .map_err(|e| Error::Model(e.to_string()))
    }

    fn embed_inputs(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(inputs.len());
        // The lock is taken per batch so a query can run between two batches.
        for batch in inputs.chunks(BATCH_SIZE) {
            let mut model = super::lock_unpoisoned(&self.model);
            let output = model
                .transform(batch, Some(batch.len()))
                .map_err(|e| Error::Embedding(e.to_string()))?;
            for raw in output.into_raw() {
                out.extend(pool_batch(&raw, self.contract.pooling)?);
            }
        }
        if let Some(v) = out.iter().find(|v| v.len() != self.contract.dim) {
            return Err(Error::Model(format!(
                "the model returned {} dimensions but the contract says {}",
                v.len(),
                self.contract.dim
            )));
        }
        Ok(out)
    }
}

impl Embedder for OnnxEmbedder {
    fn model_id(&self) -> &str {
        &self.contract.id
    }

    fn dim(&self) -> usize {
        self.contract.dim
    }

    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let inputs: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.contract.document_prefix))
            .collect();
        self.embed_inputs(&inputs)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let mut vectors = self.embed_inputs(&[format!("{}{text}", self.contract.query_prefix)])?;
        vectors
            .pop()
            .ok_or_else(|| Error::Embedding("no vector returned".into()))
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self
            .encode(&format!("{}{text}", self.contract.document_prefix), true)?
            .len())
    }

    fn token_ends(&self, text: &str) -> Result<Vec<usize>> {
        let mut ends: Vec<usize> = self
            .encode(text, false)?
            .get_offsets()
            .iter()
            .map(|&(_, end)| end)
            .collect();
        ends.retain(|&end| end > 0 && end <= text.len());
        ends.sort_unstable();
        ends.dedup();
        if ends.last() != Some(&text.len()) {
            ends.push(text.len());
        }
        Ok(ends)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelSource;
    use tempfile::TempDir;

    #[test]
    fn missing_tokenizer_file_is_named_before_loading() {
        let dir = TempDir::new().unwrap();
        let c = ModelContract {
            id: "m".into(),
            source: ModelSource::Path {
                dir: dir.path().to_path_buf(),
            },
            onnx_file: "model.onnx".into(),
            dim: 4,
            max_tokens: 64,
            pooling: Pooling::Mean,
            query_prefix: String::new(),
            document_prefix: String::new(),
        };
        let err = OnnxEmbedder::load(&c, dir.path(), 1).err().unwrap();
        assert!(matches!(err, Error::ModelFileMissing { file, .. } if file == "tokenizer.json"));
    }

    #[test]
    fn effective_limit_is_the_smaller_of_contract_and_tokenizer() {
        assert_eq!(
            effective_max_tokens(512, br#"{"model_max_length": 8192}"#).unwrap(),
            512
        );
        assert_eq!(
            effective_max_tokens(512, br#"{"model_max_length": 384}"#).unwrap(),
            384
        );
    }

    #[test]
    fn huge_hugging_face_sentinel_limit_is_accepted() {
        let config = br#"{"model_max_length": 1000000000000000019884624838656}"#;
        assert_eq!(effective_max_tokens(512, config).unwrap(), 512);
    }

    #[test]
    fn missing_model_max_length_is_a_model_error() {
        assert!(matches!(
            effective_max_tokens(512, br#"{"pad_token": "<pad>"}"#),
            Err(Error::Model(_))
        ));
    }

    fn bert_like_tokenizer_json() -> Vec<u8> {
        use std::collections::HashMap;
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::processors::bert::BertProcessing;
        let vocab: HashMap<String, u32> = [("[UNK]", 0), ("[CLS]", 1), ("[SEP]", 2)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_post_processor(Some(BertProcessing::new(
            ("[SEP]".into(), 2),
            ("[CLS]".into(), 1),
        )));
        tokenizer.to_string(false).unwrap().into_bytes()
    }

    #[test]
    fn limit_must_exceed_the_special_token_count() {
        let json = bert_like_tokenizer_json();
        for too_small in [0, 1, 2] {
            assert!(
                matches!(
                    check_room_for_special_tokens(too_small, &json),
                    Err(Error::Model(_))
                ),
                "{too_small}"
            );
        }
        assert!(check_room_for_special_tokens(3, &json).is_ok());
    }

    #[test]
    fn too_small_max_tokens_is_an_error_not_a_panic_in_load() {
        let dir = TempDir::new().unwrap();
        for (f, body) in [
            ("config.json", "{}".as_bytes()),
            ("special_tokens_map.json", b"{}"),
            ("tokenizer_config.json", br#"{"model_max_length": 512}"#),
            ("model.onnx", b"not onnx"),
        ] {
            fs::write(dir.path().join(f), body).unwrap();
        }
        fs::write(
            dir.path().join("tokenizer.json"),
            bert_like_tokenizer_json(),
        )
        .unwrap();
        let c = ModelContract {
            id: "m".into(),
            source: ModelSource::Path {
                dir: dir.path().to_path_buf(),
            },
            onnx_file: "model.onnx".into(),
            dim: 4,
            max_tokens: 1,
            pooling: Pooling::Mean,
            query_prefix: String::new(),
            document_prefix: String::new(),
        };
        let err = OnnxEmbedder::load(&c, dir.path(), 1).err().unwrap();
        assert!(
            matches!(&err, Error::Model(m) if m.contains("special tokens")),
            "{err}"
        );
    }

    #[test]
    fn zero_threads_means_half_the_cores_but_at_least_one() {
        assert_eq!(resolve_threads(3), 3);
        let auto = resolve_threads(0);
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        assert_eq!(auto, (cores / 2).max(1));
    }
}
