//! Embeddings from an OpenAI-compatible `/embeddings` endpoint (remote spec §3, §6).
use std::io::Read;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Map, Value};

use super::Embedder;
use crate::config::RemoteContract;
use crate::error::RemoteErrorKind;
use crate::{Error, Result};

/// Per request when there is no deadline (`notrieve index`).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// The waits before the first, second and third retry of a transient failure.
pub const RETRY_DELAYS: [Duration; 3] = [
    Duration::from_secs(1),
    Duration::from_secs(2),
    Duration::from_secs(4),
];
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);
const MAX_RESPONSE_BYTES: u64 = 64 << 20;
/// Server error text kept in an `Input` error.
const MAX_ERROR_BODY_CHARS: usize = 200;

type EnvLookup = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

pub struct RemoteEmbedder {
    contract: RemoteContract,
    url: String,
    agent: ureq::Agent,
    retry_delays: Vec<Duration>,
    env: EnvLookup,
}

/// A failed attempt, before it becomes an `Error::Remote`.
struct Failure {
    kind: RemoteErrorKind,
    message: String,
    retry_after: Option<Duration>,
}

impl Failure {
    fn new(kind: RemoteErrorKind, message: String) -> Self {
        Self {
            kind,
            message,
            retry_after: None,
        }
    }
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    index: usize,
    embedding: Vec<f32>,
}

impl RemoteEmbedder {
    pub fn new(contract: RemoteContract) -> Self {
        // Status codes are classified here. Redirects are not followed: the https check and the
        // API key apply to the configured host only (remote spec §6).
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .max_redirects(0)
            .build()
            .new_agent();
        Self {
            url: format!("{}/embeddings", contract.base_url),
            contract,
            agent,
            retry_delays: RETRY_DELAYS.to_vec(),
            env: Arc::new(|name| std::env::var(name).ok()),
        }
    }

    /// Replaces the waits between retries; one retry per entry.
    pub fn with_retry_delays(mut self, delays: &[Duration]) -> Self {
        self.retry_delays = delays.to_vec();
        self
    }

    /// Replaces how environment variables (the API key) are read.
    pub fn with_env(
        mut self,
        env: impl Fn(&str) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        self.env = Arc::new(env);
        self
    }

    fn embed(
        &self,
        prefix: &str,
        params: &Map<String, Value>,
        texts: &[String],
        deadline: Option<Instant>,
    ) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.contract.batch_size) {
            let inputs: Vec<String> = batch.iter().map(|t| format!("{prefix}{t}")).collect();
            out.extend(self.request(&inputs, params, deadline)?);
        }
        Ok(out)
    }

    fn request(
        &self,
        inputs: &[String],
        params: &Map<String, Value>,
        deadline: Option<Instant>,
    ) -> Result<Vec<Vec<f32>>> {
        let mut body = params.clone();
        body.insert(
            "model".into(),
            Value::String(self.contract.remote_model.clone()),
        );
        body.insert("input".into(), Value::from(inputs.to_vec()));
        body.insert("encoding_format".into(), Value::String("float".into()));
        let body = serde_json::to_vec(&Value::Object(body)).map_err(|e| {
            remote(
                RemoteErrorKind::Protocol,
                format!("cannot encode the request: {e}"),
            )
        })?;
        let key = self.api_key()?;
        // A deadline leaves no time to wait for a retry (remote spec §6).
        let retries = if deadline.is_some() {
            0
        } else {
            self.retry_delays.len()
        };
        let mut attempt = 0;
        loop {
            match self.send(&body, key.as_deref(), deadline) {
                Ok(bytes) => {
                    return self.parse(&bytes, inputs.len()).map_err(|e| match e {
                        Error::Remote { kind, message } => {
                            remote(kind, mask(message, key.as_deref()))
                        }
                        other => other,
                    });
                }
                Err(failure) if failure.kind == RemoteErrorKind::Transient && attempt < retries => {
                    let wait = retry_wait(failure.retry_after, self.retry_delays[attempt]);
                    tracing::warn!(
                        "{}; retrying in {wait:?}",
                        mask(failure.message, key.as_deref())
                    );
                    std::thread::sleep(wait);
                    attempt += 1;
                }
                Err(failure) => {
                    return Err(remote(failure.kind, mask(failure.message, key.as_deref())));
                }
            }
        }
    }

    fn api_key(&self) -> Result<Option<String>> {
        let Some(name) = &self.contract.api_key_env else {
            return Ok(None);
        };
        match (self.env)(name) {
            Some(key) if !key.is_empty() => Ok(Some(key)),
            _ => Err(remote(
                RemoteErrorKind::Auth,
                format!("the environment variable {name} (model.api_key_env) is not set"),
            )),
        }
    }

    fn send(
        &self,
        body: &[u8],
        key: Option<&str>,
        deadline: Option<Instant>,
    ) -> std::result::Result<Vec<u8>, Failure> {
        let timeout = match deadline {
            Some(deadline) => {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return Err(Failure::new(
                        RemoteErrorKind::Transient,
                        "no time was left for the request".into(),
                    ));
                }
                left
            }
            None => REQUEST_TIMEOUT,
        };
        let mut request = self
            .agent
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header(
                "User-Agent",
                concat!("notrieve/", env!("CARGO_PKG_VERSION")),
            );
        if let Some(key) = key {
            let value = if self
                .contract
                .auth_header
                .eq_ignore_ascii_case("authorization")
            {
                format!("Bearer {key}")
            } else {
                key.to_string()
            };
            request = request.header(self.contract.auth_header.as_str(), value);
        }
        let response = request
            .config()
            .timeout_global(Some(timeout))
            .build()
            .send(body.to_vec())
            .map_err(|e| {
                Failure::new(
                    RemoteErrorKind::Transient,
                    format!("request to {} failed: {e}", self.url),
                )
            })?;
        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(Duration::from_secs);
        let kind = match status {
            200..=299 => return self.read_body(response.into_body()),
            408 | 429 | 500..=599 => RemoteErrorKind::Transient,
            401 | 403 => RemoteErrorKind::Auth,
            300..=399 | 404 | 405 => RemoteErrorKind::Protocol,
            _ => RemoteErrorKind::Input,
        };
        let mut message = format!("HTTP {status} from {}", self.url);
        match kind {
            RemoteErrorKind::Input => {
                // The status already decides the kind: a body that cannot be read leaves no text.
                let bytes = self.read_body(response.into_body()).unwrap_or_default();
                // Mask before cutting, so a key at the cut point is not left half-shown.
                let text = mask(String::from_utf8_lossy(&bytes).into_owned(), key);
                let text: String = text.chars().take(MAX_ERROR_BODY_CHARS).collect();
                message.push_str(&format!(": {text}"));
            }
            RemoteErrorKind::Auth => {
                if let Some(name) = &self.contract.api_key_env {
                    message.push_str(&format!("; check the key in {name}"));
                }
            }
            _ => {}
        }
        Err(Failure {
            kind,
            message,
            retry_after,
        })
    }

    /// The body after decompression, at most `MAX_RESPONSE_BYTES`: ureq's own `limit` counts the
    /// bytes before decompression, so a small gzip body could still expand without bound.
    fn read_body(&self, body: ureq::Body) -> std::result::Result<Vec<u8>, Failure> {
        let mut bytes = Vec::new();
        body.into_reader()
            .take(MAX_RESPONSE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| {
                Failure::new(
                    RemoteErrorKind::Transient,
                    format!("reading the response from {} failed: {e}", self.url),
                )
            })?;
        if bytes.len() as u64 > MAX_RESPONSE_BYTES {
            return Err(Failure::new(
                RemoteErrorKind::Protocol,
                format!("the response from {} is larger than 64 MiB", self.url),
            ));
        }
        Ok(bytes)
    }

    /// Remote spec §6: one finite vector per input, all of one length, ordered by `index`.
    fn parse(&self, bytes: &[u8], expected: usize) -> Result<Vec<Vec<f32>>> {
        let invalid = |why: String| {
            remote(
                RemoteErrorKind::Protocol,
                format!("invalid response from {}: {why}", self.url),
            )
        };
        let response: EmbeddingResponse =
            serde_json::from_slice(bytes).map_err(|e| invalid(e.to_string()))?;
        if response.data.len() != expected {
            return Err(invalid(format!(
                "{} vectors for {expected} inputs",
                response.data.len()
            )));
        }
        let mut slots: Vec<Option<Vec<f32>>> = vec![None; expected];
        for item in response.data {
            let slot = slots
                .get_mut(item.index)
                .ok_or_else(|| invalid(format!("index {} is out of range", item.index)))?;
            if slot.is_some() {
                return Err(invalid(format!("index {} appears twice", item.index)));
            }
            if item.embedding.iter().any(|x| !x.is_finite()) {
                return Err(invalid(format!("vector {} is not finite", item.index)));
            }
            *slot = Some(item.embedding);
        }
        // Every slot is filled: as many items as slots, each index in range and seen once.
        let vectors: Vec<Vec<f32>> = slots.into_iter().flatten().collect();
        let dim = vectors.first().map_or(0, Vec::len);
        if dim == 0 || vectors.iter().any(|v| v.len() != dim) {
            return Err(invalid("the vectors are empty or differ in length".into()));
        }
        Ok(vectors)
    }
}

/// How long to wait before a retry: the server's `Retry-After`, capped, else the default.
fn retry_wait(retry_after: Option<Duration>, default: Duration) -> Duration {
    retry_after.map_or(default, |d| d.min(MAX_RETRY_AFTER))
}

/// Server text and library errors can repeat the key; it must not reach an error or a log.
fn mask(text: String, key: Option<&str>) -> String {
    let Some(key) = key else {
        return text;
    };
    // A JSON body or error repeats the key escaped (`"` as `\"`).
    let escaped = serde_json::to_string(key).unwrap_or_default();
    let escaped = escaped.trim_matches('"');
    let text = text.replace(key, "***");
    if escaped.is_empty() || escaped == key {
        text
    } else {
        text.replace(escaped, "***")
    }
}

fn remote(kind: RemoteErrorKind, message: String) -> Error {
    Error::Remote { kind, message }
}

impl Embedder for RemoteEmbedder {
    fn model_id(&self) -> &str {
        &self.contract.id
    }

    /// The dimension comes from the stored vectors (remote spec §4).
    fn dim(&self) -> usize {
        0
    }

    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_documents_within(texts, None)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_query_within(text, None)
    }

    /// Characters, prefix included, as the ONNX models count tokens with theirs.
    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.contract.document_prefix.chars().count() + text.chars().count())
    }

    fn embed_documents_within(
        &self,
        texts: &[String],
        deadline: Option<Instant>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed(
            &self.contract.document_prefix,
            &self.contract.document_params,
            texts,
            deadline,
        )
    }

    fn embed_query_within(&self, text: &str, deadline: Option<Instant>) -> Result<Vec<f32>> {
        let mut vectors = self.embed(
            &self.contract.query_prefix,
            &self.contract.query_params,
            &[text.to_string()],
            deadline,
        )?;
        // `parse` returns exactly one vector per input.
        Ok(vectors.remove(0))
    }

    fn remote_host(&self) -> Option<&str> {
        (!self.contract.loopback).then_some(self.contract.host.as_str())
    }

    fn is_remote(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_after_is_capped_at_60_seconds() {
        let wait = |after: Option<u64>, default: u64| {
            retry_wait(after.map(Duration::from_secs), Duration::from_secs(default))
        };
        assert_eq!(wait(Some(61), 1), Duration::from_secs(60));
        assert_eq!(wait(Some(2), 1), Duration::from_secs(2));
        assert_eq!(wait(None, 4), Duration::from_secs(4));
    }
}
