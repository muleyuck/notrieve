use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{Error, Result};

pub const PRESET_RURI_V3_30M: &str = "ruri-v3-30m";
pub const REMOTE_SOURCE_PREFIX: &str = "openai-compat:";
pub const DEFAULT_BATCH_SIZE: usize = 32;
pub const MAX_BATCH_SIZE: usize = 256;
/// Headers whose values ureq 3.4 logs at debug level (its NON_SENSITIVE_HEADERS list).
const UREQ_LOGGED_HEADERS: [&str; 13] = [
    "Date",
    "Content-Type",
    "Content-Length",
    "Transfer-Encoding",
    "Connection",
    "Content-Encoding",
    "Host",
    "Accept",
    "Accept-Encoding",
    "Accept-Charset",
    "Server",
    "User-Agent",
    "Location",
];
/// Set by notrieve in every request body, so `query_params` / `document_params` cannot hold them.
const RESERVED_PARAMS: [&str; 3] = ["input", "model", "encoding_format"];

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum ModelSpec {
    Preset(String),
    Table(Box<ModelTable>),
}

/// `[model]` as written. Which fields are required depends on `source` (remote spec §2).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelTable {
    pub source: String,
    pub id: Option<String>,
    pub onnx_file: Option<String>,
    pub dim: Option<usize>,
    pub max_tokens: Option<usize>,
    pub pooling: Option<Pooling>,
    pub quantization: Option<String>,
    pub query_prefix: Option<String>,
    pub document_prefix: Option<String>,
    pub remote_model: Option<String>,
    pub api_key_env: Option<String>,
    pub auth_header: Option<String>,
    pub query_params: Option<Map<String, Value>>,
    pub document_params: Option<Map<String, Value>>,
    pub batch_size: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Pooling {
    Mean,
    Cls,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSource {
    Hf { repo: String },
    Path { dir: PathBuf },
}

/// A model run here with ONNX Runtime. It serializes exactly as the contract did before remote
/// models existed, so the model fingerprints of existing indexes do not change.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OnnxContract {
    pub id: String,
    pub source: ModelSource,
    pub onnx_file: String,
    pub dim: usize,
    pub max_tokens: usize,
    pub pooling: Pooling,
    pub query_prefix: String,
    pub document_prefix: String,
}

/// A model behind an OpenAI-compatible `/embeddings` endpoint. It names the environment variable
/// that holds the API key, never the key.
#[derive(Debug, Clone, PartialEq)]
pub struct RemoteContract {
    pub id: String,
    /// Without a trailing `/`; requests go to `<base_url>/embeddings`.
    pub base_url: String,
    pub host: String,
    /// The host is localhost, 127.0.0.1 or ::1, so note text stays on this machine.
    pub loopback: bool,
    pub remote_model: String,
    pub api_key_env: Option<String>,
    pub auth_header: String,
    pub query_prefix: String,
    pub document_prefix: String,
    pub query_params: Map<String, Value>,
    pub document_params: Map<String, Value>,
    pub batch_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelContract {
    Local(OnnxContract),
    Remote(RemoteContract),
}

impl ModelContract {
    pub fn into_local(self) -> Option<OnnxContract> {
        match self {
            ModelContract::Local(contract) => Some(contract),
            ModelContract::Remote(_) => None,
        }
    }
}

impl ModelSpec {
    pub fn resolve(&self, vault_root: &Path) -> Result<ModelContract> {
        match self {
            ModelSpec::Preset(name) if name == PRESET_RURI_V3_30M => {
                Ok(ModelContract::Local(ruri_v3_30m()))
            }
            ModelSpec::Preset(name) => Err(Error::Config(format!(
                "unknown model preset `{name}` (available: {PRESET_RURI_V3_30M}); use a [model] table for other models"
            ))),
            ModelSpec::Table(table) => table.resolve(vault_root),
        }
    }
}

fn ruri_v3_30m() -> OnnxContract {
    OnnxContract {
        id: PRESET_RURI_V3_30M.to_string(),
        source: ModelSource::Hf {
            repo: "sirasagi62/ruri-v3-30m-ONNX".to_string(),
        },
        onnx_file: "onnx/model.onnx".to_string(),
        dim: 256,
        max_tokens: 512,
        pooling: Pooling::Mean,
        query_prefix: "検索クエリ: ".to_string(),
        document_prefix: "検索文書: ".to_string(),
    }
}

fn required<'a, T>(value: &'a Option<T>, field: &str) -> Result<&'a T> {
    value
        .as_ref()
        .ok_or_else(|| Error::Config(format!("model.{field} is required")))
}

/// Errors naming every field in `fields` that is set.
fn reject_set(fields: &[(&str, bool)], kind: &str) -> Result<()> {
    let set: Vec<String> = fields
        .iter()
        .filter(|(_, is_set)| *is_set)
        .map(|(name, _)| format!("model.{name}"))
        .collect();
    if set.is_empty() {
        return Ok(());
    }
    Err(Error::Config(format!(
        "{} cannot be set for a {kind} model",
        set.join(", ")
    )))
}

impl ModelTable {
    fn resolve(&self, vault_root: &Path) -> Result<ModelContract> {
        if let Some(url) = self.source.strip_prefix(REMOTE_SOURCE_PREFIX) {
            reject_set(
                &[
                    ("onnx_file", self.onnx_file.is_some()),
                    ("dim", self.dim.is_some()),
                    ("max_tokens", self.max_tokens.is_some()),
                    ("pooling", self.pooling.is_some()),
                    ("quantization", self.quantization.is_some()),
                ],
                "remote",
            )?;
            return self.resolve_remote(url).map(ModelContract::Remote);
        }
        reject_set(
            &[
                ("remote_model", self.remote_model.is_some()),
                ("api_key_env", self.api_key_env.is_some()),
                ("auth_header", self.auth_header.is_some()),
                ("query_params", self.query_params.is_some()),
                ("document_params", self.document_params.is_some()),
                ("batch_size", self.batch_size.is_some()),
            ],
            "local",
        )?;
        self.resolve_local(vault_root).map(ModelContract::Local)
    }

    fn resolve_local(&self, vault_root: &Path) -> Result<OnnxContract> {
        let quantization = required(&self.quantization, "quantization")?;
        if quantization != "none" {
            return Err(Error::Config(format!(
                "model.quantization must be \"none\" (got \"{quantization}\"); use a pre-quantized ONNX file instead"
            )));
        }
        let dim = *required(&self.dim, "dim")?;
        let max_tokens = *required(&self.max_tokens, "max_tokens")?;
        if dim == 0 || max_tokens == 0 {
            return Err(Error::Config(
                "model.dim and model.max_tokens must be positive".into(),
            ));
        }
        let onnx_file = required(&self.onnx_file, "onnx_file")?;
        validate_onnx_file(onnx_file)?;
        Ok(OnnxContract {
            id: required(&self.id, "id")?.clone(),
            source: parse_source(&self.source, vault_root)?,
            onnx_file: onnx_file.clone(),
            dim,
            max_tokens,
            pooling: *required(&self.pooling, "pooling")?,
            query_prefix: required(&self.query_prefix, "query_prefix")?.clone(),
            document_prefix: required(&self.document_prefix, "document_prefix")?.clone(),
        })
    }

    fn resolve_remote(&self, url: &str) -> Result<RemoteContract> {
        let remote_model = required(&self.remote_model, "remote_model")?.clone();
        let (base_url, host, loopback) = parse_base_url(url)?;
        let auth_header = self
            .auth_header
            .clone()
            .unwrap_or_else(|| "Authorization".to_string());
        if ureq::http::HeaderName::from_bytes(auth_header.as_bytes()).is_err() {
            return Err(Error::Config(format!(
                "model.auth_header must be an HTTP header name (got \"{auth_header}\")"
            )));
        }
        // ureq logs the values of these headers at debug level; a key sent in one would be logged.
        if UREQ_LOGGED_HEADERS
            .iter()
            .any(|name| name.eq_ignore_ascii_case(&auth_header))
        {
            return Err(Error::Config(format!(
                "model.auth_header cannot be \"{auth_header}\": the HTTP client logs the value of that header"
            )));
        }
        let batch_size = self.batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        if !(1..=MAX_BATCH_SIZE).contains(&batch_size) {
            return Err(Error::Config(format!(
                "model.batch_size must be between 1 and {MAX_BATCH_SIZE} (got {batch_size})"
            )));
        }
        Ok(RemoteContract {
            id: self.id.clone().unwrap_or_else(|| remote_model.clone()),
            base_url,
            host,
            loopback,
            remote_model,
            api_key_env: self.api_key_env.clone(),
            auth_header,
            query_prefix: self.query_prefix.clone().unwrap_or_default(),
            document_prefix: self.document_prefix.clone().unwrap_or_default(),
            query_params: checked_params(&self.query_params, "query_params")?,
            document_params: checked_params(&self.document_params, "document_params")?,
            batch_size,
        })
    }
}

fn checked_params(params: &Option<Map<String, Value>>, field: &str) -> Result<Map<String, Value>> {
    let params = params.clone().unwrap_or_default();
    if let Some(key) = RESERVED_PARAMS
        .iter()
        .find(|key| params.contains_key(**key))
    {
        return Err(Error::Config(format!(
            "model.{field} cannot set `{key}`; notrieve sets it"
        )));
    }
    Ok(params)
}

/// `https://`, or `http://` to a loopback host; no user info or query (remote spec §2).
/// Returns the URL without a trailing `/`, its host, and whether the host is a loopback one.
fn parse_base_url(url: &str) -> Result<(String, String, bool)> {
    // No message repeats the URL: its user info or query may hold a secret.
    let invalid = |why: &str| Error::Config(format!("model.source: {why}"));
    let uri: ureq::http::Uri = url
        .parse()
        .map_err(|_| invalid("not a valid URL (expected https://<host>/...)"))?;
    let authority = uri
        .authority()
        .ok_or_else(|| invalid("the URL must start with https://"))?;
    if authority.as_str().contains('@') {
        return Err(invalid(
            "the URL must not contain user info; name the variable that holds the key in model.api_key_env",
        ));
    }
    if uri.query().is_some() || url.contains('#') {
        return Err(invalid(
            "the URL must not contain a query string or a fragment",
        ));
    }
    let host = authority.host().to_string();
    let loopback = matches!(host.as_str(), "localhost" | "127.0.0.1" | "[::1]");
    match uri.scheme_str() {
        Some("https") => {}
        Some("http") if loopback => {}
        Some("http") => {
            return Err(invalid(
                "use https:// (http:// is allowed only for localhost, 127.0.0.1 and ::1)",
            ));
        }
        _ => return Err(invalid("the URL must start with https://")),
    }
    Ok((url.trim_end_matches('/').to_string(), host, loopback))
}

fn validate_onnx_file(file: &str) -> Result<()> {
    let bad = file.is_empty()
        || Path::new(file).is_absolute()
        || file.starts_with('/')
        || file
            .chars()
            .any(|c| c.is_control() || matches!(c, '\\' | '?' | '#' | '%'))
        || file.split('/').any(|part| part == "." || part == "..");
    if bad {
        return Err(Error::Config(format!(
            "model.onnx_file must be a relative path without `.`/`..` components or any of `\\ ? # %` (got \"{file}\")"
        )));
    }
    Ok(())
}

fn valid_repo_part(part: &str) -> bool {
    !part.is_empty()
        && part != "."
        && part != ".."
        && part
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn parse_source(source: &str, vault_root: &Path) -> Result<ModelSource> {
    if let Some(repo) = source.strip_prefix("hf:") {
        let parts: Vec<&str> = repo.split('/').collect();
        if parts.len() == 2 && parts.iter().all(|p| valid_repo_part(p)) {
            return Ok(ModelSource::Hf {
                repo: repo.to_string(),
            });
        }
    } else if let Some(dir) = source.strip_prefix("path:") {
        let dir = PathBuf::from(dir);
        let dir = if dir.is_absolute() {
            dir
        } else {
            vault_root.join(dir)
        };
        return Ok(ModelSource::Path { dir });
    }
    Err(Error::Config(format!(
        "model.source must be \"hf:<owner>/<repo>\", \"path:<dir>\" or \"openai-compat:<base URL>\" (got \"{source}\")"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VaultConfig;
    use serde_json::json;
    use std::path::Path;

    fn table(overrides: &str) -> String {
        let base = [
            ("id", "\"my-model\""),
            ("source", "\"path:models/m\""),
            ("onnx_file", "\"model.onnx\""),
            ("dim", "384"),
            ("max_tokens", "256"),
            ("pooling", "\"cls\""),
            ("quantization", "\"none\""),
            ("query_prefix", "\"query: \""),
            ("document_prefix", "\"passage: \""),
        ];
        let mut lines = vec!["[model]".to_string()];
        for (key, value) in base {
            let replaced = overrides
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{key} = ")));
            match replaced {
                Some("<omit>") => {}
                Some(v) => lines.push(format!("{key} = {v}")),
                None => lines.push(format!("{key} = {value}")),
            }
        }
        lines.join("\n")
    }

    fn resolve(toml: &str) -> Result<OnnxContract> {
        match VaultConfig::parse(toml)?
            .model
            .resolve(Path::new("/vault"))?
        {
            ModelContract::Local(c) => Ok(c),
            other => panic!("expected a local model: {other:?}"),
        }
    }

    /// A remote vault config: `chunk_max_tokens` and a minimal [model] table plus `extra` lines.
    fn remote_toml(source: &str, extra: &str) -> String {
        format!(
            "chunk_max_tokens = 1500\n[model]\nsource = \"{source}\"\nremote_model = \"text-embedding-3-small\"\n{extra}\n"
        )
    }

    fn resolve_remote(source: &str, extra: &str) -> Result<RemoteContract> {
        match VaultConfig::parse(&remote_toml(source, extra))?
            .model
            .resolve(Path::new("/vault"))?
        {
            ModelContract::Remote(c) => Ok(c),
            other => panic!("expected a remote model: {other:?}"),
        }
    }

    const OPENAI: &str = "openai-compat:https://api.openai.com/v1";

    fn config_error(result: Result<RemoteContract>) -> String {
        match result {
            Err(Error::Config(message)) => message,
            other => panic!("expected a configuration error: {other:?}"),
        }
    }

    #[test]
    fn preset_resolves_to_fp32_ruri_contract() {
        let c = resolve("model = \"ruri-v3-30m\"").unwrap();
        assert_eq!(c.id, "ruri-v3-30m");
        assert_eq!(
            c.source,
            ModelSource::Hf {
                repo: "sirasagi62/ruri-v3-30m-ONNX".into()
            }
        );
        assert_eq!(c.onnx_file, "onnx/model.onnx");
        assert_eq!((c.dim, c.max_tokens, c.pooling), (256, 512, Pooling::Mean));
        assert_eq!(
            (c.query_prefix.as_str(), c.document_prefix.as_str()),
            ("検索クエリ: ", "検索文書: ")
        );
    }

    #[test]
    fn table_resolves_relative_path_source_against_vault_root() {
        let c = resolve(&table("")).unwrap();
        assert_eq!(
            c.source,
            ModelSource::Path {
                dir: "/vault/models/m".into()
            }
        );
        assert_eq!((c.dim, c.max_tokens, c.pooling), (384, 256, Pooling::Cls));
        assert_eq!(c.document_prefix, "passage: ");
    }

    #[test]
    fn table_accepts_hf_source() {
        let c = resolve(&table("source = \"hf:owner/repo\"")).unwrap();
        assert_eq!(
            c.source,
            ModelSource::Hf {
                repo: "owner/repo".into()
            }
        );
    }

    #[test]
    fn table_with_missing_field_is_a_config_error() {
        assert!(matches!(
            resolve(&table("dim = <omit>")),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn unknown_preset_names_the_available_one() {
        let err = resolve("model = \"ruri-v3-130m\"").unwrap_err();
        assert!(
            matches!(&err, Error::Config(m) if m.contains("available: ruri-v3-30m")),
            "{err}"
        );
    }

    #[test]
    fn quantization_other_than_none_is_rejected() {
        assert!(matches!(
            resolve(&table("quantization = \"dynamic\"")),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn unknown_source_scheme_is_rejected() {
        assert!(matches!(
            resolve(&table("source = \"s3:bucket/m\"")),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn zero_dimension_is_rejected() {
        assert!(matches!(resolve(&table("dim = 0")), Err(Error::Config(_))));
    }

    #[test]
    fn onnx_file_accepts_nested_relative_paths() {
        let c = resolve(&table("onnx_file = \"onnx/model_fp16.onnx\"")).unwrap();
        assert_eq!(c.onnx_file, "onnx/model_fp16.onnx");
    }

    #[test]
    fn onnx_file_with_unsafe_content_is_rejected() {
        for bad in [
            "",
            "/etc/passwd",
            "../x.onnx",
            "a/../x.onnx",
            "./x.onnx",
            "a/./x.onnx",
            "a\\\\b.onnx",
            "tokenizer.json#/../../../victim",
            "m.onnx?download=1",
            "m%2e.onnx",
            "m\\u0000.onnx",
            "m\\n.onnx",
        ] {
            let err = resolve(&table(&format!("onnx_file = \"{bad}\""))).unwrap_err();
            assert!(
                matches!(&err, Error::Config(m) if m.contains("onnx_file")),
                "{bad:?}: {err}"
            );
        }
    }

    #[test]
    fn hf_repo_must_be_owner_slash_name_with_safe_characters() {
        for ok in ["owner/repo", "a-b_c.d/E.1-x"] {
            assert!(
                resolve(&table(&format!("source = \"hf:{ok}\""))).is_ok(),
                "{ok}"
            );
        }
        for bad in [
            "hf:owner",
            "hf:a/b/c",
            "hf:/repo",
            "hf:owner/",
            "hf:./repo",
            "hf:owner/..",
            "hf:ow ner/repo",
            "hf:owner/re?po",
            "hf:owner/re#po",
            "hf:owner/re%2fpo",
            "hf:owner\\\\x/repo",
            "hf:owner/répo",
        ] {
            let err = resolve(&table(&format!("source = \"{bad}\""))).unwrap_err();
            assert!(
                matches!(&err, Error::Config(m) if m.contains("source")),
                "{bad:?}: {err}"
            );
        }
    }

    #[test]
    fn a_remote_table_resolves_with_the_documented_defaults() {
        let c = resolve_remote(OPENAI, "").unwrap();
        assert_eq!(
            c,
            RemoteContract {
                id: "text-embedding-3-small".into(),
                base_url: "https://api.openai.com/v1".into(),
                host: "api.openai.com".into(),
                loopback: false,
                remote_model: "text-embedding-3-small".into(),
                api_key_env: None,
                auth_header: "Authorization".into(),
                query_prefix: String::new(),
                document_prefix: String::new(),
                query_params: Map::new(),
                document_params: Map::new(),
                batch_size: 32,
            }
        );
    }

    #[test]
    fn a_remote_table_keeps_every_explicit_setting() {
        let c = resolve_remote(
            "openai-compat:http://127.0.0.1:8080/v1",
            "id = \"ruri-tei\"\napi_key_env = \"TEI_KEY\"\nauth_header = \"api-key\"\n\
             query_prefix = \"検索クエリ: \"\ndocument_prefix = \"検索文書: \"\n\
             query_params = { input_type = \"query\" }\n\
             document_params = { input_type = \"document\", dimensions = 256 }\nbatch_size = 7",
        )
        .unwrap();
        assert_eq!(
            (c.id.as_str(), c.host.as_str(), c.loopback),
            ("ruri-tei", "127.0.0.1", true)
        );
        assert_eq!(
            (
                c.api_key_env.as_deref(),
                c.auth_header.as_str(),
                c.batch_size
            ),
            (Some("TEI_KEY"), "api-key", 7)
        );
        assert_eq!(
            (c.query_prefix.as_str(), c.document_prefix.as_str()),
            ("検索クエリ: ", "検索文書: ")
        );
        assert_eq!(
            serde_json::Value::Object(c.query_params),
            json!({"input_type": "query"})
        );
        assert_eq!(
            serde_json::Value::Object(c.document_params),
            json!({"input_type": "document", "dimensions": 256})
        );
    }

    #[test]
    fn the_trailing_slash_of_the_base_url_is_dropped() {
        let c = resolve_remote("openai-compat:https://api.example.com/v1/", "").unwrap();
        assert_eq!(c.base_url, "https://api.example.com/v1");
    }

    #[test]
    fn local_only_fields_are_rejected_for_a_remote_model() {
        for line in [
            "onnx_file = \"m.onnx\"",
            "dim = 256",
            "max_tokens = 512",
            "pooling = \"mean\"",
            "quantization = \"none\"",
        ] {
            let field = line.split(' ').next().unwrap();
            let message = config_error(resolve_remote(OPENAI, line));
            assert!(
                message.contains(&format!("model.{field}")) && message.contains("remote"),
                "{line}: {message}"
            );
        }
    }

    #[test]
    fn remote_only_fields_are_rejected_for_a_local_model() {
        for line in [
            "remote_model = \"m\"",
            "api_key_env = \"KEY\"",
            "auth_header = \"api-key\"",
            "query_params = { a = 1 }",
            "document_params = { a = 1 }",
            "batch_size = 8",
        ] {
            let field = line.split(' ').next().unwrap();
            let err = resolve(&format!("{}\n{line}", table(""))).unwrap_err();
            assert!(
                matches!(&err, Error::Config(m) if m.contains(&format!("model.{field}")) && m.contains("local")),
                "{line}: {err}"
            );
        }
    }

    #[test]
    fn remote_model_is_required() {
        let toml = "chunk_max_tokens = 1500\n[model]\nsource = \"openai-compat:https://api.openai.com/v1\"\n";
        let err = VaultConfig::parse(toml).unwrap_err();
        assert!(
            matches!(&err, Error::Config(m) if m.contains("model.remote_model")),
            "{err}"
        );
    }

    #[test]
    fn http_is_accepted_only_for_loopback_hosts() {
        for ok in [
            "http://localhost:11434/v1",
            "http://127.0.0.1:8080/v1",
            "http://[::1]:8080/v1",
            "https://api.example.com/v1",
        ] {
            let c = resolve_remote(&format!("openai-compat:{ok}"), "").unwrap();
            assert_eq!(c.loopback, ok.starts_with("http://"), "{ok}");
        }
        for bad in [
            "http://192.168.1.10:8080/v1",
            "http://api.example.com/v1",
            "ftp://api.example.com/v1",
            "api.example.com/v1",
        ] {
            let message = config_error(resolve_remote(&format!("openai-compat:{bad}"), ""));
            assert!(message.contains("model.source"), "{bad}: {message}");
        }
    }

    #[test]
    fn a_url_with_user_info_a_query_or_a_fragment_is_rejected() {
        for bad in [
            "https://user:secret@api.example.com/v1",
            "user:secret@api.example.com/v1",
            "user:secret@api.example.com",
            "https://api.example.com/v1?key=secret",
            "https://api.example.com/v1#frag",
        ] {
            let message = config_error(resolve_remote(&format!("openai-compat:{bad}"), ""));
            assert!(message.contains("model.source"), "{bad}: {message}");
            // The URL may hold a password, so the message must not repeat it.
            assert!(!message.contains("secret"), "{message}");
        }
    }

    #[test]
    fn batch_size_must_be_between_1_and_256() {
        for ok in [1, 256] {
            let c = resolve_remote(OPENAI, &format!("batch_size = {ok}")).unwrap();
            assert_eq!(c.batch_size, ok);
        }
        for bad in [0, 257] {
            let message = config_error(resolve_remote(OPENAI, &format!("batch_size = {bad}")));
            assert!(message.contains("model.batch_size"), "{bad}: {message}");
        }
    }

    #[test]
    fn params_cannot_set_the_fields_notrieve_sets() {
        for field in ["query_params", "document_params"] {
            for key in ["input", "model", "encoding_format"] {
                let message = config_error(resolve_remote(
                    OPENAI,
                    &format!("{field} = {{ {key} = \"x\" }}"),
                ));
                assert!(
                    message.contains(&format!("model.{field}")) && message.contains(key),
                    "{field}.{key}: {message}"
                );
            }
        }
    }

    #[test]
    fn auth_header_must_be_a_header_name() {
        let message = config_error(resolve_remote(OPENAI, "auth_header = \"api key\""));
        assert!(message.contains("model.auth_header"), "{message}");
    }

    #[test]
    fn auth_header_must_not_be_one_ureq_logs() {
        for name in ["Accept", "content-type", "User-Agent"] {
            let message =
                config_error(resolve_remote(OPENAI, &format!("auth_header = \"{name}\"")));
            assert!(message.contains("model.auth_header"), "{name}: {message}");
        }
    }
}
