use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

pub const PRESET_RURI_V3_130M: &str = "ruri-v3-130m";

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum ModelSpec {
    Preset(String),
    Table(ModelTable),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelTable {
    pub id: String,
    pub source: String,
    pub onnx_file: String,
    pub dim: usize,
    pub max_tokens: usize,
    pub pooling: Pooling,
    pub quantization: String,
    pub query_prefix: String,
    pub document_prefix: String,
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

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelContract {
    pub id: String,
    pub source: ModelSource,
    pub onnx_file: String,
    pub dim: usize,
    pub max_tokens: usize,
    pub pooling: Pooling,
    pub query_prefix: String,
    pub document_prefix: String,
}

impl ModelSpec {
    pub fn resolve(&self, vault_root: &Path) -> Result<ModelContract> {
        match self {
            ModelSpec::Preset(name) if name == PRESET_RURI_V3_130M => Ok(ruri_v3_130m()),
            ModelSpec::Preset(name) => Err(Error::Config(format!(
                "unknown model preset `{name}` (available: {PRESET_RURI_V3_130M}); use a [model] table for other models"
            ))),
            ModelSpec::Table(table) => table.resolve(vault_root),
        }
    }
}

fn ruri_v3_130m() -> ModelContract {
    ModelContract {
        id: PRESET_RURI_V3_130M.to_string(),
        source: ModelSource::Hf {
            repo: "sirasagi62/ruri-v3-130m-ONNX".to_string(),
        },
        onnx_file: "onnx/model.onnx".to_string(),
        dim: 512,
        max_tokens: 512,
        pooling: Pooling::Mean,
        query_prefix: "検索クエリ: ".to_string(),
        document_prefix: "検索文書: ".to_string(),
    }
}

impl ModelTable {
    fn resolve(&self, vault_root: &Path) -> Result<ModelContract> {
        if self.quantization != "none" {
            return Err(Error::Config(format!(
                "model.quantization must be \"none\" (got \"{}\"); use a pre-quantized ONNX file instead",
                self.quantization
            )));
        }
        if self.dim == 0 || self.max_tokens == 0 {
            return Err(Error::Config(
                "model.dim and model.max_tokens must be positive".into(),
            ));
        }
        validate_onnx_file(&self.onnx_file)?;
        Ok(ModelContract {
            id: self.id.clone(),
            source: parse_source(&self.source, vault_root)?,
            onnx_file: self.onnx_file.clone(),
            dim: self.dim,
            max_tokens: self.max_tokens,
            pooling: self.pooling,
            query_prefix: self.query_prefix.clone(),
            document_prefix: self.document_prefix.clone(),
        })
    }
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
        "model.source must be \"hf:<owner>/<repo>\" or \"path:<dir>\" (got \"{source}\")"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VaultConfig;
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

    fn resolve(toml: &str) -> Result<ModelContract> {
        VaultConfig::parse(toml)?.model.resolve(Path::new("/vault"))
    }

    #[test]
    fn preset_resolves_to_fp32_ruri_contract() {
        let c = resolve("model = \"ruri-v3-130m\"").unwrap();
        assert_eq!(c.id, "ruri-v3-130m");
        assert_eq!(
            c.source,
            ModelSource::Hf {
                repo: "sirasagi62/ruri-v3-130m-ONNX".into()
            }
        );
        assert_eq!(c.onnx_file, "onnx/model.onnx");
        assert_eq!((c.dim, c.max_tokens, c.pooling), (512, 512, Pooling::Mean));
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
        let err = resolve("model = \"ruri-v3-30m\"").unwrap_err();
        assert!(
            matches!(&err, Error::Config(m) if m.contains("ruri-v3-130m")),
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
}
