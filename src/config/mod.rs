mod model;
mod registry;
mod vault;

pub use model::{
    ModelContract, ModelSource, ModelSpec, ModelTable, OnnxContract, PRESET_RURI_V3_30M, Pooling,
    RemoteContract,
};
pub use registry::Registry;
pub use vault::{Vault, find_vault_root, init_vault, resolve_vault};

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{Error, Result};

pub const DEFAULT_CHUNK_MAX_TOKENS: usize = 450;
pub const CONFIG_DIR: &str = ".notrieve";
pub const CONFIG_FILE: &str = "config.toml";
pub const DEFAULT_CONFIG_TOML: &str = "\
# notrieve vault settings
model = \"ruri-v3-30m\"
exclude = []
";
pub const VAULT_GITIGNORE: &str = "index.db*\n*.embeddings.db*\n*.lock\n";

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VaultConfig {
    #[serde(default = "default_model")]
    pub model: ModelSpec,
    #[serde(default)]
    pub exclude: Vec<String>,
    #[serde(default = "default_db_path")]
    pub db_path: PathBuf,
    #[serde(default = "default_search_budget_ms")]
    pub search_budget_ms: u64,
    /// `None` when omitted. Required for a remote model, where it counts characters.
    #[serde(default)]
    pub chunk_max_tokens: Option<usize>,
    #[serde(default)]
    pub index_threads: usize,
}

fn default_model() -> ModelSpec {
    ModelSpec::Preset(PRESET_RURI_V3_30M.to_string())
}

fn default_db_path() -> PathBuf {
    PathBuf::from(".notrieve/index.db")
}

fn default_search_budget_ms() -> u64 {
    2000
}

impl VaultConfig {
    pub fn parse(text: &str) -> Result<Self> {
        let config: Self = toml::from_str(text).map_err(|e| Error::Config(e.to_string()))?;
        // Lowercased because macOS filesystems are case-insensitive; `contains` also covers the
        // cache's SQLite sidecars (-wal, -shm, -journal), which cache recovery deletes.
        let reserved = config.db_path.file_name().is_some_and(|name| {
            name.to_string_lossy()
                .to_ascii_lowercase()
                .contains(".embeddings.db")
        });
        if reserved {
            return Err(Error::Config(format!(
                "db_path {}: a file name containing `.embeddings.db` is reserved for the embedding cache",
                config.db_path.display()
            )));
        }
        // The model settings are checked when the vault loads, before anything uses them. A
        // `path:` source is only joined to the root here, never read, so any root does.
        let contract = config.model.resolve(Path::new("/"))?;
        if matches!(contract, ModelContract::Remote(_)) && config.chunk_max_tokens.is_none() {
            return Err(Error::Config(
                "chunk_max_tokens is required for a remote model; it counts characters (for example 1500)".into(),
            ));
        }
        Ok(config)
    }

    /// `chunk_max_tokens`, or 450 when a local model's config omits it.
    pub fn chunk_max_tokens(&self) -> usize {
        self.chunk_max_tokens.unwrap_or(DEFAULT_CHUNK_MAX_TOKENS)
    }
}

pub fn config_home_from(xdg: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
    dir_from(xdg, home, ".config")
}

pub fn cache_home_from(xdg: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
    dir_from(xdg, home, ".cache")
}

fn dir_from(xdg: Option<OsString>, home: Option<OsString>, fallback: &str) -> Option<PathBuf> {
    let xdg = xdg.map(PathBuf::from).filter(|p| p.is_absolute());
    match xdg {
        Some(dir) => Some(dir),
        None => home
            .filter(|v| !v.is_empty())
            .map(|h| PathBuf::from(h).join(fallback)),
    }
}

pub fn registry_path() -> Result<PathBuf> {
    config_home_from(
        std::env::var_os("XDG_CONFIG_HOME"),
        std::env::var_os("HOME"),
    )
    .map(|d| d.join("notrieve").join("vaults.toml"))
    .ok_or_else(|| Error::Config("neither XDG_CONFIG_HOME nor HOME is set".into()))
}

pub fn model_cache_dir() -> Result<PathBuf> {
    cache_home_from(std::env::var_os("XDG_CACHE_HOME"), std::env::var_os("HOME"))
        .map(|d| d.join("notrieve").join("models"))
        .ok_or_else(|| Error::Config("neither XDG_CACHE_HOME nor HOME is set".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_settings_take_documented_defaults() {
        let c = VaultConfig::parse("model = \"ruri-v3-30m\"\n").unwrap();
        assert!(c.exclude.is_empty());
        assert_eq!(c.db_path, PathBuf::from(".notrieve/index.db"));
        assert_eq!(
            (c.search_budget_ms, c.chunk_max_tokens(), c.index_threads),
            (2000, 450, 0)
        );
    }

    #[test]
    fn an_invalid_model_is_rejected_when_the_config_is_parsed() {
        let err = VaultConfig::parse("model = \"no-such-preset\"\n").unwrap_err();
        assert!(
            matches!(&err, Error::Config(m) if m.contains("no-such-preset")),
            "{err}"
        );
    }

    #[test]
    fn a_remote_model_requires_chunk_max_tokens() {
        let model =
            "[model]\nsource = \"openai-compat:https://api.openai.com/v1\"\nremote_model = \"m\"\n";
        let err = VaultConfig::parse(model).unwrap_err();
        assert!(
            matches!(&err, Error::Config(m) if m.contains("chunk_max_tokens")),
            "{err}"
        );
        let c = VaultConfig::parse(&format!("chunk_max_tokens = 1200\n{model}")).unwrap();
        assert_eq!(c.chunk_max_tokens(), 1200);
    }

    #[test]
    fn default_config_file_parses() {
        assert_eq!(
            VaultConfig::parse(DEFAULT_CONFIG_TOML).unwrap(),
            VaultConfig::parse("").unwrap()
        );
    }

    #[test]
    fn db_path_named_like_an_embedding_cache_is_rejected() {
        for name in [
            "work.db.embeddings.db",
            "work.db.EMBEDDINGS.DB",
            "work.db.embeddings.db-wal",
            "work.db.embeddings.db-shm",
        ] {
            let err = VaultConfig::parse(&format!("db_path = \"/c/{name}\"")).unwrap_err();
            assert!(
                matches!(&err, Error::Config(m) if m.contains("db_path") && m.contains(".embeddings.db")),
                "{name}: {err}"
            );
        }
        for name in ["work.sqlite", "embeddings.sqlite"] {
            let c = VaultConfig::parse(&format!("db_path = \"/c/{name}\"")).unwrap();
            assert_eq!(c.db_path, PathBuf::from(format!("/c/{name}")));
        }
    }

    #[test]
    fn unknown_key_is_rejected() {
        assert!(matches!(
            VaultConfig::parse("modle = \"x\""),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn vault_gitignore_covers_the_index_the_cache_and_the_lock() {
        use ignore::gitignore::GitignoreBuilder;
        let dir = std::path::Path::new("/vault/.notrieve");
        let mut builder = GitignoreBuilder::new(dir);
        for line in VAULT_GITIGNORE.lines() {
            builder.add_line(None, line).unwrap();
        }
        let gitignore = builder.build().unwrap();
        for name in [
            "index.db",
            "index.db-wal",
            "index.db.lock",
            "index.db.embeddings.db",
            "index.db.embeddings.db-wal",
            "index.db.embeddings.db-shm",
            "work.sqlite.embeddings.db",
        ] {
            assert!(
                gitignore.matched(dir.join(name), false).is_ignore(),
                "{name}"
            );
        }
        assert!(
            !gitignore
                .matched(dir.join("config.toml"), false)
                .is_ignore()
        );
    }

    #[test]
    fn xdg_dir_wins_over_home() {
        let got = config_home_from(Some("/xdg".into()), Some("/home/u".into()));
        assert_eq!(got, Some(PathBuf::from("/xdg")));
    }

    #[test]
    fn relative_xdg_dir_is_ignored() {
        assert_eq!(
            config_home_from(Some("rel/dir".into()), Some("/home/u".into())),
            Some(PathBuf::from("/home/u/.config"))
        );
        assert_eq!(
            cache_home_from(Some("./cache".into()), Some("/home/u".into())),
            Some(PathBuf::from("/home/u/.cache"))
        );
        assert_eq!(
            cache_home_from(Some("/abs/cache".into()), Some("/home/u".into())),
            Some(PathBuf::from("/abs/cache"))
        );
    }

    #[test]
    fn home_dot_config_is_the_fallback() {
        assert_eq!(
            config_home_from(Some("".into()), Some("/home/u".into())),
            Some(PathBuf::from("/home/u/.config"))
        );
        assert_eq!(
            cache_home_from(None, Some("/home/u".into())),
            Some(PathBuf::from("/home/u/.cache"))
        );
        assert_eq!(cache_home_from(None, None), None);
    }
}
