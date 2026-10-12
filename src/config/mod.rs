mod model;
mod registry;
mod vault;

pub use model::{ModelContract, ModelSource, ModelSpec, ModelTable, PRESET_RURI_V3_130M, Pooling};
pub use registry::Registry;
pub use vault::{Vault, find_vault_root, init_vault, resolve_vault};

use std::ffi::OsString;
use std::path::PathBuf;

use serde::Deserialize;

use crate::{Error, Result};

pub const CONFIG_DIR: &str = ".notrieve";
pub const CONFIG_FILE: &str = "config.toml";
pub const DEFAULT_CONFIG_TOML: &str = "\
# notrieve vault settings
model = \"ruri-v3-130m\"
exclude = []
";
pub const VAULT_GITIGNORE: &str = "index.db*\n*.lock\n";

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
    #[serde(default = "default_chunk_max_tokens")]
    pub chunk_max_tokens: usize,
    #[serde(default)]
    pub index_threads: usize,
}

fn default_model() -> ModelSpec {
    ModelSpec::Preset(PRESET_RURI_V3_130M.to_string())
}

fn default_db_path() -> PathBuf {
    PathBuf::from(".notrieve/index.db")
}

fn default_search_budget_ms() -> u64 {
    2000
}

fn default_chunk_max_tokens() -> usize {
    450
}

impl VaultConfig {
    pub fn parse(text: &str) -> Result<Self> {
        toml::from_str(text).map_err(|e| Error::Config(e.to_string()))
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
        let c = VaultConfig::parse("model = \"ruri-v3-130m\"\n").unwrap();
        assert!(c.exclude.is_empty());
        assert_eq!(c.db_path, PathBuf::from(".notrieve/index.db"));
        assert_eq!(
            (c.search_budget_ms, c.chunk_max_tokens, c.index_threads),
            (2000, 450, 0)
        );
    }

    #[test]
    fn default_config_file_parses() {
        assert_eq!(
            VaultConfig::parse(DEFAULT_CONFIG_TOML).unwrap(),
            VaultConfig::parse("").unwrap()
        );
    }

    #[test]
    fn unknown_key_is_rejected() {
        assert!(matches!(
            VaultConfig::parse("modle = \"x\""),
            Err(Error::Config(_))
        ));
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
