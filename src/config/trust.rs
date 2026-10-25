use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// The settings of a vault config that send data off this machine or write the index outside the
/// vault. The config travels with the notes, so these apply only as the user approved them when
/// registering the vault (`notrieve vault add`), which records them in the registry.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote: Option<RemoteTarget>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_outside: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteTarget {
    pub base_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    pub auth_header: String,
}

impl TrustedSettings {
    pub fn is_empty(&self) -> bool {
        self.remote.is_none() && self.index_outside.is_none()
    }

    /// True when every setting in `current` is one of these.
    pub fn allows(&self, current: &TrustedSettings) -> bool {
        (current.remote.is_none() || current.remote == self.remote)
            && (current.index_outside.is_none() || current.index_outside == self.index_outside)
    }

    /// One phrase per setting, completing "its config ...".
    pub fn describe(&self) -> Vec<String> {
        let mut phrases = Vec::new();
        if let Some(remote) = &self.remote {
            let key = remote
                .api_key_env
                .as_ref()
                .map(|var| format!(" with ${var} in the {} header", remote.auth_header))
                .unwrap_or_default();
            phrases.push(format!(
                "sends note text and queries to {}{key}",
                remote.base_url
            ));
        }
        if let Some(path) = &self.index_outside {
            phrases.push(format!(
                "writes the index outside the vault ({})",
                path.display()
            ));
        }
        phrases
    }
}
