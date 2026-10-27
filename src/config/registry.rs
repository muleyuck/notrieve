use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use super::TrustedSettings;
use crate::{Error, Result, lock};

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct Registry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    #[serde(default)]
    pub vaults: BTreeMap<String, PathBuf>,
    /// What the user approved when registering each vault (see `TrustedSettings`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub trusted: BTreeMap<String, TrustedSettings>,
}

impl Registry {
    pub fn load(path: &Path) -> Result<Self> {
        match fs::read_to_string(path) {
            Ok(text) => {
                toml::from_str(&text).map_err(|e| Error::Config(format!("{}: {e}", path.display())))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(Error::io(path, e)),
        }
    }

    /// Serialises load, `f` and save across processes; a failing `f` leaves the file untouched.
    /// The directory is locked, not the file: `save` replaces the file by a rename.
    pub fn update<T>(path: &Path, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let dir = path.parent().unwrap_or(Path::new("."));
        let _lock = lock::lock_directory(dir)?;
        let mut registry = Self::load(path)?;
        let out = f(&mut registry)?;
        registry.save(path)?;
        Ok(out)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let text = toml::to_string(self).map_err(|e| Error::Config(e.to_string()))?;
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let tmp = path.with_extension(format!(
            "toml.{}.{}.tmp",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&tmp, text).map_err(|e| Error::io(&tmp, e))?;
        fs::rename(&tmp, path).map_err(|e| Error::io(path, e))
    }

    pub fn add(&mut self, name: &str, root: &Path, make_default: bool) -> Result<()> {
        if self.vaults.contains_key(name) {
            return Err(Error::DuplicateVaultName(name.to_string()));
        }
        let first = self.vaults.is_empty();
        self.vaults.insert(name.to_string(), root.to_path_buf());
        if make_default || first {
            self.default = Some(name.to_string());
        }
        Ok(())
    }

    pub fn trust(&mut self, name: &str, settings: TrustedSettings) {
        if settings.is_empty() {
            self.trusted.remove(name);
        } else {
            self.trusted.insert(name.to_string(), settings);
        }
    }

    pub fn remove(&mut self, name: &str) -> Result<PathBuf> {
        self.trusted.remove(name);
        let removed = self.vaults.remove(name);
        let path = removed.ok_or_else(|| self.not_registered(name))?;
        if self.default.as_deref() == Some(name) {
            self.default = None;
        }
        Ok(path)
    }

    pub fn get(&self, name: &str) -> Result<&Path> {
        self.vaults
            .get(name)
            .map(PathBuf::as_path)
            .ok_or_else(|| self.not_registered(name))
    }

    pub fn name_of(&self, root: &Path) -> Option<String> {
        let target = root.canonicalize().ok()?;
        self.vaults
            .iter()
            .find(|(_, path)| path.canonicalize().ok().as_deref() == Some(target.as_path()))
            .map(|(name, _)| name.clone())
    }

    fn not_registered(&self, name: &str) -> Error {
        let known = if self.vaults.is_empty() {
            "none".to_string()
        } else {
            self.vaults.keys().cloned().collect::<Vec<_>>().join(", ")
        };
        Error::VaultNotRegistered {
            name: name.to_string(),
            known,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrustedSettings;
    use tempfile::TempDir;

    #[test]
    fn removing_a_vault_forgets_what_was_trusted() {
        let mut r = Registry::default();
        r.add("notes", Path::new("/a"), false).unwrap();
        r.trust(
            "notes",
            TrustedSettings {
                index_outside: Some(PathBuf::from("/cache/notes.db")),
                ..TrustedSettings::default()
            },
        );
        r.remove("notes").unwrap();
        assert!(r.trusted.is_empty());
    }

    #[test]
    fn missing_file_loads_as_empty_registry() {
        let dir = TempDir::new().unwrap();
        let r = Registry::load(&dir.path().join("vaults.toml")).unwrap();
        assert_eq!(r, Registry::default());
    }

    #[test]
    fn first_vault_becomes_default_and_later_ones_do_not() {
        let mut r = Registry::default();
        r.add("notes", Path::new("/a"), false).unwrap();
        r.add("work", Path::new("/b"), false).unwrap();
        assert_eq!(r.default.as_deref(), Some("notes"));
    }

    #[test]
    fn make_default_switches_the_default() {
        let mut r = Registry::default();
        r.add("notes", Path::new("/a"), false).unwrap();
        r.add("work", Path::new("/b"), true).unwrap();
        assert_eq!(r.default.as_deref(), Some("work"));
    }

    #[test]
    fn adding_after_the_default_was_removed_does_not_make_the_new_vault_default() {
        let mut r = Registry::default();
        r.add("a", Path::new("/a"), false).unwrap();
        r.add("b", Path::new("/b"), false).unwrap();
        r.remove("a").unwrap();
        r.add("c", Path::new("/c"), false).unwrap();
        assert_eq!(r.default, None);
        r.add("d", Path::new("/d"), true).unwrap();
        assert_eq!(r.default.as_deref(), Some("d"));
    }

    #[test]
    fn adding_to_an_emptied_registry_sets_the_default_again() {
        let mut r = Registry::default();
        r.add("a", Path::new("/a"), false).unwrap();
        r.remove("a").unwrap();
        r.add("c", Path::new("/c"), false).unwrap();
        assert_eq!(r.default.as_deref(), Some("c"));
    }

    #[test]
    fn duplicate_name_is_rejected() {
        let mut r = Registry::default();
        r.add("notes", Path::new("/a"), false).unwrap();
        assert!(
            matches!(r.add("notes", Path::new("/b"), false), Err(Error::DuplicateVaultName(n)) if n == "notes")
        );
    }

    #[test]
    fn removing_the_default_clears_it() {
        let mut r = Registry::default();
        r.add("notes", Path::new("/a"), false).unwrap();
        assert_eq!(r.remove("notes").unwrap(), PathBuf::from("/a"));
        assert_eq!(r.default, None);
    }

    #[test]
    fn unknown_name_error_lists_registered_names() {
        let mut r = Registry::default();
        r.add("notes", Path::new("/a"), false).unwrap();
        r.add("work", Path::new("/b"), false).unwrap();
        match r.get("diary") {
            Err(Error::VaultNotRegistered { name, known }) => {
                assert_eq!(name, "diary");
                assert_eq!(known, "notes, work");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn concurrent_updates_all_survive() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vaults.toml");
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let path = path.clone();
                std::thread::spawn(move || {
                    for i in 0..25 {
                        Registry::update(&path, |r| {
                            r.add(&format!("v{t}-{i}"), Path::new("/a"), false)
                        })
                        .unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(Registry::load(&path).unwrap().vaults.len(), 100);
    }

    #[test]
    fn single_update_persists_and_a_failing_one_changes_nothing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vaults.toml");
        Registry::update(&path, |r| r.add("a", Path::new("/a"), false)).unwrap();
        let err = Registry::update(&path, |r| r.add("a", Path::new("/b"), false)).unwrap_err();
        assert!(matches!(err, Error::DuplicateVaultName(_)));
        let r = Registry::load(&path).unwrap();
        assert_eq!(r.get("a").unwrap(), Path::new("/a"));
        let leftovers: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nested/vaults.toml");
        let mut r = Registry::default();
        r.add("notes", Path::new("/a"), false).unwrap();
        r.add("work", Path::new("/b"), true).unwrap();
        r.save(&path).unwrap();
        assert_eq!(Registry::load(&path).unwrap(), r);
    }
}
