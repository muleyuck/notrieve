use std::fs;
use std::path::{Path, PathBuf};

use super::{CONFIG_DIR, CONFIG_FILE, DEFAULT_CONFIG_TOML, Registry, VAULT_GITIGNORE, VaultConfig};
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Vault {
    pub name: Option<String>,
    pub root: PathBuf,
    pub config: VaultConfig,
}

impl Vault {
    pub fn load(root: &Path, name: Option<String>) -> Result<Self> {
        let path = root.join(CONFIG_DIR).join(CONFIG_FILE);
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::NotAVault(root.to_path_buf()));
            }
            Err(e) => return Err(Error::io(&path, e)),
        };
        let config = VaultConfig::parse(&text).map_err(|e| match e {
            Error::Config(message) => Error::Config(format!("{}: {message}", path.display())),
            other => other,
        })?;
        Ok(Self {
            name,
            root: root.to_path_buf(),
            config,
        })
    }

    pub fn config_dir(&self) -> PathBuf {
        self.root.join(CONFIG_DIR)
    }

    pub fn db_path(&self) -> PathBuf {
        if self.config.db_path.is_absolute() {
            self.config.db_path.clone()
        } else {
            self.root.join(&self.config.db_path)
        }
    }

    pub fn lock_path(&self) -> PathBuf {
        let mut path = self.db_path().into_os_string();
        path.push(".lock");
        PathBuf::from(path)
    }
}

pub fn find_vault_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|dir| dir.join(CONFIG_DIR).join(CONFIG_FILE).is_file())
        .map(Path::to_path_buf)
}

pub fn resolve_vault(explicit: Option<&str>, cwd: &Path, registry: &Registry) -> Result<Vault> {
    if let Some(name) = explicit {
        return load_registered(name, registry);
    }
    if let Some(root) = find_vault_root(cwd) {
        let name = registry.name_of(&root);
        return Vault::load(&root, name);
    }
    match registry.default.as_deref() {
        Some(name) => load_registered(name, registry),
        None => Err(Error::VaultNotFound),
    }
}

fn load_registered(name: &str, registry: &Registry) -> Result<Vault> {
    let root = registry.get(name)?;
    if !root.is_dir() {
        return Err(Error::VaultPathMissing {
            name: name.to_string(),
            path: root.to_path_buf(),
        });
    }
    Vault::load(root, Some(name.to_string()))
}

pub fn init_vault(root: &Path) -> Result<()> {
    let dir = root.join(CONFIG_DIR);
    if dir.exists() {
        return Err(Error::AlreadyInitialized(root.to_path_buf()));
    }
    fs::create_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
    let config = dir.join(CONFIG_FILE);
    fs::write(&config, DEFAULT_CONFIG_TOML).map_err(|e| Error::io(&config, e))?;
    let gitignore = dir.join(".gitignore");
    fs::write(&gitignore, VAULT_GITIGNORE).map_err(|e| Error::io(&gitignore, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn vault_dir(parent: &Path, name: &str) -> PathBuf {
        let root = parent.join(name);
        init_vault(&root).unwrap();
        root
    }

    #[test]
    fn init_writes_config_and_gitignore() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("notes");
        init_vault(&root).unwrap();
        assert_eq!(
            fs::read_to_string(root.join(".notrieve/config.toml")).unwrap(),
            DEFAULT_CONFIG_TOML
        );
        assert_eq!(
            fs::read_to_string(root.join(".notrieve/.gitignore")).unwrap(),
            VAULT_GITIGNORE
        );
    }

    #[test]
    fn init_twice_fails() {
        let tmp = TempDir::new().unwrap();
        let root = vault_dir(tmp.path(), "notes");
        assert!(matches!(
            init_vault(&root),
            Err(Error::AlreadyInitialized(_))
        ));
    }

    #[test]
    fn root_is_found_from_a_nested_directory() {
        let tmp = TempDir::new().unwrap();
        let root = vault_dir(tmp.path(), "notes");
        let nested = root.join("2026/10");
        fs::create_dir_all(&nested).unwrap();
        assert_eq!(find_vault_root(&nested), Some(root));
    }

    #[test]
    fn explicit_name_beats_the_enclosing_vault() {
        let tmp = TempDir::new().unwrap();
        let here = vault_dir(tmp.path(), "here");
        let other = vault_dir(tmp.path(), "other");
        let mut r = Registry::default();
        r.add("other", &other, false).unwrap();
        let v = resolve_vault(Some("other"), &here, &r).unwrap();
        assert_eq!((v.root, v.name.as_deref()), (other, Some("other")));
    }

    #[test]
    fn enclosing_vault_beats_the_default() {
        let tmp = TempDir::new().unwrap();
        let here = vault_dir(tmp.path(), "here");
        let other = vault_dir(tmp.path(), "other");
        let mut r = Registry::default();
        r.add("other", &other, true).unwrap();
        assert_eq!(resolve_vault(None, &here, &r).unwrap().root, here);
    }

    #[test]
    fn default_is_used_outside_any_vault() {
        let tmp = TempDir::new().unwrap();
        let other = vault_dir(tmp.path(), "other");
        let outside = tmp.path().join("outside");
        fs::create_dir_all(&outside).unwrap();
        let mut r = Registry::default();
        r.add("other", &other, true).unwrap();
        assert_eq!(resolve_vault(None, &outside, &r).unwrap().root, other);
    }

    #[test]
    fn nothing_found_is_vault_not_found() {
        let tmp = TempDir::new().unwrap();
        assert!(matches!(
            resolve_vault(None, tmp.path(), &Registry::default()),
            Err(Error::VaultNotFound)
        ));
    }

    #[test]
    fn registered_path_that_disappeared_is_reported() {
        let tmp = TempDir::new().unwrap();
        let mut r = Registry::default();
        r.add("gone", &tmp.path().join("gone"), false).unwrap();
        assert!(
            matches!(resolve_vault(Some("gone"), tmp.path(), &r), Err(Error::VaultPathMissing { name, .. }) if name == "gone")
        );
    }

    #[test]
    fn db_and_lock_paths_follow_db_path_setting() {
        let tmp = TempDir::new().unwrap();
        let root = vault_dir(tmp.path(), "notes");
        let mut v = Vault::load(&root, None).unwrap();
        assert_eq!(v.db_path(), root.join(".notrieve/index.db"));
        assert_eq!(v.lock_path(), root.join(".notrieve/index.db.lock"));
        v.config.db_path = PathBuf::from("/cache/notes.db");
        assert_eq!(v.db_path(), PathBuf::from("/cache/notes.db"));
    }

    #[test]
    fn directory_without_config_is_not_a_vault() {
        let tmp = TempDir::new().unwrap();
        assert!(matches!(
            Vault::load(tmp.path(), None),
            Err(Error::NotAVault(_))
        ));
    }
}
