use std::fs;
use std::path::{Component, Path, PathBuf};

use super::{
    CONFIG_DIR, CONFIG_FILE, DEFAULT_CONFIG_TOML, ModelContract, Registry, RemoteTarget,
    TrustedSettings, VAULT_GITIGNORE, VaultConfig,
};
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

    /// The settings that need the user's approval (`TrustedSettings`) as this config has them.
    pub fn trusted_settings(&self) -> TrustedSettings {
        let remote = match self.config.model.resolve(&self.root) {
            Ok(ModelContract::Remote(c)) => Some(RemoteTarget {
                base_url: c.base_url,
                api_key_env: c.api_key_env,
                auth_header: c.auth_header,
            }),
            _ => None,
        };
        let lexically_inside = self
            .config
            .db_path
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
        let location = self.index_location();
        let inside = lexically_inside
            && self
                .root
                .canonicalize()
                .is_ok_and(|root| location.starts_with(root));
        TrustedSettings {
            remote,
            index_outside: (!inside).then_some(location),
        }
    }

    /// Where the index is written once directory links are resolved: missing directories are
    /// created under the deepest one that exists, so that one is resolved. Recording this rather
    /// than `db_path` makes a relinked directory (a cloned repository can carry links) a change.
    pub fn index_location(&self) -> PathBuf {
        let db = self.db_path();
        for dir in db.ancestors().skip(1) {
            if let (Ok(mut location), Ok(rest)) = (dir.canonicalize(), db.strip_prefix(dir)) {
                // The rest does not exist yet: resolve its `..` against what it will create, so
                // that the value stays the same once those directories exist.
                for component in rest.components() {
                    match component {
                        Component::ParentDir => {
                            location.pop();
                        }
                        Component::CurDir => {}
                        other => location.push(other),
                    }
                }
                return location;
            }
        }
        db
    }

    /// For a long-running process: refuses to touch the index once its location differs from
    /// `expected` (taken at startup, after `resolve_vault` checked it) or one of its files is a
    /// link. A synced vault can swap a directory for a link while the process runs.
    pub fn ensure_index_location(&self, expected: &Path) -> Result<()> {
        if let Some(link) = self.linked_index_file() {
            return Err(Error::IndexFileLinked(link));
        }
        if self.index_location() != expected {
            return Err(Error::FingerprintChanged);
        }
        Ok(())
    }

    /// The first of the index, its lock, its embedding cache and their side files that is a
    /// symbolic link. Writing through one would reach a file nobody approved.
    fn linked_index_file(&self) -> Option<PathBuf> {
        let db = self.db_path();
        let files = [db.clone(), self.lock_path(), crate::cache::cache_path(&db)];
        files.into_iter().find_map(|file| {
            ["", "-wal", "-shm"].into_iter().find_map(|suffix| {
                let mut path = file.clone().into_os_string();
                path.push(suffix);
                let path = PathBuf::from(path);
                path.symlink_metadata()
                    .is_ok_and(|m| m.file_type().is_symlink())
                    .then_some(path)
            })
        })
    }
}

pub fn find_vault_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|dir| dir.join(CONFIG_DIR).join(CONFIG_FILE).is_file())
        .map(Path::to_path_buf)
}

pub fn resolve_vault(explicit: Option<&str>, cwd: &Path, registry: &Registry) -> Result<Vault> {
    let vault = if let Some(name) = explicit {
        load_registered(name, registry)?
    } else if let Some(root) = find_vault_root(cwd) {
        Vault::load(&root, registry.name_of(&root))?
    } else {
        match registry.default.as_deref() {
            Some(name) => load_registered(name, registry)?,
            None => return Err(Error::VaultNotFound),
        }
    };
    check_trust(&vault, registry)?;
    Ok(vault)
}

/// Refuses settings the user has not approved: any in an unregistered vault, and in a registered
/// one those its config gained after `vault add` (for example through a pull).
fn check_trust(vault: &Vault, registry: &Registry) -> Result<()> {
    if let Some(link) = vault.linked_index_file() {
        return Err(Error::IndexFileLinked(link));
    }
    let current = vault.trusted_settings();
    let approved = vault
        .name
        .as_ref()
        .and_then(|name| registry.trusted.get(name));
    if approved.cloned().unwrap_or_default().allows(&current) {
        return Ok(());
    }
    let root = vault.root.clone();
    let settings = current.describe().join(" and ");
    Err(match &vault.name {
        None => Error::UntrustedVault { root, settings },
        Some(name) => Error::VaultConfigChanged {
            name: name.clone(),
            root,
            settings,
        },
    })
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

    const REMOTE_CONFIG: &str = "chunk_max_tokens = 1500\n[model]\nsource = \"openai-compat:https://api.example.com/v1\"\nremote_model = \"m\"\napi_key_env = \"EXAMPLE_KEY\"\n";

    fn vault_with_config(parent: &Path, name: &str, config: &str) -> PathBuf {
        let root = vault_dir(parent, name);
        fs::write(root.join(".notrieve/config.toml"), config).unwrap();
        root
    }

    #[test]
    fn an_unregistered_vault_with_a_remote_model_is_refused() {
        let tmp = TempDir::new().unwrap();
        let here = vault_with_config(tmp.path(), "here", REMOTE_CONFIG);
        let err = resolve_vault(None, &here, &Registry::default()).unwrap_err();
        let message = err.to_string();
        assert!(matches!(err, Error::UntrustedVault { .. }), "{message}");
        assert!(
            message.contains("api.example.com") && message.contains("EXAMPLE_KEY"),
            "{message}"
        );
    }

    #[test]
    fn a_registered_vault_with_its_approved_remote_model_is_used() {
        let tmp = TempDir::new().unwrap();
        let here = vault_with_config(tmp.path(), "here", REMOTE_CONFIG);
        let mut r = Registry::default();
        r.add("here", &here, false).unwrap();
        r.trust("here", Vault::load(&here, None).unwrap().trusted_settings());
        assert!(resolve_vault(None, &here, &r).is_ok());
    }

    #[test]
    fn a_registered_vault_whose_config_gained_a_remote_model_is_refused() {
        let tmp = TempDir::new().unwrap();
        let here = vault_dir(tmp.path(), "here");
        let outside = tmp.path().join("outside");
        fs::create_dir_all(&outside).unwrap();
        let mut r = Registry::default();
        r.add("here", &here, true).unwrap();
        r.trust("here", Vault::load(&here, None).unwrap().trusted_settings());
        fs::write(here.join(".notrieve/config.toml"), REMOTE_CONFIG).unwrap();
        // By the working directory, by name, and as the default vault.
        for (explicit, cwd) in [(None, &here), (Some("here"), &outside), (None, &outside)] {
            let err = resolve_vault(explicit, cwd, &r).unwrap_err();
            let message = err.to_string();
            assert!(
                matches!(err, Error::VaultConfigChanged { .. }),
                "{explicit:?} {}: {message}",
                cwd.display()
            );
            assert!(message.contains("api.example.com"), "{message}");
        }
    }

    #[test]
    fn an_unregistered_vault_with_the_index_outside_it_is_refused() {
        let tmp = TempDir::new().unwrap();
        for (name, db_path) in [("abs", "/elsewhere/index.db"), ("up", "../index.db")] {
            let config = format!("db_path = \"{db_path}\"\n");
            let here = vault_with_config(tmp.path(), name, &config);
            let err = resolve_vault(None, &here, &Registry::default()).unwrap_err();
            assert!(
                matches!(err, Error::UntrustedVault { .. }),
                "{db_path}: {err}"
            );
        }
    }

    /// Registers `root` under its directory name, approving its config as it is now.
    fn register_and_approve(r: &mut Registry, root: &Path) -> String {
        let name = root.file_name().unwrap().to_string_lossy().into_owned();
        r.add(&name, root, false).unwrap();
        r.trust(&name, Vault::load(root, None).unwrap().trusted_settings());
        name
    }

    #[cfg(unix)]
    #[test]
    fn an_unregistered_vault_whose_index_directory_links_outside_is_refused() {
        let tmp = TempDir::new().unwrap();
        let victim = tmp.path().join("victim");
        fs::create_dir_all(&victim).unwrap();
        let here = vault_with_config(tmp.path(), "here", "db_path = \"data/./index.db\"\n");
        std::os::unix::fs::symlink(&victim, here.join("data")).unwrap();
        let err = resolve_vault(None, &here, &Registry::default()).unwrap_err();
        assert!(matches!(err, Error::UntrustedVault { .. }), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn relinking_an_approved_index_directory_is_refused() {
        use std::os::unix::fs::symlink;
        let tmp = TempDir::new().unwrap();
        let (first, second) = (tmp.path().join("first"), tmp.path().join("second"));
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&second).unwrap();
        let here = vault_with_config(tmp.path(), "here", "db_path = \"data/index.db\"\n");
        symlink(&first, here.join("data")).unwrap();
        let mut r = Registry::default();
        register_and_approve(&mut r, &here);
        assert!(resolve_vault(None, &here, &r).is_ok());
        fs::remove_file(here.join("data")).unwrap();
        symlink(&second, here.join("data")).unwrap();
        let err = resolve_vault(None, &here, &r).unwrap_err();
        assert!(matches!(err, Error::VaultConfigChanged { .. }), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn a_link_at_an_index_file_is_refused_even_in_a_registered_vault() {
        let tmp = TempDir::new().unwrap();
        let mut r = Registry::default();
        // The index, its lock, its embedding cache, and a side file of the index.
        for file in [
            "index.db",
            "index.db.lock",
            "index.db.embeddings.db",
            "index.db-wal",
        ] {
            let here = vault_dir(tmp.path(), file);
            std::os::unix::fs::symlink(
                tmp.path().join("app.db"),
                here.join(".notrieve").join(file),
            )
            .unwrap();
            register_and_approve(&mut r, &here);
            let err = resolve_vault(None, &here, &r).unwrap_err();
            assert!(matches!(err, Error::IndexFileLinked(_)), "{file}: {err}");
        }
    }

    #[test]
    fn an_approved_index_outside_the_vault_is_used_again_and_a_new_one_is_refused() {
        let tmp = TempDir::new().unwrap();
        // `missing` does not exist when the setting is approved; the first run creates it.
        let here = vault_with_config(
            tmp.path(),
            "here",
            "db_path = \"../missing/../outside/index.db\"\n",
        );
        let mut r = Registry::default();
        register_and_approve(&mut r, &here);
        fs::create_dir_all(tmp.path().join("missing")).unwrap();
        fs::create_dir_all(tmp.path().join("outside")).unwrap();
        assert!(resolve_vault(None, &here, &r).is_ok());
        fs::write(
            here.join(".notrieve/config.toml"),
            "db_path = \"../elsewhere/index.db\"\n",
        )
        .unwrap();
        let err = resolve_vault(None, &here, &r).unwrap_err();
        assert!(matches!(err, Error::VaultConfigChanged { .. }), "{err}");
    }

    #[test]
    fn a_change_to_an_approved_remote_setting_is_refused() {
        let tmp = TempDir::new().unwrap();
        let here = vault_with_config(tmp.path(), "here", REMOTE_CONFIG);
        let mut r = Registry::default();
        register_and_approve(&mut r, &here);
        let changes = [
            (
                "host",
                REMOTE_CONFIG.replace("api.example.com", "other.example"),
            ),
            ("key", REMOTE_CONFIG.replace("EXAMPLE_KEY", "OTHER_KEY")),
            (
                "header",
                format!("{REMOTE_CONFIG}auth_header = \"X-Other\"\n"),
            ),
        ];
        for (what, config) in changes {
            fs::write(here.join(".notrieve/config.toml"), config).unwrap();
            let err = resolve_vault(None, &here, &r).unwrap_err();
            assert!(
                matches!(err, Error::VaultConfigChanged { .. }),
                "{what}: {err}"
            );
        }
    }

    #[test]
    fn an_index_path_inside_the_vault_needs_no_registration() {
        let tmp = TempDir::new().unwrap();
        let here = vault_with_config(tmp.path(), "here", "db_path = \"data/index.db\"\n");
        assert!(resolve_vault(None, &here, &Registry::default()).is_ok());
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
