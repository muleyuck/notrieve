use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no vault found; run `notrieve init` in your notes directory or pass `--vault <name>`")]
    VaultNotFound,
    #[error("vault `{name}` is not registered (registered: {known})")]
    VaultNotRegistered { name: String, known: String },
    #[error(
        "vault `{name}` points to {path}, which does not exist; run `notrieve vault remove {name}`"
    )]
    VaultPathMissing { name: String, path: PathBuf },
    #[error("{0} already contains .notrieve/")]
    AlreadyInitialized(PathBuf),
    #[error("a vault named `{0}` is already registered")]
    DuplicateVaultName(String),
    #[error("{0} has no .notrieve/config.toml")]
    NotAVault(PathBuf),
    #[error("configuration error: {0}")]
    Config(String),
    #[error("another notrieve process is updating the index")]
    LockBusy,
    #[error("a chunk needs {tokens} tokens but the limit is {max}")]
    ChunkTooLarge { tokens: usize, max: usize },
    #[error("model error: {0}; run `notrieve model pull --force` if the files are damaged")]
    Model(String),
    #[error("model file {file} is missing in {dir}")]
    ModelFileMissing { dir: PathBuf, file: String },
    #[error("download of {url} failed: {message}")]
    Download { url: String, message: String },
    #[error("embedding failed: {0}")]
    Embedding(String),
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn io(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.as_ref().to_path_buf(),
            source,
        }
    }
}
