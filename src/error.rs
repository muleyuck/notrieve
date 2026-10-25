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
    #[error(
        "the vault at {} is not registered, and its config {settings}; check {}/.notrieve/config.toml and run `notrieve vault add {}` if you trust it",
        .root.display(),
        .root.display(),
        .root.display()
    )]
    UntrustedVault { root: PathBuf, settings: String },
    #[error(
        "{} is a symbolic link; notrieve writes its index, lock and cache only as regular files, so remove the link",
        .0.display()
    )]
    IndexFileLinked(PathBuf),
    #[error(
        "the config of vault `{name}` changed since you registered it: it now {settings}; check {}/.notrieve/config.toml and run `notrieve vault add {} --name {name}` if you trust it",
        .root.display(),
        .root.display()
    )]
    VaultConfigChanged {
        name: String,
        root: PathBuf,
        settings: String,
    },
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
    /// A failed call to a remote embedding API. `message` never contains the API key.
    #[error("embedding API error: {message}")]
    Remote {
        kind: RemoteErrorKind,
        message: String,
    },
    #[error("database error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error(
        "{0} is not a readable SQLite database; run `notrieve index --rebuild` to move it aside and rebuild"
    )]
    DbCorrupt(PathBuf),
    #[error(
        "{0} is a database of another program; notrieve leaves it untouched, so set db_path to another file"
    )]
    ForeignDatabase(PathBuf),
    #[error(
        "the index was written by a newer notrieve (schema {found}, supported {supported}); update notrieve or run `notrieve index --rebuild`"
    )]
    SchemaTooNew { found: i64, supported: i64 },
    #[error("the index settings changed while this process was running; restart it")]
    FingerprintChanged,
    #[error("invalid query: {0}")]
    InvalidQuery(String),
    #[error("the index is missing or was built with other settings; run `notrieve index`")]
    IndexNotReady,
    #[error("{0} is not valid UTF-8")]
    NotUtf8(String),
    #[error("`{0}` is outside the vault")]
    OutsideVault(String),
    #[error("no note at `{0}` in the vault")]
    NoteNotFound(String),
    #[error("`{0}` is not an indexed note; pass a path from the search results")]
    NoteNotIndexed(String),
    #[error("invalid line range: {0}")]
    InvalidLineRange(String),
    /// A background indexer failure; the message is complete (it names the indexer).
    #[error("{0}")]
    IndexerFailed(String),
    #[error(
        "the index is being rebuilt{}; search again shortly",
        .remaining.map(|n| format!(" ({n} files left)")).unwrap_or_default()
    )]
    Rebuilding { remaining: Option<usize> },
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

/// How a remote embedding failure is handled (remote spec §5.1, §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteErrorKind {
    /// Worth trying again later: no connection, a timeout, 408, 429 or 5xx.
    Transient,
    /// 401 or 403, or no API key in the environment.
    Auth,
    /// The server refused this input (another 4xx).
    Input,
    /// The endpoint or its answers are wrong: 3xx, 404, 405, an invalid response, or vectors of
    /// another dimension than the index.
    Protocol,
}
