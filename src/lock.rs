use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;

use crate::{Error, Result};

/// Exclusive write lock; released when dropped or when the process exits.
pub struct WriteLock {
    _file: File,
}

fn open(path: &Path) -> Result<File> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| Error::io(path, e))
}

pub fn try_lock(path: &Path) -> Result<Option<WriteLock>> {
    let file = open(path)?;
    match file.try_lock() {
        Ok(()) => Ok(Some(WriteLock { _file: file })),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(e)) => Err(Error::io(path, e)),
    }
}

pub fn lock_blocking(path: &Path) -> Result<WriteLock> {
    let file = open(path)?;
    file.lock().map_err(|e| Error::io(path, e))?;
    Ok(WriteLock { _file: file })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn second_lock_is_refused_until_the_first_is_dropped() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index.db.lock");
        let first = try_lock(&path).unwrap();
        assert!(first.is_some());
        assert!(try_lock(&path).unwrap().is_none());
        drop(first);
        assert!(try_lock(&path).unwrap().is_some());
    }
}
