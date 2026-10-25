use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use super::onnx::required_files;
use crate::config::{ModelSource, OnnxContract};
use crate::{Error, Result, lock};

const COMPLETE_MARKER: &str = ".complete";
pub const HF_BASE_URL: &str = "https://huggingface.co";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadEvent<'a> {
    Started { file: &'a str, total: Option<u64> },
    Finished { file: &'a str },
}

pub fn model_dir(contract: &OnnxContract, cache_root: &Path) -> PathBuf {
    match &contract.source {
        ModelSource::Hf { repo } => cache_root.join(repo.replace('/', "__")),
        ModelSource::Path { dir } => dir.clone(),
    }
}

pub fn ensure_model(
    contract: &OnnxContract,
    cache_root: &Path,
    force: bool,
    on_event: &mut dyn FnMut(DownloadEvent<'_>),
) -> Result<PathBuf> {
    ensure_model_from(HF_BASE_URL, contract, cache_root, force, on_event)
}

pub fn ensure_model_from(
    base_url: &str,
    contract: &OnnxContract,
    cache_root: &Path,
    force: bool,
    on_event: &mut dyn FnMut(DownloadEvent<'_>),
) -> Result<PathBuf> {
    with_model_from(base_url, contract, cache_root, force, on_event, |dir| {
        Ok(dir.to_path_buf())
    })
}

/// Ensures the model files and runs `f` on them while the cache lock is still held, so a
/// concurrent `model pull --force` cannot replace files between loading and fingerprinting.
pub fn with_model<T>(
    contract: &OnnxContract,
    cache_root: &Path,
    force: bool,
    on_event: &mut dyn FnMut(DownloadEvent<'_>),
    f: impl FnOnce(&Path) -> Result<T>,
) -> Result<T> {
    with_model_from(HF_BASE_URL, contract, cache_root, force, on_event, f)
}

pub fn with_model_from<T>(
    base_url: &str,
    contract: &OnnxContract,
    cache_root: &Path,
    force: bool,
    on_event: &mut dyn FnMut(DownloadEvent<'_>),
    f: impl FnOnce(&Path) -> Result<T>,
) -> Result<T> {
    let dir = model_dir(contract, cache_root);
    match &contract.source {
        ModelSource::Path { .. } => {
            if let Some(file) = required_files(contract)
                .into_iter()
                .find(|f| !dir.join(f).is_file())
            {
                return Err(Error::ModelFileMissing {
                    dir,
                    file: file.to_string(),
                });
            }
        }
        ModelSource::Hf { repo } => {
            fs::create_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
            // The cache is shared by every vault and process; one downloader at a time.
            let _lock = lock::lock_blocking(&dir.join(".download.lock"))?;
            refresh_locked(base_url, repo, contract, &dir, force, on_event)?;
            return f(&dir);
        }
    }
    f(&dir)
}

fn refresh_locked(
    base_url: &str,
    repo: &str,
    contract: &OnnxContract,
    dir: &Path,
    force: bool,
    on_event: &mut dyn FnMut(DownloadEvent<'_>),
) -> Result<()> {
    // Renaming several files is not atomic, so a complete set is recorded by a marker
    // that exists only while every file belongs to the same download.
    let marker = dir.join(COMPLETE_MARKER);
    let refresh = force || !marker.is_file();
    // All-or-nothing: nothing in the cache is replaced until every file has downloaded.
    let mut staged: Vec<(PathBuf, PathBuf)> = Vec::new();
    for file in required_files(contract) {
        let target = dir.join(file);
        if refresh || !target.is_file() {
            let url = format!("{base_url}/{repo}/resolve/main/{file}");
            match download(&url, &target, file, on_event) {
                Ok(part) => staged.push((part, target)),
                Err(e) => {
                    for (part, _) in &staged {
                        let _ = fs::remove_file(part);
                    }
                    return Err(e);
                }
            }
        }
    }
    if !staged.is_empty() {
        let discard = |staged: &[(PathBuf, PathBuf)]| {
            for (part, _) in staged {
                let _ = fs::remove_file(part);
            }
        };
        match fs::remove_file(&marker) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                discard(&staged);
                return Err(Error::io(&marker, e));
            }
            _ => {}
        }
        for (part, target) in &staged {
            if let Err(e) = fs::rename(part, target) {
                discard(&staged);
                return Err(Error::io(target, e));
            }
        }
        fs::write(&marker, "").map_err(|e| Error::io(&marker, e))?;
    }
    Ok(())
}

fn download(
    url: &str,
    target: &Path,
    file: &str,
    on_event: &mut dyn FnMut(DownloadEvent<'_>),
) -> Result<PathBuf> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    let mut part = target.as_os_str().to_os_string();
    part.push(".part");
    let part = PathBuf::from(part);
    let failed = |message: String| Error::Download {
        url: url.to_string(),
        message,
    };
    let result = (|| -> Result<()> {
        let response = ureq::get(url)
            .header(
                "User-Agent",
                concat!("notrieve/", env!("CARGO_PKG_VERSION")),
            )
            .call()
            .map_err(|e| failed(e.to_string()))?;
        let total = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok());
        on_event(DownloadEvent::Started { file, total });
        let mut reader = response.into_body().into_reader();
        let mut out = File::create(&part).map_err(|e| Error::io(&part, e))?;
        io::copy(&mut reader, &mut out).map_err(|e| failed(e.to_string()))?;
        out.sync_all().map_err(|e| Error::io(&part, e))
    })();
    match &result {
        Ok(()) => on_event(DownloadEvent::Finished { file }),
        Err(_) => {
            let _ = fs::remove_file(&part);
        }
    }
    result.map(|()| part)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Pooling;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use tempfile::TempDir;

    fn contract(source: ModelSource) -> OnnxContract {
        OnnxContract {
            id: "m".into(),
            source,
            onnx_file: "onnx/model.onnx".into(),
            dim: 4,
            max_tokens: 64,
            pooling: Pooling::Mean,
            query_prefix: "q: ".into(),
            document_prefix: "d: ".into(),
        }
    }

    #[test]
    fn hf_models_are_cached_under_a_flattened_repo_name() {
        let c = contract(ModelSource::Hf {
            repo: "owner/repo".into(),
        });
        assert_eq!(
            model_dir(&c, Path::new("/cache")),
            PathBuf::from("/cache/owner__repo")
        );
    }

    #[test]
    fn local_model_with_a_missing_file_is_reported_without_downloading() {
        let dir = TempDir::new().unwrap();
        for f in [
            "tokenizer.json",
            "config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ] {
            fs::write(dir.path().join(f), "{}").unwrap();
        }
        let c = contract(ModelSource::Path {
            dir: dir.path().to_path_buf(),
        });
        let err = ensure_model(&c, Path::new("/unused"), false, &mut |_| {}).unwrap_err();
        assert!(matches!(err, Error::ModelFileMissing { file, .. } if file == "onnx/model.onnx"));
    }

    const FILES: [&str; 5] = [
        "tokenizer.json",
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "onnx/model.onnx",
    ];

    /// Minimal HTTP/1.1 server standing in for Hugging Face. `routes` maps a file to
    /// (status, body, declared Content-Length); every requested path is recorded.
    fn serve(
        routes: Vec<(&'static str, u16, &'static [u8], usize)>,
    ) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut buf = [0u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]).into_owned();
                let path = request
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or_default()
                    .to_string();
                log.lock().unwrap().push(path.clone());
                let (status, body, declared) = routes
                    .iter()
                    .find(|(file, ..)| path == format!("/owner/repo/resolve/main/{file}"))
                    .map(|(_, s, b, d)| (*s, *b, *d))
                    .unwrap_or((404, b"".as_slice(), 0));
                let head = format!(
                    "HTTP/1.1 {status} X\r\nContent-Length: {declared}\r\nConnection: close\r\n\r\n"
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(body);
            }
        });
        (base, seen)
    }

    fn ok_routes() -> Vec<(&'static str, u16, &'static [u8], usize)> {
        FILES
            .iter()
            .map(|f| (*f, 200, b"{\"k\":1}".as_slice(), 7))
            .collect()
    }

    fn hf() -> OnnxContract {
        contract(ModelSource::Hf {
            repo: "owner/repo".into(),
        })
    }

    fn part_files(dir: &Path) -> Vec<String> {
        let mut found = Vec::new();
        for sub in [dir.to_path_buf(), dir.join("onnx")] {
            for entry in fs::read_dir(sub).into_iter().flatten().flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.ends_with(".part") {
                    found.push(name);
                }
            }
        }
        found
    }

    #[test]
    fn missing_files_are_downloaded_without_leftover_partial_files() {
        let (base, seen) = serve(ok_routes());
        let cache = TempDir::new().unwrap();
        let dir = ensure_model_from(&base, &hf(), cache.path(), false, &mut |_| {}).unwrap();
        assert_eq!(dir, cache.path().join("owner__repo"));
        for f in FILES {
            assert_eq!(fs::read(dir.join(f)).unwrap(), b"{\"k\":1}");
        }
        assert!(part_files(&dir).is_empty());
        assert_eq!(seen.lock().unwrap().len(), 5);
    }

    #[test]
    fn present_files_are_fetched_again_only_when_forced() {
        let (base, seen) = serve(ok_routes());
        let cache = TempDir::new().unwrap();
        ensure_model_from(&base, &hf(), cache.path(), false, &mut |_| {}).unwrap();
        ensure_model_from(&base, &hf(), cache.path(), false, &mut |_| {}).unwrap();
        assert_eq!(seen.lock().unwrap().len(), 5);
        ensure_model_from(&base, &hf(), cache.path(), true, &mut |_| {}).unwrap();
        assert_eq!(seen.lock().unwrap().len(), 10);
    }

    #[test]
    fn http_error_names_the_url_and_leaves_no_partial_file() {
        let mut routes = ok_routes();
        routes.retain(|(f, ..)| *f != "onnx/model.onnx");
        let (base, _) = serve(routes);
        let cache = TempDir::new().unwrap();
        let err = ensure_model_from(&base, &hf(), cache.path(), false, &mut |_| {}).unwrap_err();
        assert!(
            matches!(&err, Error::Download { url, .. } if url.ends_with("onnx/model.onnx")),
            "{err}"
        );
        let dir = cache.path().join("owner__repo");
        assert!(!dir.join("onnx/model.onnx").exists());
        assert!(part_files(&dir).is_empty());
    }

    #[test]
    fn truncated_body_is_an_error_and_is_not_kept() {
        let mut routes = ok_routes();
        routes.retain(|(f, ..)| *f != "onnx/model.onnx");
        routes.push(("onnx/model.onnx", 200, b"abc".as_slice(), 10));
        let (base, _) = serve(routes);
        let cache = TempDir::new().unwrap();
        let err = ensure_model_from(&base, &hf(), cache.path(), false, &mut |_| {}).unwrap_err();
        assert!(matches!(err, Error::Download { .. }));
        let dir = cache.path().join("owner__repo");
        assert!(!dir.join("onnx/model.onnx").exists());
        assert!(part_files(&dir).is_empty());
    }

    fn routes_with(
        body: &'static [u8],
        failing: Option<&str>,
    ) -> Vec<(&'static str, u16, &'static [u8], usize)> {
        FILES
            .iter()
            .map(|f| {
                if Some(*f) == failing {
                    (*f, 500, b"".as_slice(), 0)
                } else {
                    (*f, 200, body, body.len())
                }
            })
            .collect()
    }

    #[test]
    fn failed_forced_refresh_leaves_every_existing_file_unchanged() {
        let cache = TempDir::new().unwrap();
        let (old, _) = serve(routes_with(b"old-content", None));
        let dir = ensure_model_from(&old, &hf(), cache.path(), false, &mut |_| {}).unwrap();
        let (broken, _) = serve(routes_with(b"new-content", Some("onnx/model.onnx")));
        let err = ensure_model_from(&broken, &hf(), cache.path(), true, &mut |_| {}).unwrap_err();
        assert!(matches!(err, Error::Download { .. }));
        for f in FILES {
            assert_eq!(fs::read(dir.join(f)).unwrap(), b"old-content", "{f}");
        }
        assert!(part_files(&dir).is_empty());
    }

    #[test]
    fn successful_forced_refresh_replaces_every_file() {
        let cache = TempDir::new().unwrap();
        let (old, _) = serve(routes_with(b"old-content", None));
        let dir = ensure_model_from(&old, &hf(), cache.path(), false, &mut |_| {}).unwrap();
        let (new, _) = serve(routes_with(b"new-content", None));
        ensure_model_from(&new, &hf(), cache.path(), true, &mut |_| {}).unwrap();
        for f in FILES {
            assert_eq!(fs::read(dir.join(f)).unwrap(), b"new-content", "{f}");
        }
        assert!(part_files(&dir).is_empty());
    }

    #[test]
    fn successful_download_writes_the_marker_and_a_later_call_makes_no_requests() {
        let (base, seen) = serve(ok_routes());
        let cache = TempDir::new().unwrap();
        let dir = ensure_model_from(&base, &hf(), cache.path(), false, &mut |_| {}).unwrap();
        assert!(dir.join(".complete").is_file());
        ensure_model_from(&base, &hf(), cache.path(), false, &mut |_| {}).unwrap();
        assert_eq!(seen.lock().unwrap().len(), 5);
    }

    #[test]
    fn rename_failure_midway_leaves_no_marker_and_the_next_call_downloads_again() {
        let (base, seen) = serve(routes_with(b"new-content", None));
        let cache = TempDir::new().unwrap();
        let dir = cache.path().join("owner__repo");
        // A non-empty directory at a target makes its rename fail after tokenizer.json was renamed.
        fs::create_dir_all(dir.join("config.json")).unwrap();
        fs::write(dir.join("config.json/blocker"), "x").unwrap();
        let err = ensure_model_from(&base, &hf(), cache.path(), false, &mut |_| {}).unwrap_err();
        assert!(matches!(err, Error::Io { .. }), "{err}");
        assert!(!dir.join(".complete").exists());
        assert!(part_files(&dir).is_empty());
        assert_eq!(seen.lock().unwrap().len(), 5);

        fs::remove_dir_all(dir.join("config.json")).unwrap();
        ensure_model_from(&base, &hf(), cache.path(), false, &mut |_| {}).unwrap();
        assert_eq!(seen.lock().unwrap().len(), 10);
        assert!(dir.join(".complete").is_file());
        for f in FILES {
            assert_eq!(fs::read(dir.join(f)).unwrap(), b"new-content", "{f}");
        }
    }

    #[test]
    fn undeletable_marker_aborts_before_any_file_is_replaced() {
        let cache = TempDir::new().unwrap();
        let (old, _) = serve(routes_with(b"old-content", None));
        let dir = ensure_model_from(&old, &hf(), cache.path(), false, &mut |_| {}).unwrap();
        // A non-empty directory at the marker path cannot be removed with remove_file.
        fs::remove_file(dir.join(".complete")).unwrap();
        fs::create_dir_all(dir.join(".complete")).unwrap();
        fs::write(dir.join(".complete/blocker"), "x").unwrap();
        let (new, _) = serve(routes_with(b"new-content", None));
        let err = ensure_model_from(&new, &hf(), cache.path(), true, &mut |_| {}).unwrap_err();
        assert!(matches!(err, Error::Io { .. }), "{err}");
        for f in FILES {
            assert_eq!(fs::read(dir.join(f)).unwrap(), b"old-content", "{f}");
        }
        assert!(part_files(&dir).is_empty());
    }

    #[test]
    fn the_download_lock_is_held_while_the_callback_runs_and_released_after() {
        let (base, _) = serve(ok_routes());
        let cache = TempDir::new().unwrap();
        let lock_path = cache.path().join("owner__repo/.download.lock");
        let held = with_model_from(&base, &hf(), cache.path(), false, &mut |_| {}, |dir| {
            assert_eq!(dir, cache.path().join("owner__repo"));
            Ok(lock::try_lock(&lock_path)?.is_none())
        })
        .unwrap();
        assert!(held, "lock must be held during the callback");
        assert!(lock::try_lock(&lock_path).unwrap().is_some());
    }

    #[test]
    fn a_failing_callback_still_releases_the_lock() {
        let (base, _) = serve(ok_routes());
        let cache = TempDir::new().unwrap();
        let err = with_model_from::<()>(&base, &hf(), cache.path(), false, &mut |_| {}, |_| {
            Err(Error::Config("boom".into()))
        })
        .unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        let lock_path = cache.path().join("owner__repo/.download.lock");
        assert!(lock::try_lock(&lock_path).unwrap().is_some());
    }

    #[test]
    fn complete_local_model_is_used_in_place() {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("onnx")).unwrap();
        for f in [
            "tokenizer.json",
            "config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "onnx/model.onnx",
        ] {
            fs::write(dir.path().join(f), "{}").unwrap();
        }
        let c = contract(ModelSource::Path {
            dir: dir.path().to_path_buf(),
        });
        assert_eq!(
            ensure_model(&c, Path::new("/unused"), false, &mut |_| {}).unwrap(),
            dir.path()
        );
    }
}
