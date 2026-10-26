use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;

use crate::config::CONFIG_DIR;
use crate::store::{FileState, SkippedState, Store};
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanEntry {
    pub rel: String,
    pub abs: PathBuf,
    pub mtime_ns: i64,
    pub size: i64,
}

#[derive(Debug, Default)]
pub struct Scan {
    pub entries: Vec<ScanEntry>,
    /// False when any directory could not be listed; deletions and renames must then be skipped.
    pub complete: bool,
    pub errors: Vec<String>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ChangePlan {
    /// New files and files whose mtime or size changed.
    pub candidates: Vec<ScanEntry>,
    /// Indexed paths no longer present (empty when the scan was incomplete).
    pub deleted: Vec<String>,
}

impl ChangePlan {
    pub fn stale_files(&self) -> usize {
        self.candidates.len() + self.deleted.len()
    }
}

/// Changes `scan` found against the index, without hashing anything (spec §4.2).
pub fn count_stale(scan: &Scan, store: &Store) -> Result<usize> {
    Ok(plan_changes(scan, &store.file_states()?, &store.skipped_states()?).stale_files())
}

pub fn scan_vault(root: &Path, exclude: &[String]) -> Result<Scan> {
    let mut overrides = OverrideBuilder::new(root);
    for glob in std::iter::once(format!("{CONFIG_DIR}/")).chain(exclude.iter().cloned()) {
        overrides
            .add(&format!("!{glob}"))
            .map_err(|e| Error::Config(format!("invalid exclude pattern `{glob}`: {e}")))?;
    }
    let overrides = overrides
        .build()
        .map_err(|e| Error::Config(e.to_string()))?;
    let mut scan = Scan {
        complete: true,
        ..Scan::default()
    };
    for result in WalkBuilder::new(root)
        .hidden(true)
        .require_git(false)
        .overrides(overrides)
        .build()
    {
        let entry = match result {
            Ok(entry) => entry,
            Err(err) => {
                if is_rule_error(&err) {
                    return Err(rule_error(&err));
                }
                scan.complete = false;
                scan.errors.push(err.to_string());
                continue;
            }
        };
        if let Some(err) = entry.error() {
            return Err(rule_error(err));
        }
        let path = entry.path();
        let is_file = entry.file_type().is_some_and(|t| t.is_file());
        if !is_file || path.extension().is_none_or(|ext| ext != "md") {
            continue;
        }
        let meta = match entry.metadata() {
            Ok(meta) => meta,
            Err(err) => {
                scan.complete = false;
                scan.errors.push(err.to_string());
                continue;
            }
        };
        let mtime_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos() as i64);
        let Some(rel) = path
            .strip_prefix(root)
            .unwrap_or(path)
            .components()
            .map(|c| c.as_os_str().to_str())
            .collect::<Option<Vec<_>>>()
            .map(|parts| parts.join("/"))
        else {
            tracing::warn!(path = %path.display(), "skipped: path is not valid UTF-8");
            continue;
        };
        scan.entries.push(ScanEntry {
            rel,
            abs: path.to_path_buf(),
            mtime_ns,
            size: meta.len() as i64,
        });
    }
    scan.entries.sort_by(|a, b| a.rel.cmp(&b.rel));
    Ok(scan)
}

pub fn plan_changes(
    scan: &Scan,
    states: &HashMap<String, FileState>,
    skipped: &HashMap<String, SkippedState>,
) -> ChangePlan {
    let candidates = scan
        .entries
        .iter()
        .filter(|e| {
            if let Some(s) = skipped.get(&e.rel) {
                return s.mtime_ns != e.mtime_ns || s.size != e.size;
            }
            states
                .get(&e.rel)
                .is_none_or(|s| s.mtime_ns != e.mtime_ns || s.size != e.size)
        })
        .cloned()
        .collect();
    let deleted = if scan.complete {
        let present: HashSet<&str> = scan.entries.iter().map(|e| e.rel.as_str()).collect();
        let mut deleted: Vec<String> = states
            .keys()
            .filter(|p| !present.contains(p.as_str()))
            .cloned()
            .collect();
        deleted.sort();
        deleted
    } else {
        Vec::new()
    };
    ChangePlan {
        candidates,
        deleted,
    }
}

fn rule_error(err: &ignore::Error) -> Error {
    Error::Config(format!(
        "ignore rules could not be fully parsed ({err}); fix them so excluded files are never indexed"
    ))
}

/// True when the error says an ignore rule failed to parse (as opposed to an I/O or listing failure).
fn is_rule_error(err: &ignore::Error) -> bool {
    use ignore::Error as E;
    match err {
        E::Glob { .. } => true,
        E::Partial(errs) => errs.iter().any(is_rule_error),
        E::WithLineNumber { err, .. } | E::WithPath { err, .. } | E::WithDepth { err, .. } => {
            is_rule_error(err)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{Scan, ScanEntry, is_rule_error, plan_changes, scan_vault};
    use crate::Error;
    use crate::store::{FileState, SkippedState};
    use ignore::Error as E;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    fn write(root: &Path, rel: &str) {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "x\n").unwrap();
    }

    fn rels(root: &Path) -> Vec<String> {
        let scan = scan_vault(root, &[]).unwrap();
        scan.entries.into_iter().map(|e| e.rel).collect()
    }

    fn scan_of(size: i64) -> Scan {
        Scan {
            entries: vec![ScanEntry {
                rel: "bad.md".into(),
                abs: "bad.md".into(),
                mtime_ns: 10,
                size,
            }],
            complete: true,
            errors: Vec::new(),
        }
    }

    #[test]
    fn skipped_file_is_a_candidate_only_when_mtime_or_size_changed() {
        let skipped = HashMap::from([(
            "bad.md".to_string(),
            SkippedState {
                mtime_ns: 10,
                size: 3,
                reason: "x".into(),
            },
        )]);
        let no_states = HashMap::new();
        let no_skipped = HashMap::new();
        assert_eq!(
            plan_changes(&scan_of(3), &no_states, &skipped).stale_files(),
            0
        );
        assert_eq!(
            plan_changes(&scan_of(4), &no_states, &skipped).stale_files(),
            1
        );
        assert_eq!(
            plan_changes(&scan_of(3), &no_states, &no_skipped).stale_files(),
            1
        );
    }

    #[test]
    fn skipped_row_decides_alone_even_when_the_files_row_matches() {
        let states = HashMap::from([(
            "bad.md".to_string(),
            FileState {
                mtime_ns: 10,
                size: 3,
                content_hash: "h".into(),
                title: "bad".into(),
            },
        )]);
        let skipped = HashMap::from([(
            "bad.md".to_string(),
            SkippedState {
                mtime_ns: 20,
                size: 4,
                reason: "x".into(),
            },
        )]);
        assert_eq!(
            plan_changes(&scan_of(3), &states, &skipped).stale_files(),
            1
        );
    }

    #[test]
    fn skipped_file_with_only_a_new_mtime_is_a_candidate() {
        let skipped = HashMap::from([(
            "bad.md".to_string(),
            SkippedState {
                mtime_ns: 99,
                size: 3,
                reason: "x".into(),
            },
        )]);
        let no_states = HashMap::new();
        assert_eq!(
            plan_changes(&scan_of(3), &no_states, &skipped).stale_files(),
            1
        );
    }

    #[test]
    fn glob_errors_are_rule_errors_even_when_wrapped() {
        let glob = E::Glob {
            glob: Some("{x".into()),
            err: "unclosed".into(),
        };
        let wrapped = E::Partial(vec![E::WithLineNumber {
            line: 2,
            err: Box::new(glob),
        }]);
        assert!(is_rule_error(&wrapped));
    }

    #[test]
    fn partial_with_one_rule_error_among_io_errors_is_a_rule_error() {
        let glob = E::Glob {
            glob: None,
            err: "unclosed".into(),
        };
        let io = E::Io(std::io::Error::other("denied"));
        assert!(is_rule_error(&E::Partial(vec![io, glob])));
    }

    #[test]
    fn io_errors_are_not_rule_errors() {
        let io = E::Io(std::io::Error::other("denied"));
        assert!(!is_rule_error(&io));
        let wrapped = E::WithPath {
            path: "x".into(),
            err: Box::new(io),
        };
        assert!(!is_rule_error(&wrapped));
    }

    #[test]
    fn only_markdown_files_are_listed() {
        let dir = TempDir::new().unwrap();
        for rel in ["a.md", "sub/b.md", "c.txt"] {
            write(dir.path(), rel);
        }
        assert_eq!(rels(dir.path()), ["a.md", "sub/b.md"]);
    }

    #[test]
    fn notrieve_directory_and_hidden_directories_are_not_listed() {
        let dir = TempDir::new().unwrap();
        for rel in ["a.md", ".notrieve/x.md", ".obsidian/y.md"] {
            write(dir.path(), rel);
        }
        assert_eq!(rels(dir.path()), ["a.md"]);
    }

    #[test]
    fn gitignore_is_respected_without_a_git_repository() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join(".gitignore"), "private/\n").unwrap();
        for rel in ["a.md", "private/p.md"] {
            write(dir.path(), rel);
        }
        assert_eq!(rels(dir.path()), ["a.md"]);
    }

    #[test]
    fn malformed_parent_gitignore_is_an_error() {
        let parent = TempDir::new().unwrap();
        fs::write(parent.path().join(".gitignore"), "{unclosed\n").unwrap();
        let vault = parent.path().join("vault");
        write(&vault, "a.md");
        assert!(matches!(scan_vault(&vault, &[]), Err(Error::Config(_))));
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_file_names_are_skipped_instead_of_colliding() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let dir = TempDir::new().unwrap();
        write(dir.path(), "ok.md");
        for name in [&b"a\x80.md"[..], &b"a\x81.md"[..]] {
            if fs::write(dir.path().join(OsStr::from_bytes(name)), "x\n").is_err() {
                eprintln!("skipped: filesystem refuses non-UTF-8 names");
                return;
            }
        }
        assert_eq!(rels(dir.path()), ["ok.md"]);
    }

    #[test]
    fn stale_files_counts_new_and_deleted_files() {
        let state = |size| FileState {
            mtime_ns: 10,
            size,
            content_hash: "h".into(),
            title: "t".into(),
        };
        let scan = Scan {
            entries: ["a.md", "c.md"]
                .map(|rel| ScanEntry {
                    rel: rel.into(),
                    abs: rel.into(),
                    mtime_ns: 10,
                    size: 3,
                })
                .to_vec(),
            complete: true,
            errors: Vec::new(),
        };
        let states = HashMap::from([
            ("a.md".to_string(), state(3)),
            ("b.md".to_string(), state(3)),
        ]);
        assert_eq!(
            plan_changes(&scan, &states, &HashMap::new()).stale_files(),
            2
        );
    }
}
