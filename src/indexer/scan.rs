use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;

use crate::config::CONFIG_DIR;
use crate::store::{FileState, SkippedState};
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
    use super::{Scan, ScanEntry, is_rule_error, plan_changes};
    use crate::store::SkippedState;
    use ignore::Error as E;
    use std::collections::HashMap;

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
        use crate::store::FileState;
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
}
