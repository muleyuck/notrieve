//! `get_note` (spec §4.6): reads a note straight from disk, never outside the vault. Only the file
//! at exactly `rel` is read: no symlink, `..` or other spelling of it, which the indexer never
//! stores, so a path the index holds cannot be turned into a way to read a file it leaves out.
use std::io::Read;
use std::path::Path;

use serde::Serialize;

use crate::markdown::chunk::line_starts;
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NoteText {
    pub path: String,
    /// 1-based, inclusive (the same numbering as search hits).
    pub start_line: usize,
    pub end_line: usize,
    pub total_lines: usize,
    pub text: String,
}

/// `rel` is vault-relative. Lines are 1-based and inclusive; `end_line` is clamped to the note.
/// `root` must be canonical, resolved once by the caller: resolving it per read would follow a
/// vault directory later swapped for a symlink. Any other `root` refuses every note.
pub fn read_note(
    root: &Path,
    rel: &str,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> Result<NoteText> {
    read_note_with(root, rel, start_line, end_line, open_nonblocking)
}

/// O_NONBLOCK keeps the open of a file swapped for a FIFO from blocking until a writer shows up.
#[cfg(unix)]
fn open_nonblocking(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)
}

#[cfg(not(unix))]
fn open_nonblocking(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::File::open(path)
}

/// `open` is the seam between the containment check and the read, where tests swap the file.
fn read_note_with(
    root: &Path,
    rel: &str,
    start_line: Option<usize>,
    end_line: Option<usize>,
    open: impl FnOnce(&Path) -> std::io::Result<std::fs::File>,
) -> Result<NoteText> {
    let path = resolve_inside(root, rel)?;
    let mut file = match open(&path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(Error::NoteNotFound(rel.to_string()));
        }
        Err(e) => return Err(Error::io(&path, e)),
    };
    verify_open(&file, root, rel)?;
    // The path was a regular file at the check; the opened handle may be something else by now.
    if !file.metadata().map_err(|e| Error::io(&path, e))?.is_file() {
        return Err(Error::NoteNotFound(rel.to_string()));
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| Error::io(&path, e))?;
    let text = String::from_utf8(bytes).map_err(|_| Error::NotUtf8(rel.to_string()))?;
    let starts = line_starts(&text);
    let total = starts.len();
    let start = start_line.unwrap_or(1);
    let requested_end = end_line.unwrap_or(usize::MAX);
    if start == 0 || requested_end < start || start > total.max(1) {
        return Err(Error::InvalidLineRange(format!(
            "start_line {start}, end_line {end_line:?}; the note has {total} lines"
        )));
    }
    let end = requested_end.min(total);
    let from = starts.get(start - 1).copied().unwrap_or(text.len());
    let to = starts.get(end).copied().unwrap_or(text.len());
    Ok(NoteText {
        path: rel.to_string(),
        start_line: start,
        end_line: end,
        total_lines: total,
        text: text[from..to].to_string(),
    })
}

/// The canonical path of `rel`, which must be a file inside `root` (already canonical).
fn resolve_inside(root: &Path, rel: &str) -> Result<std::path::PathBuf> {
    if Path::new(rel).is_absolute() {
        return Err(Error::OutsideVault(rel.to_string()));
    }
    let joined = root.join(rel);
    // Canonicalizing resolves `..` and symlinks, so the containment check sees the real target.
    let path = match joined.canonicalize() {
        Ok(path) => path,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(Error::NoteNotFound(rel.to_string()));
        }
        Err(e) => return Err(Error::io(&joined, e)),
    };
    if !path.starts_with(root) {
        return Err(Error::OutsideVault(rel.to_string()));
    }
    // `canonicalize` returns the names as stored on disk, so another case or Unicode form differs.
    if path != joined || !path.is_file() {
        return Err(Error::NoteNotFound(rel.to_string()));
    }
    Ok(path)
}

/// The containment check above runs on the path; the read runs on the opened file. A file or
/// directory swapped for a symlink in between would let the open follow it out of the vault, so the
/// real path of the open descriptor must be inside the vault, and be `rel` itself.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn verify_open(file: &std::fs::File, root: &Path, rel: &str) -> Result<()> {
    let real = fd_path(file).map_err(|e| Error::io(root.join(rel), e))?;
    if !real.starts_with(root) {
        return Err(Error::OutsideVault(rel.to_string()));
    }
    if real != root.join(rel) {
        return Err(Error::NoteNotFound(rel.to_string()));
    }
    Ok(())
}

/// notrieve ships for macOS and Linux only (spec §8); elsewhere the pre-open check stands alone.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn verify_open(_file: &std::fs::File, _root: &Path, _rel: &str) -> Result<()> {
    Ok(())
}

/// The path the kernel holds for the open descriptor (symlinks resolved).
#[cfg(target_os = "macos")]
fn fd_path(file: &std::fs::File) -> std::io::Result<std::path::PathBuf> {
    use std::os::fd::AsRawFd;
    use std::os::unix::ffi::OsStrExt;
    let mut buf = vec![0u8; libc::PATH_MAX as usize];
    // SAFETY: F_GETPATH writes at most PATH_MAX bytes, NUL-terminated, into `buf`.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETPATH, buf.as_mut_ptr()) } == -1 {
        return Err(std::io::Error::last_os_error());
    }
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(std::path::PathBuf::from(std::ffi::OsStr::from_bytes(
        &buf[..len],
    )))
}

#[cfg(target_os = "linux")]
fn fd_path(file: &std::fs::File) -> std::io::Result<std::path::PathBuf> {
    use std::os::fd::AsRawFd;
    std::fs::read_link(format!("/proc/self/fd/{}", file.as_raw_fd()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// `<tmp>/notes` is the vault (canonical, as `read_note` requires); `<tmp>/outside.md` sits
    /// next to it.
    fn vault_dir() -> (TempDir, PathBuf) {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("notes");
        fs::create_dir_all(root.join("sub")).unwrap();
        fs::write(root.join("sub/a.md"), "one\ntwo\nthree\nfour\n").unwrap();
        fs::write(tmp.path().join("outside.md"), "SECRET\n").unwrap();
        (tmp, root.canonicalize().unwrap())
    }

    #[cfg(unix)]
    #[test]
    fn a_vault_root_swapped_for_a_symlink_is_not_followed() {
        let (tmp, root) = vault_dir();
        let moved = tmp.path().join("moved");
        fs::rename(&root, &moved).unwrap();
        fs::create_dir_all(moved.join("private/sub")).unwrap();
        fs::write(moved.join("private/sub/a.md"), "SECRET\n").unwrap();
        std::os::unix::fs::symlink(moved.join("private"), &root).unwrap();
        let err = read_note(&root, "sub/a.md", None, None).unwrap_err();
        assert!(matches!(err, Error::OutsideVault(_)), "{err}");
    }

    #[test]
    fn whole_note_is_returned_by_default_with_its_line_count() {
        let (_tmp, root) = vault_dir();
        let n = read_note(&root, "sub/a.md", None, None).unwrap();
        assert_eq!(
            (n.path.as_str(), n.start_line, n.end_line, n.total_lines),
            ("sub/a.md", 1, 4, 4)
        );
        assert_eq!(n.text, "one\ntwo\nthree\nfour\n");
    }

    #[test]
    fn line_range_is_inclusive_and_the_end_is_clamped_to_the_note() {
        let (_tmp, root) = vault_dir();
        let mid = read_note(&root, "sub/a.md", Some(2), Some(3)).unwrap();
        assert_eq!((mid.start_line, mid.end_line), (2, 3));
        assert_eq!(mid.text, "two\nthree\n");
        let tail = read_note(&root, "sub/a.md", Some(3), Some(99)).unwrap();
        assert_eq!((tail.end_line, tail.text.as_str()), (4, "three\nfour\n"));
    }

    #[test]
    fn last_line_alone_can_be_read() {
        let (_tmp, root) = vault_dir();
        let n = read_note(&root, "sub/a.md", Some(4), None).unwrap();
        assert_eq!(
            (n.start_line, n.end_line, n.text.as_str()),
            (4, 4, "four\n")
        );
    }

    #[test]
    fn start_after_end_zero_or_past_the_note_is_an_invalid_range() {
        let (_tmp, root) = vault_dir();
        for (start, end) in [(Some(3), Some(2)), (Some(0), None), (Some(5), None)] {
            let err = read_note(&root, "sub/a.md", start, end).unwrap_err();
            assert!(
                matches!(err, Error::InvalidLineRange(_)),
                "{start:?}..{end:?}: {err}"
            );
        }
    }

    #[test]
    fn empty_note_reads_as_zero_lines() {
        let (_tmp, root) = vault_dir();
        fs::write(root.join("empty.md"), "").unwrap();
        let n = read_note(&root, "empty.md", None, None).unwrap();
        assert_eq!(
            (n.start_line, n.end_line, n.total_lines, n.text.as_str()),
            (1, 0, 0, "")
        );
    }

    #[test]
    fn parent_directory_escape_is_rejected_without_reading_the_file() {
        let (_tmp, root) = vault_dir();
        let err = read_note(&root, "../outside.md", None, None).unwrap_err();
        assert!(
            matches!(&err, Error::OutsideVault(p) if p == "../outside.md"),
            "{err}"
        );
        assert!(!err.to_string().contains("SECRET"));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_to_a_file_outside_the_vault_is_rejected() {
        let (tmp, root) = vault_dir();
        std::os::unix::fs::symlink(tmp.path().join("outside.md"), root.join("link.md")).unwrap();
        let err = read_note(&root, "link.md", None, None).unwrap_err();
        assert!(matches!(err, Error::OutsideVault(_)), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_is_not_read_even_when_it_points_into_the_vault() {
        let (_tmp, root) = vault_dir();
        std::os::unix::fs::symlink(root.join("sub/a.md"), root.join("alias.md")).unwrap();
        let err = read_note(&root, "alias.md", Some(1), Some(1)).unwrap_err();
        assert!(
            matches!(&err, Error::NoteNotFound(p) if p == "alias.md"),
            "{err}"
        );
    }

    #[test]
    fn absolute_path_is_rejected_even_when_it_points_into_the_vault() {
        let (_tmp, root) = vault_dir();
        let abs = root.join("sub/a.md");
        let err = read_note(&root, abs.to_str().unwrap(), None, None).unwrap_err();
        assert!(matches!(err, Error::OutsideVault(_)), "{err}");
    }

    #[test]
    fn missing_note_and_directory_are_not_found() {
        let (_tmp, root) = vault_dir();
        for rel in ["sub/none.md", "sub"] {
            let err = read_note(&root, rel, None, None).unwrap_err();
            assert!(matches!(err, Error::NoteNotFound(_)), "{rel}: {err}");
        }
    }

    #[test]
    fn non_utf8_note_is_reported_as_such() {
        let (_tmp, root) = vault_dir();
        fs::write(root.join("bin.md"), [0xff, 0xfe, 0x00]).unwrap();
        let err = read_note(&root, "bin.md", None, None).unwrap_err();
        assert!(matches!(&err, Error::NotUtf8(p) if p == "bin.md"), "{err}");
    }

    // The swap between the containment check and the open cannot be timed from a test, so the
    // opener handed to `read_note_with` performs the swap right before it opens.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn a_file_swapped_for_a_symlink_out_of_the_vault_before_the_open_is_rejected() {
        let (tmp, root) = vault_dir();
        let outside = tmp.path().join("outside.md");
        let err = read_note_with(&root, "sub/a.md", None, None, |p| {
            fs::remove_file(p).unwrap();
            std::os::unix::fs::symlink(&outside, p).unwrap();
            std::fs::File::open(p)
        })
        .unwrap_err();
        assert!(
            matches!(&err, Error::OutsideVault(p) if p == "sub/a.md"),
            "{err}"
        );
        assert!(!err.to_string().contains("SECRET"));
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn a_handle_to_a_vault_note_passes_the_post_open_check() {
        let (_tmp, root) = vault_dir();
        let root = root.canonicalize().unwrap();
        let file = std::fs::File::open(root.join("sub/a.md")).unwrap();
        assert!(verify_open(&file, &root, "sub/a.md").is_ok());
        let n = read_note(&root, "sub/a.md", Some(1), Some(1)).unwrap();
        assert_eq!(n.text, "one\n");
    }

    #[cfg(unix)]
    #[test]
    fn a_file_swapped_for_a_fifo_before_the_open_is_not_found_and_does_not_hang() {
        use std::os::unix::ffi::OsStrExt;
        let (_tmp, root) = vault_dir();
        let err = read_note_with(&root, "sub/a.md", None, None, |p| {
            fs::remove_file(p).unwrap();
            let c = std::ffi::CString::new(p.as_os_str().as_bytes()).unwrap();
            assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
            open_nonblocking(p)
        })
        .unwrap_err();
        assert!(
            matches!(&err, Error::NoteNotFound(p) if p == "sub/a.md"),
            "{err}"
        );
    }

    #[test]
    fn carriage_returns_break_lines_like_the_chunker_does() {
        let (_tmp, root) = vault_dir();
        fs::write(root.join("cr.md"), "# A\rtext\r# B\rmore\r").unwrap();
        let n = read_note(&root, "cr.md", Some(3), Some(4)).unwrap();
        assert_eq!(n.text, "# B\rmore\r");
        assert_eq!(n.total_lines, 4);
        fs::write(root.join("crlf.md"), "a\r\nb\r\n").unwrap();
        let n = read_note(&root, "crlf.md", None, None).unwrap();
        assert_eq!((n.total_lines, n.text.as_str()), (2, "a\r\nb\r\n"));
    }

    #[test]
    fn end_line_equal_to_the_line_count_is_accepted() {
        let (_tmp, root) = vault_dir();
        let n = read_note(&root, "sub/a.md", Some(1), Some(4)).unwrap();
        assert_eq!(
            (n.end_line, n.text.as_str()),
            (4, "one\ntwo\nthree\nfour\n")
        );
    }

    #[test]
    fn a_path_through_dot_dot_is_not_read_even_when_it_stays_inside_the_vault() {
        let (_tmp, root) = vault_dir();
        let err = read_note(&root, "sub/../sub/a.md", Some(1), Some(1)).unwrap_err();
        assert!(matches!(err, Error::NoteNotFound(_)), "{err}");
    }

    #[test]
    fn a_decomposed_mixed_case_name_is_read_only_in_its_own_spelling() {
        let (_tmp, root) = vault_dir();
        // ガ as カ + U+3099, the form names copied from HFS+ keep.
        let rel = "sub/Gu\u{30ab}\u{3099}ide.md";
        fs::write(root.join(rel), "guide\n").unwrap();
        let n = read_note(&root, rel, None, None).unwrap();
        assert_eq!(n.text, "guide\n");
        let err = read_note(&root, "sub/GU\u{30ab}\u{3099}IDE.md", None, None).unwrap_err();
        assert!(matches!(err, Error::NoteNotFound(_)), "{err}");
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn a_file_swapped_for_a_symlink_to_another_vault_file_before_the_open_is_rejected() {
        let (_tmp, root) = vault_dir();
        fs::write(root.join(".env"), "SECRET\n").unwrap();
        let env = root.join(".env");
        let err = read_note_with(&root, "sub/a.md", None, None, |p| {
            fs::remove_file(p).unwrap();
            std::os::unix::fs::symlink(&env, p).unwrap();
            std::fs::File::open(p)
        })
        .unwrap_err();
        assert!(
            matches!(&err, Error::NoteNotFound(p) if p == "sub/a.md"),
            "{err}"
        );
        assert!(!err.to_string().contains("SECRET"));
    }

    #[test]
    fn sibling_directory_sharing_the_vault_name_prefix_is_outside() {
        let (tmp, root) = vault_dir();
        fs::create_dir_all(tmp.path().join("notes2")).unwrap();
        fs::write(tmp.path().join("notes2/x.md"), "X\n").unwrap();
        let err = read_note(&root, "../notes2/x.md", None, None).unwrap_err();
        assert!(matches!(err, Error::OutsideVault(_)), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_directory_pointing_outside_the_vault_is_rejected() {
        let (tmp, root) = vault_dir();
        fs::create_dir_all(tmp.path().join("outdir")).unwrap();
        fs::write(tmp.path().join("outdir/y.md"), "Y\n").unwrap();
        std::os::unix::fs::symlink(tmp.path().join("outdir"), root.join("ext")).unwrap();
        let err = read_note(&root, "ext/y.md", None, None).unwrap_err();
        assert!(matches!(err, Error::OutsideVault(_)), "{err}");
    }
}
