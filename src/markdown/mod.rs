mod frontmatter;
mod tags;

pub use tags::{inline_tags, is_valid_tag};

use std::path::Path;

use frontmatter::Properties;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoteMeta {
    pub title: String,
    pub aliases: Vec<String>,
    pub tags: Vec<String>,
    pub frontmatter: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedNote<'a> {
    pub meta: NoteMeta,
    pub body: &'a str,
    pub body_start_line: usize,
}

pub fn title_from_path(rel_path: &str) -> String {
    Path::new(rel_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

pub fn parse_note<'a>(rel_path: &str, text: &'a str) -> ParsedNote<'a> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut warnings = Vec::new();
    let (properties, raw, body, body_start_line) = match frontmatter::split(text) {
        Some(split) => {
            let (properties, mut w) = frontmatter::parse_properties(split.yaml);
            warnings.append(&mut w);
            (
                properties,
                Some(split.yaml.to_string()),
                split.body,
                split.body_start_line,
            )
        }
        None => (Properties::default(), None, text, 1),
    };
    let tags = tags::merge_tags(&properties.tags, &tags::inline_tags(body), &mut warnings);
    let aliases = properties
        .aliases
        .into_iter()
        .map(|a| a.trim().to_string())
        .filter(|a| !a.is_empty())
        .collect();
    ParsedNote {
        meta: NoteMeta {
            title: title_from_path(rel_path),
            aliases,
            tags,
            frontmatter: raw,
            warnings,
        },
        body,
        body_start_line,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOTE: &str =
        "---\ntitle: Ignored\naliases: [税, tax]\ntags: [Tax]\n---\n# Heading\nbody #tax #extra\n";

    #[test]
    fn title_is_the_file_stem_not_frontmatter_or_h1() {
        assert_eq!(parse_note("2026/税金メモ.md", NOTE).meta.title, "税金メモ");
        assert_eq!(title_from_path("a/b.c.md"), "b.c");
    }

    #[test]
    fn frontmatter_and_inline_tags_are_merged() {
        assert_eq!(parse_note("n.md", NOTE).meta.tags, ["Tax", "extra"]);
    }

    #[test]
    fn aliases_and_raw_frontmatter_are_kept() {
        let note = parse_note("n.md", NOTE);
        assert_eq!(note.meta.aliases, ["税", "tax"]);
        assert_eq!(
            note.meta.frontmatter.as_deref(),
            Some("title: Ignored\naliases: [税, tax]\ntags: [Tax]\n")
        );
    }

    #[test]
    fn body_starts_after_the_frontmatter() {
        let note = parse_note("n.md", NOTE);
        assert_eq!(
            (note.body, note.body_start_line),
            ("# Heading\nbody #tax #extra\n", 6)
        );
    }

    #[test]
    fn note_without_frontmatter_is_all_body() {
        let note = parse_note("n.md", "# A\ntext\n");
        assert_eq!(
            (note.body, note.body_start_line, note.meta.frontmatter),
            ("# A\ntext\n", 1, None)
        );
    }

    #[test]
    fn byte_order_mark_is_ignored() {
        let note = parse_note("n.md", "\u{feff}---\ntags: [a]\n---\nx\n");
        assert_eq!(note.meta.tags, ["a"]);
    }

    #[test]
    fn numeric_only_inline_tag_is_dropped_with_a_warning() {
        let note = parse_note("n.md", "#1984 #y1984\n");
        assert_eq!(note.meta.tags, ["y1984"]);
        assert_eq!(note.meta.warnings.len(), 1);
    }

    #[test]
    fn invalid_frontmatter_tag_is_dropped_with_a_warning() {
        let note = parse_note("n.md", "---\ntags: [\"1984\", ok]\n---\n");
        assert_eq!(note.meta.tags, ["ok"]);
        assert_eq!(note.meta.warnings.len(), 1);
    }
}
