use std::collections::HashSet;
use std::ops::Range;

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

/// Characters allowed in Obsidian tags (https://obsidian.md/help/tags).
pub fn is_tag_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '-' | '/') || is_emoji(c)
}

fn is_emoji(c: char) -> bool {
    matches!(c as u32, 0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0xFE0F | 0x200D)
}

/// Obsidian rejects tags made only of digits (`#1984`).
pub fn is_valid_tag(tag: &str) -> bool {
    !tag.is_empty() && tag.chars().all(is_tag_char) && tag.chars().any(|c| !c.is_ascii_digit())
}

pub fn inline_tags(body: &str) -> Vec<String> {
    let mut tags = Vec::new();
    let mut in_code_block = false;
    for (event, range) in Parser::new_ext(body, Options::empty()).into_offset_iter() {
        match event {
            Event::Start(Tag::CodeBlock(_)) => in_code_block = true,
            Event::End(TagEnd::CodeBlock) => in_code_block = false,
            Event::Text(_) if !in_code_block => collect(body, range, &mut tags),
            _ => {}
        }
    }
    tags
}

fn collect(source: &str, range: Range<usize>, tags: &mut Vec<String>) {
    let segment = &source[range.clone()];
    for (index, _) in segment.match_indices('#') {
        let at = range.start + index;
        // Not documented by Obsidian: require line start or whitespace before `#`
        // so headings, URL fragments and `issue#12` are not tags.
        if source[..at]
            .chars()
            .next_back()
            .is_some_and(|c| !c.is_whitespace())
        {
            continue;
        }
        let tag: String = segment[index + 1..]
            .chars()
            .take_while(|&c| is_tag_char(c))
            .collect();
        // Validity (e.g. digits only) is checked in `merge_tags`, which also warns.
        if !tag.is_empty() {
            tags.push(tag);
        }
    }
}

pub fn merge_tags(
    frontmatter: &[String],
    inline: &[String],
    warnings: &mut Vec<String>,
) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut tags = Vec::new();
    for tag in frontmatter.iter().chain(inline) {
        if !is_valid_tag(tag) {
            warnings.push(format!(
                "tag `{tag}` is not a valid Obsidian tag; it was skipped"
            ));
            continue;
        }
        if seen.insert(tag.to_lowercase()) {
            tags.push(tag.clone());
        }
    }
    tags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_after_whitespace_or_line_start_are_found() {
        assert_eq!(inline_tags("#first and #second\n"), ["first", "second"]);
    }

    #[test]
    fn hash_inside_a_word_is_not_a_tag() {
        assert!(inline_tags("issue#12 and a#b\n").is_empty());
    }

    #[test]
    fn heading_marker_is_not_a_tag_but_tags_in_headings_are() {
        assert_eq!(inline_tags("# Heading\n## Sub #tag\n"), ["tag"]);
    }

    #[test]
    fn code_is_excluded() {
        assert_eq!(
            inline_tags("`#code` text\n\n```\n#fenced\n```\n\n#real\n"),
            ["real"]
        );
    }

    #[test]
    fn nested_unicode_and_emoji_tags_are_kept() {
        assert_eq!(
            inline_tags("#inbox/to-read #税金 #🎉\n"),
            ["inbox/to-read", "税金", "🎉"]
        );
    }

    #[test]
    fn numeric_only_inline_tags_are_candidates_for_validation() {
        assert_eq!(inline_tags("#1984 #y1984\n"), ["1984", "y1984"]);
    }

    #[test]
    fn punctuation_ends_a_tag() {
        assert_eq!(inline_tags("#tag, #b. #税金。\n"), ["tag", "b", "税金"]);
    }

    #[test]
    fn validity_follows_obsidian_rules() {
        assert!(is_valid_tag("y1984"));
        assert!(is_valid_tag("a_b-c/d"));
        assert!(!is_valid_tag("1984"));
        assert!(!is_valid_tag("has space"));
        assert!(!is_valid_tag(""));
    }

    #[test]
    fn merge_drops_invalid_inline_tags_with_a_warning() {
        let mut w = Vec::new();
        assert_eq!(
            merge_tags(&[], &["1984".into(), "ok".into()], &mut w),
            ["ok"]
        );
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("1984"), "{w:?}");
    }

    #[test]
    fn merge_dedupes_case_insensitively_keeping_first_spelling() {
        let mut w = Vec::new();
        let tags = merge_tags(
            &["Tax".into(), "1984".into()],
            &["tax".into(), "other".into()],
            &mut w,
        );
        assert_eq!(tags, ["Tax", "other"]);
        assert_eq!(w.len(), 1);
    }
}
