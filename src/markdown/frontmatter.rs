use serde::Deserialize;
use serde_json::Value;
use serde_saphyr::{DuplicateKeyPolicy, Spanned};

#[derive(Debug, PartialEq, Eq)]
pub struct Split<'a> {
    pub yaml: &'a str,
    pub body: &'a str,
    pub body_start_line: usize,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Properties {
    pub tags: Vec<String>,
    pub aliases: Vec<String>,
}

pub fn split(text: &str) -> Option<Split<'_>> {
    let mut lines = text.split_inclusive('\n');
    let first = lines.next()?;
    if first.trim_end_matches(['\r', '\n']) != "---" {
        return None;
    }
    let yaml_start = first.len();
    let mut offset = yaml_start;
    for (index, line) in lines.enumerate() {
        if line.trim_end_matches(['\r', '\n']) == "---" {
            return Some(Split {
                yaml: &text[yaml_start..offset],
                body: &text[offset + line.len()..],
                body_start_line: index + 3,
            });
        }
        offset += line.len();
    }
    None
}

pub fn parse_properties(yaml: &str) -> (Properties, Vec<String>) {
    let mut warnings = Vec::new();
    if yaml.trim().is_empty() {
        return (Properties::default(), warnings);
    }
    // Whole-document pass: detects syntax errors, non-mapping documents and duplicate keys.
    let (document, duplicate_keys) = match serde_saphyr::from_str_with_options::<Value>(
        yaml,
        options(DuplicateKeyPolicy::Error),
    ) {
        Ok(document) => (document, DuplicateKeyPolicy::Error),
        Err(_) => match serde_saphyr::from_str_with_options::<Value>(
            yaml,
            options(DuplicateKeyPolicy::LastWins),
        ) {
            Ok(document) => {
                warnings.push(
                    "frontmatter has duplicate property names; the last value is used".into(),
                );
                (document, DuplicateKeyPolicy::LastWins)
            }
            Err(e) => {
                warnings.push(format!(
                    "frontmatter is not valid YAML; it was ignored: {e}"
                ));
                return (Properties::default(), warnings);
            }
        },
    };
    match document {
        Value::Object(_) => {}
        Value::Null => return (Properties::default(), warnings),
        _ => {
            warnings.push("frontmatter is not a mapping; it was ignored".into());
            return (Properties::default(), warnings);
        }
    }
    // Per-field typed passes keep scalar text (`007`, `1.10`, `on`). A field that fits neither form
    // is read item by item with source spans, so its valid items keep their text and only the
    // invalid items (nested lists, maps) are dropped.
    let mut field =
        |name: &str, typed: Option<(Vec<String>, usize)>, spanned: Option<Vec<Spanned<Value>>>| {
            match typed {
                Some((items, skipped)) => {
                    if skipped > 0 {
                        warnings.push(format!("{skipped} empty item(s) in `{name}` were skipped"));
                    }
                    items
                }
                None => {
                    warnings.push(format!(
                        "`{name}` has values of unexpected types; they were skipped"
                    ));
                    spanned
                        .map(|items| spanned_items(yaml, items))
                        .unwrap_or_default()
                }
            }
        };
    let tags = field(
        "tags",
        read_field::<SeqTags, OneTags>(yaml, duplicate_keys, |s| s.tags, |o| o.tags),
        read_spanned::<SpannedTags>(yaml, duplicate_keys, |s| s.tags),
    );
    let aliases = field(
        "aliases",
        read_field::<SeqAliases, OneAliases>(yaml, duplicate_keys, |s| s.aliases, |o| o.aliases),
        read_spanned::<SpannedAliases>(yaml, duplicate_keys, |s| s.aliases),
    );
    (Properties { tags, aliases }, warnings)
}

fn options(duplicate_keys: DuplicateKeyPolicy) -> serde_saphyr::Options {
    serde_saphyr::options! {
        strict_booleans: true,
        duplicate_keys: duplicate_keys,
        reject_non_finite_typeless_float: false,
    }
}

/// Reads one property as a list of strings, or as a single string; `None` when neither form fits.
fn read_field<S, O>(
    yaml: &str,
    duplicate_keys: DuplicateKeyPolicy,
    seq: fn(S) -> Option<Vec<Option<String>>>,
    one: fn(O) -> Option<String>,
) -> Option<(Vec<String>, usize)>
where
    S: for<'de> Deserialize<'de>,
    O: for<'de> Deserialize<'de>,
{
    if let Ok(parsed) = serde_saphyr::from_str_with_options::<S>(yaml, options(duplicate_keys)) {
        let items = seq(parsed).unwrap_or_default();
        let skipped = items.iter().filter(|item| item.is_none()).count();
        return Some((items.into_iter().flatten().collect(), skipped));
    }
    serde_saphyr::from_str_with_options::<O>(yaml, options(duplicate_keys))
        .ok()
        .map(|parsed| (one(parsed).into_iter().collect(), 0))
}

fn read_spanned<S>(
    yaml: &str,
    duplicate_keys: DuplicateKeyPolicy,
    pick: fn(S) -> Option<Vec<Spanned<Value>>>,
) -> Option<Vec<Spanned<Value>>>
where
    S: for<'de> Deserialize<'de>,
{
    serde_saphyr::from_str_with_options::<S>(yaml, options(duplicate_keys))
        .ok()
        .and_then(pick)
}

/// Strings keep their parsed content (quotes removed); numbers and booleans keep their source text.
fn spanned_items(yaml: &str, items: Vec<Spanned<Value>>) -> Vec<String> {
    items
        .into_iter()
        .filter_map(|item| match &item.value {
            Value::String(s) => Some(s.clone()),
            Value::Number(_) | Value::Bool(_) => {
                let span = item.referenced.span();
                let start = usize::try_from(span.byte_offset()?).ok()?;
                let len = usize::try_from(span.byte_len()?).ok()?;
                yaml.get(start..start + len).map(str::to_string)
            }
            _ => None,
        })
        .collect()
}

#[derive(Deserialize)]
struct SeqTags {
    #[serde(default)]
    tags: Option<Vec<Option<String>>>,
}

#[derive(Deserialize)]
struct OneTags {
    #[serde(default)]
    tags: Option<String>,
}

#[derive(Deserialize)]
struct SeqAliases {
    #[serde(default)]
    aliases: Option<Vec<Option<String>>>,
}

#[derive(Deserialize)]
struct OneAliases {
    #[serde(default)]
    aliases: Option<String>,
}

#[derive(Deserialize)]
struct SpannedTags {
    #[serde(default)]
    tags: Option<Vec<Spanned<Value>>>,
}

#[derive(Deserialize)]
struct SpannedAliases {
    #[serde(default)]
    aliases: Option<Vec<Spanned<Value>>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(yaml: &str) -> (Properties, Vec<String>) {
        parse_properties(yaml)
    }

    #[test]
    fn block_on_the_first_line_is_split_from_the_body() {
        let s = split("---\ntags: [a]\n---\n# Body\n").unwrap();
        assert_eq!(
            (s.yaml, s.body, s.body_start_line),
            ("tags: [a]\n", "# Body\n", 4)
        );
    }

    #[test]
    fn block_not_on_the_first_line_is_body() {
        assert_eq!(split("\n---\na: 1\n---\n"), None);
    }

    #[test]
    fn unclosed_block_is_body() {
        assert_eq!(split("---\na: 1\n"), None);
    }

    #[test]
    fn crlf_delimiters_are_recognised() {
        let s = split("---\r\na: 1\r\n---\r\nx").unwrap();
        assert_eq!((s.body, s.body_start_line), ("x", 4));
    }

    #[test]
    fn obsidian_list_properties_are_read() {
        let (p, w) = props("tags:\n  - 税金\n  - 確定申告\naliases:\n  - tax\n");
        assert_eq!(p.tags, ["税金", "確定申告"]);
        assert_eq!(p.aliases, ["tax"]);
        assert!(w.is_empty());
    }

    #[test]
    fn scalars_keep_their_source_text() {
        let (p, w) = props("tags: [on, no, y, 007, 1.10]\n");
        assert_eq!(p.tags, ["on", "no", "y", "007", "1.10"]);
        assert!(w.is_empty());
    }

    #[test]
    fn single_value_becomes_one_item() {
        assert_eq!(props("tags: 日記\n").0.tags, ["日記"]);
        assert_eq!(props("tags: on\n").0.tags, ["on"]);
    }

    #[test]
    fn null_items_are_skipped_with_a_warning() {
        let (p, w) = props("tags: [a, ~, b]\n");
        assert_eq!(p.tags, ["a", "b"]);
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn nested_list_item_is_skipped_and_other_fields_keep_their_text() {
        let (p, w) = props("tags: [a, [b]]\naliases: [007, 1.10]\n");
        assert_eq!(p.tags, ["a"]);
        assert_eq!(p.aliases, ["007", "1.10"]);
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("`tags`"), "{w:?}");
    }

    #[test]
    fn valid_items_next_to_an_invalid_one_keep_their_text() {
        let (p, w) = props("aliases: [1.10, [bad], ok, \"q\", 007]\n");
        assert_eq!(p.aliases, ["1.10", "ok", "q", "007"]);
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn single_scalar_keeps_its_text() {
        assert_eq!(props("aliases: 1.10\n").0.aliases, ["1.10"]);
        assert_eq!(props("tags: 007\n").0.tags, ["007"]);
    }

    #[test]
    fn duplicate_key_takes_the_last_value_with_a_warning() {
        let (p, w) = props("tags: [a]\ntags: [b]\n");
        assert_eq!(p.tags, ["b"]);
        assert!(w[0].contains("duplicate"), "{w:?}");
    }

    #[test]
    fn non_mapping_frontmatter_is_ignored_with_a_warning() {
        let (p, w) = props("- a\n- b\n");
        assert_eq!(p, Properties::default());
        assert!(w[0].contains("not a mapping"), "{w:?}");
    }

    #[test]
    fn syntax_error_is_ignored_with_a_warning() {
        let (p, w) = props("tags: [unclosed\n");
        assert_eq!(p, Properties::default());
        assert!(w[0].contains("not valid YAML"), "{w:?}");
    }

    #[test]
    fn other_properties_are_not_interpreted() {
        let (p, w) = props("title: 007\ncssclasses: [wide]\ndate: 2026-10-12\ntags: [a]\n");
        assert_eq!(p.tags, ["a"]);
        assert!(w.is_empty());
    }

    #[test]
    fn empty_yaml_has_no_properties() {
        assert_eq!(props(""), (Properties::default(), vec![]));
    }
}
