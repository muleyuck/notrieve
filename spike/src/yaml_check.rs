use std::panic;

use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json::Value;

use crate::CheckResult;

type Extracted = (Option<String>, Vec<String>);

fn scalar_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn extract(v: &Value) -> Extracted {
    let title = v.get("title").and_then(scalar_to_string);
    let tags = match v.get("tags") {
        Some(Value::Array(items)) => items.iter().filter_map(scalar_to_string).collect(),
        Some(other) => scalar_to_string(other).into_iter().collect(),
        None => Vec::new(),
    };
    (title, tags)
}

fn parse(yaml: &str) -> Result<Extracted, String> {
    let yaml = yaml.to_owned();
    panic::catch_unwind(move || serde_saphyr::from_str::<Value>(&yaml))
        .map_err(|_| "panicked".to_string())?
        .map(|v| extract(&v))
        .map_err(|e| format!("error: {e}"))
}

struct StrOrSeq(Vec<String>);

impl<'de> Deserialize<'de> for StrOrSeq {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = StrOrSeq;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a string or a sequence of strings")
            }
            fn visit_str<E: de::Error>(self, s: &str) -> Result<StrOrSeq, E> {
                Ok(StrOrSeq(vec![s.to_owned()]))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<StrOrSeq, A::Error> {
                let mut out = Vec::new();
                while let Some(s) = a.next_element::<String>()? {
                    out.push(s);
                }
                Ok(StrOrSeq(out))
            }
        }
        d.deserialize_any(V)
    }
}

#[derive(Deserialize, Default)]
struct Frontmatter {
    title: Option<String>,
    tags: Option<StrOrSeq>,
}

fn parse_typed(yaml: &str) -> Result<Extracted, String> {
    let yaml = yaml.to_owned();
    panic::catch_unwind(move || serde_saphyr::from_str::<Option<Frontmatter>>(&yaml))
        .map_err(|_| "panicked".to_string())?
        .map(|f| {
            let f = f.unwrap_or_default();
            (f.title, f.tags.map(|t| t.0).unwrap_or_default())
        })
        .map_err(|e| format!("error: {e}"))
}

enum Expect {
    Ok(Extracted),
    Err(Option<&'static str>),
}

fn owned(title: Option<&str>, tags: &[&str]) -> Extracted {
    (title.map(str::to_owned), tags.iter().map(|t| (*t).to_owned()).collect())
}

pub fn run(_args: &[String]) -> CheckResult {
    let cases: [(&str, &str, Expect); 13] = [
        ("flow_list", "title: 税金メモ\ntags: [税金, 確定申告]\n", Expect::Ok(owned(Some("税金メモ"), &["税金", "確定申告"]))),
        ("block_list", "title: 旅行\ntags:\n  - 京都\n  - 紅葉\n", Expect::Ok(owned(Some("旅行"), &["京都", "紅葉"]))),
        ("tags_as_string", "tags: 日記\n", Expect::Ok(owned(None, &["日記"]))),
        ("numeric_title", "title: 1984\n", Expect::Ok(owned(Some("1984"), &[]))),
        ("quoted_colon", "title: \"コロン: 含む\"\n", Expect::Ok(owned(Some("コロン: 含む"), &[]))),
        ("empty", "", Expect::Ok(owned(None, &[]))),
        ("bool_like_tags", "tags: [on, no, y]\n", Expect::Ok(owned(None, &["on", "no", "y"]))),
        ("leading_zero_title", "title: 007\n", Expect::Ok(owned(Some("007"), &[]))),
        ("decimal_title", "title: 1.10\n", Expect::Ok(owned(Some("1.10"), &[]))),
        ("scalar_bool_like_tag", "tags: on\n", Expect::Ok(owned(None, &["on"]))),
        ("scalar_numeric_tag", "tags: 5\n", Expect::Ok(owned(None, &["5"]))),
        ("broken_flow_tags", "title: ok\ntags: [a, b\n", Expect::Err(Some("unclosed"))),
        ("broken_is_error", "title: [unclosed\n", Expect::Err(None)),
    ];
    let mut ok = true;
    for (mode, f) in [("value", parse as fn(&str) -> Result<Extracted, String>), ("typed", parse_typed)] {
        for (name, yaml, expected) in &cases {
            let actual = f(yaml);
            let pass = match (expected, &actual) {
                (Expect::Ok(e), Ok(a)) => e == a,
                (Expect::Err(sub), Err(e)) => {
                    e.starts_with("error:") && sub.is_none_or(|s| e.contains(s))
                }
                _ => false,
            };
            println!("{} [{mode}] {name}: {actual:?}", if pass { "PASS" } else { "FAIL" });
            if mode == "typed" {
                ok &= pass;
            }
        }
    }
    Ok(ok)
}
