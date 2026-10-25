use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

use crate::{Error, Result};

/// Bump when the splitting rules change; it is part of the index fingerprint.
pub const CHUNKER_VERSION: u32 = 1;
pub const MERGE_BELOW_TOKENS: usize = 30;

pub trait TokenCount {
    /// Tokens of the final model input (prefix and special tokens included).
    fn document_tokens(&self, input: &str) -> Result<usize>;

    /// Byte offsets in `text` where its tokens end, ascending and ending at `text.len()`.
    /// The default treats every character as a token.
    fn token_ends(&self, text: &str) -> Result<Vec<usize>> {
        Ok(text
            .char_indices()
            .skip(1)
            .map(|(o, _)| o)
            .chain(std::iter::once(text.len()))
            .collect())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub heading_path: Vec<String>,
    pub start_line: usize,
    pub end_line: usize,
    pub body: String,
}

impl Chunk {
    pub fn heading_path_text(&self) -> String {
        self.heading_path.join(" > ")
    }

    pub fn embedding_input(&self, title: &str) -> String {
        embedding_input(title, &self.heading_path, &self.body)
    }
}

pub fn embedding_input(title: &str, heading_path: &[String], body: &str) -> String {
    if heading_path.is_empty() {
        format!("{title}\n{body}")
    } else {
        format!("{title} > {}\n{body}", heading_path.join(" > "))
    }
}

pub fn chunk_note<C: TokenCount + ?Sized>(
    body: &str,
    body_start_line: usize,
    title: &str,
    max_tokens: usize,
    merge_below: usize,
    counter: &C,
) -> Result<Vec<Chunk>> {
    let lines = Lines::new(body);
    if lines.len() == 0 {
        return Ok(Vec::new());
    }
    let in_code = code_lines(body, &lines);
    let ctx = Ctx {
        title,
        max_tokens,
        counter,
        lines: &lines,
        in_code: &in_code,
    };
    let mut pieces = Vec::new();
    for section in sections(body, &lines) {
        ctx.split_section(&section, &mut pieces)?;
    }
    ctx.merge_small(&mut pieces, merge_below)?;
    pieces
        .into_iter()
        .map(|p| {
            let tokens = ctx.tokens(&p.heading_path, &p.text)?;
            if tokens > max_tokens {
                return Err(Error::ChunkTooLarge {
                    tokens,
                    max: max_tokens,
                });
            }
            Ok(Chunk {
                heading_path: p.heading_path,
                start_line: body_start_line + p.first,
                end_line: body_start_line + p.last,
                body: p.text,
            })
        })
        .collect()
}

/// Byte offsets where each line starts. A line ends at `\n`, at a lone `\r`, or at `\r\n` (one
/// break). `get_note` numbers lines with this too, so hit line numbers address the same lines.
pub(crate) fn line_starts(text: &str) -> Vec<usize> {
    let mut starts = Vec::new();
    if !text.is_empty() {
        starts.push(0);
        let bytes = text.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            let breaks = b == b'\n' || (b == b'\r' && bytes.get(i + 1) != Some(&b'\n'));
            if breaks && i + 1 < text.len() {
                starts.push(i + 1);
            }
        }
    }
    starts
}

struct Lines<'a> {
    text: &'a str,
    starts: Vec<usize>,
}

impl<'a> Lines<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            starts: line_starts(text),
        }
    }

    fn len(&self) -> usize {
        self.starts.len()
    }

    fn line_of(&self, offset: usize) -> usize {
        self.starts.partition_point(|&s| s <= offset) - 1
    }

    fn slice(&self, first: usize, last: usize) -> &'a str {
        let end = self
            .starts
            .get(last + 1)
            .copied()
            .unwrap_or(self.text.len());
        self.text[self.starts[first]..end].trim_end_matches(['\n', '\r'])
    }

    fn line(&self, index: usize) -> &'a str {
        self.slice(index, index)
    }
}

/// Lines that belong to a fenced or indented code block, as parsed by pulldown-cmark.
fn code_lines(body: &str, lines: &Lines) -> Vec<bool> {
    let mut mask = vec![false; lines.len()];
    for (event, range) in Parser::new_ext(body, Options::empty()).into_offset_iter() {
        if let Event::Start(Tag::CodeBlock(_)) = event {
            let first = lines.line_of(range.start);
            let last = lines.line_of(range.end.saturating_sub(1).max(range.start));
            mask[first..=last].iter_mut().for_each(|m| *m = true);
        }
    }
    mask
}

struct Section {
    heading_path: Vec<String>,
    first: usize,
    last: usize,
}

fn sections(body: &str, lines: &Lines) -> Vec<Section> {
    let mut heads: Vec<(usize, Vec<String>)> = Vec::new();
    let mut stack: Vec<(u8, String)> = Vec::new();
    let mut current: Option<(u8, usize, String)> = None;
    for (event, range) in Parser::new_ext(body, Options::empty()).into_offset_iter() {
        match event {
            Event::Start(Tag::Heading { level, .. }) if (level as u8) <= 3 => {
                current = Some((level as u8, lines.line_of(range.start), String::new()));
            }
            Event::Text(text) | Event::Code(text) => {
                if let Some((_, _, name)) = current.as_mut() {
                    name.push_str(&text);
                }
            }
            Event::End(TagEnd::Heading(_)) => {
                if let Some((level, line, name)) = current.take() {
                    while stack.last().is_some_and(|(l, _)| *l >= level) {
                        stack.pop();
                    }
                    stack.push((level, name.trim().to_string()));
                    heads.push((line, stack.iter().map(|(_, n)| n.clone()).collect()));
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    let first_head = heads.first().map_or(lines.len(), |(line, _)| *line);
    if first_head > 0 {
        out.push(Section {
            heading_path: Vec::new(),
            first: 0,
            last: first_head - 1,
        });
    }
    for (i, (line, path)) in heads.iter().enumerate() {
        let end = heads.get(i + 1).map_or(lines.len(), |(next, _)| *next);
        if end <= *line {
            continue;
        }
        let last = end - 1;
        out.push(Section {
            heading_path: path.clone(),
            first: *line,
            last,
        });
    }
    out
}

struct Piece {
    heading_path: Vec<String>,
    first: usize,
    last: usize,
    text: String,
    /// Part of a line split at token boundaries; cannot be merged by line range.
    partial: bool,
}

#[derive(Clone, Copy)]
enum Level {
    Blocks,
    Lines,
}

struct Ctx<'a, C: ?Sized> {
    title: &'a str,
    max_tokens: usize,
    counter: &'a C,
    lines: &'a Lines<'a>,
    in_code: &'a [bool],
}

impl<C: TokenCount + ?Sized> Ctx<'_, C> {
    fn tokens(&self, path: &[String], text: &str) -> Result<usize> {
        self.counter
            .document_tokens(&embedding_input(self.title, path, text))
    }

    fn fits(&self, path: &[String], text: &str) -> Result<bool> {
        Ok(self.tokens(path, text)? <= self.max_tokens)
    }

    fn whole(&self, path: &[String], first: usize, last: usize) -> Piece {
        Piece {
            heading_path: path.to_vec(),
            first,
            last,
            text: self.lines.slice(first, last).to_string(),
            partial: false,
        }
    }

    fn split_section(&self, section: &Section, out: &mut Vec<Piece>) -> Result<()> {
        let path = &section.heading_path;
        let text = self.lines.slice(section.first, section.last);
        if text.trim().is_empty() {
            return Ok(());
        }
        if self.fits(path, text)? {
            out.push(self.whole(path, section.first, section.last));
            return Ok(());
        }
        let blocks = self.blocks(section.first, section.last);
        self.pack(path, &blocks, Level::Blocks, out)
    }

    /// Blank-line separated blocks; blank lines inside a code block do not split it.
    fn blocks(&self, first: usize, last: usize) -> Vec<(usize, usize)> {
        let mut blocks = Vec::new();
        let mut start: Option<usize> = None;
        for i in first..=last {
            if !self.in_code[i] && self.lines.line(i).trim().is_empty() {
                if let Some(s) = start.take() {
                    blocks.push((s, i - 1));
                }
                continue;
            }
            if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start {
            blocks.push((s, last));
        }
        blocks
    }

    fn pack(
        &self,
        path: &[String],
        ranges: &[(usize, usize)],
        level: Level,
        out: &mut Vec<Piece>,
    ) -> Result<()> {
        let mut current: Option<(usize, usize)> = None;
        for &(first, last) in ranges {
            if let Some((start, end)) = current {
                if self.fits(path, self.lines.slice(start, last))? {
                    current = Some((start, last));
                    continue;
                }
                out.push(self.whole(path, start, end));
                current = None;
            }
            if self.fits(path, self.lines.slice(first, last))? {
                current = Some((first, last));
                continue;
            }
            match level {
                Level::Blocks => {
                    let lines: Vec<(usize, usize)> = (first..=last).map(|i| (i, i)).collect();
                    self.pack(path, &lines, Level::Lines, out)?;
                }
                Level::Lines => self.split_line(path, first, out)?,
            }
        }
        if let Some((start, end)) = current {
            out.push(self.whole(path, start, end));
        }
        Ok(())
    }

    fn split_line(&self, path: &[String], index: usize, out: &mut Vec<Piece>) -> Result<()> {
        let mut rest = self.lines.line(index);
        while !rest.is_empty() {
            let ends: Vec<usize> = self
                .counter
                .token_ends(rest)?
                .into_iter()
                .filter(|&e| e > 0 && e <= rest.len() && rest.is_char_boundary(e))
                .collect();
            if ends.is_empty() {
                return Err(Error::ChunkTooLarge {
                    tokens: self.tokens(path, rest)?,
                    max: self.max_tokens,
                });
            }
            let (mut lo, mut hi) = (0, ends.len());
            while lo < hi {
                let mid = (lo + hi) / 2;
                if self.fits(path, &rest[..ends[mid]])? {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if lo == 0 {
                let tokens = self.tokens(path, &rest[..ends[0]])?;
                return Err(Error::ChunkTooLarge {
                    tokens,
                    max: self.max_tokens,
                });
            }
            let end = ends[lo - 1];
            out.push(Piece {
                heading_path: path.to_vec(),
                first: index,
                last: index,
                text: rest[..end].to_string(),
                partial: true,
            });
            rest = &rest[end..];
        }
        Ok(())
    }

    fn merge_small(&self, pieces: &mut Vec<Piece>, merge_below: usize) -> Result<()> {
        let mut i = 0;
        while i < pieces.len() {
            let small = pieces.len() > 1
                && !pieces[i].partial
                && self.tokens(&pieces[i].heading_path, &pieces[i].text)? < merge_below;
            if small {
                let (a, b, keep) = if i + 1 < pieces.len() {
                    (i, i + 1, i + 1)
                } else {
                    (i - 1, i, i - 1)
                };
                if !pieces[a].partial && !pieces[b].partial {
                    let path = pieces[keep].heading_path.clone();
                    let text = self.lines.slice(pieces[a].first, pieces[b].last);
                    if self.fits(&path, text)? {
                        let merged = Piece {
                            heading_path: path,
                            first: pieces[a].first,
                            last: pieces[b].last,
                            text: text.to_string(),
                            partial: false,
                        };
                        pieces.splice(a..=b, [merged]);
                        i = a;
                        continue;
                    }
                }
            }
            i += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token per character of the final input.
    struct CharCounter;

    impl TokenCount for CharCounter {
        fn document_tokens(&self, input: &str) -> Result<usize> {
            Ok(input.chars().count())
        }
    }

    fn chunks(body: &str, max: usize, merge_below: usize) -> Vec<Chunk> {
        chunk_note(body, 1, "T", max, merge_below, &CharCounter).unwrap()
    }

    fn summary(chunks: &[Chunk]) -> Vec<(Vec<&str>, usize, usize)> {
        chunks
            .iter()
            .map(|c| {
                (
                    c.heading_path.iter().map(String::as_str).collect(),
                    c.start_line,
                    c.end_line,
                )
            })
            .collect()
    }

    #[test]
    fn h1_to_h3_start_sections_and_h4_stays_in_its_parent() {
        let body =
            "intro line\n# A\na text\n## B\nb text\n#### D\nd text\n### C\nc text\n# E\ne text\n";
        assert_eq!(
            summary(&chunks(body, 1000, 0)),
            vec![
                (vec![], 1, 1),
                (vec!["A"], 2, 3),
                (vec!["A", "B"], 4, 7),
                (vec!["A", "B", "C"], 8, 9),
                (vec!["E"], 10, 11),
            ]
        );
    }

    #[test]
    fn body_is_the_source_slice_without_trailing_newline() {
        let c = chunks("# A\na text\n\n", 1000, 0);
        assert_eq!(c[0].body, "# A\na text");
    }

    #[test]
    fn line_numbers_are_offset_by_the_body_start() {
        let c = chunk_note("# A\nx\n", 5, "T", 1000, 0, &CharCounter).unwrap();
        assert_eq!((c[0].start_line, c[0].end_line), (5, 6));
    }

    #[test]
    fn lone_carriage_returns_are_line_breaks() {
        assert_eq!(
            summary(&chunks("# A\rtext\r# B\rmore\r", 1000, 0)),
            vec![(vec!["A"], 1, 2), (vec!["B"], 3, 4)]
        );
    }

    #[test]
    fn crlf_line_endings_keep_line_numbers() {
        let c = chunks("# A\r\nx\r\n# B\r\ny\r\n", 1000, 0);
        assert_eq!(summary(&c), vec![(vec!["A"], 1, 2), (vec!["B"], 3, 4)]);
        assert_eq!(c[0].body, "# A\r\nx");
    }

    #[test]
    fn blank_preamble_produces_no_chunk() {
        assert_eq!(
            summary(&chunks("\n\n# A\nx\n", 1000, 0)),
            vec![(vec!["A"], 3, 4)]
        );
    }

    #[test]
    fn empty_body_produces_no_chunks() {
        assert!(chunks("", 1000, 0).is_empty());
    }

    #[test]
    fn heading_inside_a_code_block_does_not_start_a_section() {
        assert_eq!(chunks("# A\n```\n# not\n```\n", 1000, 0).len(), 1);
    }

    #[test]
    fn oversized_section_splits_at_blank_lines_but_never_inside_a_fence() {
        let body = "# A\naaaaaaaaaa\n\n```\nbbbbbbbbbb\n\ncccccccccc\n```\n\ndddddddddd\n";
        let c = chunks(body, 40, 0);
        let bodies: Vec<&str> = c.iter().map(|c| c.body.as_str()).collect();
        assert_eq!(
            bodies,
            [
                "# A\naaaaaaaaaa",
                "```\nbbbbbbbbbb\n\ncccccccccc\n```",
                "dddddddddd"
            ]
        );
        assert_eq!(
            summary(&c),
            vec![(vec!["A"], 1, 2), (vec!["A"], 4, 8), (vec!["A"], 10, 10)]
        );
    }

    #[test]
    fn a_longer_fence_containing_a_shorter_one_is_kept_whole() {
        let body = "# A\nintro\n\n````\n```\n\ninner\n````\n\nafter\n";
        let c = chunks(body, 30, 0);
        assert_eq!(
            summary(&c),
            vec![(vec!["A"], 1, 2), (vec!["A"], 4, 8), (vec!["A"], 10, 10)]
        );
        assert_eq!(c[1].body, "````\n```\n\ninner\n````");
    }

    #[test]
    fn oversized_block_splits_by_lines() {
        let line = "x".repeat(20);
        let body = format!("# A\n{line}\n{line}\n{line}\n");
        assert_eq!(
            summary(&chunks(&body, 30, 0)),
            vec![(vec!["A"], 1, 2), (vec!["A"], 3, 3), (vec!["A"], 4, 4)]
        );
    }

    #[test]
    fn oversized_line_splits_at_token_boundaries() {
        let body = "x".repeat(50);
        let c = chunks(&body, 22, 0);
        let lens: Vec<usize> = c.iter().map(|c| c.body.chars().count()).collect();
        assert_eq!(lens, [20, 20, 10]);
        assert!(c.iter().all(|c| (c.start_line, c.end_line) == (1, 1)));
        assert_eq!(c.iter().map(|c| c.body.as_str()).collect::<String>(), body);
    }

    /// Tokens are three characters long, so cuts must land on multiples of three.
    struct TriCounter;

    impl TokenCount for TriCounter {
        fn document_tokens(&self, input: &str) -> Result<usize> {
            Ok(input.chars().count().div_ceil(3))
        }

        fn token_ends(&self, text: &str) -> Result<Vec<usize>> {
            let n = text.len();
            Ok((1..=n.div_ceil(3)).map(|i| (i * 3).min(n)).collect())
        }
    }

    #[test]
    fn oversized_line_is_cut_on_token_ends_not_on_characters() {
        let c = chunk_note("abcdefghij", 1, "T", 3, 0, &TriCounter).unwrap();
        let bodies: Vec<&str> = c.iter().map(|c| c.body.as_str()).collect();
        assert_eq!(bodies, ["abcdef", "ghij"]);
    }

    #[test]
    fn small_section_merges_into_the_next_and_takes_its_heading() {
        let body = format!("# A\nx\n# B\n{}\n", "y".repeat(40));
        assert_eq!(summary(&chunks(&body, 1000, 20)), vec![(vec!["B"], 1, 4)]);
    }

    #[test]
    fn small_last_section_merges_into_the_previous() {
        let body = format!("# A\n{}\n# B\nx\n", "y".repeat(40));
        assert_eq!(summary(&chunks(&body, 1000, 20)), vec![(vec!["A"], 1, 4)]);
    }

    #[test]
    fn merge_is_skipped_when_the_result_would_exceed_the_limit() {
        let body = format!("# A\nx\n# B\n{}\n", "y".repeat(40));
        assert_eq!(chunks(&body, 50, 20).len(), 2);
    }

    #[test]
    fn every_chunk_input_fits_the_limit() {
        let body = format!(
            "# A\n{}\n\n## B\n{}\n{}\n# C\nshort\n",
            "a".repeat(70),
            "b".repeat(45),
            "c".repeat(90)
        );
        let c = chunks(&body, 60, 10);
        assert!(
            c.iter()
                .all(|c| c.embedding_input("T").chars().count() <= 60)
        );
    }

    #[test]
    fn title_longer_than_the_limit_is_an_error() {
        let err = chunk_note("x", 1, &"t".repeat(30), 10, 0, &CharCounter).unwrap_err();
        assert!(matches!(err, Error::ChunkTooLarge { max: 10, .. }));
    }

    #[test]
    fn embedding_input_joins_title_heading_path_and_body() {
        let c = Chunk {
            heading_path: vec!["A".into(), "B".into()],
            start_line: 1,
            end_line: 1,
            body: "x".into(),
        };
        assert_eq!(c.embedding_input("T"), "T > A > B\nx");
        let top = Chunk {
            heading_path: vec![],
            ..c
        };
        assert_eq!(top.embedding_input("T"), "T\nx");
    }
}
