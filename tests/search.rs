use std::fs;
use std::path::PathBuf;

use notrieve::cache::{EmbeddingCache, cache_path};
use notrieve::embed::Embedder;
use notrieve::embed::fake::FakeEmbedder;
use notrieve::fingerprint::Fingerprints;
use notrieve::indexer::{IndexOptions, index_vault};
use notrieve::search::{Hit, MAX_K, SearchQuery, open_reader, search};
use notrieve::store::Store;
use notrieve::{Error, Result};
use tempfile::TempDir;

const FP: &str = "fp-a";

struct Fixture {
    vault: TempDir,
    data: TempDir,
    embedder: Box<dyn Embedder>,
}

impl Fixture {
    fn new() -> Self {
        Self::with_embedder(Box::new(FakeEmbedder::new(16)))
    }

    fn with_embedder(embedder: Box<dyn Embedder>) -> Self {
        Self {
            vault: TempDir::new().unwrap(),
            data: TempDir::new().unwrap(),
            embedder,
        }
    }

    fn write(&self, rel: &str, text: &str) {
        let path = self.vault.path().join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    fn db(&self) -> PathBuf {
        self.data.path().join("index.db")
    }

    fn index(&self) {
        let mut store = Store::open(&self.db()).unwrap();
        let mut cache = EmbeddingCache::open(&cache_path(&self.db())).unwrap();
        let opts = IndexOptions {
            exclude: &[],
            chunk_max_tokens: 450,
            budget: None,
            deadline: None,
            rebuild: false,
        };
        let fps = Fingerprints {
            model: "model-a".into(),
            index: FP.into(),
        };
        let r = index_vault(
            self.vault.path(),
            &mut store,
            &mut cache,
            &*self.embedder,
            &fps,
            &opts,
            &mut |_| {},
        )
        .unwrap();
        assert!(r.skipped.is_empty(), "{:?}", r.skipped);
    }

    fn search(&self, query: &SearchQuery) -> Result<Vec<Hit>> {
        let reader = open_reader(&self.db()).unwrap();
        search(&reader, &*self.embedder, FP, query)
    }

    fn query(&self, text: &str) -> Vec<Hit> {
        self.search(&SearchQuery::new(text)).unwrap()
    }
}

/// Paths of the hits found by the text side, in `fts_rank` order.
fn fts_paths(hits: &[Hit]) -> Vec<String> {
    let mut ranked: Vec<(usize, String)> = hits
        .iter()
        .filter_map(|h| h.fts_rank.map(|r| (r, h.path.clone())))
        .collect();
    ranked.sort();
    ranked.into_iter().map(|(_, p)| p).collect()
}

fn paths(hits: &[Hit]) -> Vec<String> {
    let mut v: Vec<String> = hits.iter().map(|h| h.path.clone()).collect();
    v.sort();
    v
}

fn with_tags(text: &str, tags: &[&str]) -> SearchQuery {
    SearchQuery {
        tags: tags.iter().map(|t| t.to_string()).collect(),
        ..SearchQuery::new(text)
    }
}

#[test]
fn japanese_word_of_three_characters_is_found_by_full_text_search() {
    let f = Fixture::new();
    f.write(
        "税金.md",
        "# 医療費\n医療費控除の対象になる領収書を保管する。\n",
    );
    f.write("旅行.md", "# 京都\n紅葉の季節に嵐山を歩いた。\n");
    f.index();
    let hits = f.query("医療費控除");
    assert_eq!(fts_paths(&hits), ["税金.md"]);
    let top = &hits[0];
    assert_eq!(
        (
            top.path.as_str(),
            top.title.as_str(),
            top.heading_path.as_str()
        ),
        ("税金.md", "税金", "医療費")
    );
    assert_eq!(
        (top.start_line, top.end_line, top.fts_rank),
        (1, 2, Some(1))
    );
    assert_eq!(
        top.body,
        "# 医療費\n医療費控除の対象になる領収書を保管する。"
    );
}

#[test]
fn two_character_word_matches_the_title_an_alias_or_the_body() {
    let f = Fixture::new();
    f.write(
        "税金.md",
        "---\naliases: [TX]\n---\n確定申告の準備を始める。\n",
    );
    f.write("料理.md", "# 献立\n鍋に昆布を入れて出汁をとる。\n");
    f.write("旅行.md", "# 京都\n紅葉の季節に嵐山を歩いた。\n");
    f.index();
    assert_eq!(fts_paths(&f.query("税金")), ["税金.md"]);
    assert_eq!(fts_paths(&f.query("tx")), ["税金.md"]);
    assert_eq!(fts_paths(&f.query("昆布")), ["料理.md"]);
}

#[test]
fn like_matches_are_appended_after_the_fts_ranking_without_duplicates() {
    let f = Fixture::new();
    f.write("both.md", "医療費控除の領収書をまとめる。\n");
    f.write("fts.md", "医療費控除の対象になる費用を調べる。\n");
    f.write("like.md", "領収書を封筒に入れて保管する。\n");
    f.write("none.md", "紅葉の季節に嵐山を歩いた。\n");
    f.index();
    let hits = f.query("医療費控除 領収");
    let rank = |path: &str| -> Vec<Option<usize>> {
        hits.iter()
            .filter(|h| h.path == path)
            .map(|h| h.fts_rank)
            .collect()
    };
    // both.md and fts.md come from BM25 (ranks 1 and 2 in some order); like.md only from LIKE.
    let mut fts_ranks = [rank("both.md"), rank("fts.md")].concat();
    fts_ranks.sort();
    assert_eq!(fts_ranks, [Some(1), Some(2)]);
    assert_eq!(rank("like.md"), [Some(3)]);
    assert_eq!(rank("none.md"), [None]);
}

/// Gives documents containing `marker` the vector opposite to `query`'s, so they rank last by
/// cosine distance; every other document keeps its fake vector.
struct OppositeEmbedder {
    inner: FakeEmbedder,
    marker: &'static str,
    query: &'static str,
}

impl Embedder for OppositeEmbedder {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut vectors = self.inner.embed_documents(texts)?;
        let opposite: Vec<f32> = self
            .inner
            .embed_query(self.query)?
            .iter()
            .map(|x| -x)
            .collect();
        for (text, vector) in texts.iter().zip(&mut vectors) {
            if text.contains(self.marker) {
                *vector = opposite.clone();
            }
        }
        Ok(vectors)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.inner.embed_query(text)
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        self.inner.count_tokens(text)
    }
}

#[test]
fn tag_filter_restricts_candidates_before_the_top_fifty_are_taken() {
    let f = Fixture::with_embedder(Box::new(OppositeEmbedder {
        inner: FakeEmbedder::new(16),
        marker: "遠い",
        query: "共通語句",
    }));
    for i in 0..120 {
        f.write(&format!("n{i:03}.md"), "共通語句のメモ。\n");
    }
    let filler = "関係のない文章を長く続ける。".repeat(20);
    f.write(
        "tagged.md",
        &format!("---\ntags: [work/proj]\n---\n共通語句。遠い。{filler}\n"),
    );
    f.index();
    // Without the filter the note is last in both rankings (the longest text for BM25, the
    // opposite vector for cosine), far below the 50 candidates of each.
    let unfiltered = SearchQuery {
        k: MAX_K,
        ..SearchQuery::new("共通語句")
    };
    assert!(
        f.search(&unfiltered)
            .unwrap()
            .iter()
            .all(|h| h.path != "tagged.md")
    );
    let hits = f.search(&with_tags("共通語句", &["work"])).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(
        (hits[0].path.as_str(), hits[0].fts_rank, hits[0].vec_rank),
        ("tagged.md", Some(1), Some(1))
    );
    assert_eq!(hits[0].tags, ["work/proj"]);
    // A two-character term goes through LIKE, whose candidates are taken in chunk id order: the
    // tagged note was indexed last (the scan is sorted by path), 121st of 121.
    let short = SearchQuery {
        k: MAX_K,
        ..SearchQuery::new("語句")
    };
    assert!(
        f.search(&short)
            .unwrap()
            .iter()
            .all(|h| h.path != "tagged.md" || h.fts_rank.is_none())
    );
    let hits = f.search(&with_tags("語句", &["work"])).unwrap();
    assert_eq!(
        (hits.len(), hits[0].path.as_str(), hits[0].fts_rank),
        (1, "tagged.md", Some(1))
    );
}

#[test]
fn tag_filter_matches_the_tag_and_its_children_ignoring_case_and_ands_tags() {
    let f = Fixture::new();
    let body = "会議の議事録をまとめる。\n";
    f.write("child.md", &format!("---\ntags: [work/proj]\n---\n{body}"));
    f.write("upper.md", &format!("---\ntags: [Work]\n---\n{body}"));
    f.write(
        "both.md",
        &format!("---\ntags: [work, urgent]\n---\n{body}"),
    );
    f.write(
        "lookalike.md",
        &format!("---\ntags: [workshop]\n---\n{body}"),
    );
    f.write(
        "accent.md",
        &format!("---\ntags: [Équipe/projet]\n---\n{body}"),
    );
    f.write("none.md", body);
    f.index();
    assert_eq!(
        paths(&f.search(&with_tags("議事録", &["work"])).unwrap()),
        ["both.md", "child.md", "upper.md"]
    );
    assert_eq!(
        paths(
            &f.search(&with_tags("議事録", &["work", "#urgent"]))
                .unwrap()
        ),
        ["both.md"]
    );
    // Case folding is not limited to ASCII.
    assert_eq!(
        paths(&f.search(&with_tags("議事録", &["équipe"])).unwrap()),
        ["accent.md"]
    );
}

#[test]
fn two_character_word_found_only_in_a_heading_path_matches_through_like() {
    let f = Fixture::new();
    f.write(
        "a.md",
        "# 家計\n毎月の支出を食費や光熱費などに分類して記録しておくことにした。\n## 保険\n契約している保険の内容と更新の時期を一覧にまとめておくことにした。\n",
    );
    f.index();
    let hits = f.query("家計");
    let mut ranked: Vec<(Option<usize>, &str)> = hits
        .iter()
        .map(|h| (h.fts_rank, h.heading_path.as_str()))
        .collect();
    ranked.sort();
    // The second chunk's text lacks 家計; only its heading path "家計 > 保険" has it.
    assert_eq!(ranked, [(Some(1), "家計"), (Some(2), "家計 > 保険")]);
}

#[test]
fn path_prefix_filter_keeps_only_notes_under_the_prefix() {
    let f = Fixture::new();
    for rel in ["projects/a.md", "projects-old/b.md", "diary/c.md"] {
        f.write(rel, "議事録の書き方を学ぶ。\n");
    }
    f.index();
    let q = SearchQuery {
        path_prefix: Some("projects/".into()),
        ..SearchQuery::new("議事録")
    };
    assert_eq!(paths(&f.search(&q).unwrap()), ["projects/a.md"]);
}

#[test]
fn path_prefix_filter_restricts_every_ranking_before_the_top_fifty_are_taken() {
    let f = Fixture::with_embedder(Box::new(OppositeEmbedder {
        inner: FakeEmbedder::new(16),
        marker: "遠い",
        query: "議事録 メモ",
    }));
    for i in 0..120 {
        f.write(&format!("other/n{i:03}.md"), "議事録のメモ。\n");
    }
    // Sorted after every decoy (so last for LIKE), the longest (last for BM25) and the opposite
    // vector of the query (last for cosine).
    let filler = "関係のない文章を長く続ける。".repeat(20);
    f.write("projects/a.md", &format!("議事録のメモ。遠い。{filler}\n"));
    f.index();
    let unfiltered = SearchQuery {
        k: MAX_K,
        ..SearchQuery::new("議事録 メモ")
    };
    assert!(
        f.search(&unfiltered)
            .unwrap()
            .iter()
            .all(|h| h.path != "projects/a.md")
    );
    let filtered = |text: &str| -> Vec<Hit> {
        let q = SearchQuery {
            path_prefix: Some("projects/".into()),
            ..SearchQuery::new(text)
        };
        f.search(&q).unwrap()
    };
    let hits = filtered("議事録 メモ");
    assert_eq!(
        (
            hits.len(),
            hits[0].path.as_str(),
            hits[0].fts_rank,
            hits[0].vec_rank
        ),
        (1, "projects/a.md", Some(1), Some(1))
    );
    // The two-character term alone goes through LIKE only.
    let like = filtered("メモ");
    assert_eq!((like.len(), like[0].fts_rank), (1, Some(1)));
}

#[test]
fn query_without_text_matches_ranks_every_chunk_by_cosine_similarity() {
    let f = Fixture::new();
    f.write("a.md", "alpha beta\n");
    f.write("b.md", "gamma delta\n");
    f.write("c.md", "epsilon zeta\n");
    f.index();
    let text = "まったく無関係";
    let hits = f.query(text);
    assert_eq!(hits.len(), 3);
    assert!(hits.iter().all(|h| h.fts_rank.is_none()));
    let q = f.embedder.embed_query(text).unwrap();
    let similarity = |input: &str| -> f32 {
        let d = f
            .embedder
            .embed_documents(&[input.to_string()])
            .unwrap()
            .remove(0);
        d.iter().zip(&q).map(|(a, b)| a * b).sum()
    };
    let mut expected = [
        ("a.md", similarity("a\nalpha beta")),
        ("b.md", similarity("b\ngamma delta")),
        ("c.md", similarity("c\nepsilon zeta")),
    ];
    expected.sort_by(|x, y| y.1.total_cmp(&x.1));
    let mut by_vec_rank = hits.clone();
    by_vec_rank.sort_by_key(|h| h.vec_rank);
    assert_eq!(
        by_vec_rank
            .iter()
            .map(|h| h.path.as_str())
            .collect::<Vec<_>>(),
        expected.iter().map(|e| e.0).collect::<Vec<_>>()
    );
    assert_eq!(
        by_vec_rank.iter().map(|h| h.vec_rank).collect::<Vec<_>>(),
        [Some(1), Some(2), Some(3)]
    );
}

#[test]
fn chunk_whose_vector_is_missing_from_the_cache_is_left_out_of_vector_results() {
    let f = Fixture::new();
    f.write("a.md", "alpha beta\n");
    f.write("b.md", "gamma delta\n");
    f.index();
    let store = Store::open(&f.db()).unwrap();
    let key: Vec<u8> = store
        .conn()
        .query_row(
            "SELECT c.embedding_key FROM chunks c JOIN files f ON f.id = c.file_id
             WHERE f.path = 'a.md'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let cache = EmbeddingCache::open(&cache_path(&f.db())).unwrap();
    cache
        .conn()
        .execute("DELETE FROM embeddings WHERE key = ?1", [key])
        .unwrap();
    assert_eq!(paths(&f.query("まったく無関係")), ["b.md"]);
}

#[test]
fn other_index_fingerprint_or_a_missing_cache_is_index_not_ready() {
    let f = Fixture::new();
    assert!(matches!(open_reader(&f.db()), Err(Error::IndexNotReady)));
    f.write("a.md", "alpha beta\n");
    f.index();
    let reader = open_reader(&f.db()).unwrap();
    assert!(matches!(
        search(
            &reader,
            &*f.embedder,
            "fp-other",
            &SearchQuery::new("alpha")
        ),
        Err(Error::IndexNotReady)
    ));
}

#[test]
fn k_limits_the_results_and_invalid_queries_are_rejected() {
    let f = Fixture::new();
    for i in 0..5 {
        f.write(&format!("n{i}.md"), &format!("note number {i}\n"));
    }
    f.index();
    let two = SearchQuery {
        k: 2,
        ..SearchQuery::new("note")
    };
    assert_eq!(f.search(&two).unwrap().len(), 2);
    for k in [0, 51] {
        let q = SearchQuery {
            k,
            ..SearchQuery::new("note")
        };
        assert!(matches!(f.search(&q), Err(Error::InvalidQuery(_))), "{k}");
    }
    assert!(matches!(
        f.search(&SearchQuery::new("   ")),
        Err(Error::InvalidQuery(_))
    ));
}

/// Deletes `a.md` through another connection while the query is being embedded.
struct DeletingEmbedder<'a> {
    inner: &'a dyn Embedder,
    db: PathBuf,
}

impl Embedder for DeletingEmbedder<'_> {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed_documents(texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let mut other = Store::open(&self.db).unwrap();
        other.delete_file(FP, "a.md").unwrap();
        self.inner.embed_query(text)
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        self.inner.count_tokens(text)
    }
}

#[test]
fn a_commit_from_another_connection_during_the_search_is_not_seen() {
    let f = Fixture::new();
    f.write("a.md", "議事録の書き方を学ぶ。\n");
    f.write("b.md", "週報の書き方を学ぶ。\n");
    f.index();
    let reader = open_reader(&f.db()).unwrap();
    let deleting = DeletingEmbedder {
        inner: &*f.embedder,
        db: f.db(),
    };
    let during = search(&reader, &deleting, FP, &SearchQuery::new("議事録")).unwrap();
    assert_eq!(fts_paths(&during), ["a.md"]);
    let after = f.query("議事録");
    assert!(after.iter().all(|h| h.path != "a.md"));
}
