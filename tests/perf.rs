//! Performance targets (spec §4.1, §4.5). Runs only with NOTRIEVE_TEST_MODEL=1 and NOTRIEVE_PERF=1
//! on Apple Silicon. Vectors are faked: a rebuild served by the cache never runs the model, while
//! chunking still needs the real tokenizer.
mod support;

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use notrieve::Result;
use notrieve::cache::{EmbeddingCache, cache_path};
use notrieve::config::{ModelSpec, PRESET_RURI_V3_30M, model_cache_dir};
use notrieve::embed::download::ensure_model;
use notrieve::embed::fake::FakeEmbedder;
use notrieve::embed::onnx::resolve_threads;
use notrieve::embed::{Embedder, OnnxEmbedder};
use notrieve::fingerprint::{Fingerprints, fingerprints};
use notrieve::indexer::{IndexOptions, index_vault};
use notrieve::search::{DEFAULT_K, SearchQuery, open_reader, search};
use notrieve::store::{RebuildReason, Store};
use support::enabled;
use tempfile::TempDir;

const FILES: usize = 10_000;
const FILE_BYTES: usize = 5_000;
const REBUILD_LIMIT: Duration = Duration::from_secs(30);
const SEARCH_BUDGET: Duration = Duration::from_millis(2000);

struct RealTokensFakeVectors {
    real: OnnxEmbedder,
    fake: FakeEmbedder,
}

impl Embedder for RealTokensFakeVectors {
    fn model_id(&self) -> &str {
        self.real.model_id()
    }

    fn dim(&self) -> usize {
        self.real.dim()
    }

    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.fake.embed_documents(texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.real.embed_query(text)
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        self.real.count_tokens(text)
    }

    fn token_ends(&self, text: &str) -> Result<Vec<usize>> {
        self.real.token_ends(text)
    }
}

const PHRASES: [&str; 24] = [
    "今日は会議の議事録をまとめた。",
    "領収書を封筒に入れて保管する。",
    "SQLite の WAL を有効にした。",
    "観葉植物に水をやった。",
    "週末に京都で紅葉を見る予定。",
    "Rust の借用規則を復習する。",
    "カレーの玉ねぎを飴色に炒める。",
    "英単語を間隔反復で覚える。",
    "障害対応の手順書を更新した。",
    "睡眠の質を上げるため室温を下げた。",
    "引っ越しの見積もりを三社に頼んだ。",
    "ぬか床に塩を足してかき混ぜた。",
    "Docker イメージをマルチステージで小さくした。",
    "ふるさと納税の上限額を計算した。",
    "読書メモを要点ごとに整理する。",
    "電車の遅延で打ち合わせに遅れた。",
    "新しいノートアプリを試している。",
    "家計簿の分類を見直した。",
    "健康診断の結果が届いた。",
    "プロジェクトの締め切りが近い。",
    "朝のランニングを三十分続けた。",
    "図書館で本を三冊借りた。",
    "夕飯の献立を一週間分考えた。",
    "検索の精度を評価セットで測る。",
];

/// About FILE_BYTES of deterministic Japanese text in three sections.
fn note(i: usize) -> String {
    let mut state = (i as u64)
        .wrapping_add(1)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as usize
    };
    let mut text = format!("# メモ {i}\n");
    for section in 1..=3 {
        text.push_str(&format!("\n## 節 {section}\n"));
        let goal = FILE_BYTES * section / 3;
        while text.len() < goal {
            text.push_str(PHRASES[next() % PHRASES.len()]);
            if next() % 4 == 0 {
                text.push('\n');
            }
        }
        text.push('\n');
    }
    text
}

fn index(
    root: &Path,
    store: &mut Store,
    db: &Path,
    embedder: &RealTokensFakeVectors,
    fps: &Fingerprints,
) -> notrieve::indexer::IndexReport {
    let mut cache = EmbeddingCache::open(&cache_path(db)).unwrap();
    let opts = IndexOptions::default();
    index_vault(root, store, &mut cache, embedder, fps, &opts, &mut |_| {}).unwrap()
}

#[test]
fn ten_thousand_files_rebuild_from_the_cache_in_30_seconds_and_search_within_the_budget() {
    if !(enabled("NOTRIEVE_TEST_MODEL") && enabled("NOTRIEVE_PERF")) {
        eprintln!("skipped: set NOTRIEVE_TEST_MODEL=1 and NOTRIEVE_PERF=1");
        return;
    }
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        eprintln!("skipped: the targets are defined for Apple Silicon");
        return;
    }
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("vault");
    let mut total = 0;
    for i in 0..FILES {
        let dir = root.join(format!("d{:02}", i % 100));
        fs::create_dir_all(&dir).unwrap();
        let text = note(i);
        total += text.len();
        fs::write(dir.join(format!("n{i:05}.md")), text).unwrap();
    }
    assert!((45_000_000..=55_000_000).contains(&total), "{total} bytes");

    let contract = ModelSpec::Preset(PRESET_RURI_V3_30M.into())
        .resolve(&root)
        .unwrap()
        .into_local()
        .unwrap();
    let dir = ensure_model(&contract, &model_cache_dir().unwrap(), false, &mut |_| {}).unwrap();
    let real = OnnxEmbedder::load(&contract, &dir, resolve_threads(0)).unwrap();
    let fps = fingerprints(&contract, &dir, 450).unwrap();
    let embedder = RealTokensFakeVectors {
        fake: FakeEmbedder::new(real.dim()),
        real,
    };
    let db = tmp.path().join("index.db");
    let mut store = Store::open(&db).unwrap();

    let started = Instant::now();
    let first = index(&root, &mut store, &db, &embedder, &fps);
    eprintln!("first index (vectors faked): {:?}", started.elapsed());
    assert_eq!(first.indexed, FILES);
    let (_, chunks) = store.stats().unwrap();
    assert!(chunks >= 10_000, "{chunks} chunks");
    let embedded = embedder.fake.embedded_count();

    // What a notrieve update looks like: the same model, another index fingerprint.
    let updated = Fingerprints {
        model: fps.model.clone(),
        index: format!("{}-next", fps.index),
    };
    let started = Instant::now();
    let rebuild = index(&root, &mut store, &db, &embedder, &updated);
    let took = started.elapsed();
    eprintln!("rebuild from the cache: {took:?} ({FILES} files, {total} bytes, {chunks} chunks)");
    assert_eq!(
        (rebuild.rebuild_reason, rebuild.indexed),
        (Some(RebuildReason::SettingsChanged), FILES)
    );
    assert_eq!(embedder.fake.embedded_count(), embedded);
    assert!(took < REBUILD_LIMIT, "rebuild took {took:?}");
    drop(store);

    let reader = open_reader(&db).unwrap();
    for text in [
        "会議の議事録",
        "紅葉を見に行く",
        "領収書の保管",
        "SQLite WAL",
        "睡眠",
    ] {
        let started = Instant::now();
        let hits = search(
            &reader,
            &embedder,
            &updated.index,
            &SearchQuery::new(text),
            None,
        )
        .unwrap()
        .hits;
        let took = started.elapsed();
        eprintln!("search {text:?}: {took:?}");
        assert_eq!(hits.len(), DEFAULT_K);
        assert!(took < SEARCH_BUDGET, "search {text:?} took {took:?}");
    }
}
