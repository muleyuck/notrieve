# notrieve タスク 0 検証スパイク

本体とは独立した検証用クレート。結果は `docs/plans/2026-10-12-notrieve-design.md` に反映済み。

## 再実行

    uv run --script reference/reference.py model_int8
    uv run --script reference/reference.py model
    cargo build --release
    target/release/notrieve-spike embed-check models/ruri-v3-130m onnx/model_int8.onnx out/reference-model_int8.json
    target/release/notrieve-spike bench models/ruri-v3-130m onnx/model_int8.onnx 16 4 128
    target/release/notrieve-spike sqlite-check
    target/release/notrieve-spike yaml-check
    docker/linux-check.sh arm64
    docker/linux-check.sh amd64

## 結果（2026-10-12 実測）

FAIL は失敗ではなく結果として記録している。

### 1. ruri-v3-130m の Rust 推論（仕様 §9.1）

Python（onnxruntime 1.28.0）。トークン数は [11, 46, 13, 38, 7, 198]。

| | int8 | fp32 |
|---|---|---|
| `onnx_single_vs_batch_min_cos` | 0.9334844350814819 | 1.0 |
| `sentence_embedding_vs_mean_max_abs` | 0.0 | 0.0 |
| `st_vs_onnx_min_cos` | 0.9547500014305115 | 0.9999998807907104 |

- int8 モデルには `DynamicQuantizeLinear` と `MatMulInteger` が 76 個ずつ含まれる（fp32 には無い）。量子化のスケールがバッチ全体で決まるため、Python の onnxruntime でも int8 はバッチ依存になる。
- `sentence_embedding` 出力は masked mean pooling と一致する（差の最大値 0.0）。
- `st_vs_onnx_min_cos` の int8 の値は、量子化誤差と padding の影響が混ざっている。

Rust（fastembed 7.1.1、`intra_threads` 1、`max_length` 512）の `embed-check`:

int8:

    PASS dim: rust=512 python=512
    PASS rust_vs_python_min_cos: 0.9999998
    FAIL rust_single_vs_batch_min_cos: 0.9334848
    PASS l2_norm_max_deviation: 2.3841858e-7
    exit=1

fp32:

    PASS dim: rust=512 python=512
    PASS rust_vs_python_min_cos: 0.9999999
    PASS rust_single_vs_batch_min_cos: 0.9999998
    PASS l2_norm_max_deviation: 2.3841858e-7
    exit=0

- 未実施: 評価セットでの検索品質への影響の測定（本スパイクの範囲外）。

### 2. HF リポジトリとファイル（§9.2）

- 採用: `sirasagi62/ruri-v3-130m-ONNX`。`onnx/model_int8.onnx`（133MB）と `onnx/model.onnx`（529MB）、tokenizer 系 4 ファイルがそろっている。
- `input_names`: `input_ids`, `attention_mask`
- `output_names`: `token_embeddings`, `sentence_embedding`（int8 / fp32 とも）
- fastembed は `sentence_embedding` を選ぶ（出力選択の優先順位による。レビューでソースを確認）。
- 未実施: 130m 以外のプリセット（30m / 70m / 310m / multilingual-e5-small）のファイル確認。
- 未確認: HF からのダウンロードにリビジョン固定はしていない。

### 3. 静的リンク（§9.3）

| ターゲット | ビルド | libonnxruntime 依存 | 実行 | 備考 |
|---|---|---|---|---|
| aarch64-apple-darwin | 成功 | `otool -L` に現れない（libiconv, libc++, Foundation, CoreML, libSystem, CoreFoundation, libobjc） | ホストで Task 2 と同じ結果 | `minos 11.0`（Mach-O の宣言値。ort の文書は macOS 13.4 以上を要件とする。実行での確認はしていない）。ホストに onnxruntime の共有ライブラリは無い |
| x86_64-apple-darwin | **失敗** | — | — | `error: ort-sys@2.0.0-rc.13: no prebuilt binaries available for target x86_64-apple-darwin`（exit 101） |
| aarch64-unknown-linux-gnu | ubuntu:24.04 で成功（1m55s） | `ldd` に現れない | ubuntu:24.04: PASS dim / rust_vs_python 0.9999998 / l2、FAIL single_vs_batch 0.9334848（ホストと同じ値）。`Unknown CPU vendor` の警告あり | 最大 `GLIBC_2.38` / `GLIBCXX_3.4.31` / `CXXABI_1.3.15`。ubuntu:22.04 はこれらが無く起動しない |
| x86_64-unknown-linux-gnu | arm64 コンテナからのクロスコンパイルで成功（1m37s） | `NEEDED` は libc / ld-linux-x86-64 / libm / libstdc++ / libgcc_s のみ | qemu 上の ubuntu:24.04: PASS dim / l2、FAIL rust_vs_python 0.9657305、FAIL single_vs_batch 0.9410530。ubuntu:22.04 は `GLIBC_2.38` / `GLIBCXX_3.4.31` が無く起動しない | 最大 `GLIBC_2.38` / `GLIBCXX_3.4.31` / `CXXABI_1.3.11`。qemu 上の amd64 コンテナでは rustc が SIGSEGV（rc=139、2 回再現）したため、クロスコンパイルにした |

- 両 Linux のビルドログに `cargo:rustc-link-lib=static=onnxruntime` がある（Codex レビューで確認）。
- 未確認: x86_64 Linux の int8 が参照と一致しない原因。qemu の影響か、非 VNNI の x86 での U8S8 飽和（ORT の文書に記載。推測であり未検証）かは切り分けていない。切り分け用の fp32 実行は `exit=137`（SIGKILL）で終了し、原因は不明。ネイティブの x86 Linux での実行は未実施（この環境に無い。Rosetta は入れない方針）。
- 未実施: ネイティブ x86_64 Linux での実行、x86_64-apple-darwin（ビルド不可）。

### 4. sqlite-vec のフィルタ付き KNN（§9.4）

`sqlite-check`（SQLite 3.53.2、sqlite-vec v0.1.9、rusqlite 経由）。10 項目すべて PASS、exit 0。

    sqlite=3.53.2 sqlite-vec=v0.1.9
    PASS wal
    PASS knn [1,2,3]
    PASS knn_rowid_in_json_each [2,5,7]
    PASS knn_rowid_in_subquery [1,3,5]
    PASS vec_distance_cosine 1
    PASS vec0_drop_create_in_transaction
    PASS vec0_rejects_old_dim ("Dimension mismatch ... Expected 8 dimensions but received 4.")
    PASS fts_trigram_ja_3plus_chars [2]
    PASS fts_trigram_2_chars_matches_nothing []
    PASS like_2_chars_hits_title [1]

- `rowid IN` の検査は、事前絞り込みと「多めに取ってから絞る」実装を区別できない。sqlite-vec 0.1.9 の `sqlite-vec.c` を読んで、距離計算の前に配列にない行を飛ばすこと（事前絞り込み）を確認した（レビュー）。
- `vec0_drop_create_in_transaction` は原子性を示していない。レビュー担当の in-memory での追加検証では、`BEGIN; DROP; CREATE float[8]; ROLLBACK` で元のテーブルに戻った。WAL のファイル DB では未実施。
- FTS の検査は trigram と unicode61 を区別できない（`全文検索` は単語全体のため）。追加検証では `医療費` が trigram でだけヒットした（レビュー）。
- 未実施: 1024 行を超えるチャンク境界。

### 5. スループット（§9.5）

`bench`（n=128、約 200 トークンのチャンク、Apple Silicon 8 コア / 8 GB、int8）。

    batch=8 threads=2 load_ms=328 chunks_per_sec=11.78 batch_ms=679 query_ms=7.7 rss=925581312
    batch=16 threads=2 load_ms=355 chunks_per_sec=11.56 batch_ms=1384 query_ms=7.6 rss=1146814464
    batch=32 threads=2 load_ms=342 chunks_per_sec=10.34 batch_ms=3093 query_ms=7.9 rss=1394851840
    batch=8 threads=4 load_ms=362 chunks_per_sec=17.34 batch_ms=461 query_ms=6.3 rss=925941760
    batch=16 threads=4 load_ms=322 chunks_per_sec=16.21 batch_ms=987 query_ms=6.8 rss=1348517888
    batch=32 threads=4 load_ms=330 chunks_per_sec=15.00 batch_ms=2133 query_ms=7.7 rss=1558429696
    fp32 batch=16 threads=4 load_ms=591 chunks_per_sec=5.68 batch_ms=2819 query_ms=17.7 rss=1441824768

計画の規則で決めた既定値:
- バッチサイズ: 8
- `index_threads`: 「0 = CPU コア数の半分」を維持（threads 4 / threads 2 = 17.34 / 11.78 = 1.47 倍）
- `search_budget_ms`: 2000（batch 8 / threads 4 の 1 バッチ 461 ms）
- CLI `search` の固定コスト: int8 で 368.3 ms（load 362 + query 6.3）、fp32 で 609 ms。ページキャッシュが温まった状態の値。

- 注意: チャンクは約 200 トークンのため、450 トークンのチャンクではスループットは下がる（上限値）。測定点ごとに 1 回の実行。
- 未実施: ページキャッシュが冷えた状態での load 時間、x86 環境での測定。

### 6. YAML パーサー（§9.6）

`yaml-check`（serde-saphyr 1.3.0、パーサーは granit-parser 1.3.0）。`[value]`（`serde_json::Value` 経由）と `[typed]`（型付き構造体）の 2 方式、13 ケース。終了コードは `[typed]` の結果で決まる（exit 1）。

- 計画の 7 ケース（flow_list / block_list / tags_as_string / numeric_title("1984") / quoted_colon / empty / broken_is_error）は PASS した。これだけだと `Value` 経由の値の書き換えを検出できないため、ケースを追加した。
- `[value]` で値が書き換わったケース:

| ケース | 入力 | 取り出した値 |
|---|---|---|
| `bool_like_tags` | `tags: [on, no, y]` | `["true","false","true"]` |
| `leading_zero_title` | `title: 007` | `"7.0"` |
| `decimal_title` | `title: 1.10` | `"1.1"` |
| `scalar_bool_like_tag` | `tags: on` | `["true"]` |

- `[typed]`: 13 ケース中 11 ケース PASS。上の書き換えは起きず、入力どおりの文字列が残る。FAIL の 2 ケースは frontmatter 全体がエラーになる。
  - `scalar_bool_like_tag`: `tags: on` → `invalid type: boolean`
  - `scalar_numeric_tag`: `tags: 5` → `invalid type: integer`
- 構文エラー（`broken_flow_tags`）は `unclosed bracket` として検出した。
- 追加の検証（レビュー）:
  - `[typed]` は `tags: [a, ~]`、空のブロック要素、`tags: [a,[b]]`、`title: [a,b]`、mapping でない文書で frontmatter 全体を拒否する。
  - 重複キーは両方式ともエラー（`DuplicateKeyPolicy::Error` が既定）。
  - 既定の Budget で billion laughs を 57 ms で停止する（深さ 64、ノード 250k）。
  - `forbid(unsafe_code)` が serde-saphyr と granit-parser にある。依存は 13 クレート増える。
- 未実施: `cargo audit`。
