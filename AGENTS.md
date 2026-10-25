# AGENTS.md

notrieve is local hybrid search (FTS5 text plus vector search, fused with reciprocal rank fusion) over a vault of Markdown notes, served over MCP to AI agents such as Claude Code and Codex. User-facing behavior is in README.md.

## Commands

```bash
make                          # cargo check, test, clippy, fmt --check (what unit-test CI runs)
cargo test                    # all tests (gated ones below skip themselves)
cargo test --test cli         # CLI E2E tests only
cargo test <name>             # tests whose name contains <name>
cargo clippy --all-targets --all-features
cargo fmt --all
uv run --script scripts/reference.py   # regenerate tests/fixtures/reference-ruri-v3-30m.json
```

- Toolchain and tools are pinned in `mise.toml` (Rust, prek). Run `mise install` if they are missing.
- The prek pre-commit hook (`.pre-commit-config.yml`) runs `cargo fmt --check`, `cargo check`, `cargo clippy` and `cargo test` on every commit.
- `NOTRIEVE_LOG` sets the tracing filter (default `warn`).

## Architecture

| Module | Role |
|---|---|
| `main.rs` | clap CLI: `init`, `index`, `search`, `model pull`, `vault add/list/remove`, `mcp`. |
| `app.rs` | The operations behind the CLI: vault open/init/registry, model pull and load, `index`, `search`, `search_vault`. |
| `config/` | `Vault` and `.notrieve/config.toml`, the vault registry, and the model spec (`PRESET_RURI_V3_30M`, local ONNX contract or remote contract). |
| `embed/` | The `Embedder` trait with `OnnxEmbedder` (fastembed/ort, local), `RemoteEmbedder` (OpenAI-compatible `/embeddings`), the model downloader, and a hash-based `FakeEmbedder` for tests. |
| `markdown/` | Note parsing: frontmatter properties, tags, and heading-based chunking by token count. |
| `indexer/` | `scan` finds changed notes; `index_vault` chunks, embeds and writes them; `updater` is the background updater for long-running processes. |
| `store.rs` | The SQLite index (FTS5 and sqlite-vec) and its rebuild reasons and schema version. |
| `cache.rs` | Embedding cache keyed by model and model input, reused across index rebuilds. |
| `fingerprint.rs` | Fingerprints of the model and the index settings; a change forces a rebuild. |
| `search.rs` | Hybrid search: FTS5 trigram and LIKE for text, cosine distance for vectors, fused with reciprocal rank fusion. |
| `session.rs` | A long-running session over one vault: the model stays loaded and a background thread keeps the index current. |
| `mcp.rs` | MCP server over stdio. It calls only the `session` API; stdout carries the protocol and nothing else. |
| `note.rs` | `get_note`: reads a note from disk, never outside the vault. |
| `lock.rs` | Exclusive write lock on the index. |
| `error.rs` | `Error` and `RemoteErrorKind`. |

## Testing

- Unit tests live in each module (`#[cfg(test)]`). Integration tests are in `tests/*.rs`; `tests/support/` holds shared helpers, including a fake embeddings API.
- Tests that need the real model or a real API skip themselves unless their variable is set:

| Variable | Enables |
|---|---|
| `NOTRIEVE_TEST_MODEL=1` | `tests/real_model.rs`, `tests/reference_vectors.rs` (model comparison), `tests/eval.rs`, `tests/perf.rs`. Downloads ~150 MB on the first run. |
| `NOTRIEVE_PERF=1` | `tests/perf.rs` as well (needs `NOTRIEVE_TEST_MODEL=1`; Apple Silicon). |
| `NOTRIEVE_EVAL_COMPARE=1` | The candidate-model comparison in `tests/eval.rs` (with `NOTRIEVE_TEST_MODEL=1`). |
| `NOTRIEVE_TEST_REMOTE=1` | `tests/real_remote.rs` against a real API, with `NOTRIEVE_TEST_REMOTE_SOURCE`, `NOTRIEVE_TEST_REMOTE_MODEL` and, if the API needs a key, `NOTRIEVE_TEST_REMOTE_KEY_ENV`. |
| `NOTRIEVE_TEST_TEI_SOURCE` | `tests/real_remote.rs` against Text Embeddings Inference serving `cl-nagoya/ruri-v3-30m` (with `NOTRIEVE_TEST_MODEL=1`). |

  Example: `NOTRIEVE_TEST_MODEL=1 cargo test --locked --test reference_vectors --test real_model --test eval -- --nocapture`.
- **Reference fixture**: `tests/fixtures/reference-ruri-v3-30m.json` holds Python (onnxruntime) vectors for the preset, plus the inputs they came from (`files`: model file hashes, prefixes, max tokens). `tests/reference_vectors.rs` compares the Rust embeddings with it. Regenerate it with `uv run --script scripts/reference.py` when the preset changes or the check fails with `fixture is stale: ...`, then run the gated test again. The CI model cache is keyed on this file.
- `tests/eval/` (`queries.toml` and a sample vault) feeds the search-quality test in `tests/eval.rs`.

## Commits and releases

- Conventional Commits (`feat:`, `fix:`, `docs:`, `build:`, `refactor:`, `test:`, `chore:`; `!` for breaking changes). release-please builds the CHANGELOG and release PRs from them (`.github/release-please-config.json`).
- CI (`.github/workflows/`): `unit-test.yml` runs `make test` through muleyuck/github-actions' ci-rust; `eval.yml` runs the gated model tests (`reference_vectors`, `real_model`, `eval`) on macos-14 and ubuntu-24.04; `audit-check.yml` runs cargo audit. unit-test and eval run on every pull request and on pushes to `main`.
- Making `eval` a required status check on `main` is a repository setting; the owner applies it (spec §7).
- Releases: release-please creates the tag and the GitHub release after a successful `unit-test` run on a push to `main` (`.github/workflows/release-please.yml`). It then builds three targets (`aarch64-apple-darwin`, `aarch64-unknown-linux-gnu`, `x86_64-unknown-linux-gnu`) in `build-release.yml`, packed by `scripts/package.sh`, attaches the archives, then renders the Homebrew formula from `packaging/notrieve.rb.in` and pushes it to muleyuck/homebrew-tap with muleyuck/github-actions' `actions/homebrew/publish`. The GitHub App token for the tap comes from the secrets `TAP_APP_CLIENT_ID` and `TAP_APP_PRIVATE_KEY`.
- To try the builds without releasing, run `build-release.yml` with `workflow_dispatch`.
