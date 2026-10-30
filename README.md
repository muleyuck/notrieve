[![unit-test](https://github.com/muleyuck/notrieve/actions/workflows/unit-test.yml/badge.svg)](https://github.com/muleyuck/notrieve/actions/workflows/unit-test.yml)
[![eval](https://github.com/muleyuck/notrieve/actions/workflows/eval.yml/badge.svg)](https://github.com/muleyuck/notrieve/actions/workflows/eval.yml)
![Software License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)
[![Release](https://img.shields.io/github/release/muleyuck/notrieve.svg)](https://github.com/muleyuck/notrieve/releases/latest)

# <img src="assets/icon.svg" width="40" height="40" alt="" align="top"> notrieve

Local hybrid search over your Markdown notes, for you and for AI agents. Full-text and vector search
run on your machine against a SQLite index, and `notrieve mcp` serves them to Claude Code, Codex
and other MCP clients.

![demo](tapes/demo.gif)

The queries share almost no words with the notes they find.

## Install

Homebrew:

```sh
brew install muleyuck/tap/notrieve
```

From source (needs a Rust toolchain):

```sh
cargo install --git https://github.com/muleyuck/notrieve
```

Or download `notrieve-<target>.tar.xz` from [GitHub Releases](https://github.com/muleyuck/notrieve/releases),
unpack it and put `notrieve` on your `PATH`. Targets: `aarch64-apple-darwin`,
`aarch64-unknown-linux-gnu`, `x86_64-unknown-linux-gnu`.

## Requirements

- macOS 13.4 or later on Apple Silicon. Intel Macs are not supported.
- Linux x86_64 or aarch64 with glibc 2.38 or later and `GLIBCXX_3.4.31` or later (Ubuntu 24.04 or
  later). Ubuntu 22.04 does not work.

## Quick start

```sh
cd ~/notes
notrieve init          # writes .notrieve/, registers the vault, downloads the model, builds the index
notrieve search "バックアップについて書いたこと"
```

`notrieve search` first indexes notes changed since the last run, for about `search_budget_ms` (2
seconds by default), then prints the matching chunks with their path, line range and heading path.
Notes it did not get to, or all of them while another process is indexing, are searched as last
indexed. `-k` sets the number of results (default 8), `--tag` keeps notes with that tag or one of
its child tags (repeat it to require several), `--path` keeps notes whose vault-relative path starts
with the given text, and `--json` prints JSON. Run it anywhere inside the vault, or pass `--vault
<name>`. `notrieve index` updates the index without searching.

The built-in model, ruri-v3-30m, is a Japanese embedding model. Full-text matching works for any
language, but for notes mostly in another language, set up [another local
model](#using-another-local-model) or [an API](#embedding-through-an-api-instead-of-a-local-model)
for the semantic half.

`notrieve vault list` shows the registered vaults, `notrieve vault add <dir>` registers a
directory that already has `.notrieve/` (a cloned or synced vault), and `notrieve vault remove
<name>` unregisters one and keeps its files.

## What gets indexed

Every `*.md` file under the vault, except hidden files and directories, files ignored by
`.gitignore` or `.ignore` (whether or not the vault is a git repository), and files matching
`exclude` in `.notrieve/config.toml`. Notes are split into chunks at headings, and long sections by size. Tags come from the
frontmatter `tags` property and from inline `#tags`, following Obsidian's tag rules.

## Use from AI agents (MCP)

`notrieve mcp` is a stdio MCP server, so any client that can launch one can use it. Claude Code and
Codex are shown below.

Run `notrieve init` in your notes directory first. It downloads the model and builds the index, so
the MCP server starts quickly.

Claude Code (`--scope user` makes the notes searchable from every project; without it the
server is added to the current project only):

```sh
claude mcp add --scope user notrieve -- notrieve mcp --vault <name>
```

Codex (`~/.codex/config.toml`, or `codex mcp add notrieve -- notrieve mcp --vault <name>`):

```toml
[mcp_servers.notrieve]
command = "notrieve"
args = ["mcp", "--vault", "<name>"]
```

The server serves one vault. Register one server per vault (`notrieve vault list` shows the names).
Without `--vault` it uses the vault containing the working directory, then the default vault.

The server offers two tools:

- `search`: hybrid search with `query`, and optionally `k`, `tags` and `path_prefix`. Results
  carry the path, line range, heading path and text of each chunk.
- `get_note`: reads the note at `path` (as `search` returns it), whole or from `start_line` to
  `end_line`, to see the context around a hit. It never reads outside the vault.

When a search finds notes changed since they were indexed, it answers from the index as it is,
counts them in `stale_files`, and asks a background thread to index them for later searches.

## Configuration

`.notrieve/config.toml` in the vault root. Every key has a default, except that an API model
needs `chunk_max_tokens`:

| Key | Default | Meaning |
|---|---|---|
| `model` | `"ruri-v3-30m"` | The embedding model: the built-in preset, or a `[model]` table (below). |
| `exclude` | `[]` | Gitignore-style patterns, relative to the vault root, for notes not to index. |
| `db_path` | `".notrieve/index.db"` | The index file, relative to the vault root unless absolute. |
| `search_budget_ms` | `2000` | About how long `notrieve search` spends indexing changed notes before it searches. With the local model, a note already started is finished, so it can run over. |
| `chunk_max_tokens` | `450` | Chunk size in tokens. Required for an API model, where it counts characters. |
| `index_threads` | `0` | Threads for the local model; `0` uses half the CPU cores. |

## Using another local model

`model = "ruri-v3-30m"` in `.notrieve/config.toml` names the only built-in preset. To run another
ONNX embedding model on this machine, replace that line with a `[model]` table. Every field is
required. This is ruri-v3-130m (529 MB, 512 dimensions):

```toml
[model]
source = "hf:sirasagi62/ruri-v3-130m-ONNX"
id = "ruri-v3-130m"
onnx_file = "onnx/model.onnx"
dim = 512
max_tokens = 512
pooling = "mean"
quantization = "none"
query_prefix = "検索クエリ: "
document_prefix = "検索文書: "
```

- `source` is `hf:<owner>/<repo>` (downloaded from Hugging Face) or `path:<dir>` (a directory,
  relative to the vault root unless absolute). Either must hold `onnx_file` and `tokenizer.json`,
  `config.json`, `special_tokens_map.json` and `tokenizer_config.json`.
- `pooling` is `mean` or `cls`. `quantization` must be `none`; use an ONNX file that is already
  quantized instead.
- `notrieve model pull` downloads the files ahead of the first index. Changing the model rebuilds
  the index.

## Embedding through an API instead of a local model

By default notrieve runs `ruri-v3-30m` locally (a 147 MB download). To keep no model on this
machine, point `[model]` at an OpenAI-compatible `/embeddings` endpoint. **The text of your notes
is sent to that endpoint.** Start with `notrieve init --no-index`, edit
`.notrieve/config.toml`, then run `notrieve index`.

`chunk_max_tokens` is required here and counts characters. With a model limited to about 512
tokens, use a smaller value than the examples.

OpenAI:

```toml
chunk_max_tokens = 1500

[model]
source = "openai-compat:https://api.openai.com/v1"
remote_model = "text-embedding-3-small"
api_key_env = "OPENAI_API_KEY"
```

Azure OpenAI (the key goes in an `api-key` header):

```toml
chunk_max_tokens = 1500

[model]
source = "openai-compat:https://<resource>.openai.azure.com/openai/v1"
remote_model = "<deployment>"
api_key_env = "AZURE_OPENAI_API_KEY"
auth_header = "api-key"
```

Text Embeddings Inference on this machine, serving ruri-v3-30m with its prefixes:

```toml
chunk_max_tokens = 450

[model]
source = "openai-compat:http://127.0.0.1:8080/v1"
remote_model = "cl-nagoya/ruri-v3-30m"
query_prefix = "検索クエリ: "
document_prefix = "検索文書: "
```

Ollama on this machine:

```toml
chunk_max_tokens = 1500

[model]
source = "openai-compat:http://localhost:11434/v1"
remote_model = "nomic-embed-text"
```

The key itself never goes in the file; `api_key_env` names the environment variable that holds
it. `http://` is accepted only for `localhost`, `127.0.0.1` and `::1`. The base URL cannot carry
a query string or a fragment. The API key never appears in error messages or logs, and
configuration errors do not repeat the URL.

`.notrieve/config.toml` travels with your notes, so a vault you cloned or synced may carry
someone else's settings. A remote model, or an index written outside the vault, is used only as
you approved it. `notrieve vault add <dir>` registers a vault and prints where its config sends
notes and which environment variable it sends; that is what you approve. Commands stop with an
error before contacting anything when an unregistered vault has such settings, or when a
registered vault's config changed them since (for example after a pull). Review the change, then
run `notrieve vault add <dir> --name <name>` again to approve it.

When the API is slow or unreachable, `notrieve mcp` keeps serving: searches fall back to
full-text matching, and the background indexer waits and retries. While it waits, search
responses carry `indexer_error: "the background indexer paused for <delay>: <reason>"`.

Stored vectors are reused across `notrieve index --rebuild`. After moving to another server or a
different precision of the same model, set a new `id` under `[model]` to compute every vector
again.

## Where notrieve keeps its data

| What | Where |
|---|---|
| Settings | `<vault>/.notrieve/config.toml` |
| Index and its embedding cache | `<vault>/.notrieve/index.db` and `index.db.embeddings.db` (or `db_path` and the same name with `.embeddings.db` added). The `.notrieve/.gitignore` that `init` writes covers files inside `.notrieve/`; ignore a `db_path` elsewhere yourself. |
| Downloaded models | `$XDG_CACHE_HOME/notrieve/models` (default `~/.cache/notrieve/models`) |
| Registered vaults and approvals | `$XDG_STATE_HOME/notrieve/vaults.toml` (default `~/.local/state/notrieve/vaults.toml`) |

To uninstall, remove the binary and these directories. Your notes are never modified.
