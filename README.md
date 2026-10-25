# notrieve

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

## Use from AI agents (MCP)

`notrieve mcp` is a stdio MCP server, so any client that can launch one can use it. Claude Code and
Codex are shown below.

Run `notrieve init` in your notes directory first. It downloads the model and builds the index, so
the MCP server starts quickly.

Claude Code:

```sh
claude mcp add notrieve -- notrieve mcp --vault <name>
```

Codex (`~/.codex/config.toml`, or `codex mcp add notrieve -- notrieve mcp --vault <name>`):

```toml
[mcp_servers.notrieve]
command = "notrieve"
args = ["mcp", "--vault", "<name>"]
```

The server serves one vault. Register one server per vault (`notrieve vault list` shows the names).
Without `--vault` it uses the vault containing the working directory, then the default vault.

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
someone else's settings. A remote model, or a `db_path` outside the vault, is used only for a
vault in your registry. `notrieve init` registers the vault it creates; `notrieve vault add <dir>`
registers an existing one and prints where its config sends notes and which environment variable
it sends. In an unregistered vault, commands stop with an error before contacting anything.
Registering trusts the directory, not one version of its config, so review changes to
`.notrieve/config.toml` that you pull from others.

When the API is slow or unreachable, `notrieve mcp` keeps serving: searches fall back to
full-text matching, and the background indexer waits and retries. While it waits, search
responses carry `indexer_error: "the background indexer paused for <delay>: <reason>"`.

Stored vectors are reused across `notrieve index --rebuild`. After moving to another server or a
different precision of the same model, set a new `id` under `[model]` to compute every vector
again.
