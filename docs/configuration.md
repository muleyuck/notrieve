# Configuration

## Settings

Settings live in `.notrieve/config.toml` in the vault root. Every key has a default, except that
an API model needs `chunk_max_tokens`.

| Key | Default | Meaning |
|---|---|---|
| `model` | `"ruri-v3-30m"` | The embedding model: the built-in preset, or a `[model]` table (below). |
| `exclude` | `[]` | Gitignore-style patterns, relative to the vault root, for notes not to index. |
| `db_path` | `".notrieve/index.db"` | The index file, relative to the vault root unless absolute. |
| `search_budget_ms` | `2000` | About how long `notrieve search` spends indexing changed notes before it searches. With the local model, a note already started is finished, so it can run over. |
| `chunk_max_tokens` | `450` | Chunk size in tokens. Required for an API model, where it counts characters. |
| `index_threads` | `0` | Threads for the local model. `0` uses half the CPU cores. |

## Choosing a model

| Model | When to use it |
|---|---|
| `ruri-v3-30m` (default, the only built-in preset) | Notes mostly in Japanese. Runs locally, a 147 MB download. |
| [Another local model](#another-local-model) | Another language, or a larger model, kept on this machine. |
| [An embedding API](#an-embedding-api) | notrieve downloads and loads no model; a server does the embedding (OpenAI, or Ollama on this machine). **The text of your notes is sent to it.** |

Full-text matching works for any language whichever model you choose. Changing the model rebuilds
the index.

## Another local model

Replace `model = "ruri-v3-30m"` with a `[model]` table for any ONNX embedding model. Every field is
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
- `pooling` is `mean` or `cls`.
- `quantization` must be `none`. To use a quantized model, point `onnx_file` at an ONNX file that
  is already quantized.
- `notrieve model pull` downloads the files ahead of the first index.

## An embedding API

Point `[model]` at an OpenAI-compatible `/embeddings` endpoint. **The text of your notes is sent
to that endpoint.**

1. `notrieve init --no-index`
2. Edit `.notrieve/config.toml` with one of the examples below.
3. `notrieve index`

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

### Keys and URLs

- The key never goes in the file. `api_key_env` names the environment variable that holds it.
- The key never appears in error messages or logs, and configuration errors do not repeat the URL.
- `http://` is accepted only for `localhost`, `127.0.0.1` and `::1`.
- The base URL cannot carry a query string or a fragment.

### When the API is slow or down

`notrieve mcp` keeps serving: searches fall back to full-text matching, and the background indexer
waits and retries. While it waits, search responses carry
`indexer_error: "the background indexer paused for <delay>: <reason>"`.

### Recomputing vectors

Stored vectors are reused across `notrieve index --rebuild`. After moving to another server or a
different precision of the same model, set a new `id` under `[model]` to compute every vector
again.

## Vaults from someone else

`.notrieve/config.toml` travels with your notes, so a vault you cloned or synced may carry someone
else's settings. A remote model, or an index written outside the vault, is used only as you
approved it.

- `notrieve vault add <dir>` registers a vault and prints where its config sends notes and which
  environment variable it sends. That is what you approve.
- Commands stop with an error, before contacting anything, when an unregistered vault has such
  settings, or when a registered vault's config changed them since (for example after a pull).
- Review the change, then run `notrieve vault add <dir> --name <name>` again to approve it.
