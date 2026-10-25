# notrieve

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

When the API is slow or unreachable, `notrieve mcp` keeps serving: searches fall back to
full-text matching, and the background indexer waits and retries. While it waits, search
responses carry `indexer_error: "the background indexer paused for <delay>: <reason>"`.

Stored vectors are reused across `notrieve index --rebuild`. After moving to another server or a
different precision of the same model, set a new `id` under `[model]` to compute every vector
again.
