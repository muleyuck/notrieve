# Usage

## Requirements

- macOS 13.4 or later on Apple Silicon. Intel Macs are not supported.
- Linux x86_64 or aarch64 with glibc 2.38 or later and `GLIBCXX_3.4.31` or later (Ubuntu 24.04 or
  later). Ubuntu 22.04 does not work.

Release archives are named `notrieve-<target>.tar.xz`, for `aarch64-apple-darwin`,
`aarch64-unknown-linux-gnu` and `x86_64-unknown-linux-gnu`. Unpack one and put `notrieve` on your
`PATH`. Building from source needs a Rust toolchain.

## Searching

```sh
notrieve search "バックアップについて書いたこと"
notrieve search -k 3 --path 仕事/ "障害のあとにやること"
```

Each result is a chunk of a note: its path, line range and heading path, then the text.

| Option | Meaning |
|---|---|
| `-k <n>` | Number of results, 1 to 50 (default 8). |
| `--tag <tag>` | Only notes with this tag or one of its child tags. Repeat it to require several. |
| `--path <prefix>` | Only notes whose vault-relative path starts with this. |
| `--json` | Print the results as JSON. |
| `--vault <name>` | Search this registered vault (see below). |

Before searching, `notrieve search` indexes notes changed since the last run, for about
`search_budget_ms` (2 seconds by default). Notes it does not get to are searched as last indexed,
and so are all notes while another process is indexing. `notrieve index` updates the index
without searching.

## Vaults

A vault is a directory of notes with a `.notrieve/` directory in it. `notrieve init` creates one
and registers it under the directory's name.

| Command | What it does |
|---|---|
| `notrieve vault list` | List the registered vaults. |
| `notrieve vault add <dir>` | Register a directory that already has `.notrieve/`, such as a cloned or synced vault. |
| `notrieve vault remove <name>` | Unregister a vault. Its files are kept. |

Without `--vault`, commands use the vault containing the current directory, then the default
vault (the first one registered, or the one given `--default`).

## What gets indexed

- Every `*.md` file under the vault, except:
  - hidden files and directories,
  - files ignored by `.gitignore` or `.ignore` (whether or not the vault is a git repository),
  - files matching `exclude` in `.notrieve/config.toml`.
- Notes are split into chunks at headings, and long sections by size.
- Tags come from the frontmatter `tags` property and from inline `#tags`, following Obsidian's tag
  rules.

## The MCP server

`notrieve mcp` is a stdio MCP server, so any client that can launch one can use it. Run
`notrieve init` first, so the server starts with the model downloaded and the index built.

- One server serves one vault. Register one server per vault.
- Claude Code: `--scope user` makes the notes searchable from every project. Without it, the
  server is added to the current project only.
- Codex: `codex mcp add notrieve -- notrieve mcp --vault <name>` writes the same entry as the
  `config.toml` in the README.

The server offers two tools:

| Tool | What it does |
|---|---|
| `search` | Hybrid search with `query`, and optionally `k`, `tags` and `path_prefix`. Returns the path, line range, heading path and text of each chunk. |
| `get_note` | Reads the note at `path` (as `search` returns it), whole or from `start_line` to `end_line`, to see the context around a hit. It never reads outside the vault. |

When a search finds notes changed since they were indexed, it answers from the index as it is,
counts them in `stale_files`, and asks a background thread to index them for later searches.

## Where notrieve keeps its data

| What | Where |
|---|---|
| Settings | `<vault>/.notrieve/config.toml` |
| Index | `<vault>/.notrieve/index.db`, or `db_path` |
| Embedding cache | Next to the index, with `.embeddings.db` added to its name |
| Downloaded models | `$XDG_CACHE_HOME/notrieve/models` (default `~/.cache/notrieve/models`) |
| Registered vaults and approvals | `$XDG_STATE_HOME/notrieve/vaults.toml` (default `~/.local/state/notrieve/vaults.toml`) |

`init` writes a `.notrieve/.gitignore` that ignores the index and the cache inside `.notrieve/`.
Ignore a `db_path` elsewhere yourself.

To uninstall, remove the binary, each vault's `.notrieve/` directory (and its index and cache if
`db_path` points elsewhere), the models directory and `vaults.toml`.
