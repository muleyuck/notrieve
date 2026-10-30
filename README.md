[![unit-test](https://github.com/muleyuck/notrieve/actions/workflows/unit-test.yml/badge.svg)](https://github.com/muleyuck/notrieve/actions/workflows/unit-test.yml)
[![eval](https://github.com/muleyuck/notrieve/actions/workflows/eval.yml/badge.svg)](https://github.com/muleyuck/notrieve/actions/workflows/eval.yml)
![Software License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)
[![Release](https://img.shields.io/github/release/muleyuck/notrieve.svg)](https://github.com/muleyuck/notrieve/releases/latest)

# <img src="assets/icon.svg" width="40" height="40" alt="" align="top"> notrieve

Search your Markdown notes by meaning, from the terminal or from AI agents such as Claude Code and
Codex.

![demo](tapes/demo.gif)

- **Finds notes by meaning**: full-text and vector search combined. The queries above share
  almost no words with the notes they find.
- **Runs on your machine**: by default the model and the SQLite index stay local. Your notes are
  never modified.
- **Built for AI agents**: `notrieve mcp` is an MCP server for Claude Code, Codex and other
  clients.
- **Japanese by default**: the built-in model is ruri-v3-30m, a Japanese embedding model.
  Full-text matching works in any language; for search by meaning in another language,
  [choose another model](docs/configuration.md#choosing-a-model).

## Install

```sh
brew install muleyuck/tap/notrieve
```

Or `cargo install --git https://github.com/muleyuck/notrieve`, or download an archive from
[GitHub Releases](https://github.com/muleyuck/notrieve/releases).

Runs on macOS 13.4+ (Apple Silicon) and Linux x86_64 / aarch64 with glibc 2.38+ (Ubuntu 24.04+).

## Quick start

```sh
cd ~/notes
notrieve init                                  # set up this directory and build the index
notrieve search "バックアップについて書いたこと"
```

`init` downloads the model (147 MB) once.

## Use from AI agents

Claude Code:

```sh
claude mcp add --scope user notrieve -- notrieve mcp --vault <name>
```

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.notrieve]
command = "notrieve"
args = ["mcp", "--vault", "<name>"]
```

`<name>` is the vault name `init` printed (`notrieve vault list` shows them all). The agent can
then search your notes and read the ones it finds.

## Learn more

- [Usage](docs/usage.md): search options, vaults, what gets indexed, the MCP tools, where data is
  kept
- [Configuration](docs/configuration.md): settings, choosing another local model or an embedding
  API, vaults from someone else
