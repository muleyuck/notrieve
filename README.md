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
