# Changelog

## [0.1.1](https://github.com/muleyuck/notrieve/compare/v0.1.0...v0.1.1) (2026-10-31)


### Bug Fixes

* **build:** build the macOS release for macOS 13.4 ([#30](https://github.com/muleyuck/notrieve/issues/30)) ([90d414e](https://github.com/muleyuck/notrieve/commit/90d414e993b721758a88eed7c08b62ae0e819747))

## 0.1.0 (2026-10-30)


### Features

* embed through an OpenAI-compatible API ([#5](https://github.com/muleyuck/notrieve/issues/5)) ([948c0f0](https://github.com/muleyuck/notrieve/commit/948c0f040a805b56358fd8fdbd33d36f567a7763))
* hybrid search with an embedding cache and ruri-v3-30m (plan 2) ([#3](https://github.com/muleyuck/notrieve/issues/3)) ([d775305](https://github.com/muleyuck/notrieve/commit/d775305142f27aadf0357e9dbf279299ff4f2941))
* index Markdown vaults into a local SQLite vector store (plan 1) ([#1](https://github.com/muleyuck/notrieve/issues/1)) ([98b27c8](https://github.com/muleyuck/notrieve/commit/98b27c8ad87aea27a578b908ae9fc66d9dd44858))
* serve a vault to AI agents over MCP (plan 3) ([#4](https://github.com/muleyuck/notrieve/issues/4)) ([b9af159](https://github.com/muleyuck/notrieve/commit/b9af159d674e05b6cb93e5b6f1604215e38131a5))


### Bug Fixes

* close three gaps in trusting a cloned or synced vault ([#22](https://github.com/muleyuck/notrieve/issues/22)) ([26cf066](https://github.com/muleyuck/notrieve/commit/26cf06665228f0f7ef6a9ba797d71636828b5c53))
* keep the vault registry in XDG_STATE_HOME without a lock file ([#21](https://github.com/muleyuck/notrieve/issues/21)) ([ac4a48c](https://github.com/muleyuck/notrieve/commit/ac4a48c4a07ce4c4d352490fd2c1c4662173d135))
* trust a vault's remote and outside-index settings only as the user approved them ([#11](https://github.com/muleyuck/notrieve/issues/11)) ([829c190](https://github.com/muleyuck/notrieve/commit/829c1902f5d86cece044d60063c7db256bacf4e8))
