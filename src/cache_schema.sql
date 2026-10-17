-- The embedding cache outlives index rebuilds and is the only place vectors are stored (spec §4.1).
-- Changing this file means bumping CACHE_SCHEMA_VERSION in src/cache.rs, which discards old caches.
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE embeddings(key BLOB PRIMARY KEY, vector BLOB NOT NULL);
