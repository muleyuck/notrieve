-- index.db holds only data derived from the vault's Markdown files, so a schema change rebuilds it
-- from scratch (Store::rebuild) instead of migrating. Vectors live in the embedding cache
-- (src/cache_schema.sql); chunks.embedding_key points there.
DROP TABLE IF EXISTS chunks_vec;  -- schema 1 kept vectors here
DROP TABLE IF EXISTS chunks_fts;
DROP TABLE IF EXISTS file_tags;
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS files;
DROP TABLE IF EXISTS skipped_files;
DROP TABLE IF EXISTS meta;

CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- Files that could not be indexed; unchanged ones (same mtime and size) are not retried.
CREATE TABLE skipped_files(path TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL, size INTEGER NOT NULL, reason TEXT NOT NULL);
CREATE TABLE files(
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  mtime_ns INTEGER NOT NULL,
  size INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  title TEXT NOT NULL,
  aliases TEXT NOT NULL,
  frontmatter TEXT);
-- tag keeps the note's spelling; tag_key is its Unicode lowercase form, used by the search filter.
CREATE TABLE file_tags(
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  tag TEXT NOT NULL,
  tag_key TEXT NOT NULL);
CREATE INDEX file_tags_tag_key ON file_tags(tag_key);
CREATE TABLE chunks(
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  ord INTEGER NOT NULL,
  heading_path TEXT NOT NULL,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  body TEXT NOT NULL,
  embedding_key BLOB NOT NULL);
CREATE INDEX chunks_file ON chunks(file_id);
CREATE VIRTUAL TABLE chunks_fts USING fts5(title, heading_path, body, tokenize='trigram');
