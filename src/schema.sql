-- index.db holds only data derived from the vault's Markdown files, so a schema change rebuilds it
-- from scratch (Store::rebuild) instead of migrating. {dim} is replaced with the model dimension.
DROP TABLE IF EXISTS chunks_vec;
DROP TABLE IF EXISTS chunks_fts;
DROP TABLE IF EXISTS file_tags;
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS files;
DROP TABLE IF EXISTS meta;

CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE files(
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  mtime_ns INTEGER NOT NULL,
  size INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  title TEXT NOT NULL,
  aliases TEXT NOT NULL,
  frontmatter TEXT);
CREATE TABLE file_tags(
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  tag TEXT NOT NULL);
CREATE INDEX file_tags_tag ON file_tags(tag COLLATE NOCASE);
CREATE TABLE chunks(
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  ord INTEGER NOT NULL,
  heading_path TEXT NOT NULL,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  body TEXT NOT NULL);
CREATE INDEX chunks_file ON chunks(file_id);
CREATE VIRTUAL TABLE chunks_fts USING fts5(title, heading_path, body, tokenize='trigram');
CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{dim}]);
