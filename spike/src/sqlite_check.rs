use rusqlite::{Connection, ToSql, params};

use crate::CheckResult;

fn register_sqlite_vec() {
    // sqlite-vec の公式 Rust 例と同じ登録方法。sqlite3_vec_init は引数なしで宣言されているため transmute が必要。
    unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    }
}

fn report(ok: &mut bool, name: &str, pass: bool, detail: String) {
    println!("{} {name}: {detail}", if pass { "PASS" } else { "FAIL" });
    *ok &= pass;
}

fn ids(conn: &Connection, sql: &str, params: &[&dyn ToSql]) -> rusqlite::Result<Vec<i64>> {
    let mut stmt = conn.prepare(sql)?;
    stmt.query_map(params, |row| row.get(0))?.collect()
}

pub fn run(_args: &[String]) -> CheckResult {
    register_sqlite_vec();
    let dir = std::env::temp_dir().join(format!("notrieve-spike-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let conn = Connection::open(dir.join("check.db"))?;
    let mut ok = true;

    let (sqlite, vec): (String, String) =
        conn.query_row("SELECT sqlite_version(), vec_version()", [], |row| Ok((row.get(0)?, row.get(1)?)))?;
    println!("sqlite={sqlite} sqlite-vec={vec}");
    let mode: String = conn.query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))?;
    report(&mut ok, "wal", mode == "wal", mode);

    conn.execute_batch(
        "CREATE TABLE chunks(id INTEGER PRIMARY KEY, file_id INTEGER);
         CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[4]);",
    )?;
    for id in 1..=10i64 {
        let file_id = if id % 2 == 0 { 1 } else { 2 };
        conn.execute("INSERT INTO chunks(id, file_id) VALUES (?1, ?2)", params![id, file_id])?;
        conn.execute(
            "INSERT INTO chunks_vec(rowid, embedding) VALUES (?1, ?2)",
            params![id, format!("[{id}, 0, 0, 0]")],
        )?;
    }
    let query = "[0, 0, 0, 0]";

    let plain = ids(
        &conn,
        "SELECT rowid FROM chunks_vec WHERE embedding MATCH ?1 AND k = 3 ORDER BY distance",
        &[&query],
    );
    report(&mut ok, "knn", matches!(&plain, Ok(v) if *v == [1, 2, 3]), format!("{plain:?}"));

    let json_in = ids(
        &conn,
        "SELECT rowid FROM chunks_vec WHERE embedding MATCH ?1 AND k = 3
           AND rowid IN (SELECT value FROM json_each(?2)) ORDER BY distance",
        &[&query, &"[2, 5, 7, 9]"],
    );
    report(&mut ok, "knn_rowid_in_json_each", matches!(&json_in, Ok(v) if *v == [2, 5, 7]), format!("{json_in:?}"));

    let sub_in = ids(
        &conn,
        "SELECT rowid FROM chunks_vec WHERE embedding MATCH ?1 AND k = 3
           AND rowid IN (SELECT id FROM chunks WHERE file_id = ?2) ORDER BY distance",
        &[&query, &2i64],
    );
    report(&mut ok, "knn_rowid_in_subquery", matches!(&sub_in, Ok(v) if *v == [1, 3, 5]), format!("{sub_in:?}"));

    let distance: f64 = conn.query_row(
        "SELECT vec_distance_cosine(?1, ?2)",
        params!["[1, 0, 0, 0]", "[0, 1, 0, 0]"],
        |row| row.get(0),
    )?;
    report(&mut ok, "vec_distance_cosine", (distance - 1.0).abs() < 1e-6, format!("{distance}"));

    let rebuilt = conn
        .execute_batch(
            "BEGIN;
             DROP TABLE chunks_vec;
             CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[8]);
             COMMIT;",
        )
        .and_then(|_| conn.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (1, '[1,0,0,0,0,0,0,0]')", []));
    report(&mut ok, "vec0_drop_create_in_transaction", rebuilt.is_ok(), format!("{rebuilt:?}"));
    let old_dim = conn.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (2, '[1,0,0,0]')", []);
    report(&mut ok, "vec0_rejects_old_dim", old_dim.is_err(), format!("{old_dim:?}"));

    conn.execute_batch("CREATE VIRTUAL TABLE chunks_fts USING fts5(title, heading_path, body, tokenize='trigram');")?;
    let rows = [
        (1i64, "税金メモ", "確定申告", "医療費控除の領収書を集める。"),
        (2, "検索の設計", "全文検索", "FTS5 の trigram で全文検索を行う。"),
        (3, "旅行", "京都", "全文を読む。"),
    ];
    for (id, title, heading, body) in rows {
        conn.execute(
            "INSERT INTO chunks_fts(rowid, title, heading_path, body) VALUES (?1, ?2, ?3, ?4)",
            params![id, title, heading, body],
        )?;
    }
    let three = ids(
        &conn,
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?1 ORDER BY bm25(chunks_fts)",
        &[&"\"全文検索\""],
    );
    report(&mut ok, "fts_trigram_ja_3plus_chars", matches!(&three, Ok(v) if *v == [2]), format!("{three:?}"));
    let two = ids(&conn, "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?1", &[&"\"税金\""]);
    report(&mut ok, "fts_trigram_2_chars_matches_nothing", matches!(&two, Ok(v) if v.is_empty()), format!("{two:?}"));
    let like = ids(
        &conn,
        "SELECT rowid FROM chunks_fts
          WHERE title LIKE ?1 ESCAPE '\\' OR heading_path LIKE ?1 ESCAPE '\\' OR body LIKE ?1 ESCAPE '\\'
          ORDER BY rowid",
        &[&"%税金%"],
    );
    report(&mut ok, "like_2_chars_hits_title", matches!(&like, Ok(v) if *v == [1]), format!("{like:?}"));

    std::fs::remove_dir_all(&dir)?;
    Ok(ok)
}
