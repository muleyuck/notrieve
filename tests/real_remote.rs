//! Against real embedding APIs; skipped unless the environment enables them (remote spec §7).
//! - NOTRIEVE_TEST_REMOTE=1 with NOTRIEVE_TEST_REMOTE_SOURCE (`openai-compat:<base URL>`),
//!   NOTRIEVE_TEST_REMOTE_MODEL and, when the API needs a key, NOTRIEVE_TEST_REMOTE_KEY_ENV.
//! - NOTRIEVE_TEST_TEI_SOURCE (`openai-compat:<base URL>` of TEI serving cl-nagoya/ruri-v3-30m)
//!   with NOTRIEVE_TEST_MODEL=1 (downloads the local model).
mod support;

use std::path::Path;

use notrieve::app::{self, IndexRequest, Paths, search_vault};
use notrieve::config::{
    ModelContract, ModelSpec, PRESET_RURI_V3_30M, VaultConfig, model_cache_dir,
};
use notrieve::embed::download::ensure_model;
use notrieve::embed::{Embedder, OnnxEmbedder, RemoteEmbedder};
use notrieve::search::{SearchQuery, VectorSearch};
use support::{cosine, remote_fps, write, write_config};

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

#[test]
fn a_real_api_indexes_and_searches_a_vault() {
    if var("NOTRIEVE_TEST_REMOTE").as_deref() != Some("1") {
        eprintln!("skipped: set NOTRIEVE_TEST_REMOTE=1");
        return;
    }
    let source = var("NOTRIEVE_TEST_REMOTE_SOURCE").expect("NOTRIEVE_TEST_REMOTE_SOURCE");
    let model = var("NOTRIEVE_TEST_REMOTE_MODEL").expect("NOTRIEVE_TEST_REMOTE_MODEL");
    let key = var("NOTRIEVE_TEST_REMOTE_KEY_ENV")
        .map(|name| format!("api_key_env = \"{name}\"\n"))
        .unwrap_or_default();
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path().join("notes");
    write_config(
        &root,
        &format!(
            "chunk_max_tokens = 1500\n[model]\nsource = \"{source}\"\nremote_model = \"{model}\"\n{key}"
        ),
    );
    let vault = notrieve::config::Vault::load(&root, None).unwrap();
    write(
        &vault,
        "税金.md",
        "# 医療費\n医療費控除の領収書をまとめる。\n",
    );
    write(&vault, "旅行.md", "# 京都\n紅葉の季節に嵐山を歩いた。\n");
    let contract = match vault.config.model.resolve(&vault.root).unwrap() {
        ModelContract::Remote(c) => c,
        other => panic!("{other:?}"),
    };
    let paths = Paths {
        registry: tmp.path().join("vaults.toml"),
        model_cache: tmp.path().join("models"),
    };
    // Indexed without the search budget, which a slow API could exceed.
    let request = IndexRequest {
        rebuild: false,
        wait: true,
        budget: None,
    };
    app::index(&vault, &paths, &request, &mut |_| {}).unwrap();
    let e = RemoteEmbedder::new(contract);
    let r = search_vault(
        &vault,
        &e,
        &remote_fps(&vault),
        &SearchQuery::new("紅葉を見に行った"),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(r.vector_search, VectorSearch::Ok, "{:?}", r.vector_error);
    assert_eq!(r.results[0].path, "旅行.md");
}

#[test]
fn ruri_on_tei_matches_the_local_onnx_model() {
    let (Some(source), Some("1")) = (
        var("NOTRIEVE_TEST_TEI_SOURCE"),
        var("NOTRIEVE_TEST_MODEL").as_deref(),
    ) else {
        eprintln!("skipped: set NOTRIEVE_TEST_TEI_SOURCE and NOTRIEVE_TEST_MODEL=1");
        return;
    };
    let toml = format!(
        "chunk_max_tokens = 450\n[model]\nsource = \"{source}\"\nremote_model = \"cl-nagoya/ruri-v3-30m\"\n\
         query_prefix = \"検索クエリ: \"\ndocument_prefix = \"検索文書: \"\n"
    );
    let remote = match VaultConfig::parse(&toml)
        .unwrap()
        .model
        .resolve(Path::new("/"))
        .unwrap()
    {
        ModelContract::Remote(c) => RemoteEmbedder::new(c),
        other => panic!("{other:?}"),
    };
    let contract = ModelSpec::Preset(PRESET_RURI_V3_30M.into())
        .resolve(Path::new("."))
        .unwrap()
        .into_local()
        .unwrap();
    let dir = ensure_model(&contract, &model_cache_dir().unwrap(), false, &mut |_| {}).unwrap();
    let local = OnnxEmbedder::load(&contract, &dir, 2).unwrap();
    let texts = vec!["医療費控除の領収書をまとめる。".to_string()];
    let a = remote.embed_documents(&texts).unwrap().remove(0);
    let b = local.embed_documents(&texts).unwrap().remove(0);
    let cos = cosine(&a, &b);
    assert!(cos >= 0.999, "cos = {cos}");
}
