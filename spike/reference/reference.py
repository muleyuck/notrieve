# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface_hub>=0.25",
#   "onnxruntime==1.28.*",
#   "transformers>=4.48",
#   "sentence-transformers>=3.4",
#   "numpy",
# ]
# ///
"""ruri-v3-130m ONNX の参照ベクトルを作る。Rust 側 embed-check の比較対象。

usage: uv run --script spike/reference/reference.py <model_int8|model>
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
REPO = "sirasagi62/ruri-v3-130m-ONNX"
MODEL_DIR = ROOT / "models" / "ruri-v3-130m"
OUT_DIR = ROOT / "out"
TOKENIZER_FILES = [
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
]
PARAGRAPH = "ハイブリッド検索では、全文検索の順位とベクトル検索の順位を Reciprocal Rank Fusion で統合する。"
TEXTS = [
    "検索クエリ: 瑠璃色はどんな色？",
    "検索文書: 瑠璃色（るりいろ）は、紫みを帯びた濃い青。名は、半貴石の瑠璃（ラピスラズリ、英: lapis lazuli）による。",
    "検索クエリ: SQLite で日本語の全文検索をしたい",
    "検索文書: FTS5 の trigram トークナイザは 3 文字単位で索引を作るため、日本語でも分かち書きなしで部分一致検索ができる。",
    "検索文書: 短い",
    "検索文書: " + PARAGRAPH * 8,
]
MAX_TOKENS = 512


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def run_onnx(sess, tok, texts):
    enc = tok(texts, padding="longest", return_tensors="np")
    feeds = {}
    for inp in sess.get_inputs():
        if inp.name in enc:
            feeds[inp.name] = enc[inp.name].astype(np.int64)
        elif inp.name == "token_type_ids":
            feeds[inp.name] = np.zeros_like(enc["input_ids"], dtype=np.int64)
        else:
            raise SystemExit(f"unexpected model input: {inp.name}")
    names = [o.name for o in sess.get_outputs()]
    outs = dict(zip(names, sess.run(None, feeds)))
    tokens = outs.get("token_embeddings", outs.get("last_hidden_state"))
    mask = enc["attention_mask"][..., None].astype(np.float32)
    mean = (tokens * mask).sum(1) / mask.sum(1)
    pooled = outs.get("sentence_embedding")
    return l2(pooled if pooled is not None else mean), l2(mean), pooled is not None


def main() -> None:
    variant = sys.argv[1]
    onnx_file = f"onnx/{variant}.onnx"
    snapshot_download(REPO, local_dir=MODEL_DIR, allow_patterns=TOKENIZER_FILES + [onnx_file])

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    token_counts = [len(tok(t)["input_ids"]) for t in TEXTS]
    if max(token_counts) > MAX_TOKENS:
        raise SystemExit(f"fixture exceeds {MAX_TOKENS} tokens: {token_counts}")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(MODEL_DIR / onnx_file), options, providers=["CPUExecutionProvider"])
    batch, batch_mean, has_sentence_embedding = run_onnx(sess, tok, TEXTS)
    single = np.stack([run_onnx(sess, tok, [t])[0][0] for t in TEXTS])

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer("cl-nagoya/ruri-v3-130m", device="cpu").encode(
        TEXTS, normalize_embeddings=True, batch_size=1
    )

    def cos(a, b):
        return (a * b).sum(-1)

    result = {
        "repo": REPO,
        "onnx_file": onnx_file,
        "onnxruntime_version": ort.__version__,
        "input_names": [i.name for i in sess.get_inputs()],
        "output_names": [o.name for o in sess.get_outputs()],
        "texts": TEXTS,
        "token_counts": token_counts,
        "onnx_vectors": batch.tolist(),
        "onnx_single_vs_batch_min_cos": float(cos(single, batch).min()),
        "sentence_embedding_vs_mean_max_abs": float(np.abs(batch - batch_mean).max()) if has_sentence_embedding else None,
        "st_vs_onnx_min_cos": float(cos(st, batch).min()),
    }
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"reference-{variant}.json"
    out.write_text(json.dumps(result, ensure_ascii=False))
    summary = {k: v for k, v in result.items() if k not in ("texts", "onnx_vectors")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
