# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface_hub>=0.25",
#   "onnxruntime==1.28.*",
#   "transformers>=4.48",
#   "numpy",
#   "blake3",
# ]
# ///
"""Generate the Python reference vectors for the ruri-v3-30m preset.

usage: uv run --script scripts/reference.py
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from blake3 import blake3
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO = "sirasagi62/ruri-v3-30m-ONNX"
ONNX_FILE = "onnx/model.onnx"
FILES = [
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    ONNX_FILE,
]
QUERY_PREFIX = "検索クエリ: "
DOCUMENT_PREFIX = "検索文書: "
MAX_TOKENS = 512
DIM = 256
PARAGRAPH = "ハイブリッド検索では、全文検索の順位とベクトル検索の順位を Reciprocal Rank Fusion で統合する。"
CASES = [
    ("query", "瑠璃色はどんな色？"),
    ("document", "瑠璃色（るりいろ）は、紫みを帯びた濃い青。名は、半貴石の瑠璃（ラピスラズリ、英: lapis lazuli）による。"),
    ("query", "SQLite で日本語の全文検索をしたい"),
    ("document", "FTS5 の trigram トークナイザは 3 文字単位で索引を作るため、日本語でも分かち書きなしで部分一致検索ができる。"),
    ("document", "短い"),
    ("document", PARAGRAPH * 8),
]
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "reference-ruri-v3-30m.json"


def embed(sess, tok, text: str) -> list[float]:
    enc = tok(text, return_tensors="np")
    feeds = {}
    for inp in sess.get_inputs():
        if inp.name in enc:
            feeds[inp.name] = enc[inp.name].astype(np.int64)
        elif inp.name == "token_type_ids":
            feeds[inp.name] = np.zeros_like(enc["input_ids"], dtype=np.int64)
        else:
            raise SystemExit(f"unexpected model input: {inp.name}")
    outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
    mask = enc["attention_mask"][..., None].astype(np.float32)
    mean = (outs["token_embeddings"] * mask).sum(1) / mask.sum(1)
    vec = (mean / np.linalg.norm(mean, axis=-1, keepdims=True))[0]
    if vec.shape != (DIM,):
        raise SystemExit(f"expected dimension {DIM}, got {vec.shape}")
    return vec.tolist()


def main() -> None:
    paths = {name: Path(hf_hub_download(REPO, name)) for name in FILES}
    model_dir = paths["tokenizer.json"].parent

    tok = AutoTokenizer.from_pretrained(model_dir)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(paths[ONNX_FILE]), options, providers=["CPUExecutionProvider"])

    prefixes = {"query": QUERY_PREFIX, "document": DOCUMENT_PREFIX}
    cases = []
    for kind, text in CASES:
        prefixed = prefixes[kind] + text
        tokens = len(tok(prefixed)["input_ids"])
        if tokens > MAX_TOKENS:
            raise SystemExit(f"case exceeds {MAX_TOKENS} tokens ({tokens}): {text[:30]}")
        cases.append({"kind": kind, "text": text, "vector": embed(sess, tok, prefixed)})

    fixture = {
        "repo": REPO,
        "onnx_file": ONNX_FILE,
        "files": [
            {"blake3": blake3(paths[name].read_bytes()).hexdigest(), "name": name}
            for name in FILES
        ],
        "query_prefix": QUERY_PREFIX,
        "document_prefix": DOCUMENT_PREFIX,
        "max_tokens": MAX_TOKENS,
        "onnxruntime_version": ort.__version__,
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, ensure_ascii=False, indent=1, sort_keys=False) + "\n")
    print(f"onnxruntime {ort.__version__}; wrote {OUT}")


if __name__ == "__main__":
    main()
