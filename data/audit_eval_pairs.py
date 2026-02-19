"""
Audit eval (or any) pairs for possible mislabels using base model cosine similarity.

Flags:
- Case A: label=1 but cosine < 0.45  (labeled compatible but weak similarity)
- Case B: label=0 but cosine > 0.65   (labeled incompatible but high similarity)

Usage:
  python data/audit_eval_pairs.py
  python data/audit_eval_pairs.py --data data/eval_pairs.jsonl --out data/eval_flagged.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "eval_pairs.jsonl"
DEFAULT_OUT = SCRIPT_DIR / "eval_flagged.jsonl"

THRESHOLD_WEAK_POSITIVE = 0.45   # label=1 but cos below → suspicious
THRESHOLD_STRONG_NEGATIVE = 0.65  # label=0 but cos above → suspicious


def load_pairs(path):
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pairs.append(json.loads(line))
    return pairs


def cosine_similarities(emb_a, emb_b):
    norm_a = np.linalg.norm(emb_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(emb_b, axis=1, keepdims=True)
    norm_a = np.where(norm_a > 0, norm_a, 1.0)
    norm_b = np.where(norm_b > 0, norm_b, 1.0)
    a_n = emb_a / norm_a
    b_n = emb_b / norm_b
    return (a_n * b_n).sum(axis=1)


def main():
    parser = argparse.ArgumentParser(description="Audit pairs for possible mislabels (cosine vs label).")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Input JSONL (text_1, text_2, label)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSONL for flagged pairs")
    parser.add_argument("--threshold-pos", type=float, default=THRESHOLD_WEAK_POSITIVE, help="Flag label=1 if cos below this")
    parser.add_argument("--threshold-neg", type=float, default=THRESHOLD_STRONG_NEGATIVE, help="Flag label=0 if cos above this")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Missing: {args.data}")
        return

    pairs = load_pairs(args.data)
    print(f"Loaded {len(pairs)} pairs from {args.data}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts_1 = [p["text_1"] for p in pairs]
    texts_2 = [p["text_2"] for p in pairs]
    emb_1 = model.encode(texts_1)
    emb_2 = model.encode(texts_2)
    sims = cosine_similarities(emb_1, emb_2)

    flagged = []
    for i in range(len(pairs)):
        p = pairs[i]
        label = int(p["label"])
        cos = float(sims[i])
        if label == 1 and cos < args.threshold_pos:
            flagged.append({
                "index": i,
                "case": "A",
                "reason": f"label=1 but cosine < {args.threshold_pos} (weak positive)",
                "cosine": round(cos, 4),
                "label": label,
                "text_1": p["text_1"],
                "text_2": p["text_2"],
            })
        elif label == 0 and cos > args.threshold_neg:
            flagged.append({
                "index": i,
                "case": "B",
                "reason": f"label=0 but cosine > {args.threshold_neg} (strong negative)",
                "cosine": round(cos, 4),
                "label": label,
                "text_1": p["text_1"],
                "text_2": p["text_2"],
            })

    print(f"\nFlagged {len(flagged)} pairs (possible mislabels):\n")
    for item in flagged[:20]:
        print(f"  [{item['index']}] {item['case']}  cos={item['cosine']}  {item['reason']}")
        print(f"    text_1: {item['text_1'][:80]}{'...' if len(item['text_1']) > 80 else ''}")
        print(f"    text_2: {item['text_2'][:80]}{'...' if len(item['text_2']) > 80 else ''}")
        print()
    if len(flagged) > 20:
        print(f"  ... and {len(flagged) - 20} more (see {args.out})")

    with args.out.open("w", encoding="utf-8") as f:
        for item in flagged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(flagged)} flagged pairs to {args.out}")


if __name__ == "__main__":
    main()
