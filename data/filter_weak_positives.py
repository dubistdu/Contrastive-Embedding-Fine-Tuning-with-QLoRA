"""
Filter weak positives from a contrastive dataset without deleting anything.

Rule: label=1 and cosine < threshold → candidate for removal.
Exception: if both text_1 and text_2 share at least one value-domain keyword, we KEEP
the pair (treat as value-alignment, not weak). Only pairs that are low-cosine and
do NOT share a value keyword are written to weak_positive_candidates.jsonl.

Usage:
  python data/filter_weak_positives.py --data data/dating_pairs_expanded.jsonl
  python data/filter_weak_positives.py --data data/dating_pairs_expanded.jsonl --threshold 0.40
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Pairs that mention these in BOTH texts are treated as value-alignment and preserved
# even when cosine is below threshold (we do not mark them as weak_positive_candidate).
VALUE_KEYWORDS = [
    "family", "kids", "children", "marriage", "honesty", "loyalty", "commitment",
    "career", "ambition", "stability", "entrepreneurship", "religion", "religious",
    "smoking", "drinking", "alcohol", "dealbreaker",
]

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "weak_positive_candidates.jsonl"


def load_pairs(path):
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pairs.append(json.loads(line))
    return pairs


def shares_value_keyword(text_1: str, text_2: str, keywords: list) -> bool:
    """True if both texts contain at least one of the same keyword (case-insensitive)."""
    t1_lower = text_1.lower()
    t2_lower = text_2.lower()
    for kw in keywords:
        if kw in t1_lower and kw in t2_lower:
            return True
    return False


def cosine_similarities(emb_a, emb_b):
    """L2-normalize then dot product (cosine sim)."""
    norm_a = np.linalg.norm(emb_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(emb_b, axis=1, keepdims=True)
    norm_a = np.where(norm_a > 0, norm_a, 1.0)
    norm_b = np.where(norm_b > 0, norm_b, 1.0)
    a_n = emb_a / norm_a
    b_n = emb_b / norm_b
    return (a_n * b_n).sum(axis=1)


def main():
    parser = argparse.ArgumentParser(description="Identify weak positive candidates; preserve value-alignment pairs.")
    parser.add_argument("--data", type=str, default=str(SCRIPT_DIR / "dating_pairs_expanded.jsonl"), help="Path to JSONL (text_1, text_2, label).")
    parser.add_argument("--threshold", type=float, default=0.45, help="Cosine below this with label=1 → weak positive candidate (default 0.45).")
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"File not found: {path}")
        return

    pairs = load_pairs(path)
    print(f"Loaded {len(pairs)} pairs from {path}")
    print(f"Threshold: {args.threshold} (label=1 and cosine < this → candidate unless value keyword shared)")

    # Compute cosine similarities
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts_1 = [p["text_1"] for p in pairs]
    texts_2 = [p["text_2"] for p in pairs]
    emb_1 = model.encode(texts_1)
    emb_2 = model.encode(texts_2)
    sims = cosine_similarities(emb_1, emb_2)

    total_positives = sum(1 for p in pairs if int(p["label"]) == 1)
    weak_candidates = []
    preserved_value = 0

    for i in range(len(pairs)):
        p = pairs[i]
        if int(p["label"]) != 1:
            continue
        cos = float(sims[i])
        if cos >= args.threshold:
            continue
        # Low-cosine positive: keep if they share a value keyword, else candidate
        if shares_value_keyword(p["text_1"], p["text_2"], VALUE_KEYWORDS):
            preserved_value += 1
        else:
            weak_candidates.append({
                "index": i,
                "cosine": round(cos, 4),
                "label": 1,
                "text_1": p["text_1"],
                "text_2": p["text_2"],
            })

    # Summary
    print("\n--- Summary ---")
    print(f"  Total positives (label=1):           {total_positives}")
    print(f"  Preserved (low-cosine + value kw):   {preserved_value}")
    print(f"  Weak positive candidates:           {len(weak_candidates)}")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for item in weak_candidates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(weak_candidates)} candidates to {OUTPUT_PATH}")
    print("(No rows were deleted from the original file.)")


if __name__ == "__main__":
    main()
