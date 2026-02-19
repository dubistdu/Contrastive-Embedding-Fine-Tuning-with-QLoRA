"""
Audit data/dating_pairs_expanded.jsonl for potentially neutral or mislabeled pairs.

Uses the base model (all-MiniLM-L6-v2) to compute cosine similarity, then flags:
- Case A: label=1 but cosine < 0.45  (labeled compatible but weak similarity → possible mislabel or neutral)
- Case B: label=0 but cosine > 0.65  (labeled incompatible but high similarity → possible mislabel)

Threshold rationale:
- 0.45: compatible pairs typically sit well above this; below suggests the pair may not be truly similar.
- 0.65: incompatible pairs typically sit below this; above suggests the model sees strong overlap (or label is wrong).

Usage: python data/audit_training_pairs.py
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Paths relative to this script (script lives in data/)
SCRIPT_DIR = Path(__file__).resolve().parent
EXPANDED_PATH = SCRIPT_DIR / "dating_pairs_expanded.jsonl"
FLAGGED_PATH = SCRIPT_DIR / "flagged_pairs.jsonl"
WEAK_POSITIVES_PATH = SCRIPT_DIR / "weak_positives_bottom100.jsonl"

# Thresholds for flagging (see docstring)
THRESHOLD_WEAK_POSITIVE = 0.45   # label=1 but cos below this → suspicious
THRESHOLD_STRONG_NEGATIVE = 0.65  # label=0 but cos above this → suspicious


def load_pairs(path):
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pairs.append(json.loads(line))
    return pairs


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
    if not EXPANDED_PATH.exists():
        print(f"Missing: {EXPANDED_PATH}")
        return

    pairs = load_pairs(EXPANDED_PATH)
    print(f"Loaded {len(pairs)} pairs from {EXPANDED_PATH}")

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

        # Case A: positive label but weak similarity
        if label == 1 and cos < THRESHOLD_WEAK_POSITIVE:
            flagged.append({
                "index": i,
                "case": "A",
                "reason": "label=1 but cosine < 0.45 (weak positive)",
                "cosine": round(cos, 4),
                "label": label,
                "text_1": p["text_1"],
                "text_2": p["text_2"],
            })
        # Case B: negative label but high similarity
        elif label == 0 and cos > THRESHOLD_STRONG_NEGATIVE:
            flagged.append({
                "index": i,
                "case": "B",
                "reason": "label=0 but cosine > 0.65 (strong negative)",
                "cosine": round(cos, 4),
                "label": label,
                "text_1": p["text_1"],
                "text_2": p["text_2"],
            })

    # Print
    print(f"\nFlagged {len(flagged)} pairs:\n")
    for item in flagged:
        print(f"Index: {item['index']}  Case: {item['case']}  Cosine: {item['cosine']}  Label: {item['label']}")
        print(f"  Reason: {item['reason']}")
        print(f"  text_1: {item['text_1'][:100]}{'...' if len(item['text_1']) > 100 else ''}")
        print(f"  text_2: {item['text_2'][:100]}{'...' if len(item['text_2']) > 100 else ''}")
        print()

    # Save (one JSON object per line for easy re-use)
    with FLAGGED_PATH.open("w", encoding="utf-8") as f:
        for item in flagged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(flagged)} flagged pairs to {FLAGGED_PATH}")

    # --- Label=1 pairs sorted by cosine ascending; sample bottom 100 (weakest positives) ---
    compat_idx = [i for i in range(len(pairs)) if int(pairs[i]["label"]) == 1]
    compat_idx_sorted = sorted(compat_idx, key=lambda i: sims[i])  # ascending (lowest first)
    bottom_100_idx = compat_idx_sorted[:100]

    weak_positives = []
    for i in bottom_100_idx:
        p = pairs[i]
        weak_positives.append({
            "index": i,
            "cosine": round(float(sims[i]), 4),
            "label": 1,
            "text_1": p["text_1"],
            "text_2": p["text_2"],
        })

    print(f"\nBottom 100 label=1 pairs (by cosine ascending, weakest positives):")
    for item in weak_positives[:10]:
        print(f"  index={item['index']} cos={item['cosine']}  {item['text_1'][:50]}... / {item['text_2'][:50]}...")
    print(f"  ... and {len(weak_positives) - 10} more")

    with WEAK_POSITIVES_PATH.open("w", encoding="utf-8") as f:
        for item in weak_positives:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(weak_positives)} to {WEAK_POSITIVES_PATH}")


if __name__ == "__main__":
    main()
