"""
Inspect cosine similarity scores for compatible vs incompatible pairs.
Usage: python inspect_pairs.py --data path/to/pairs.jsonl
"""

import argparse
import json
import random

import numpy as np
from sentence_transformers import SentenceTransformer


def load_pairs(path):
    """Load pairs. Supports text_1/text_2/label (0/1) or text_a/text_b/compatible."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if path.endswith(".jsonl"):
        items = [json.loads(line) for line in raw.splitlines() if line]
    else:
        items = json.loads(raw)
    if not isinstance(items, list):
        items = [items]
    pairs = []
    for p in items:
        t1 = p.get("text_1") or p.get("text_a")
        t2 = p.get("text_2") or p.get("text_b")
        if "label" in p:
            compatible = bool(int(p["label"]))
        else:
            c = p.get("compatible")
            compatible = c if isinstance(c, bool) else str(c).lower() in ("1", "true", "yes")
        pairs.append({"text_a": t1, "text_b": t2, "compatible": compatible})
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to JSON or JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    pairs = load_pairs(args.data)
    random.seed(args.seed)

    texts_a = [p["text_a"] for p in pairs]
    texts_b = [p["text_b"] for p in pairs]
    compatible = np.array([p["compatible"] for p in pairs])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_a = model.encode(texts_a)
    emb_b = model.encode(texts_b)

    # Explicit L2 normalization
    norm_a = np.linalg.norm(emb_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(emb_b, axis=1, keepdims=True)
    norm_a = np.where(norm_a > 0, norm_a, 1.0)
    norm_b = np.where(norm_b > 0, norm_b, 1.0)
    emb_a = emb_a / norm_a
    emb_b = emb_b / norm_b

    sims = (emb_a * emb_b).sum(axis=1)

    # Confirm first embedding norm ≈ 1.0
    first_norm = float(np.linalg.norm(emb_a[0]))
    print(f"First embedding L2 norm: {first_norm:.6f}\n")

    compat_mask = compatible.astype(bool)
    incompat_idx = np.where(~compat_mask)[0]
    compat_idx = np.where(compat_mask)[0]

    n_incompat = min(5, len(incompat_idx))
    n_compat = min(5, len(compat_idx))
    sample_incompat = random.sample(list(incompat_idx), n_incompat)
    sample_compat = random.sample(list(compat_idx), n_compat)

    print("--- 5 random incompatible pairs (label=0) ---")
    for i in sample_incompat:
        s = sims[i]
        print(f"  sim = {s:.4f}")
        print(f"    a: {pairs[i]['text_a'][:60]}{'...' if len(pairs[i]['text_a']) > 60 else ''}")
        print(f"    b: {pairs[i]['text_b'][:60]}{'...' if len(pairs[i]['text_b']) > 60 else ''}\n")

    print("--- 5 random compatible pairs (label=1) ---")
    for i in sample_compat:
        s = sims[i]
        print(f"  sim = {s:.4f}")
        print(f"    a: {pairs[i]['text_a'][:60]}{'...' if len(pairs[i]['text_a']) > 60 else ''}")
        print(f"    b: {pairs[i]['text_b'][:60]}{'...' if len(pairs[i]['text_b']) > 60 else ''}\n")

    mean_incompat = np.mean(sims[~compat_mask]) if np.any(~compat_mask) else float("nan")
    mean_compat = np.mean(sims[compat_mask]) if np.any(compat_mask) else float("nan")
    print("--- Means ---")
    print(f"  Incompatible mean similarity: {mean_incompat:.4f}")
    print(f"  Compatible mean similarity:   {mean_compat:.4f}")


if __name__ == "__main__":
    main()
