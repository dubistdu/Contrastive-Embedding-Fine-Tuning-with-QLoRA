"""
Small helpers for evaluation metrics.

We keep this separate so `understand_embeddings.py` stays under 200 lines.
"""

from typing import List

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute AUC-ROC without external deps.

    Uses the rank-based formula:
      AUC = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    """
    labels = labels.astype(int)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    """Simple accuracy with cosine-similarity threshold."""
    labels = labels.astype(int)
    preds = (scores >= threshold).astype(int)
    return float((preds == labels).mean())


def print_top_pairs(pairs: List[dict], sims: np.ndarray, compat_mask: np.ndarray, k: int = 10) -> None:
    """Print hardest incompatible + weakest compatible examples."""
    # Hardest negatives: incompatible with highest similarity
    incompat_idx = np.where(~compat_mask)[0]
    compat_idx = np.where(compat_mask)[0]

    # Pairs may use either text_1/text_2 or text_a/text_b depending on source.
    def _get_text(p: dict, which: int) -> str:
        if which == 1:
            return p.get("text_1") or p.get("text_a") or ""
        return p.get("text_2") or p.get("text_b") or ""

    print("\nTop incompatible pairs with highest similarity (hard negatives):")
    if len(incompat_idx) == 0:
        print("  [none]")
    else:
        hard_neg_idx = incompat_idx[np.argsort(sims[incompat_idx])[::-1][:k]]
        for i in hard_neg_idx:
            print(f"  sim={sims[i]:.4f}")
            print(f"    1: {_get_text(pairs[i], 1)[:80]}")
            print(f"    2: {_get_text(pairs[i], 2)[:80]}")

    print("\nTop compatible pairs with lowest similarity (weak positives):")
    if len(compat_idx) == 0:
        print("  [none]")
    else:
        weak_pos_idx = compat_idx[np.argsort(sims[compat_idx])[:k]]
        for i in weak_pos_idx:
            print(f"  sim={sims[i]:.4f}")
            print(f"    1: {_get_text(pairs[i], 1)[:80]}")
            print(f"    2: {_get_text(pairs[i], 2)[:80]}")


