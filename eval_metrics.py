"""Evaluation helpers: roc_auc, accuracy, precision_recall_f1, print_top_pairs, load_pairs, cosine_sims."""

from typing import List, Union

import numpy as np
import pandas as pd


def load_pairs(path: Union[str, "Path"]) -> pd.DataFrame:
    """Load JSONL pairs into DataFrame with text_a, text_b, compatible. Supports text_1/text_2 or text_a/text_b, label (0/1)."""
    p = str(path)
    df = pd.read_json(p, lines=True)
    df["text_a"] = df["text_1"] if "text_1" in df.columns else df["text_a"]
    df["text_b"] = df["text_2"] if "text_2" in df.columns else df["text_b"]
    df["compatible"] = (df["label"].astype(int) == 1)
    return df


def cosine_sims(model, df: pd.DataFrame) -> np.ndarray:
    """Cosine similarity per pair (encode both sides, L2-normalize, dot product)."""
    t1 = df["text_a"].astype(str).tolist()
    t2 = df["text_b"].astype(str).tolist()
    e1 = model.encode(t1)
    e2 = model.encode(t2)
    e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-9)
    e2 = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-9)
    return (e1 * e2).sum(axis=1)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC-ROC via rank-based formula (no sklearn)."""
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


def precision_recall_f1(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    """Precision, recall, F1 at a given threshold (binary: 1 = compatible)."""
    labels = labels.astype(int)
    preds = (scores >= threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


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


