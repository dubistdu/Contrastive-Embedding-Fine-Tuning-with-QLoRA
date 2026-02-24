"""Post-training: category-wise metrics, false positives, before/after vs baseline_metrics.json."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval_metrics import accuracy, cosine_sims, load_pairs, precision_recall_f1, roc_auc
from sentence_transformers import SentenceTransformer


def metrics_for_sims(sims: np.ndarray, compat: np.ndarray, threshold: float = 0.5) -> dict:
    labels = compat.astype(int)
    s1, s0 = sims[compat], sims[~compat]
    n = len(sims)
    n1, n0 = len(s1), len(s0)
    margin = float(np.mean(s1) - np.mean(s0))
    pooled = np.sqrt((np.var(s1) * (n1 - 1) + np.var(s0) * (n0 - 1)) / (n - 2)) if n > 2 else 1e-9
    effect = float(margin / pooled) if pooled > 0 else 0.0
    try:
        from scipy import stats
        pval = float(stats.ttest_ind(s1, s0).pvalue)
    except Exception:
        pval = float("nan")
    acc = accuracy(sims, labels, threshold)
    prec, rec, f1 = precision_recall_f1(sims, labels, threshold)
    auc = roc_auc(sims, labels)
    fp_rate = float(((~compat) & (sims >= threshold)).sum() / n0) if n0 > 0 else 0.0
    return {
        "margin": margin, "effect_size": effect, "p_value": pval,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        "false_positive_rate": fp_rate,
        "mean_compat": float(np.mean(s1)), "mean_incompat": float(np.mean(s0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Post-training evaluation vs baseline")
    parser.add_argument("--eval-data", type=Path, default=REPO_ROOT / "data" / "eval_pairs.jsonl")
    parser.add_argument("--finetuned-model", type=Path, default=REPO_ROOT / "training" / "model")
    parser.add_argument("--baseline-metrics", type=Path, default=REPO_ROOT / "eval" / "visualizations" / "baseline" / "baseline_metrics.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "eval" / "visualizations" / "post_training")
    args = parser.parse_args()

    if not args.eval_data.exists():
        print(f"Eval data not found: {args.eval_data}")
        return
    if not Path(args.finetuned_model).exists():
        print(f"Fine-tuned model not found: {args.finetuned_model}")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_pairs(args.eval_data)
    compat = df["compatible"].values

    # Fine-tuned model
    print(f"Loading fine-tuned model: {args.finetuned_model}")
    model = SentenceTransformer(str(args.finetuned_model))
    sims_ft = cosine_sims(model, df)
    ft_metrics = metrics_for_sims(sims_ft, compat, args.threshold)

    # Baseline metrics (from Step 2)
    baseline_metrics = None
    if args.baseline_metrics.exists():
        with open(args.baseline_metrics) as f:
            baseline_metrics = json.load(f)
        print(f"Loaded baseline metrics from {args.baseline_metrics}")
    else:
        print("No baseline_metrics.json found; run eval/baseline_analysis.py first for before/after comparison.")

    # --- Before/after comparison ---
    print("\n--- Before/After comparison ---")
    if baseline_metrics:
        for key in ["margin", "effect_size", "accuracy", "f1", "auc", "false_positive_rate"]:
            b = baseline_metrics.get(key)
            f = ft_metrics.get(key)
            if b is not None and f is not None:
                delta = f - b
                print(f"  {key}: {b:.4f} → {f:.4f}  (Δ {delta:+.4f})")
    else:
        print("  Fine-tuned only:", {k: round(v, 4) for k, v in ft_metrics.items()})

    # --- Category-wise performance ---
    if "category" in df.columns and df["category"].notna().any():
        df = df.copy()
        df["cosine_sim"] = sims_ft
        df["pred"] = (sims_ft >= args.threshold).astype(int)
        df["label_int"] = compat.astype(int)
        cat_rows = []
        for cat, g in df.groupby("category", dropna=False):
            if len(g) < 2:
                continue
            c = g["compatible"].values
            s = g["cosine_sim"].values
            m = metrics_for_sims(s, c, args.threshold)
            cat_rows.append({
                "category": cat or "(missing)",
                "n": len(g),
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "auc": m["auc"],
                "margin": m["margin"],
            })
        cat_df = pd.DataFrame(cat_rows)
        cat_path = out_dir / "category_wise_metrics.csv"
        cat_df.to_csv(cat_path, index=False)
        print("\n--- Category-wise (fine-tuned) ---")
        print(cat_df.to_string(index=False))
        print(f"\nSaved {cat_path}")
    else:
        print("\nNo 'category' column in eval data; skipping category-wise breakdown.")

    # --- False positive analysis (high-similarity incompatible pairs) ---
    incompat_idx = np.where(~compat)[0]
    if len(incompat_idx) > 0:
        order = np.argsort(-sims_ft[incompat_idx])
        top_fp_idx = incompat_idx[order[:15]]
        fp_rows = []
        for i in top_fp_idx:
            fp_rows.append({
                "text_1": df.iloc[i]["text_a"][:100],
                "text_2": df.iloc[i]["text_b"][:100],
                "cosine_sim": round(float(sims_ft[i]), 4),
                "category": df.iloc[i].get("category", ""),
                "pair_type": df.iloc[i].get("pair_type", ""),
            })
        fp_df = pd.DataFrame(fp_rows)
        fp_path = out_dir / "false_positives_high_similarity.csv"
        fp_df.to_csv(fp_path, index=False)
        print("\n--- False positives (top 15 incompatible by similarity) ---")
        print(fp_df[["cosine_sim", "category", "pair_type"]].to_string(index=False))
        print(f"\nSaved {fp_path}")

    # Save post-training summary for reports
    summary = {"fine_tuned": ft_metrics}
    if baseline_metrics:
        summary["baseline"] = {k: baseline_metrics.get(k) for k in ft_metrics if k in baseline_metrics}
    with open(out_dir / "post_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_dir / 'post_training_summary.json'}")


if __name__ == "__main__":
    main()
