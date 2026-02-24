"""Pre-train baseline: metrics (margin, effect size, p-value, FPR), optional UMAP; writes eval/visualizations/baseline/."""

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


def main():
    parser = argparse.ArgumentParser(description="Baseline analysis before fine-tuning")
    parser.add_argument("--eval-data", type=Path, default=REPO_ROOT / "data" / "eval_pairs.jsonl")
    parser.add_argument("--baseline-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "eval" / "visualizations" / "baseline")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.eval_data.exists():
        print(f"Eval data not found: {args.eval_data}")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_pairs(args.eval_data)
    compat = df["compatible"].values
    labels = compat.astype(int)
    n_compat = int(compat.sum())
    n_incompat = int((~compat).sum())
    n = len(df)

    print(f"Loading baseline model: {args.baseline_model}")
    model = SentenceTransformer(args.baseline_model)
    sims = cosine_sims(model, df)

    t1 = df["text_a"].astype(str).tolist()
    t2 = df["text_b"].astype(str).tolist()
    emb1 = model.encode(t1)
    emb2 = model.encode(t2)
    embeddings = (emb1 + emb2) / 2.0

    # Statistical separation
    s1, s0 = sims[compat], sims[~compat]
    mean_compat = float(np.mean(s1))
    mean_incompat = float(np.mean(s0))
    std_compat = float(np.std(s1)) if len(s1) > 0 else 0.0
    std_incompat = float(np.std(s0)) if len(s0) > 0 else 0.0
    margin = mean_compat - mean_incompat
    pooled_std = np.sqrt((np.var(s1) * (n_compat - 1) + np.var(s0) * (n_incompat - 1)) / (n - 2)) if n > 2 else 1e-9
    effect_size = float(margin / pooled_std) if pooled_std > 0 else 0.0

    try:
        from scipy import stats as scipy_stats
        p_value = float(scipy_stats.ttest_ind(s1, s0).pvalue)
    except Exception:
        p_value = float("nan")

    # Classification metrics
    acc = accuracy(sims, labels, args.threshold)
    prec, rec, f1 = precision_recall_f1(sims, labels, args.threshold)
    auc = roc_auc(sims, labels)

    # False positive rate: incompatible pairs with similarity >= threshold
    fp_count = ((~compat) & (sims >= args.threshold)).sum()
    false_positive_rate = float(fp_count / n_incompat) if n_incompat > 0 else 0.0

    metrics = {
        "mean_compat": mean_compat,
        "mean_incompat": mean_incompat,
        "std_compat": std_compat,
        "std_incompat": std_incompat,
        "margin": float(margin),
        "effect_size": effect_size,
        "p_value": p_value,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "false_positive_rate": false_positive_rate,
        "n_compat": n_compat,
        "n_incompat": n_incompat,
        "threshold": args.threshold,
    }

    # Detailed CSV
    detail = df.copy()
    detail["cosine_similarity"] = sims
    detail["pred_compatible"] = (sims >= args.threshold).astype(int)
    if "category" not in detail.columns:
        detail["category"] = ""
    if "pair_type" not in detail.columns:
        detail["pair_type"] = ""
    detail.to_csv(out_dir / "detailed_analysis.csv", index=False)

    # UMAP (optional)
    umap_xy = None
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
        umap_xy = reducer.fit_transform(embeddings)
    except ImportError:
        print("UMAP not installed; skipping UMAP visualization. pip install umap-learn")

    # HDBSCAN (optional)
    cluster_labels = None
    if umap_xy is not None:
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean")
            cluster_labels = clusterer.fit_predict(umap_xy)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            metrics["n_umap_clusters"] = n_clusters
            metrics["n_noise_points"] = int((cluster_labels == -1).sum())
        except ImportError:
            print("HDBSCAN not installed; skipping clustering. pip install hdbscan")

    # Save JSON for post-training comparison
    with open(out_dir / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # HTML report
    html_lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Baseline Analysis</title></head><body>",
        "<h1>Baseline embedding analysis (pre-training)</h1>",
        f"<p>Model: <code>{args.baseline_model}</code> | Eval: <code>{args.eval_data.name}</code></p>",
        "<h2>Key metrics</h2>",
        "<table border='1' cellpadding='6'>",
        "<tr><th>Metric</th><th>Value</th></tr>",
        f"<tr><td>Compatible mean similarity</td><td>{mean_compat:.4f} ± {std_compat:.4f}</td></tr>",
        f"<tr><td>Incompatible mean similarity</td><td>{mean_incompat:.4f} ± {std_incompat:.4f}</td></tr>",
        f"<tr><td>Margin (compat − incompat)</td><td>{margin:.4f}</td></tr>",
        f"<tr><td>Effect size (Cohen's d)</td><td>{effect_size:.4f}</td></tr>",
        f"<tr><td>P-value (t-test)</td><td>{p_value:.6f}</td></tr>",
        f"<tr><td>Accuracy (threshold {args.threshold})</td><td>{acc:.4f}</td></tr>",
        f"<tr><td>F1-Score</td><td>{f1:.4f}</td></tr>",
        f"<tr><td>AUC</td><td>{auc:.4f}</td></tr>",
        f"<tr><td>False positive rate</td><td>{false_positive_rate:.4f}</td></tr>",
        "</table>",
        "<p><a href='detailed_analysis.csv'>Download detailed_analysis.csv</a></p>",
    ]

    if umap_xy is not None:
        # Save UMAP scatter as PNG for HTML embedding
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(umap_xy[compat, 0], umap_xy[compat, 1], c="steelblue", alpha=0.6, label="Compatible", s=20)
            ax.scatter(umap_xy[~compat, 0], umap_xy[~compat, 1], c="coral", alpha=0.6, label="Incompatible", s=20)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title("Baseline embedding space (UMAP)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "umap_baseline.png", dpi=120)
            plt.close(fig)
            html_lines.append("<h2>UMAP visualization</h2>")
            html_lines.append("<img src='umap_baseline.png' alt='UMAP' width='700' />")
        except Exception as e:
            html_lines.append(f"<p>UMAP plot skipped: {e}</p>")

    html_lines.append("</body></html>")
    with open(out_dir / "baseline_report.html", "w") as f:
        f.write("\n".join(html_lines))

    print("\n--- Baseline metrics ---")
    print(f"  Compatible mean:   {mean_compat:.4f} ± {std_compat:.4f}")
    print(f"  Incompatible mean: {mean_incompat:.4f} ± {std_incompat:.4f}")
    print(f"  Margin:            {margin:.4f}")
    print(f"  Effect size:       {effect_size:.4f}")
    print(f"  P-value:          {p_value:.6f}")
    print(f"  Accuracy:          {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")
    print(f"  False positive rate: {false_positive_rate:.4f}")
    print(f"\nSaved to {out_dir}/")
    print("  baseline_metrics.json, detailed_analysis.csv, baseline_report.html")


if __name__ == "__main__":
    main()
