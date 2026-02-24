"""Baseline vs fine-tuned comparison: metrics, table, optional 4 plots to eval/visualizations/comparison/."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval_metrics import accuracy, cosine_sims, load_pairs, precision_recall_f1, roc_auc
from sentence_transformers import SentenceTransformer


def _resolve_finetuned_path(model_path: Path) -> Path:
    """If path is a training output dir with only checkpoints (no config.json), use the latest checkpoint."""
    if not model_path.is_dir():
        return model_path
    if (model_path / "config.json").is_file():
        return model_path
    checkpoints = [
        d for d in model_path.iterdir()
        if d.is_dir() and re.match(r"checkpoint-\d+", d.name)
    ]
    if not checkpoints:
        return model_path
    latest = max(checkpoints, key=lambda d: int(d.name.split("-")[1]))
    return latest


def roc_curve(scores: np.ndarray, labels: np.ndarray, n_pts: int = 101):
    """Return (fpr, tpr) for ROC plot. labels: 1 = positive."""
    labels = labels.astype(int)
    order = np.argsort(-scores)  # descending
    srt_scores = scores[order]
    srt_labels = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1])
    thresholds = np.linspace(scores.max() + 1e-6, scores.min() - 1e-6, n_pts)
    tpr = np.zeros(n_pts)
    fpr = np.zeros(n_pts)
    for i, t in enumerate(thresholds):
        pred_pos = srt_scores >= t
        tpr[i] = (srt_labels[pred_pos].sum()) / n_pos
        fpr[i] = ((1 - srt_labels)[pred_pos].sum()) / n_neg
    return fpr, tpr


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs fine-tuned dating embeddings")
    parser.add_argument("--eval-data", type=Path, default=REPO_ROOT / "data" / "eval_pairs.jsonl")
    parser.add_argument("--baseline-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--finetuned-model", type=Path, default=REPO_ROOT / "training" / "model")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-plots", type=Path, default=REPO_ROOT / "eval" / "visualizations" / "comparison", help="Save 4 plots here")
    args = parser.parse_args()

    if not args.eval_data.exists():
        print(f"Eval data not found: {args.eval_data}")
        return

    df = load_pairs(args.eval_data)
    compat = df["compatible"].values
    labels = compat.astype(int)
    n_compat, n_incompat = compat.sum(), (~compat).sum()
    print(f"Loaded {len(df)} eval pairs (compatible: {n_compat}, incompatible: {n_incompat})\n")

    try:
        from scipy import stats
    except ImportError:
        stats = None

    def margin_and_effect(sims: np.ndarray):
        s1, s0 = sims[compat], sims[~compat]
        m1, m0 = np.mean(s1), np.mean(s0)
        std1 = np.std(s1) if len(s1) > 0 else 0.0
        std0 = np.std(s0) if len(s0) > 0 else 0.0
        margin = m1 - m0
        pooled_std = np.sqrt((np.var(s1) * (n_compat - 1) + np.var(s0) * (n_incompat - 1)) / (len(sims) - 2)) if len(sims) > 2 else 0.0
        effect = margin / pooled_std if pooled_std > 0 else 0.0
        pval = stats.ttest_ind(s1, s0).pvalue if stats is not None else float("nan")
        return m1, m0, std1, std0, margin, effect, pval

    results = {}

    # Baseline
    print("Loading baseline model:", args.baseline_model)
    baseline = SentenceTransformer(args.baseline_model)
    sims_b = cosine_sims(baseline, df)
    m1_b, m0_b, std1_b, std0_b, margin_b, effect_b, pval_b = margin_and_effect(sims_b)
    results["baseline"] = {
        "mean_compat": m1_b, "std_compat": std1_b, "mean_incompat": m0_b, "std_incompat": std0_b,
        "margin": margin_b, "effect_size": effect_b, "p_value": pval_b,
        "accuracy": accuracy(sims_b, labels, args.threshold),
        "auc": roc_auc(sims_b, labels),
    }
    results["baseline"]["precision"], results["baseline"]["recall"], results["baseline"]["f1"] = precision_recall_f1(
        sims_b, labels, args.threshold
    )

    # Fine-tuned (may be checkpoint dir: resolve to latest checkpoint if no config.json)
    finetuned_path = Path(args.finetuned_model)
    if not finetuned_path.exists():
        print(f"Fine-tuned model not found: {finetuned_path}. Run training/train_dating_embeddings.py first.")
        print("\n--- Baseline only ---\n")
    else:
        resolved = _resolve_finetuned_path(finetuned_path)
        if resolved != finetuned_path:
            print("Loading fine-tuned model:", finetuned_path, "->", resolved)
        else:
            print("Loading fine-tuned model:", finetuned_path)
        finetuned = SentenceTransformer(str(resolved))
        sims_f = cosine_sims(finetuned, df)
        m1_f, m0_f, std1_f, std0_f, margin_f, effect_f, pval_f = margin_and_effect(sims_f)
        results["finetuned"] = {
            "mean_compat": m1_f, "std_compat": std1_f, "mean_incompat": m0_f, "std_incompat": std0_f,
            "margin": margin_f, "effect_size": effect_f, "p_value": pval_f,
            "accuracy": accuracy(sims_f, labels, args.threshold),
            "auc": roc_auc(sims_f, labels),
        }
        results["finetuned"]["precision"], results["finetuned"]["recall"], results["finetuned"]["f1"] = precision_recall_f1(
            sims_f, labels, args.threshold
        )

    # Print course-style table
    def print_metrics(name: str, r: dict):
        print(f"{name}:")
        print(f"  • Compatible mean:   {r['mean_compat']:.4f} ± {r.get('std_compat', 0):.4f}")
        print(f"  • Incompatible mean: {r['mean_incompat']:.4f} ± {r.get('std_incompat', 0):.4f}")
        print(f"  • Margin:            {r['margin']:.4f}")
        print(f"  • Effect size:      {r['effect_size']:.4f}")
        print(f"  • P-value:          {r['p_value']:.6f}")
        print("  • Classification (threshold 0.5):")
        print(f"    Accuracy:         {r['accuracy']:.4f}")
        print(f"    Precision:        {r['precision']:.4f}")
        print(f"    Recall:           {r['recall']:.4f}")
        print(f"    F1-Score:         {r['f1']:.4f}")
        print(f"    AUC:              {r['auc']:.4f}")
        print()

    print("--- Comparing similarity metrics ---\n")
    print_metrics("Baseline Model", results["baseline"])
    if "finetuned" in results:
        print_metrics("Fine-tuned Model", results["finetuned"])
        b, f = results["baseline"], results["finetuned"]
        margin_imp = f["margin"] - b["margin"]
        pct = 100 * margin_imp / abs(b["margin"]) if b["margin"] != 0 else 0
        print("--- Improvement ---")
        print(f"  • Margin improvement:    {margin_imp:.4f}")
        print(f"  • Effect size improvement: {f['effect_size'] - b['effect_size']:.4f}")
        print(f"  • Margin improvement %:  {pct:.1f}%")
        print(f"  • Accuracy: {b['accuracy']:.4f} → {f['accuracy']:.4f}")

    # --- Baseline vs Fine-tuned output comparison (side-by-side) ---
    if "finetuned" in results:
        b, f = results["baseline"], results["finetuned"]
        print("--- Baseline vs Fine-tuned comparison ---\n")
        headers = ("Metric", "Baseline", "Fine-tuned", "Δ")
        row_fmt = "  {:22}  {:>10}  {:>10}  {:>10}"
        print(row_fmt.format(*headers))
        print("  " + "-" * 58)
        print(row_fmt.format("Compatible mean", f"{b['mean_compat']:.4f}", f"{f['mean_compat']:.4f}", _delta(b["mean_compat"], f["mean_compat"])))
        print(row_fmt.format("Incompatible mean", f"{b['mean_incompat']:.4f}", f"{f['mean_incompat']:.4f}", _delta(b["mean_incompat"], f["mean_incompat"])))
        print(row_fmt.format("Margin", f"{b['margin']:.4f}", f"{f['margin']:.4f}", _delta(b["margin"], f["margin"])))
        print(row_fmt.format("Effect size", f"{b['effect_size']:.4f}", f"{f['effect_size']:.4f}", _delta(b["effect_size"], f["effect_size"])))
        print(row_fmt.format("Accuracy", f"{b['accuracy']:.4f}", f"{f['accuracy']:.4f}", _delta(b["accuracy"], f["accuracy"])))
        print(row_fmt.format("F1-Score", f"{b['f1']:.4f}", f"{f['f1']:.4f}", _delta(b["f1"], f["f1"])))
        print(row_fmt.format("AUC", f"{b['auc']:.4f}", f"{f['auc']:.4f}", _delta(b["auc"], f["auc"])))
        print()

    # Save 4 graphics if requested
    if args.save_plots and "finetuned" in results:
        _save_plots(
            compat=compat,
            sims_baseline=sims_b,
            sims_finetuned=sims_f,
            labels=labels,
            results_baseline=results["baseline"],
            results_finetuned=results["finetuned"],
            out_dir=args.save_plots,
        )
    elif args.save_plots and "finetuned" not in results:
        print("Skipping plots (fine-tuned model not loaded); run with a valid --finetuned-model to save graphics.")


def _delta(baseline_val, finetuned_val):
    if finetuned_val is None:
        return "—"
    d = finetuned_val - baseline_val
    return f"{d:+.4f}"


def _save_plots(
    compat: np.ndarray,
    sims_baseline: np.ndarray,
    sims_finetuned: np.ndarray,
    labels: np.ndarray,
    results_baseline: dict,
    results_finetuned: dict,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_b, s0_b = sims_baseline[compat], sims_baseline[~compat]
    s1_f, s0_f = sims_finetuned[compat], sims_finetuned[~compat]

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.hist(s1_b, bins=25, alpha=0.7, label="Compatible", color="steelblue", density=True, edgecolor="white")
    ax1.hist(s0_b, bins=25, alpha=0.7, label="Incompatible", color="coral", density=True, edgecolor="white")
    ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.8)
    ax1.set_xlabel("Cosine similarity")
    ax1.set_ylabel("Density")
    ax1.set_title("1. Baseline model — score distribution")
    ax1.legend()
    ax1.set_xlim(-0.05, 1.05)
    fig1.tight_layout()
    fig1.savefig(out_dir / "1_baseline_distribution.png", dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(s1_f, bins=25, alpha=0.7, label="Compatible", color="steelblue", density=True, edgecolor="white")
    ax2.hist(s0_f, bins=25, alpha=0.7, label="Incompatible", color="coral", density=True, edgecolor="white")
    ax2.axvline(0.5, color="gray", linestyle="--", alpha=0.8)
    ax2.set_xlabel("Cosine similarity")
    ax2.set_ylabel("Density")
    ax2.set_title("2. Fine-tuned model — score distribution")
    ax2.legend()
    ax2.set_xlim(-0.05, 1.05)
    fig2.tight_layout()
    fig2.savefig(out_dir / "2_finetuned_distribution.png", dpi=150)
    plt.close(fig2)

    fpr_b, tpr_b = roc_curve(sims_baseline, labels)
    fpr_f, tpr_f = roc_curve(sims_finetuned, labels)
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.plot(fpr_b, tpr_b, label=f"Baseline (AUC={results_baseline['auc']:.3f})", color="gray", lw=2)
    ax3.plot(fpr_f, tpr_f, label=f"Fine-tuned (AUC={results_finetuned['auc']:.3f})", color="green", lw=2)
    ax3.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax3.set_xlabel("False positive rate")
    ax3.set_ylabel("True positive rate")
    ax3.set_title("3. ROC curves — baseline vs fine-tuned")
    ax3.legend()
    ax3.set_aspect("equal")
    fig3.tight_layout()
    fig3.savefig(out_dir / "3_roc_curves.png", dpi=150)
    plt.close(fig3)

    metrics_names = ["Accuracy", "F1", "AUC", "Margin", "Effect size"]
    keys = ["accuracy", "f1", "auc", "margin", "effect_size"]
    baseline_vals = [results_baseline[k] for k in keys]
    finetuned_vals = [results_finetuned[k] for k in keys]
    x = np.arange(len(metrics_names))
    w = 0.35
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.bar(x - w / 2, baseline_vals, w, label="Baseline", color="gray", alpha=0.9)
    ax4.bar(x + w / 2, finetuned_vals, w, label="Fine-tuned", color="green", alpha=0.9)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_names)
    ax4.set_ylabel("Score")
    ax4.set_title("4. Metrics comparison — baseline vs fine-tuned")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(out_dir / "4_metrics_comparison.png", dpi=150)
    plt.close(fig4)

    print(f"Saved 4 plots to {out_dir}/")


if __name__ == "__main__":
    main()
