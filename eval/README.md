# Evaluation

All outputs under `eval/visualizations/`:

| Dir | Script | Contents |
|-----|--------|----------|
| **baseline/** | `baseline_analysis.py` | baseline_metrics.json, detailed_analysis.csv, baseline_report.html, umap_baseline.png |
| **comparison/** | `evaluate_dating.py` | 4 plots: score distributions, ROC, metrics bar chart |
| **post_training/** | `post_training_evaluation.py` | category_wise_metrics.csv, false_positives_high_similarity.csv, post_training_summary.json |

**Regenerate comparison plots** (from project root):

```bash
# Optional: refresh baseline metrics first
python eval/baseline_analysis.py

# Writes 4 PNGs to eval/visualizations/comparison/
python eval/evaluate_dating.py
```

Full eval sequence:

```bash
python eval/baseline_analysis.py
python eval/evaluate_dating.py
python eval/post_training_evaluation.py
```
