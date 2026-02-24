# Contrastive Embedding Fine-Tuning (Dating Compatibility)

Contrastive fine-tuning of sentence embeddings for dating compatibility: compatible pairs (label=1) embed close, incompatible (label=0) far apart. Phase 1 is the core workflow; Phase 2 adds expanded data and human-in-the-loop curation ([PHASE2.md](PHASE2.md)).

## Project structure

```
├── data/              dating_pairs.jsonl, eval_pairs.jsonl + audit/removal scripts
├── training/          train_dating_embeddings.py (LoRA), train_dating_embeddings_qlora.py (4-bit QLoRA), simple_trainer.py, train_contrastive.py (full)
├── eval/              baseline_analysis.py, evaluate_dating.py, post_training_evaluation.py, understand_embeddings.py
├── scripts/           generate_dating_pairs.py, evaluate_synthetic_data.py, inspect_pairs.py
├── eval_metrics.py    shared metric helpers (roc_auc, accuracy, etc.)
└── eval/visualizations/   baseline/, comparison/, post_training/
```

## Quick start

```bash
pip install -r requirements-embeddings.txt
python eval/baseline_analysis.py
python training/train_dating_embeddings.py
python eval/evaluate_dating.py
python eval/post_training_evaluation.py
```

- **Train:** LoRA + `CosineSimilarityLoss` on `data/dating_pairs.jsonl` → `training/model/`
- **Eval:** `evaluate_dating.py` compares baseline vs fine-tuned, prints metrics, writes 4 plots to `eval/visualizations/comparison/`

## Workflow (Steps 1–4)

| Step | Command | Output |
|------|---------|--------|
| 1a | `scripts/generate_dating_pairs.py` | `dating_pairs.jsonl`, `eval_pairs.jsonl` |
| 1b | `scripts/evaluate_synthetic_data.py --data data/dating_pairs.jsonl --output data/eval_report.json` | Full JSON report; overall_scores.overall_score (require > 60%) |
| 2 | `eval/baseline_analysis.py` | `eval/visualizations/baseline/` (metrics, report, UMAP) |
| 3 | `training/train_dating_embeddings.py` or `training/simple_trainer.py` | `training/model/` or `training/model/final_model/` |
| 4 | `eval/evaluate_dating.py` + `eval/post_training_evaluation.py` | Comparison plots, category-wise, false positives |

## Data alteration + full model (proven setup)

For best results: curate data, then full fine-tune.

1. **Curate data** — Remove unrelated/mislabeled pairs:
   - Training: `data/audit_training_pairs.py` (flags by cosine); or Phase 2 `filter_weak_positives.py` + N/X review.
   - Eval: `data/audit_eval_pairs.py` → `eval_flagged.jsonl`; then `data/remove_eval_by_index.py --indices ...`
2. **Full model** — `training/train_contrastive.py --data data/dating_pairs.jsonl` or `training/simple_trainer.py ...` (no LoRA).

## Repo layout

| Path | Purpose |
|------|---------|
| `training/train_dating_embeddings.py` | LoRA + contrastive (default) |
| `training/train_dating_embeddings_qlora.py` | 4-bit quantized LoRA (QLoRA); requires bitsandbytes, accelerate |
| `training/simple_trainer.py` | Full fine-tune, 4 epochs |
| `eval/baseline_analysis.py` | Pre-train metrics, UMAP, HTML report |
| `eval/evaluate_dating.py` | Baseline vs fine-tuned + 4 plots |
| `eval/post_training_evaluation.py` | Category-wise, false positives |
| `data/audit_*.py`, `remove_eval_by_index.py` | Data curation |
| `scripts/generate_dating_pairs.py` | Synthetic pairs |
| `scripts/evaluate_synthetic_data.py` | Full JSON report (data_quality, diversity, bias, linguistic, overall_scores); use --output for file |
| `scripts/inspect_pairs.py` | Inspect cosine scores for any pairs file |
| `training/train_contrastive.py` | Full fine-tune (CosineSimilarityLoss, no LoRA) |
| `eval/understand_embeddings.py` | Ad-hoc eval: any model + pairs file → cosine stats, AUC, accuracy |
| [PHASE2.md](PHASE2.md) | Expanded pipeline, human review |

## Docs

- [PROCESS.md](PROCESS.md) — Data pipeline
- [PHASE2.md](PHASE2.md) — Phase 2 workflow
- [eval/README.md](eval/README.md) — Eval outputs
- [data/README.md](data/README.md) — Data files
