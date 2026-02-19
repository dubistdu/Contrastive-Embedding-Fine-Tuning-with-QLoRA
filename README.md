# Contrastive Embedding Fine-Tuning (Dating Compatibility)

Sentence embedding contrastive fine-tuning for dating-compatibility style pairs: compatible pairs (label=1) should have high cosine similarity, incompatible (label=0) low.

- **Model**: `all-MiniLM-L6-v2` (SentenceTransformers), trained with **CosineSimilarityLoss**.
- **Training data**: `data/dating_pairs_expanded.jsonl` (823 pairs after manual curation).

## Quick start

```bash
# Install
pip install -r requirements-embeddings.txt

# Train (uses data/dating_pairs_expanded.jsonl)
python3 train_contrastive.py --data data/dating_pairs_expanded.jsonl

# Evaluate fine-tuned model
python3 understand_embeddings.py --model training/model/expanded_model --data data/eval_pairs.jsonl

# Compare with base model
python3 understand_embeddings.py --model all-MiniLM-L6-v2 --data data/eval_pairs.jsonl
```

## Documentation

- **[PROCESS.md](PROCESS.md)** — Full process: what was done, data pipeline (generate → expand → filter weak positives → manual N/X review → apply edits → train → evaluate), file manifest, and summary for initial commit.
- **data/README.md** — Data folder: expected files and paths.

## Repo layout (main)

| Path | Description |
|------|-------------|
| `train_contrastive.py` | Contrastive training (CosineSimilarityLoss). |
| `understand_embeddings.py` | Eval: embed pairs, cosine similarity, metrics. |
| `eval_metrics.py` | Metric helpers (accuracy, ROC-AUC, etc.). |
| `inspect_pairs.py` | Inspect cosine scores for any pairs file. |
| `data/` | All datasets and data scripts (see data/README.md). |
| `scripts/generate_dating_pairs.py` | Synthetic dating pairs generator. |
| `training/model/expanded_model/` | Fine-tuned model output (optional to track in git). |


