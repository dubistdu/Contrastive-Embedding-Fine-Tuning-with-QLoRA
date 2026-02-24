# Data

Use only files in this `data/` folder so paths stay self-contained.

| File | Purpose |
|------|---------|
| `dating_pairs.jsonl` | Training pairs (`text_1`, `text_2`, `label`, optional `category`, `pair_type`). Generate via `scripts/generate_dating_pairs.py`. |
| `eval_pairs.jsonl` | Hold-out eval. Used by `evaluate_dating.py`, `eval/understand_embeddings.py`. |
| `ood_eval_pairs.jsonl` | Optional OOD eval (no boy:/girl:). `eval/understand_embeddings.py --data data/ood_eval_pairs.jsonl --model <path>` |
