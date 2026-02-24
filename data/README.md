# Data

Use only files in this `data/` folder so paths stay self-contained.

## Main files

| File | Purpose |
|------|---------|
| `dating_pairs.jsonl` | Training pairs (`text_1`, `text_2`, `label`, optional `category`, `pair_type`). Generate via `scripts/generate_dating_pairs.py`. |
| `eval_pairs.jsonl` | Hold-out eval. Used by `evaluate_dating.py`, `eval/understand_embeddings.py`. |
| `ood_eval_pairs.jsonl` | Optional OOD eval (no boy:/girl:). `eval/understand_embeddings.py --data data/ood_eval_pairs.jsonl --model <path>` |

## Optional: weak-positive curation

If you use an **expanded** training set (`dating_pairs_expanded.jsonl` from `build_expanded_training_set.py`), you can clean it with human-in-the-loop review:

- **`filter_weak_positives.py`** — Finds compatible pairs (label=1) with low cosine similarity (and no shared value keyword). Writes `weak_positive_candidates.jsonl` with an `index` per row. You then mark lines with **N** (remove from training) or **X** (relabel to incompatible). Indices go into `n_marked_indices.json` and `x_marked_indices.json`; you apply those edits to the expanded set to get a final curated training file.

  ```bash
  python data/filter_weak_positives.py --data data/dating_pairs_expanded.jsonl
  ```

Full pipeline (build expanded set → filter → review → apply N/X) is in **PROCESS.md** (§3.2–3.5). If you only train on `dating_pairs.jsonl`, you can ignore this.
