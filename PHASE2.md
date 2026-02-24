# Phase 2: More sophisticated pipeline (human-in-the-loop curation)

Phase 1 is a **simpler, basic fine-tuning process**: generate synthetic pairs, train with LoRA + contrastive loss on that data, evaluate on a hold-out set. It matches the core assignment and gets you a working compatibility model quickly.

**Phase 2** adds a **more sophisticated data pipeline**: an expanded training mix (synthetic + natural + hard negatives), model-assisted filtering to surface weak or ambiguous positives, and **human-in-the-loop review** so you can remove or relabel candidates before training. The result is a higher-quality, human-curated training set and optional eval cleanup. This phase is for when you want to push data quality and robustness rather than only ship the minimal workflow.

---

## Phase 2 files (grouped)

### Expanded data pipeline

| File | Purpose |
|------|---------|
| `data/build_expanded_training_set.py` | Builds `dating_pairs_expanded.jsonl`: 70% synthetic + 20% natural (OOD) + 10% hard negatives. Run after you have `dating_pairs.jsonl` and `data/ood_eval_pairs.jsonl`. |
| `data/dating_pairs_expanded.jsonl` | Output of the above (or the curated version after N-removal / X-relabel). |
| `data/ood_eval_pairs.jsonl` | Out-of-distribution pairs (different style/phrasing). Used as the 20% natural slice in the expanded set; can also be used for OOD evaluation. |

### Weak-positive filtering and human review

| File | Purpose |
|------|---------|
| `data/filter_weak_positives.py` | From the expanded set: finds label=1 pairs with low cosine (and no value keyword). Writes `weak_positive_candidates.jsonl` with an `index` per row. |
| `data/weak_positive_candidates.jsonl` | Candidates for removal/relabel; you mark lines with **N** (remove) or **X** (relabel to 0). |
| `data/n_marked_indices.json` | List of 0-based indices to **remove** from the expanded set (N-marked). |
| `data/x_marked_indices.json` | List of 0-based indices to **relabel to 0** in the expanded set (X-marked). |

### Auditing and cleanup

| File | Purpose |
|------|---------|
| `data/audit_training_pairs.py` | Flags rows in `dating_pairs_expanded.jsonl`: label=1 but low cosine, or label=0 but high cosine. Writes `flagged_pairs.jsonl`, `weak_positives_bottom100.jsonl`. |
| `data/audit_eval_pairs.py` | Same idea for `eval_pairs.jsonl`; writes `data/eval_flagged.jsonl`. |
| `data/remove_eval_by_index.py` | Removes rows from a JSONL by 0-based index (e.g. after reviewing eval_flagged). |
| `data/flagged_pairs.jsonl` | Audit output (training expanded). |
| `data/eval_flagged.jsonl` | Audit output (eval). |
| `data/weak_positives_bottom100.jsonl` | Bottom-100 weak positives from audit. |

### Other

| File | Purpose |
|------|---------|
| `training/train_contrastive.py` | Full fine-tuning (no LoRA) with CosineSimilarityLoss. |
| `eval/understand_embeddings.py` | Ad-hoc eval: load any model + pairs file, print cosine stats and metrics. |
| `scripts/inspect_pairs.py` | Inspect cosine scores for any pairs file. |
| `PROCESS.md` | Full pipeline walkthrough (including Phase 2 steps). |

---

## Phase 2 workflow

1. **Build expanded set**  
   `python data/build_expanded_training_set.py`  
   → needs `dating_pairs.jsonl`, `ood_eval_pairs.jsonl`.

2. **Filter weak positives**  
   `python data/filter_weak_positives.py --data data/dating_pairs_expanded.jsonl`  
   → `weak_positive_candidates.jsonl`.

3. **Human review**  
   In `weak_positive_candidates.jsonl`, prefix lines with **N** (remove) or **X** (relabel to 0). Gather N indices → `n_marked_indices.json`, X indices → `x_marked_indices.json`.

4. **Apply to expanded set**  
   Relabel X to 0 at those indices; remove rows at N indices → final training set (e.g. 823 lines).

5. **Train**  
   e.g. `python training/train_contrastive.py --data data/dating_pairs_expanded.jsonl` or use the same file with `training/train_dating_embeddings.py` (LoRA) for the curated set.

6. **Eval audit (optional)**  
   `python data/audit_eval_pairs.py` → review `eval_flagged.jsonl`; optionally drop bad rows with `data/remove_eval_by_index.py`.
