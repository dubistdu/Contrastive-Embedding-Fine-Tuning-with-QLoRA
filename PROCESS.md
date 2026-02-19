# Process & What Was Done

This document lists everything that was done in this project, in order, so you can reproduce it or document the initial commit.

---

## 1. Project goal

- **Contrastive fine-tuning** of a sentence embedding model (e.g. `all-MiniLM-L6-v2`) for **dating compatibility**: pairs with `label=1` should have high cosine similarity, pairs with `label=0` should have low similarity.
- Training uses **CosineSimilarityLoss** (MSE between predicted cosine and target 0/1).
- Data: synthetic dating pairs plus expanded mix (natural paraphrases, hard negatives), then **manual curation** to drop weak positives and fix mislabels.

---

## 2. Setup

- **Python**: 3.9 (or compatible).
- **Dependencies**: `requirements-embeddings.txt` (sentence-transformers, numpy). For data generation, the `scripts/` pipeline may need extra deps (e.g. pydantic); use a venv and install as needed.
- **Data**: All under `data/`. Paths in scripts assume running from project root.

---

## 3. Data pipeline (in order)

### 3.1 Generate synthetic pairs

- **Script**: `scripts/generate_dating_pairs.py`
- **Output**: `data/dating_pairs.jsonl`, `data/eval_pairs.jsonl` (and metadata).
- **What it does**: Generates synthetic boy/girl preference statements with compatibility labels (LLM-as-judge style). Optional: OOD eval set.

### 3.2 Build expanded training set

- **Script**: `data/build_expanded_training_set.py`
- **Input**: `data/dating_pairs.jsonl`, `data/ood_eval_pairs.jsonl` (for natural paraphrase-compatible pairs).
- **Output**: `data/dating_pairs_expanded.jsonl`
- **What it does**: Mixes ~70% synthetic pairs, ~20% natural paraphrase-compatible pairs, ~10% hand-written hard negatives (topic-aligned but incompatible). No index field; rows are 0-based by line order.

### 3.3 Filter weak positive candidates

- **Script**: `data/filter_weak_positives.py`
- **Input**: `data/dating_pairs_expanded.jsonl`
- **Output**: `data/weak_positive_candidates.jsonl`
- **What it does**: For each row with `label=1`, computes cosine similarity with the base model. If cosine &lt; threshold (default 0.40) and the pair does **not** share a value-domain keyword (family, career, etc.), it is written as a “weak positive candidate” with an **`index`** field: the 0-based row index in `dating_pairs_expanded.jsonl`. Does not modify the expanded file.

### 3.4 Manual review of weak positive candidates

- **File**: `data/weak_positive_candidates.jsonl`
- **What was done**:
  - **N** prefix: mark a line as “remove from training” (not a real compatible pair).
  - **X** prefix: mark as “mislabel” — treat as incompatible (set label to 0 in expanded set).
  - No prefix: keep as-is (still a weak positive but not removed/relabeled).
- Indices were then gathered into:
  - **N-marked indices** → `data/n_marked_indices.json` (173 indices).
  - **X-marked indices** → `data/x_marked_indices.json` (7 indices: 335, 342, 415, 429, 576, 671, 768).

### 3.5 Apply manual edits to expanded set

- **File**: `data/dating_pairs_expanded.jsonl`
- **What was done** (0-based indices):
  1. **Relabel X to 0**: For each index in the X list, set `label` to `0` at that row (e.g. indices 335, 342, 415, 429, 576, 671, 768).
  2. **Remove N-marked rows**: Delete every row whose 0-based index is in `n_marked_indices.json` (173 rows removed).
- **Result**: `dating_pairs_expanded.jsonl` went from 997 lines to **823 lines**; those 823 pairs are the final training set used for contrastive fine-tuning.

---

## 4. Training

- **Script**: `train_contrastive.py`
- **Command**:  
  `python3 train_contrastive.py --data data/dating_pairs_expanded.jsonl`
- **What it does**: Loads pairs (text_1, text_2, label), converts to (sentence1, sentence2, score 0.0/1.0), trains `SentenceTransformer("all-MiniLM-L6-v2")` with **CosineSimilarityLoss**, 3 epochs, batch 16, lr 2e-5.
- **Output**: `training/model/expanded_model/` (and checkpoints). No commit was made on your behalf; you can add this path to `.gitignore` if you don’t want to track checkpoints.

---

## 5. Evaluation

- **Script**: `understand_embeddings.py`
- **Usage**:
  - Fine-tuned model on eval set:  
    `python3 understand_embeddings.py --model training/model/expanded_model --data data/eval_pairs.jsonl`
  - Base model (comparison):  
    `python3 understand_embeddings.py --model all-MiniLM-L6-v2 --data data/eval_pairs.jsonl`
  - OOD eval:  
    `python3 understand_embeddings.py --model training/model/expanded_model --data data/ood_eval_pairs.jsonl`
- **What it does**: Loads pairs, embeds with the given model, computes cosine similarity, and reports metrics (e.g. accuracy, ROC-AUC) and optionally top/bottom pairs. Uses `eval_metrics.py` for metric helpers.

---

## 6. Other scripts (no order dependency)

- **`data/audit_training_pairs.py`**: Flags rows in `dating_pairs_expanded.jsonl` where label=1 but cosine &lt; 0.45 or label=0 but cosine &gt; 0.65; writes `data/flagged_pairs.jsonl` and can sample `data/weak_positives_bottom100.jsonl`.
- **`inspect_pairs.py`**: Inspects cosine scores for any pairs file; `--data path/to/pairs.jsonl`.
- **`eval_metrics.py`**: Shared helpers (roc_auc, accuracy, print_top_pairs) used by `understand_embeddings.py`.

---

## 7. File manifest (key files)

| File | Role |
|------|------|
| `data/dating_pairs.jsonl` | Original synthetic training pairs (from generator). |
| `data/dating_pairs_expanded.jsonl` | **Final training set** (823 rows after N-removal and X-relabel). |
| `data/eval_pairs.jsonl` | In-distribution evaluation pairs. |
| `data/ood_eval_pairs.jsonl` | Out-of-distribution evaluation pairs. |
| `data/weak_positive_candidates.jsonl` | Candidates from filter script; manually marked with N/X. |
| `data/n_marked_indices.json` | List of 0-based indices to **remove** from expanded set (N-marked). |
| `data/x_marked_indices.json` | List of 0-based indices to **relabel to 0** in expanded set (X-marked). |
| `data/flagged_pairs.jsonl` | Audit output: possible mislabels (weak positives / strong negatives). |
| `data/weak_positives_bottom100.jsonl` | Optional; bottom-100 weak positives from audit. |
| `data/dating_pairs_metadata.json`, `data/eval_pairs_metadata.json` | Metadata for generated datasets. |
| `training/model/expanded_model/` | Fine-tuned model (and checkpoints). |

---

## 8. Summary for initial commit

- **Done**: Data generation → expanded set → weak-positive filtering → manual N/X review → apply N-removal and X-relabel → train on 823 pairs → evaluate with `understand_embeddings.py`.
- **Not done by assistant**: No `git commit` was run; you can run your initial commit yourself with this process and file manifest as reference.
