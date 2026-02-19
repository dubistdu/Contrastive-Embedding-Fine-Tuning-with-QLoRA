# Data (inside this repo only)

This project uses **only** the `data/` folder inside `Contrastive-Embedding-Fine-Tuning-with-QLoRA`.  
If you have another data folder elsewhere (e.g. `~/Downloads/data` or a sibling folder), copy the files you need **here** so the repo is self-contained and paths stay simple.

**Expected files:**

- **`eval_pairs.jsonl`** – evaluation pairs (e.g. `{"text_1": "...", "text_2": "...", "label": 0|1}`).  
  If present, `understand_embeddings.py` uses it by default.
- **`dating_pairs.jsonl`** (optional) – training pairs for `train_contrastive.py --data data/dating_pairs.jsonl`.
- **`ood_eval_pairs.jsonl`** – out-of-distribution eval (108 pairs). No boy:/girl: prefixes; indirect phrasing; four buckets: `paraphrase_compatible`, `topical_incompatible`, `dealbreaker_soft`, `ambiguous`. Use for OOD evaluation:  
  `python understand_embeddings.py --data data/ood_eval_pairs.jsonl --model training/model/test_model`

**Using a file outside this folder:**  
You can still pass an explicit path, e.g.  
`python understand_embeddings.py --data /path/to/pairs.jsonl`
