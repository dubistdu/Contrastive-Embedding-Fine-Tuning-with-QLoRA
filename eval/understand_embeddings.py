"""
Minimal script to understand embeddings:
- Load evaluation pairs (compatible vs incompatible) from built-in list or from a file
- Embed with all-MiniLM-L6-v2
- Compute cosine similarity
- Compare mean similarity for compatible vs incompatible
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# --- Step 1: Evaluation pairs ---
# Each pair has text_a, text_b, and compatible (True = should be similar, False = should be dissimilar).
# Default: 20 hand-written pairs. Use --data path/to/pairs.json to evaluate the entire dataset.

EVAL_PAIRS = [
    # Compatible pairs (semantically similar)
    {"text_a": "The cat sat on the mat.", "text_b": "A feline rested on the rug.", "compatible": True},
    {"text_a": "Python is a programming language.", "text_b": "Python is used for coding.", "compatible": True},
    {"text_a": "It is raining outside.", "text_b": "The weather is wet and rainy.", "compatible": True},
    {"text_a": "She loves reading books.", "text_b": "She enjoys reading.", "compatible": True},
    {"text_a": "The meeting starts at nine.", "text_b": "The meeting begins at 9 AM.", "compatible": True},
    {"text_a": "Coffee keeps me awake.", "text_b": "Caffeine helps me stay alert.", "compatible": True},
    {"text_a": "Dogs are loyal pets.", "text_b": "Canines are faithful companions.", "compatible": True},
    {"text_a": "Learning takes time.", "text_b": "Acquiring skills requires patience.", "compatible": True},
    {"text_a": "The sun rises in the east.", "text_b": "Sunrise happens in the east.", "compatible": True},
    {"text_a": "Water boils at 100 degrees.", "text_b": "At 100 Celsius water boils.", "compatible": True},
    # Incompatible pairs (semantically different)
    {"text_a": "The cat sat on the mat.", "text_b": "Quantum physics is complex.", "compatible": False},
    {"text_a": "Python is a programming language.", "text_b": "I ate a banana for lunch.", "compatible": False},
    {"text_a": "It is raining outside.", "text_b": "The concert was sold out.", "compatible": False},
    {"text_a": "She loves reading books.", "text_b": "The engine overheated.", "compatible": False},
    {"text_a": "The meeting starts at nine.", "text_b": "Mountains are beautiful.", "compatible": False},
    {"text_a": "Coffee keeps me awake.", "text_b": "The ship sailed at dawn.", "compatible": False},
    {"text_a": "Dogs are loyal pets.", "text_b": "Algebra uses variables.", "compatible": False},
    {"text_a": "Learning takes time.", "text_b": "Pizza has cheese and tomato.", "compatible": False},
    {"text_a": "The sun rises in the east.", "text_b": "She bought a new laptop.", "compatible": False},
    {"text_a": "Water boils at 100 degrees.", "text_b": "Birds migrate in autumn.", "compatible": False},
]


# --- Step 2: Load model and generate embeddings ---
# Dimensionality: Each sentence becomes one vector of 384 floats (shape (384,)).
# Why 384: all-MiniLM-L6-v2 is a 6-layer BERT-style model whose hidden size is 384;
# the embedding is the [CLS] (or mean-pooled) output of that last hidden layer.
# What the numbers mean: No single dimension has a fixed "meaning" (e.g. "dim 0 = sentiment").
# The model learned 384 directions in space; similarity is encoded in the overall
# direction of the vector. Similar sentences get similar directions; magnitude is
# normalized so we use cosine similarity (angle) rather than distance.

def _resolve_model_path(model_name_or_path):
    """If path is a training output dir with only checkpoints, use the latest checkpoint."""
    if not os.path.isdir(model_name_or_path):
        return model_name_or_path
    config_path = os.path.join(model_name_or_path, "config.json")
    if os.path.isfile(config_path):
        return model_name_or_path
    import re
    checkpoints = [
        d for d in os.listdir(model_name_or_path)
        if os.path.isdir(os.path.join(model_name_or_path, d)) and re.match(r"checkpoint-\d+", d)
    ]
    if not checkpoints:
        return model_name_or_path
    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
    return os.path.join(model_name_or_path, latest)


def load_model_and_embed(pairs, model_name_or_path="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    import numpy as np

    path = _resolve_model_path(model_name_or_path)
    if path != model_name_or_path:
        print("  Resolved to checkpoint:", path)
    model = SentenceTransformer(path)
    texts_a = [p["text_a"] for p in pairs]
    texts_b = [p["text_b"] for p in pairs]
    # encode() returns a numpy array of shape (n_sentences, 384). Each row is one embedding.
    emb_a = model.encode(texts_a)   # shape: (20, 384)
    emb_b = model.encode(texts_b)   # shape: (20, 384)
    return emb_a, emb_b, np.array([p["compatible"] for p in pairs])


# --- Step 3: Cosine similarity ---
# cos_sim = (a·b) / (||a|| * ||b||). We explicitly L2-normalize so this is correct
# regardless of whether the encoder returns unit-norm vectors.
def cosine_similarities(emb_a, emb_b):
    import numpy as np
    norm_a = np.linalg.norm(emb_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(emb_b, axis=1, keepdims=True)
    # Avoid div by zero (shouldn't happen for non-zero embeddings)
    norm_a = np.where(norm_a > 0, norm_a, 1.0)
    norm_b = np.where(norm_b > 0, norm_b, 1.0)
    a_n = emb_a / norm_a
    b_n = emb_b / norm_b
    return (a_n * b_n).sum(axis=1)


if __name__ == "__main__":
    import numpy as np
    from eval_metrics import accuracy, load_pairs as load_pairs_df, print_top_pairs, roc_auc

    parser = argparse.ArgumentParser(description="Evaluate embeddings on compatible/incompatible pairs.")
    default_data = str(REPO_ROOT / "data" / "eval_pairs.jsonl")
    parser.add_argument("--data", type=str, default=default_data, help="Path to JSON/JSONL with pairs. Default: data/eval_pairs.jsonl if present.")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Model name or path. Use training/model/test_model to evaluate the fine-tuned model.")
    args = parser.parse_args()

    if os.path.isfile(args.data):
        df = load_pairs_df(args.data)
        pairs = df[["text_a", "text_b", "compatible"]].to_dict("records")
        print("Loaded", len(pairs), "evaluation pairs from", args.data)
    else:
        pairs = EVAL_PAIRS
        if args.data != default_data:
            print("Warning: --data file not found, using built-in 20 pairs.")
        else:
            print("No data/eval_pairs.jsonl found; using built-in 20 pairs. To evaluate all pairs, create data/eval_pairs.jsonl or pass --data /path/to/your_pairs.jsonl")
        print("Loaded", len(pairs), "evaluation pairs (built-in).")

    n_compat = sum(1 for p in pairs if p.get("compatible") is True)
    n_incompat = len(pairs) - n_compat
    print("  Compatible:", n_compat, "  Incompatible:", n_incompat)
    print("  Model:", args.model)

    emb_a, emb_b, compatible = load_model_and_embed(pairs, model_name_or_path=args.model)

    # Print one full embedding vector only for small eval sets
    if len(pairs) <= 30:
        sample_text = pairs[0]["text_a"]
        sample_emb = emb_a[0]
        print("\nOne embedding vector (sentence: %r):" % sample_text)
        print("  shape:", sample_emb.shape, "  dtype:", sample_emb.dtype)
        print("  vector:", np.array(sample_emb))
        print("  length (L2 norm):", round(float(np.linalg.norm(sample_emb)), 6))
    print("\nEmbedding shape per sentence:", emb_a.shape[1], "dimensions.")

    # --- Step 4: Similarity means and margin ---
    # The margin (compatible mean - incompatible mean) is the main signal: we want the model
    # to assign higher similarity to compatible pairs than incompatible. Larger margin = better
    # separation. Cohen's d = margin / pooled_std tells us how many standard deviations apart
    # the two groups are; it's scale-invariant so we can compare across models or datasets.
    sims = cosine_similarities(emb_a, emb_b)
    compat_mask = compatible.astype(bool)
    sim_compat = sims[compat_mask]
    sim_incompat = sims[~compat_mask]
    mean_compatible = np.mean(sim_compat)
    mean_incompatible = np.mean(sim_incompat)
    margin = mean_compatible - mean_incompatible
    # Cohen's d: (mean1 - mean2) / pooled_std
    n1, n2 = np.sum(compat_mask), np.sum(~compat_mask)
    pooled_std = np.sqrt(((n1 - 1) * np.var(sim_compat) + (n2 - 1) * np.var(sim_incompat)) / (n1 + n2 - 2))
    cohens_d = margin / pooled_std if pooled_std > 0 else 0.0
    print("\nCosine similarity (higher = more similar):")
    print("  Compatible pairs (should be similar):     mean =", round(mean_compatible, 4))
    print("  Incompatible pairs (should be dissimilar): mean =", round(mean_incompatible, 4))
    print("  Margin (compatible - incompatible):       ", round(margin, 4))
    print("  Cohen's d (margin / pooled_std):           ", round(cohens_d, 4))

    # Additional OOD-focused metrics: AUC and accuracy at 0.5.
    labels = compat_mask.astype(int)
    auc = roc_auc(sims, labels)
    acc = accuracy(sims, labels, threshold=0.5)
    print("\nClassification-style view (threshold = 0.5):")
    print("  AUC-ROC:", round(auc, 4))
    print("  Accuracy:", round(acc, 4))

    # Hardest incompatible / weakest compatible examples help debug OOD failures.
    print_top_pairs(pairs, sims, compat_mask, k=10)
