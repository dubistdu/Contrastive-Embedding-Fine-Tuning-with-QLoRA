"""
Minimal contrastive fine-tuning with CosineSimilarityLoss.
Usage: python train_contrastive.py --data data/dating_pairs.jsonl
"""

import argparse
import json
import os

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


def load_pairs(path):
    """Load pairs from JSON/JSONL. Supports text_1/text_2/label or text_a/text_b/compatible."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if path.endswith(".jsonl"):
        items = [json.loads(line) for line in raw.splitlines() if line]
    else:
        items = json.loads(raw)
    if not isinstance(items, list):
        items = [items]
    pairs = []
    for p in items:
        t1 = p.get("text_1") or p.get("text_a")
        t2 = p.get("text_2") or p.get("text_b")
        if "label" in p:
            compatible = bool(int(p["label"]))
        else:
            c = p.get("compatible")
            compatible = c if isinstance(c, bool) else str(c).lower() in ("1", "true", "yes")
        pairs.append({"text_a": t1, "text_b": t2, "compatible": compatible})
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Contrastive fine-tuning with CosineSimilarityLoss")
    parser.add_argument("--data", required=True, help="Path to training pairs (JSON/JSONL: text_1, text_2, label 0/1)")
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Training data not found: {args.data}")

    # Load pairs and convert to format expected by CosineSimilarityLoss:
    # sentence1, sentence2, score (1.0 = compatible, 0.0 = incompatible)
    pairs = load_pairs(args.data)
    train_dataset = Dataset.from_dict({
        "sentence1": [p["text_a"] for p in pairs],
        "sentence2": [p["text_b"] for p in pairs],
        "score": [1.0 if p["compatible"] else 0.0 for p in pairs],
    })
    print(f"Loaded {len(pairs)} training pairs from {args.data}")

    # Model: same as baseline so we can compare before/after
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Loss: MSE between predicted cosine similarity and target (1 or 0)
    loss = losses.CosineSimilarityLoss(model)

    # Training args: 3 epochs, batch 16, lr 2e-5.
    # For the expanded experiment, pass --data data/dating_pairs_expanded.jsonl
    # and we save to training/model/expanded_model so we can compare to baseline.
    training_args = SentenceTransformerTrainingArguments(
        output_dir="training/model/expanded_model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        save_strategy="epoch",
        logging_steps=20,
        remove_unused_columns=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    # Model is saved to output_dir (training/model/expanded_model) at end of training
    print("Saved model to training/model/expanded_model")


if __name__ == "__main__":
    main()
