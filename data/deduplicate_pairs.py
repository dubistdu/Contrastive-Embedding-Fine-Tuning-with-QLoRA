"""
Deduplicate dating pairs JSONL by (text_1, text_2) / (text_a, text_b).

Usage (from repo root):
  python data/deduplicate_pairs.py \
      --input data/dating_pairs.jsonl \
      --output data/dating_pairs_dedup.jsonl
"""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(records, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_pair(rec: dict):
    """Return a key used for deduplication: (text_1/text_a, text_2/text_b) stripped."""
    t1 = (rec.get("text_1") or rec.get("text_a") or "").strip()
    t2 = (rec.get("text_2") or rec.get("text_b") or "").strip()
    return t1, t2


def main():
    parser = argparse.ArgumentParser(description="Deduplicate JSONL of dating pairs by (text_1, text_2).")
    parser.add_argument("--input", type=Path, default=Path("data") / "dating_pairs.jsonl", help="Input JSONL file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "dating_pairs_dedup.jsonl",
        help="Output JSONL file (deduplicated)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    records = load_jsonl(args.input)
    seen = set()
    deduped = []
    dup_count = 0

    for rec in records:
        key = normalize_pair(rec)
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        deduped.append(rec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(deduped, args.output)

    print(f"Input records:      {len(records)}")
    print(f"Unique pairs kept:  {len(deduped)}")
    print(f"Duplicates removed: {dup_count}")
    print(f"Wrote deduplicated file to: {args.output}")


if __name__ == "__main__":
    main()

