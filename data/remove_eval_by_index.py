"""
Remove specific rows from a JSONL by 0-based index.
Use this after reviewing eval_flagged.jsonl: decide which indices to drop, then run:

  python data/remove_eval_by_index.py --data data/eval_pairs.jsonl --indices 1 5 12 --out data/eval_pairs.jsonl
  # or from a file (one index per line):
  python data/remove_eval_by_index.py --data data/eval_pairs.jsonl --indices-file data/indices_to_remove.txt

Backup: copies input to <input>.bak before overwriting (when --out equals --data).
"""

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "eval_pairs.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Remove rows from JSONL by 0-based index.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Input JSONL")
    parser.add_argument("--out", type=Path, default=None, help="Output JSONL (default: overwrite --data)")
    parser.add_argument("--indices", type=int, nargs="*", help="0-based indices to remove")
    parser.add_argument("--indices-file", type=Path, help="File with one 0-based index per line")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak when overwriting")
    args = parser.parse_args()

    out = args.out if args.out is not None else args.data
    to_remove = set()
    if args.indices is not None:
        to_remove.update(args.indices)
    if args.indices_file is not None:
        if not args.indices_file.exists():
            print(f"Missing: {args.indices_file}")
            return
        with args.indices_file.open("r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    to_remove.add(int(line))

    if not to_remove:
        print("No indices given. Use --indices 1 2 3 or --indices-file path.")
        return

    if not args.data.exists():
        print(f"Missing: {args.data}")
        return

    lines = []
    dropped = 0
    with args.data.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i not in to_remove:
                lines.append(line)
            else:
                dropped += 1
                print(f"  Dropping index {i}")

    if out.resolve() == args.data.resolve() and not args.no_backup:
        backup = args.data.with_suffix(args.data.suffix + ".bak")
        with args.data.open("r", encoding="utf-8") as f:
            backup.write_text(f.read(), encoding="utf-8")
        print(f"Backed up to {backup}")

    with out.open("w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Removed {dropped} rows. Before: {len(lines) + dropped}, after: {len(lines)}. Written to {out}")


if __name__ == "__main__":
    main()
