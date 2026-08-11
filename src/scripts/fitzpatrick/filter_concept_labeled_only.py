"""
filter_concept_labeled_only.py - Restrict an already-prepared Fitzpatrick
index (train/val/test.csv from prepare_dataset.py) down to only images with
REAL SkinCon concept labels (concept_mask=1) -- matches the exact data scope
the official CRL repo trains on (their architecture requires real concept
supervision for every image, no partial/masked support).

Pure filtering, no re-splitting/re-grouping needed: the original split
already grouped near-duplicates and stratified by (label, has_concept), so
keeping only concept_mask=1 rows from each split file preserves that
structure -- no new train/test leakage introduced.

Usage:
    python -m src.scripts.fitzpatrick.filter_concept_labeled_only \
        --data_dir data/fitzpatrick17k_prepared \
        --output_dir data/fitzpatrick17k_prepared_concept_only
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/fitzpatrick17k_prepared")
    p.add_argument("--output_dir", type=str, default="data/fitzpatrick17k_prepared_concept_only")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_stats = {}
    for split in ("train", "val", "test"):
        with open(data_dir / f"{split}.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = [r for r in reader if r["concept_mask"] == "1"]

        with open(out_dir / f"{split}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        label_dist = Counter(r["label"] for r in rows)
        split_stats[split] = {"n": len(rows), "label_dist": dict(label_dist)}
        print(f"[INFO] {split}: {len(rows)} images (concept_mask=1 only)  label_dist={dict(label_dist)}")

    meta = {
        "source": str(data_dir),
        "note": "Filtered to concept_mask=1 only (real SkinCon labels), matching the "
                "official CRL repo's data scope. Same 3-class/35-concept schema as the "
                "source -- no other changes.",
        "split_stats": split_stats,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Saved to {out_dir}")


if __name__ == "__main__":
    main()
