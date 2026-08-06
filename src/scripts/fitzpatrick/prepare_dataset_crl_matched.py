"""
prepare_dataset_crl_matched.py — Index prep matching the REAL CRL repo's own
filtering (obiyoag/crl, data/split_dataset.py), so ICRL can be run on the
EXACT same data scope as an official CRL run, for an apples-to-apples
accuracy comparison (not just a rule-quality-only comparison).

Filters replicated EXACTLY from their split_dataset.py:
  1. QC: drop "Do not consider this image"=1 (SkinCon) -- same flag our own
     prepare_dataset.py already excludes.
  2. Drop "non-neoplastic" entirely -- their task is 2-class (benign/malignant).
  3. INNER JOIN label with concepts -- only images that have a real SkinCon
     concept annotation are kept at all (no concept_mask=0 rows with
     S1-guessed concepts, unlike our main pipeline).
  4. All 48 concept columns kept (we normally drop 13 rare ones for the main
     35-concept pipeline -- not here, to match their code exactly).

Deliberately NOT replicated (documented, not silent):
  - "Wrongly labelled" qc exclusion (17 images) from fitzpatrick17k.csv's
    own quality flag -- their script never reads this column at all. We
    keep applying it: it is real author-flagged bad data, dropping it is
    a strict improvement, not an arbitrary scope choice.
  - Near-duplicate (dHash) grouping before splitting -- their script has
    no such step (plain per-row 5-fold assignment), a real leakage risk in
    their own pipeline. We keep grouping to avoid leaking near-duplicate
    lesion photos across splits.
  - Split protocol: they do 5-fold CV where "test" == "valid" (no held-out
    test set at all). We use a stratified train/val/test 3-way split
    instead, consistent with the rest of this project's convention.

Usage:
    python -m src.scripts.fitzpatrick.prepare_dataset_crl_matched \\
        --data_dir data/fitzpatrick17k --output_dir data/fitzpatrick17k_crl_matched
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.utils.fitzpatrick_dedup import UnionFind, compute_dhashes, find_near_duplicates

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
LABEL_TO_IDX = {"benign": 0, "malignant": 1}
NON_CONCEPT_SKINCON_COLUMNS = {"ImageID", "Do not consider this image", ""}
# The "" entry matters: skincon.csv has a leading unnamed index column
# (pandas' own read_csv(..., index_col=0) in the official repo silently
# drops it as the DataFrame index -- csv.DictReader doesn't, so it must be
# excluded explicitly here or it leaks in as a spurious 49th "concept").


def parse_args():
    p = argparse.ArgumentParser(description="Prep Fitzpatrick17k index matching the real CRL repo's data filtering.")
    p.add_argument("--data_dir", type=str, default="data/fitzpatrick17k")
    p.add_argument("--output_dir", type=str, default="data/fitzpatrick17k_crl_matched")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_fitzpatrick_csv(data_dir: Path) -> dict[str, dict]:
    with open(data_dir / "fitzpatrick17k.csv", encoding="utf-8") as f:
        return {r["md5hash"]: r for r in csv.DictReader(f)}


def load_skincon_csv(data_dir: Path) -> tuple[list[str], list[dict]]:
    with open(data_dir / "skincon.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        concept_names = [c for c in reader.fieldnames if c not in NON_CONCEPT_SKINCON_COLUMNS]
    return concept_names, rows


def build_concept_vectors(skincon_rows: list[dict], concept_names: list[str]) -> dict[str, list[int]]:
    """hash -> concept vector, QC-filtered ('Do not consider this image'), all 48 concepts."""
    out = {}
    for r in skincon_rows:
        if r["Do not consider this image"] == "1":
            continue
        h = r["ImageID"].replace(".jpg", "")
        out[h] = [int(r[c]) for c in concept_names]
    return out


def stratified_group_split(all_hashes, group_of, label_of, seed) -> dict[str, str]:
    """Same grouped/stratified logic as prepare_dataset.py, minus the
    has_concept stratification key (every image here has real concepts)."""
    rng = random.Random(seed)
    members_of_group: dict[int, list[str]] = defaultdict(list)
    for h in all_hashes:
        members_of_group[group_of[h]].append(h)

    def group_key(members: list[str]) -> str:
        return Counter(label_of[h] for h in members).most_common(1)[0][0]

    groups_by_key: dict[str, list[list[str]]] = defaultdict(list)
    for members in members_of_group.values():
        groups_by_key[group_key(members)].append(members)

    split_assignment: dict[str, str] = {}
    for key, group_list in groups_by_key.items():
        rng.shuffle(group_list)
        key_total = sum(len(m) for m in group_list)
        target = {s: r * key_total for s, r in SPLIT_RATIOS.items()}
        assigned = {s: 0 for s in SPLIT_RATIOS}
        for members in group_list:
            deficit = {s: target[s] - assigned[s] for s in SPLIT_RATIOS}
            best_split = max(deficit, key=lambda s: deficit[s])
            for h in members:
                split_assignment[h] = best_split
            assigned[best_split] += len(members)
    return split_assignment


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    img_dir = data_dir / "data" / "finalfitz17k"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading CSVs...")
    fp_rows = load_fitzpatrick_csv(data_dir)
    concept_names, skincon_rows = load_skincon_csv(data_dir)
    print(f"[INFO] {len(concept_names)} concept columns found in skincon.csv "
          f"(expect 48, matching official CRL's df.columns[2:50])")
    concept_vectors = build_concept_vectors(skincon_rows, concept_names)

    # ── Filter 1: our own extra QC flag (their code doesn't check this, but
    #    it's real author-flagged bad data -- see module docstring) ────────
    wrongly_labelled = {h for h, r in fp_rows.items() if r["qc"].startswith("3 Wrongly")}

    # ── Filter 2: drop non-neoplastic -- 2-class task only ──────────────────
    two_class_hashes = {
        h for h, r in fp_rows.items()
        if r["three_partition_label"] in LABEL_TO_IDX and h not in wrongly_labelled
    }

    # ── Filter 3: INNER JOIN -- only images with a real concept annotation ──
    keep_hashes = sorted(h for h in two_class_hashes if h in concept_vectors)
    print(f"[INFO] {len(fp_rows)} total -> {len(two_class_hashes)} after "
          f"(QC + non-neoplastic dropped) -> {len(keep_hashes)} after requiring real concept label "
          f"(inner join, matches official CRL scope)")

    missing_files = [h for h in keep_hashes if not (img_dir / f"{h}.jpg").exists()]
    if missing_files:
        print(f"[WARN] {len(missing_files)} images missing JPG file on disk, excluding them too.")
        keep_hashes = [h for h in keep_hashes if h not in set(missing_files)]

    # ── Near-dup grouping (our own addition, not in official CRL script) ────
    group_of = build_groups(img_dir, keep_hashes)
    label_of = {h: fp_rows[h]["three_partition_label"] for h in keep_hashes}

    n_groups = len(set(group_of.values()))
    print(f"[INFO] {len(keep_hashes)} images -> {n_groups} near-dup groups.")

    split_of = stratified_group_split(keep_hashes, group_of, label_of, args.seed)

    fieldnames = ["md5hash", "filename", "label", "label_idx", "concept_mask"] + concept_names
    split_stats = {}
    for split in SPLIT_RATIOS:
        rows = [h for h in keep_hashes if split_of[h] == split]
        with open(output_dir / f"{split}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for h in rows:
                writer.writerow({
                    "md5hash": h,
                    "filename": f"{h}.jpg",
                    "label": label_of[h],
                    "label_idx": LABEL_TO_IDX[label_of[h]],
                    "concept_mask": 1,  # always 1 here -- inner join guarantees a real label
                    **{name: v for name, v in zip(concept_names, concept_vectors[h])},
                })
        label_dist = Counter(label_of[h] for h in rows)
        split_stats[split] = {
            "n": len(rows),
            "label_dist_pct": {k: round(v / len(rows) * 100, 1) for k, v in label_dist.items()},
        }
        print(f"[INFO] {split}: n={len(rows)}  label={split_stats[split]['label_dist_pct']}")

    meta = {
        "source": str(data_dir),
        "purpose": "Matches official CRL repo (obiyoag/crl) data scope for apples-to-apples "
                   "accuracy comparison against ICRL -- see module docstring for exact filters "
                   "replicated / deliberately not replicated.",
        "n_total_csv_rows": len(fp_rows),
        "n_kept": len(keep_hashes),
        "num_concepts": len(concept_names),
        "concept_names": concept_names,
        "label_names": list(LABEL_TO_IDX.keys()),
        "split_ratios_target": SPLIT_RATIOS,
        "n_near_dup_groups": n_groups,
        "split_stats": split_stats,
        "seed": args.seed,
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Saved train.csv / val.csv / test.csv / meta.json to {output_dir}")


def build_groups(img_dir: Path, hashes_ordered: list[str]) -> dict[str, int]:
    files = [h + ".jpg" for h in hashes_ordered]
    print(f"[INFO] Computing dHash for {len(files)} images...")
    hashes = compute_dhashes(img_dir, files)
    strict_pairs, loose_pairs = find_near_duplicates(hashes)
    print(f"[INFO] {len(strict_pairs)} strict pairs, {len(loose_pairs)} loose pairs -> grouping")

    uf = UnionFind(len(hashes_ordered))
    for i, j, _ in strict_pairs + loose_pairs:
        uf.union(i, j)

    group_of = {}
    for root, members in uf.groups().items():
        for m in members:
            group_of[hashes_ordered[m]] = root
    return group_of


if __name__ == "__main__":
    main()
