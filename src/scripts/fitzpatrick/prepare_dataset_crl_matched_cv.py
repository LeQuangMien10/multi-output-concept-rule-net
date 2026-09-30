"""
prepare_dataset_crl_matched_cv.py - k-fold CV variant of
prepare_dataset_crl_matched.py, so ICRL's CRL-matched evaluation can be
compared against CRL's own published numbers (5-fold CV mean+-std) instead
of a single fixed train/val/test split.

Reuses the exact same filtering as prepare_dataset_crl_matched.py (QC drop,
non-neoplastic drop, inner join with real SkinCon concepts) by importing its
helper functions directly -- the two scripts must stay in lockstep on WHICH
images are in scope, only the split logic differs.

Two --protocol modes, since they answer different questions:

  official (default): replicates CRL's own split_dataset.py exactly --
    images are grouped by label, then assigned to folds by a per-label
    round-robin cycle (fold_idx = i % k, in sorted-hash order as a
    deterministic stand-in for their original dataframe row order, since
    we don't have their exact row ordering). No near-duplicate grouping
    (they don't do this either -- a real leakage risk in their own
    pipeline, kept faithfully here rather than silently fixed). val and
    test are the SAME set for a given fold (their protocol has no
    separate held-out validation split within a fold): train.csv is the
    other k-1 folds, val.csv and test.csv are byte-identical copies of
    the held-out fold. Use this to compare ICRL against CRL under their
    exact evaluation protocol.

  grouped: our own stricter protocol -- near-dup groups are kept whole
    within a single fold (avoiding the leakage risk above), and each
    fold's non-test portion is further split into disjoint train/val
    (--val_frac), so the reported test accuracy never touches any data
    used to pick a checkpoint or threshold. Use this to see how much
    CRL's numbers might benefit from val==test, by evaluating ICRL (or,
    later, a reproduction of CRL) under a fair, leakage-free protocol.

Usage:
    python -m src.scripts.fitzpatrick.prepare_dataset_crl_matched_cv \\
        --data_dir data/fitzpatrick17k --protocol official \\
        --output_dir data/fitzpatrick17k_crl_matched_cv_official
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from src.scripts.fitzpatrick.prepare_dataset_crl_matched import (
    LABEL_TO_IDX,
    build_concept_vectors,
    build_groups,
    load_fitzpatrick_csv,
    load_skincon_csv,
)


def parse_args():
    p = argparse.ArgumentParser(description="Prep Fitzpatrick17k CRL-matched scope as k-fold CV.")
    p.add_argument("--data_dir", type=str, default="data/fitzpatrick17k")
    p.add_argument("--output_dir", type=str, default=None,
                    help="Default: data/fitzpatrick17k_crl_matched_cv_<protocol>.")
    p.add_argument("--protocol", type=str, default="official", choices=["official", "grouped"],
                    help="'official' replicates CRL's own split_dataset.py (val==test, no dedup "
                         "grouping). 'grouped' is our own stricter protocol (held-out val, "
                         "near-dup-group-safe). See module docstring.")
    p.add_argument("--k", type=int, default=5, help="Number of CV folds (CRL uses 5).")
    p.add_argument("--val_frac", type=float, default=0.15,
                    help="'grouped' protocol only: fraction of each fold's train+val pool held "
                         "out as validation.")
    p.add_argument("--seed", type=int, default=42,
                    help="Seed for the fold PARTITION itself (which images land in which fold), "
                         "not the training randomness of downstream S1/ICRL runs. Unused by "
                         "'official' (its round-robin assignment is deterministic).")
    return p.parse_args()


def round_robin_fold_split(hashes: list[str], label_of: dict[str, str], k: int) -> dict[str, int]:
    """CRL's own protocol: group by label, then cycle fold_idx = i % k within
    each label group. `hashes` must already be in a fixed deterministic
    order (sorted by hash here) since we don't have their original
    dataframe row order to replicate bit-for-bit -- this matches their
    ALGORITHM (label-stratified round robin), not necessarily their exact
    fold membership."""
    label_groups: dict[str, list[str]] = defaultdict(list)
    for h in hashes:
        label_groups[label_of[h]].append(h)
    fold_of: dict[str, int] = {}
    for label, members in label_groups.items():
        for i, h in enumerate(members):
            fold_of[h] = i % k
    return fold_of


def balanced_group_split(hashes: list[str], group_of: dict[str, int], label_of: dict[str, str],
                          ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Generic group-aware, label-stratified split into named buckets sized
    by `ratios` (must sum to ~1.0). Same deficit-maximization heuristic as
    prepare_dataset_crl_matched.py's stratified_group_split, generalized to
    an arbitrary bucket-name/ratio dict so it can build both the k-fold
    partition and the per-fold train/val carve-out ('grouped' protocol)."""
    rng = random.Random(seed)
    members_of_group: dict[int, list[str]] = defaultdict(list)
    for h in hashes:
        members_of_group[group_of[h]].append(h)

    def group_key(members: list[str]) -> str:
        return Counter(label_of[h] for h in members).most_common(1)[0][0]

    groups_by_key: dict[str, list[list[str]]] = defaultdict(list)
    for members in members_of_group.values():
        groups_by_key[group_key(members)].append(members)

    assignment: dict[str, str] = {}
    for key, group_list in groups_by_key.items():
        rng.shuffle(group_list)
        key_total = sum(len(m) for m in group_list)
        target = {s: r * key_total for s, r in ratios.items()}
        assigned = {s: 0 for s in ratios}
        for members in group_list:
            deficit = {s: target[s] - assigned[s] for s in ratios}
            best_bucket = max(deficit, key=lambda s: deficit[s])
            for h in members:
                assignment[h] = best_bucket
            assigned[best_bucket] += len(members)
    return assignment


def write_split_csv(path: Path, hashes: list[str], label_of: dict[str, str],
                     concept_names: list[str], concept_vectors: dict[str, list[int]]) -> dict:
    fieldnames = ["md5hash", "filename", "label", "label_idx", "concept_mask"] + concept_names
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in hashes:
            writer.writerow({
                "md5hash": h,
                "filename": f"{h}.jpg",
                "label": label_of[h],
                "label_idx": LABEL_TO_IDX[label_of[h]],
                "concept_mask": 1,
                **{name: v for name, v in zip(concept_names, concept_vectors[h])},
            })
    label_dist = Counter(label_of[h] for h in hashes)
    return {
        "n": len(hashes),
        "label_dist_pct": {k: round(v / len(hashes) * 100, 1) for k, v in label_dist.items()},
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    img_dir = data_dir / "data" / "finalfitz17k"
    output_dir = Path(args.output_dir or f"data/fitzpatrick17k_crl_matched_cv_{args.protocol}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Protocol: {args.protocol}")
    print("[INFO] Loading CSVs (same filters as prepare_dataset_crl_matched.py)...")
    fp_rows = load_fitzpatrick_csv(data_dir)
    concept_names, skincon_rows = load_skincon_csv(data_dir)
    concept_vectors = build_concept_vectors(skincon_rows, concept_names)

    wrongly_labelled = {h for h, r in fp_rows.items() if r["qc"].startswith("3 Wrongly")}
    two_class_hashes = {
        h for h, r in fp_rows.items()
        if r["three_partition_label"] in LABEL_TO_IDX and h not in wrongly_labelled
    }
    keep_hashes = sorted(h for h in two_class_hashes if h in concept_vectors)
    missing_files = [h for h in keep_hashes if not (img_dir / f"{h}.jpg").exists()]
    if missing_files:
        print(f"[WARN] {len(missing_files)} images missing JPG file on disk, excluding them too.")
        keep_hashes = [h for h in keep_hashes if h not in set(missing_files)]

    label_of = {h: fp_rows[h]["three_partition_label"] for h in keep_hashes}

    fold_stats = {}

    if args.protocol == "official":
        fold_of = round_robin_fold_split(keep_hashes, label_of, args.k)
        for i in range(args.k):
            fold_dir = output_dir / f"fold_{i}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            held_out = [h for h in keep_hashes if fold_of[h] == i]
            train_hashes = [h for h in keep_hashes if fold_of[h] != i]

            stats = {"train": write_split_csv(fold_dir / "train.csv", train_hashes, label_of,
                                               concept_names, concept_vectors),
                     "test": write_split_csv(fold_dir / "test.csv", held_out, label_of,
                                              concept_names, concept_vectors)}
            shutil.copy2(fold_dir / "test.csv", fold_dir / "val.csv")
            stats["val"] = stats["test"]
            fold_stats[f"fold_{i}"] = stats
            print(f"[INFO] fold_{i}: train={stats['train']['n']}  val=test={stats['test']['n']}  "
                  f"held_out_label={stats['test']['label_dist_pct']}")

        all_held_out = [h for i in range(args.k) for h in keep_hashes if fold_of[h] == i]
        assert len(all_held_out) == len(keep_hashes) == len(set(all_held_out)), \
            "official protocol: every image must appear in exactly one fold's held-out set"

    else:  # grouped
        group_of = build_groups(img_dir, keep_hashes)
        n_groups = len(set(group_of.values()))
        print(f"[INFO] {len(keep_hashes)} images -> {n_groups} near-dup groups.")

        fold_ratios = {f"fold{i}": 1.0 / args.k for i in range(args.k)}
        fold_of = balanced_group_split(keep_hashes, group_of, label_of, fold_ratios, args.seed)

        for i in range(args.k):
            fold_dir = output_dir / f"fold_{i}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            test_hashes = [h for h in keep_hashes if fold_of[h] == f"fold{i}"]
            pool_hashes = [h for h in keep_hashes if fold_of[h] != f"fold{i}"]

            pool_ratios = {"train": 1.0 - args.val_frac, "val": args.val_frac}
            pool_assignment = balanced_group_split(pool_hashes, group_of, label_of, pool_ratios,
                                                    seed=args.seed * 1000 + i)
            train_hashes = [h for h in pool_hashes if pool_assignment[h] == "train"]
            val_hashes = [h for h in pool_hashes if pool_assignment[h] == "val"]

            stats = {}
            for split_name, hashes in (("train", train_hashes), ("val", val_hashes), ("test", test_hashes)):
                stats[split_name] = write_split_csv(fold_dir / f"{split_name}.csv", hashes, label_of,
                                                     concept_names, concept_vectors)
            fold_stats[f"fold_{i}"] = stats
            print(f"[INFO] fold_{i}: train={stats['train']['n']} val={stats['val']['n']} "
                  f"test={stats['test']['n']}  test_label={stats['test']['label_dist_pct']}")

            test_groups = {group_of[h] for h in test_hashes}
            pool_groups = {group_of[h] for h in pool_hashes}
            overlap = test_groups & pool_groups
            assert not overlap, f"fold_{i}: {len(overlap)} near-dup groups leak between test and train/val pool"

    meta = {
        "source": str(data_dir),
        "protocol": args.protocol,
        "purpose": "k-fold CV variant of fitzpatrick17k_crl_matched, for comparison against "
                   "CRL's own 5-fold evaluation protocol -- see module docstring for what "
                   "'official' vs 'grouped' each does and does not replicate.",
        "k": args.k,
        "val_frac": args.val_frac if args.protocol == "grouped" else None,
        "n_kept": len(keep_hashes),
        "num_concepts": len(concept_names),
        "concept_names": concept_names,
        "label_names": list(LABEL_TO_IDX.keys()),
        "seed": args.seed,
        "fold_stats": fold_stats,
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Saved fold_0..fold_{args.k - 1}/{{train,val,test}}.csv and meta.json to {output_dir}")


if __name__ == "__main__":
    main()
