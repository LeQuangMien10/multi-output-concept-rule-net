"""
prepare_dataset.py — Bước 1: chuẩn bị index train/val/test cho Fitzpatrick17k
================================================================================

KHÔNG train gì, KHÔNG đụng vào ảnh gốc. Chỉ đọc 2 CSV + hash ảnh, rồi xuất ra
3 file index (train/val/test.csv) để Dataset (Bước 3) load ảnh trực tiếp từ
JPG lúc training — không bake ảnh thật (kích thước lệch 130-2825px, RGB) thành
1 tensor .pt khổng lồ như đã làm với MNIST-MultiConcept (ảnh đó nhỏ, đồng nhất,
hợp lý để bake; ảnh thật thì không).

Các quyết định đã chốt (xem trao đổi thiết kế trước khi triển khai):
  1. Loại 17 ảnh "Wrongly labelled" (qc) + 460 ảnh "Do not consider this image"
     (SkinCon) — dữ liệu nhiễu do chính tác giả gốc đánh dấu.
  2. Concept: dùng 35/48 concept SkinCon (đã bỏ 13 concept hiếm <20 ảnh —
     xem src/utils/fitzpatrick_concepts.py::DROPPED_RARE_CONCEPTS).
  3. Nhãn: three_partition_label (3 lớp) — khớp baseline paper gốc (~62.4% acc).
  4. Split train/val/test: GOM theo near-duplicate (dHash strict+loose, xem
     src/utils/fitzpatrick_dedup.py) trước khi chia, để 1 lesion không vừa có
     ảnh ở train vừa có ở test/val (leak). Sau đó stratify theo
     (three_partition_label, có SkinCon hay không) để tỉ lệ nhãn và tỉ lệ
     concept-coverage đồng đều giữa 3 split.

Usage:
    python -m src.scripts.fitzpatrick.prepare_dataset \\
        --data_dir data/fitzpatrick17k --output_dir data/fitzpatrick17k_prepared
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.utils.fitzpatrick_concepts import (
    CONCEPT_NAMES,
    DROPPED_RARE_CONCEPTS,
    LABEL_TO_IDX,
    QC_WRONGLY_LABELLED_PREFIX,
)
from src.utils.fitzpatrick_dedup import UnionFind, compute_dhashes, find_near_duplicates

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def parse_args():
    p = argparse.ArgumentParser(description="Chuẩn bị index train/val/test cho Fitzpatrick17k.")
    p.add_argument("--data_dir", type=str, default="data/fitzpatrick17k",
                    help="Thư mục chứa fitzpatrick17k.csv, skincon.csv, data/finalfitz17k/*.jpg")
    p.add_argument("--output_dir", type=str, default="data/fitzpatrick17k_prepared")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_fitzpatrick_csv(data_dir: Path) -> dict[str, dict]:
    with open(data_dir / "fitzpatrick17k.csv", encoding="utf-8") as f:
        return {r["md5hash"]: r for r in csv.DictReader(f)}


def load_skincon_csv(data_dir: Path) -> list[dict]:
    with open(data_dir / "skincon.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_concept_vectors(skincon_rows: list[dict]) -> dict[str, list[int]]:
    """hash -> concept vector [NUM_CONCEPTS] (0/1), chỉ cho ảnh usable (không bị
    'Do not consider this image')."""
    out = {}
    for r in skincon_rows:
        if r["Do not consider this image"] == "1":
            continue
        h = r["ImageID"].replace(".jpg", "")
        out[h] = [int(r[c]) for c in CONCEPT_NAMES]
    return out


def build_groups(img_dir: Path, hashes_ordered: list[str]) -> dict[str, int]:
    """Chạy dHash + union-find trên toàn bộ ảnh usable -> hash -> group_id.
    Dùng cả strict lẫn loose pair để gộp nhóm (ưu tiên an toàn, tránh leak hơn
    là tối ưu kích thước train)."""
    files = [h + ".jpg" for h in hashes_ordered]
    print(f"[INFO] Computing dHash for {len(files)} images (a few minutes)...")
    hashes = compute_dhashes(img_dir, files)
    strict_pairs, loose_pairs = find_near_duplicates(hashes)
    print(f"[INFO] {len(strict_pairs)} strict pairs, {len(loose_pairs)} loose pairs -> grouping")

    uf = UnionFind(len(hashes_ordered))
    for i, j, _ in strict_pairs + loose_pairs:
        uf.union(i, j)

    idx_to_hash = hashes_ordered
    group_of = {}
    for root, members in uf.groups().items():
        for m in members:
            group_of[idx_to_hash[m]] = root
    return group_of


def stratified_group_split(
    all_hashes: list[str],
    group_of: dict[str, int],
    label_of: dict[str, str],
    has_concept: dict[str, bool],
    seed: int,
) -> dict[str, str]:
    """Chia all_hashes vào train/val/test theo nhóm (near-dup) — cả nhóm đi
    cùng 1 split. Stratify bằng round-robin theo key (label, has_concept) để
    giữ tỉ lệ nhãn + tỉ lệ concept-coverage gần đúng SPLIT_RATIOS ở mọi split."""
    rng = random.Random(seed)

    # Gom hash theo group.
    members_of_group: dict[int, list[str]] = defaultdict(list)
    for h in all_hashes:
        members_of_group[group_of[h]].append(h)

    # Key phân tầng của 1 nhóm = (label, has_concept) đa số trong nhóm.
    def group_key(members: list[str]) -> tuple[str, bool]:
        labels = Counter(label_of[h] for h in members)
        concepts = Counter(has_concept[h] for h in members)
        return labels.most_common(1)[0][0], concepts.most_common(1)[0][0]

    groups_by_key: dict[tuple, list[list[str]]] = defaultdict(list)
    for gid, members in members_of_group.items():
        groups_by_key[group_key(members)].append(members)

    split_assignment: dict[str, str] = {}

    for key, group_list in groups_by_key.items():
        rng.shuffle(group_list)
        # Round-robin theo tỉ lệ mục tiêu TRONG PHẠM VI đúng key này, để mỗi
        # (label, has_concept) đều được rải đều 3 split theo đúng SPLIT_RATIOS.
        key_total = sum(len(m) for m in group_list)
        target = {s: r * key_total for s, r in SPLIT_RATIOS.items()}
        assigned = {s: 0 for s in SPLIT_RATIOS}
        for members in group_list:
            # split đang thiếu hụt nhiều nhất so với target (theo tỉ lệ) được ưu tiên.
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
    skincon_rows = load_skincon_csv(data_dir)
    concept_vectors = build_concept_vectors(skincon_rows)

    # ── Loại ảnh nhiễu ────────────────────────────────────────
    wrongly_labelled = {h for h, r in fp_rows.items() if r["qc"].startswith(QC_WRONGLY_LABELLED_PREFIX)}
    keep_hashes = sorted(h for h in fp_rows if h not in wrongly_labelled)
    print(f"[INFO] Excluded {len(wrongly_labelled)} 'Wrongly labelled' images. Kept {len(keep_hashes)}/{len(fp_rows)}.")

    missing_files = [h for h in keep_hashes if not (img_dir / f"{h}.jpg").exists()]
    if missing_files:
        print(f"[WARN] {len(missing_files)} images missing JPG file on disk, excluding them too.")
        keep_hashes = [h for h in keep_hashes if h not in set(missing_files)]

    # ── Gom nhóm near-duplicate trên đúng tập ảnh giữ lại ──────
    group_of = build_groups(img_dir, keep_hashes)

    # ── Chuẩn bị label/concept cho từng ảnh ────────────────────
    label_of = {h: fp_rows[h]["three_partition_label"] for h in keep_hashes}
    has_concept = {h: h in concept_vectors for h in keep_hashes}

    n_groups = len(set(group_of.values()))
    n_multi_member_groups = sum(1 for g in Counter(group_of.values()).values() if g > 1)
    print(f"[INFO] {len(keep_hashes)} images -> {n_groups} near-dup groups "
          f"({n_multi_member_groups} groups with >1 image).")

    # ── Chia split ──────────────────────────────────────────────
    split_of = stratified_group_split(keep_hashes, group_of, label_of, has_concept, args.seed)

    # ── Xuất index CSV mỗi split ────────────────────────────────
    fieldnames = ["md5hash", "filename", "label", "label_idx", "concept_mask"] + CONCEPT_NAMES
    split_stats = {}
    for split in SPLIT_RATIOS:
        rows = [h for h in keep_hashes if split_of[h] == split]
        with open(output_dir / f"{split}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for h in rows:
                cv = concept_vectors.get(h, [0] * len(CONCEPT_NAMES))
                writer.writerow({
                    "md5hash": h,
                    "filename": f"{h}.jpg",
                    "label": label_of[h],
                    "label_idx": LABEL_TO_IDX[label_of[h]],
                    "concept_mask": int(has_concept[h]),
                    **{name: v for name, v in zip(CONCEPT_NAMES, cv)},
                })

        label_dist = Counter(label_of[h] for h in rows)
        n_concept = sum(has_concept[h] for h in rows)
        split_stats[split] = {
            "n": len(rows),
            "label_dist_pct": {k: round(v / len(rows) * 100, 1) for k, v in label_dist.items()},
            "concept_coverage_pct": round(n_concept / len(rows) * 100, 2),
        }
        print(f"[INFO] {split}: n={len(rows)}  label={split_stats[split]['label_dist_pct']}  "
              f"concept_coverage={split_stats[split]['concept_coverage_pct']}%")

    meta = {
        "source": str(data_dir),
        "n_total_csv_rows": len(fp_rows),
        "n_wrongly_labelled_excluded": len(wrongly_labelled),
        "n_missing_files_excluded": len(missing_files),
        "n_kept": len(keep_hashes),
        "num_concepts": len(CONCEPT_NAMES),
        "concept_names": CONCEPT_NAMES,
        "dropped_rare_concepts": DROPPED_RARE_CONCEPTS,
        "label_names": list(LABEL_TO_IDX.keys()),
        "split_ratios_target": SPLIT_RATIOS,
        "n_near_dup_groups": n_groups,
        "n_near_dup_groups_with_gt1_member": n_multi_member_groups,
        "split_stats": split_stats,
        "seed": args.seed,
        "note": "Split theo nhóm near-dup (dHash strict+loose) rồi stratify theo "
                "(three_partition_label, có SkinCon concept hay không). Ảnh thật KHÔNG "
                "được copy/resize ở bước này -- Dataset (Bước 3) đọc trực tiếp từ "
                "data/finalfitz17k/ qua cột 'filename'.",
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Saved train.csv / val.csv / test.csv / meta.json to {output_dir}")


if __name__ == "__main__":
    main()
