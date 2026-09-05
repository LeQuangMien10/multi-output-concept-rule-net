"""
prepare_dataset.py — Bước 1: chuẩn bị index train/val/test cho PAD-UFES-20
================================================================================

KHÔNG train gì, KHÔNG đụng vào ảnh gốc. Đọc metadata.csv rồi xuất ra 3 file
index (train/val/test.csv) để Dataset (PadUfes20Dataset) load ảnh trực tiếp
từ PNG lúc training, theo đúng pattern của
src/scripts/fitzpatrick/prepare_dataset.py.

Các quyết định đã chốt (xem ghi chú dự án trước khi triển khai):
  1. Concept: 6 cột triệu chứng bệnh nhân tự khai có sẵn trong metadata.csv
     (itch, grew, hurt, changed, bleed, elevation) — xem
     src/utils/pad_ufes20_concepts.py.
  2. Nhãn: cột "diagnostic" gốc, giữ nguyên 6 lớp (không gộp benign/malignant).
  3. UNK xảy ra THEO TỪNG CONCEPT riêng lẻ (khác Fitzpatrick, nơi 1 ảnh hoặc
     có đủ concept hoặc không có concept nào) — mỗi concept có 1 cột mask
     riêng ("<name>_mask") thay vì 1 "concept_mask" chung cho cả ảnh.
  4. Split train/val/test: GOM theo (patient_id, lesion_id) trước khi chia,
     để 1 lesion (361 lesion có >1 ảnh, tối đa 6 ảnh/lesion) không vừa có
     ảnh ở train vừa ở val/test (leak). Stratify theo nhãn "diagnostic" để
     tỉ lệ 6 lớp đồng đều giữa 3 split.
  5. 804/2,298 dòng không có biopsy (chẩn đoán lâm sàng thuần, đợt thu thập
     riêng) thiếu nhiều cột nhân khẩu/lâm sàng KHÔNG liên quan tới 6 concept
     đã chọn -- giữ lại nguyên vẹn, không loại.

Usage:
    python -m src.scripts.pad_ufes20.prepare_dataset \\
        --data_dir data/PAD-UFES-20 --output_dir data/PAD-UFES-20_prepared
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.utils.pad_ufes20_concepts import CONCEPT_NAMES, LABEL_TO_IDX, UNK_VALUE

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def parse_args():
    p = argparse.ArgumentParser(description="Chuẩn bị index train/val/test cho PAD-UFES-20.")
    p.add_argument("--data_dir", type=str, default="data/PAD-UFES-20",
                    help="Thư mục chứa metadata.csv, images/*.png")
    p.add_argument("--output_dir", type=str, default="data/PAD-UFES-20_prepared")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_metadata_csv(data_dir: Path) -> list[dict]:
    with open(data_dir / "metadata.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def concept_value_and_mask(row: dict, concept: str) -> tuple[int, int]:
    """Cell rỗng cũng coi như UNK (phòng trường hợp encode khác 'UNK') --
    trong thực tế metadata.csv chỉ dùng 'UNK', không có cell rỗng cho 6 cột
    này, nhưng xử lý an toàn."""
    raw = row[concept]
    if raw == UNK_VALUE or raw == "":
        return 0, 0
    return (1 if raw == "True" else 0), 1


def stratified_group_split(
    group_ids: list[tuple[str, str]],
    label_of_group: dict[tuple[str, str], str],
    seed: int,
) -> dict[tuple[str, str], str]:
    """Chia các lesion-group vào train/val/test, round-robin theo label để
    giữ tỉ lệ 6 lớp gần đúng SPLIT_RATIOS ở mọi split — cùng thuật toán với
    stratified_group_split trong fitzpatrick/prepare_dataset.py, chỉ khác
    key phân tầng (ở đây chỉ có "diagnostic", không có "has_concept" vì
    concept ở PAD-UFES-20 luôn có sẵn theo từng field, không theo cả ảnh)."""
    rng = random.Random(seed)

    groups_by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for gid in group_ids:
        groups_by_label[label_of_group[gid]].append(gid)

    split_assignment: dict[tuple[str, str], str] = {}
    for label, gids in groups_by_label.items():
        rng.shuffle(gids)
        target = {s: r * len(gids) for s, r in SPLIT_RATIOS.items()}
        assigned = {s: 0 for s in SPLIT_RATIOS}
        for gid in gids:
            deficit = {s: target[s] - assigned[s] for s in SPLIT_RATIOS}
            best_split = max(deficit, key=lambda s: deficit[s])
            split_assignment[gid] = best_split
            assigned[best_split] += 1

    return split_assignment


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    img_dir = data_dir / "images"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading metadata.csv...")
    rows = load_metadata_csv(data_dir)
    print(f"[INFO] {len(rows)} rows.")

    missing_files = [r["img_id"] for r in rows if not (img_dir / r["img_id"]).exists()]
    if missing_files:
        print(f"[WARN] {len(missing_files)} images missing on disk, excluding them.")
        missing_set = set(missing_files)
        rows = [r for r in rows if r["img_id"] not in missing_set]

    # ── Gom theo lesion (patient_id, lesion_id) ─────────────────
    group_of_row: dict[str, tuple[str, str]] = {
        r["img_id"]: (r["patient_id"], r["lesion_id"]) for r in rows
    }
    rows_by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        rows_by_group[group_of_row[r["img_id"]]].append(r)

    label_of_group: dict[tuple[str, str], str] = {}
    for gid, members in rows_by_group.items():
        diags = {m["diagnostic"] for m in members}
        assert len(diags) == 1, f"Lesion {gid} has inconsistent diagnostic labels: {diags}"
        label_of_group[gid] = members[0]["diagnostic"]

    n_multi = sum(1 for m in rows_by_group.values() if len(m) > 1)
    print(f"[INFO] {len(rows)} images -> {len(rows_by_group)} lesions "
          f"({n_multi} lesions with >1 image).")

    # ── Chia split ──────────────────────────────────────────────
    split_of_group = stratified_group_split(list(rows_by_group.keys()), label_of_group, args.seed)

    # ── Xuất index CSV mỗi split ────────────────────────────────
    concept_fields = []
    for name in CONCEPT_NAMES:
        concept_fields += [name, f"{name}_mask"]
    fieldnames = ["img_id", "filename", "patient_id", "lesion_id", "label", "label_idx"] + concept_fields

    split_stats = {}
    for split in SPLIT_RATIOS:
        gids = [gid for gid in rows_by_group if split_of_group[gid] == split]
        split_rows = [r for gid in gids for r in rows_by_group[gid]]

        with open(output_dir / f"{split}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in split_rows:
                out_row = {
                    "img_id": r["img_id"],
                    "filename": r["img_id"],
                    "patient_id": r["patient_id"],
                    "lesion_id": r["lesion_id"],
                    "label": r["diagnostic"],
                    "label_idx": LABEL_TO_IDX[r["diagnostic"]],
                }
                for name in CONCEPT_NAMES:
                    value, mask = concept_value_and_mask(r, name)
                    out_row[name] = value
                    out_row[f"{name}_mask"] = mask
                writer.writerow(out_row)

        label_dist = Counter(r["diagnostic"] for r in split_rows)
        concept_coverage = {
            name: round(sum(concept_value_and_mask(r, name)[1] for r in split_rows) / len(split_rows) * 100, 1)
            for name in CONCEPT_NAMES
        }
        split_stats[split] = {
            "n": len(split_rows),
            "n_lesions": len(gids),
            "label_dist_pct": {k: round(v / len(split_rows) * 100, 1) for k, v in label_dist.items()},
            "concept_coverage_pct": concept_coverage,
        }
        print(f"[INFO] {split}: n={len(split_rows)}  n_lesions={len(gids)}  "
              f"label={split_stats[split]['label_dist_pct']}")
        print(f"       concept_coverage={concept_coverage}")

    meta = {
        "source": str(data_dir),
        "n_total_csv_rows": len(rows) + len(missing_files),
        "n_missing_files_excluded": len(missing_files),
        "n_kept": len(rows),
        "n_lesions": len(rows_by_group),
        "n_lesions_with_gt1_image": n_multi,
        "num_concepts": len(CONCEPT_NAMES),
        "concept_names": CONCEPT_NAMES,
        "label_names": list(LABEL_TO_IDX.keys()),
        "split_ratios_target": SPLIT_RATIOS,
        "split_stats": split_stats,
        "seed": args.seed,
        "note": "Split theo lesion (patient_id, lesion_id) rồi stratify theo "
                "'diagnostic' (6 lớp gốc). Concept mask theo TỪNG concept "
                "(<name>_mask), khác 1 concept_mask chung/ảnh của Fitzpatrick, "
                "vì UNK ở đây xảy ra theo từng field riêng lẻ. Ảnh KHÔNG được "
                "copy/resize ở bước này -- Dataset đọc trực tiếp từ images/ "
                "qua cột 'filename'.",
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Saved train.csv / val.csv / test.csv / meta.json to {output_dir}")


if __name__ == "__main__":
    main()
