"""
prepare_dataset.py — Bước 1: chuẩn bị index train/val/test cho CUB-200-2011
============================================================================

KHÔNG train gì, KHÔNG đụng vào ảnh gốc. Đọc các file text của CUB-200-2011
(bản tải từ data.caltech.edu) rồi xuất index CSV + meta.json cho CubDataset.

Quy trình theo Koh et al. (ICML 2020, Section 3 và Appendix A.2):
  1. Split: giữ tập test chính thức (train_test_split.txt, 5,794 ảnh); chia tập
     train chính thức (5,994 ảnh) thành train/val theo tỉ lệ 80/20, phân tầng theo
     loài (Koh: 4,796 / 1,198; ở đây xấp xỉ, seed cố định).
  2. Concept class-level: với mỗi loài và mỗi attribute, majority vote của
     is_present trên các ảnh TRAIN (hòa thì tính là có mặt, giống code gốc của
     Koh). Mọi ảnh của loài đó, ở cả train/val/test, dùng chung vector này.
  3. Chọn concept: mặc định dùng đúng 112 chỉ số Koh đã chọn
     (KOH_SELECTED_CONCEPTS). Script cũng tự tính lại phép chọn "attribute có
     mặt ở >= 10 loài sau majority vote" và in mức trùng khớp để kiểm tra. Đặt
     --selection recompute để dùng phép chọn tự tính thay vì danh sách chuẩn.

--concept_level instance: dùng nhãn attribute GỐC theo từng ảnh (nhiễu) trên
cùng tập concept, thay cho bản class-level. Không phải cấu hình chuẩn của
literature, chỉ để thử sau nếu cần.

Bố cục đầu vào (thư mục --cub_dir là thư mục CUB_200_2011 sau khi giải nén):
    CUB_200_2011/images.txt, image_class_labels.txt, train_test_split.txt,
    classes.txt, attributes/image_attribute_labels.txt, images/...
    attributes.txt nằm ở thư mục CHA của CUB_200_2011 trong file tgz gốc
    (script tự tìm ở cả hai chỗ).

Usage (local):
    python -m src.scripts.cub.prepare_dataset \\
        --cub_dir data/CUB/CUB_200_2011 --output_dir data/CUB_prepared
Usage (Kaggle):
    python -m src.scripts.cub.prepare_dataset \\
        --cub_dir /kaggle/input/<dataset>/CUB_200_2011 --output_dir /kaggle/working/CUB_prepared
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils.cub_concepts import (
    KOH_SELECTED_CONCEPTS, MIN_CLASS_COUNT, NUM_CLASSES, NUM_RAW_ATTRIBUTES,
)

VAL_RATIO = 0.2


def parse_args():
    p = argparse.ArgumentParser(description="Chuẩn bị index train/val/test cho CUB-200-2011.")
    p.add_argument("--cub_dir", type=str, default="data/CUB/CUB_200_2011")
    p.add_argument("--output_dir", type=str, default="data/CUB_prepared")
    p.add_argument("--selection", type=str, default="koh", choices=["koh", "recompute"])
    p.add_argument("--concept_level", type=str, default="class", choices=["class", "instance"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_pairs(path: Path) -> dict[int, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                k, v = line.split(" ", 1)
                out[int(k)] = v
    return out


def find_attributes_txt(cub_dir: Path) -> Path:
    for cand in (cub_dir / "attributes" / "attributes.txt", cub_dir / "attributes.txt",
                 cub_dir.parent / "attributes.txt"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"attributes.txt not found next to or inside {cub_dir}")


def read_image_attributes(path: Path, num_images: int) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (is_present, certainty_id), mỗi mảng [num_images+1, 312] (chỉ số
    ảnh/attribute 1-based, hàng 0 bỏ trống). certainty_id: 1 not visible,
    2 guessing, 3 probably, 4 definitely. Một số dòng của file gốc có thừa cột,
    chỉ đọc 4 cột đầu."""
    present = np.full((num_images + 1, NUM_RAW_ATTRIBUTES), -1, dtype=np.int8)
    certainty = np.zeros((num_images + 1, NUM_RAW_ATTRIBUTES), dtype=np.int8)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            img_id, attr_id = int(parts[0]), int(parts[1])
            present[img_id, attr_id - 1] = int(parts[2])
            certainty[img_id, attr_id - 1] = int(parts[3])
    missing = int((present[1:] < 0).sum())
    if missing:
        raise ValueError(f"{missing} (image, attribute) cells missing in {path}")
    return present, certainty


def stratified_val_split(train_ids: list[int], label_of: dict[int, int], seed: int) -> set[int]:
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i in train_ids:
        by_class[label_of[i]].append(i)
    val = set()
    for c in sorted(by_class):
        ids = sorted(by_class[c])
        rng.shuffle(ids)
        val.update(ids[:round(len(ids) * VAL_RATIO)])
    return val


def main():
    args = parse_args()
    cub_dir, out_dir = Path(args.cub_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = read_pairs(cub_dir / "images.txt")
    label_of = {i: int(c) - 1 for i, c in read_pairs(cub_dir / "image_class_labels.txt").items()}
    is_train = {i: v == "1" for i, v in read_pairs(cub_dir / "train_test_split.txt").items()}
    class_names = [n for _, n in sorted(read_pairs(cub_dir / "classes.txt").items())]
    attr_names = [n for _, n in sorted(read_pairs(find_attributes_txt(cub_dir)).items())]
    assert len(class_names) == NUM_CLASSES and len(attr_names) == NUM_RAW_ATTRIBUTES
    num_images = len(paths)
    attrs, certainty = read_image_attributes(cub_dir / "attributes" / "image_attribute_labels.txt",
                                             num_images)

    official_train = sorted(i for i in paths if is_train[i])
    test_ids = sorted(i for i in paths if not is_train[i])
    val_ids = stratified_val_split(official_train, label_of, args.seed)
    train_ids = [i for i in official_train if i not in val_ids]
    split_of = {**{i: "train" for i in train_ids}, **{i: "val" for i in val_ids},
                **{i: "test" for i in test_ids}}

    # Majority vote theo loài trên TRAIN, theo code gốc của Koh
    # (ConceptBottleneck/CUB/generate_new_data.py, get_class_attributes_data):
    # bỏ qua phiếu "vắng mặt" khi attribute không nhìn thấy (certainty 1), và
    # hòa (kể cả khi không còn phiếu nào) thì tính là có mặt.
    pos = np.zeros((NUM_CLASSES, NUM_RAW_ATTRIBUTES), dtype=np.int64)
    neg = np.zeros((NUM_CLASSES, NUM_RAW_ATTRIBUTES), dtype=np.int64)
    for i in train_ids:
        counted = ~((attrs[i] == 0) & (certainty[i] == 1))
        pos[label_of[i]] += (attrs[i] == 1) & counted
        neg[label_of[i]] += (attrs[i] == 0) & counted
    class_attr = (pos >= neg).astype(np.int8)                       # [200, 312]

    recomputed = [a for a in range(NUM_RAW_ATTRIBUTES) if class_attr[:, a].sum() >= MIN_CLASS_COUNT]
    overlap = len(set(recomputed) & set(KOH_SELECTED_CONCEPTS))
    print(f"[INFO] Recomputed selection (>= {MIN_CLASS_COUNT} classes): {len(recomputed)} attributes; "
          f"overlap with Koh's 112 = {overlap}")
    selected = KOH_SELECTED_CONCEPTS if args.selection == "koh" else recomputed
    concept_names = [attr_names[a] for a in selected]

    rows = {"train": [], "val": [], "test": []}
    for i in sorted(paths):
        vec = class_attr[label_of[i], selected] if args.concept_level == "class" else attrs[i, selected]
        rows[split_of[i]].append({"filename": paths[i], "label_idx": label_of[i],
                                  "concepts": "".join(str(int(v)) for v in vec)})
    for split, rs in rows.items():
        with open(out_dir / f"{split}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "label_idx", "concepts"])
            w.writeheader()
            w.writerows(rs)

    sig = class_attr[:, selected]
    n_unique_signatures = len({tuple(r) for r in sig})
    meta = {
        "dataset": "CUB-200-2011",
        "concept_level": args.concept_level,
        "selection": args.selection,
        "concept_names": concept_names,
        "concept_attr_ids_0based": list(map(int, selected)),
        "label_names": class_names,
        "num_concepts": len(concept_names),
        "num_labels": NUM_CLASSES,
        "split_sizes": {k: len(v) for k, v in rows.items()},
        "seed": args.seed,
        "recomputed_selection_size": len(recomputed),
        "recomputed_overlap_with_koh": overlap,
        "class_level_concept_positive_rate": float(sig.mean()),
        "num_unique_class_signatures": n_unique_signatures,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[DONE] {out_dir}: train/val/test = {meta['split_sizes']}, {len(concept_names)} concepts, "
          f"{NUM_CLASSES} classes, positive rate {meta['class_level_concept_positive_rate']:.3f}, "
          f"{n_unique_signatures}/{NUM_CLASSES} distinct class concept signatures")


if __name__ == "__main__":
    main()
