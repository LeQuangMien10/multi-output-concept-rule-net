"""
pad_ufes20_dataset.py — Dataset cho PAD-UFES-20 (ảnh thật)
============================================================================

Cùng pattern I/O với FitzpatrickDataset (src/datasets/fitzpatrick/
fitzpatrick_dataset.py) -- đọc PNG trực tiếp mỗi __getitem__ từ index CSV do
prepare_dataset.py xuất ra. build_transforms dùng lại nguyên xi từ module đó
(ImageNet-normalize augmentation, không có gì đặc thù riêng cho Fitzpatrick).

Khác FitzpatrickDataset ở đúng 1 điểm: concept_mask là VECTOR
[NUM_CONCEPTS] thay vì 1 scalar/ảnh, vì UNK ở PAD-UFES-20 xảy ra theo TỪNG
concept riêng lẻ (1 ảnh có thể biết "itch" nhưng UNK "grew") -- xem
src/utils/pad_ufes20_concepts.py.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.pad_ufes20_concepts import CONCEPT_NAMES, NUM_CONCEPTS


class PadUfes20Dataset(Dataset):
    """
    Mỗi item trả về (image, labels_dict):
        labels_dict["concepts"]     FloatTensor[NUM_CONCEPTS]  multi-hot (0 khi UNK)
        labels_dict["concept_mask"] FloatTensor[NUM_CONCEPTS]  1=biết, 0=UNK (theo từng concept)
        labels_dict["label"]        scalar long (0..5, xem LABEL_TO_IDX)
    """

    def __init__(self, index_csv: str | Path, img_dir: str | Path, transform=None,
                 concept_names: list[str] | None = None):
        self.index_csv = Path(index_csv)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.concept_names = concept_names if concept_names is not None else CONCEPT_NAMES

        with open(self.index_csv, encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(self.img_dir / row["filename"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        concepts = torch.tensor([float(row[c]) for c in self.concept_names], dtype=torch.float32)
        concept_mask = torch.tensor([float(row[f"{c}_mask"]) for c in self.concept_names], dtype=torch.float32)
        labels = {
            "concepts": concepts,
            "concept_mask": concept_mask,
            "label": torch.tensor(int(row["label_idx"]), dtype=torch.long),
        }
        return img, labels


assert NUM_CONCEPTS == len(CONCEPT_NAMES)
