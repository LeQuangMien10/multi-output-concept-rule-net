"""
fitzpatrick_dataset.py — Dataset/transform cho Fitzpatrick17k (ảnh thật)
============================================================================

Khác MNISTMultiConceptPTDataset: ảnh KHÔNG bake sẵn vào .pt (kích thước lệch
130-2825px, RGB, ~16.5k ảnh — quá nặng để load hết vào RAM/1 tensor). Đọc JPG
trực tiếp mỗi __getitem__, giống pattern chuẩn khi fine-tune backbone
pretrained ImageNet.

Augmentation theo đúng policy paper gốc Groh et al. 2021 (RandomResizedCrop
scale 0.8-1.0, xoay ±15°, lật ngang/dọc) — không tự bịa augmentation mới, để
số liệu train/val còn so sánh được với baseline ~62.4% accuracy của paper gốc.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.fitzpatrick_concepts import CONCEPT_NAMES, NUM_CONCEPTS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """split='train' -> augment đầy đủ. split khác -> resize + center-crop only
    (không augment, giống cách val/test luôn được đánh giá "sạch")."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),   # cạnh ngắn -> 256 nếu image_size=224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FitzpatrickDataset(Dataset):
    """
    Load từ index CSV do prepare_dataset.py xuất ra (train.csv/val.csv/test.csv).

    Mỗi item trả về (image, labels_dict) — CÙNG CONTRACT với
    MNISTMultiConceptPTDataset để train_system1_baseline.py dùng lại được
    gần như nguyên logic training loop:
        labels_dict["concepts"]     FloatTensor[NUM_CONCEPTS]  multi-hot
        labels_dict["concept_mask"] scalar float (0/1)
        labels_dict["label"]        scalar long (0/1/2, xem LABEL_TO_IDX)
    """

    _NON_CONCEPT_COLUMNS = {"md5hash", "filename", "label", "label_idx", "concept_mask"}

    def __init__(self, index_csv: str | Path, img_dir: str | Path, transform=None,
                 concept_names: list[str] | None = None):
        self.index_csv = Path(index_csv)
        self.img_dir = Path(img_dir)
        self.transform = transform

        with open(self.index_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)
            # Self-describing: any prepared variant (35-concept/3-class,
            # 48-concept/2-class CRL-matched, ...) just works, no import of a
            # fixed constant needed -- derive concept columns from whatever
            # this CSV actually has, in file order.
            csv_columns = reader.fieldnames or []
        self.concept_names = concept_names if concept_names is not None else [
            c for c in csv_columns if c not in self._NON_CONCEPT_COLUMNS
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(self.img_dir / row["filename"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        concepts = torch.tensor([float(row[c]) for c in self.concept_names], dtype=torch.float32)
        labels = {
            "concepts": concepts,
            "concept_mask": torch.tensor(float(row["concept_mask"])),
            "label": torch.tensor(int(row["label_idx"]), dtype=torch.long),
        }
        return img, labels


assert NUM_CONCEPTS == len(CONCEPT_NAMES)
