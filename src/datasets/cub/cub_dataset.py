"""
cub_dataset.py — Dataset cho CUB-200-2011
=========================================

Cùng contract (image, labels_dict) với PadUfes20Dataset để dùng lại nguyên các
vòng train/eval của PAD-UFES-20 (concept_mask dạng vector theo từng concept):
    labels_dict["concepts"]     FloatTensor[num_concepts]  multi-hot
    labels_dict["concept_mask"] FloatTensor[num_concepts]  luôn 1 (class-level, không thiếu)
    labels_dict["label"]        scalar long (0..199)

Index CSV (output của src/scripts/cub/prepare_dataset.py) có 3 cột:
    filename   đường dẫn tương đối dưới thư mục images/ của CUB (vd. 001.Black_footed_Albatross/xxx.jpg)
    label_idx  0..199
    concepts   chuỗi 0/1 dài num_concepts, theo thứ tự meta.json["concept_names"]

Transform riêng cho chim (build_cub_transforms): không lật dọc, không xoay như
bản Fitzpatrick (ảnh da liễu không có chiều trên/dưới, ảnh chim thì có).
Augment theo tinh thần Koh 2020: RandomResizedCrop + lật ngang + ColorJitter.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.cub_concepts import load_schema

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_cub_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CubDataset(Dataset):
    def __init__(self, index_csv: str | Path, img_dir: str | Path, transform=None,
                 concept_names: list[str] | None = None):
        self.index_csv = Path(index_csv)
        self.img_dir = Path(img_dir)
        self.transform = transform
        schema = load_schema(self.index_csv.parent)
        self.concept_names = concept_names if concept_names is not None else schema["concept_names"]
        self.label_names = schema["label_names"]
        with open(self.index_csv, encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        n = len(self.concept_names)
        bad = [r["filename"] for r in self.rows if len(r["concepts"]) != n]
        if bad:
            raise ValueError(f"{self.index_csv}: {len(bad)} rows have a concept string whose length "
                             f"differs from {n} concept names (first: {bad[0]})")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(self.img_dir / row["filename"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        concepts = torch.tensor([float(ch) for ch in row["concepts"]], dtype=torch.float32)
        labels = {
            "concepts": concepts,
            "concept_mask": torch.ones_like(concepts),
            "label": torch.tensor(int(row["label_idx"]), dtype=torch.long),
        }
        return img, labels
