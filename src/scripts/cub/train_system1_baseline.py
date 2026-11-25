"""
train_system1_baseline.py - Train S1 cho CUB-200-2011
=====================================================

Lớp bọc mỏng quanh src/scripts/pad_ufes20/train_system1_baseline.py: dùng lại
nguyên vòng train/eval, masked BCE theo từng concept, monitor
label_acc_gated và FitzpatrickSystem1 (ResNet + 2 đầu song song). Chỉ thay:
  - make_loaders -> CubDataset + build_cub_transforms (không lật dọc/xoay);
  - mặc định --num_concepts/--num_labels đọc từ <data_dir>/meta.json nếu không
    truyền vào (112 concept, 200 loài với cấu hình chuẩn);
  - mặc định đường dẫn local của CUB.

Usage:
    python -m src.scripts.cub.train_system1_baseline \\
        --data_dir data/CUB_prepared --img_dir data/CUB/CUB_200_2011/images \\
        --output_dir outputs/cub_system1 --epochs 40 --lr 1e-3 --lr_schedule cosine
"""
from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import DataLoader

import src.scripts.pad_ufes20.train_system1_baseline as base
from src.datasets.cub.cub_dataset import CubDataset, build_cub_transforms
from src.utils.cub_concepts import load_schema

DEFAULTS = {
    "--data_dir": "data/CUB_prepared",
    "--img_dir": "data/CUB/CUB_200_2011/images",
    "--output_dir": "outputs/cub_system1",
}


def make_loaders(data_dir: Path, img_dir: Path, image_size: int, batch_size: int, num_workers: int):
    def _ds(split):
        return CubDataset(Path(data_dir) / f"{split}.csv", img_dir, build_cub_transforms(split, image_size))

    train_loader = DataLoader(_ds("train"), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(_ds("val"), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(_ds("test"), batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def _arg_value(argv: list[str], flag: str) -> str | None:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def inject_defaults(argv: list[str]) -> list[str]:
    argv = list(argv)
    for flag, value in DEFAULTS.items():
        if _arg_value(argv, flag) is None:
            argv += [flag, value]
    schema = load_schema(_arg_value(argv, "--data_dir"))
    if _arg_value(argv, "--num_concepts") is None:
        argv += ["--num_concepts", str(len(schema["concept_names"]))]
    if _arg_value(argv, "--num_labels") is None:
        argv += ["--num_labels", str(len(schema["label_names"]))]
    return argv


def main():
    sys.argv = inject_defaults(sys.argv)
    base.make_loaders = make_loaders
    base.main()


if __name__ == "__main__":
    main()
