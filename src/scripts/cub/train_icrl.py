"""
train_icrl.py - ICRL Stage 2/3 cho CUB-200-2011
===============================================

Lớp bọc mỏng quanh src/scripts/pad_ufes20/train_icrl.py: dùng lại nguyên
Stage 2 (build rule memory), Stage 3 (head trên centroid, chấm accuracy rule
trên val, prune conf_min, dedupe), --n_min_sweep và --theta auto. Chỉ thay:
  - make_loaders -> CubDataset với transform sạch (không augment) cho cả 3
    split, cùng lý do như bản gốc: augment ngẫu nhiên làm nhiễu vector concept
    dùng để cluster;
  - mặc định --label_names đọc từ <data_dir>/meta.json (200 tên loài);
  - mặc định đường dẫn local của CUB.

Các hàm load_system1, build_full_concept_layout, make_loaders được export ở mức
module để các script đánh giá (eval_ablation_scope.py, ...) import theo cùng
quy ước "src.scripts.<module>.train_icrl" như Fitzpatrick/PAD-UFES-20.

Usage:
    python -m src.scripts.cub.train_icrl \\
        --data_dir data/CUB_prepared --img_dir data/CUB/CUB_200_2011/images \\
        --system1_ckpt outputs/cub_system1/best_model.pt --output_dir outputs/cub_icrl \\
        --theta auto --n_min_sweep 2,3,5,8,12 --exclude_label_slot
"""
from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import DataLoader

import src.scripts.pad_ufes20.train_icrl as base
from src.datasets.cub.cub_dataset import CubDataset, build_cub_transforms
from src.scripts.cub.train_system1_baseline import _arg_value
from src.utils.cub_concepts import load_schema

load_system1 = base.load_system1
build_full_concept_layout = base.build_full_concept_layout

DEFAULTS = {
    "--data_dir": "data/CUB_prepared",
    "--img_dir": "data/CUB/CUB_200_2011/images",
    "--system1_ckpt": "outputs/cub_system1/best_model.pt",
    "--output_dir": "outputs/cub_icrl",
}


def make_loaders(data_dir: Path, img_dir: Path, image_size: int, batch_size: int, num_workers: int):
    clean = build_cub_transforms("val", image_size)

    def _loader(split, shuffle):
        ds = CubDataset(Path(data_dir) / f"{split}.csv", img_dir, clean)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return _loader("train", True), _loader("val", False), _loader("test", False)


def inject_defaults(argv: list[str]) -> list[str]:
    argv = list(argv)
    for flag, value in DEFAULTS.items():
        if _arg_value(argv, flag) is None:
            argv += [flag, value]
    if _arg_value(argv, "--label_names") is None:
        names = load_schema(_arg_value(argv, "--data_dir"))["label_names"]
        assert not any("," in n for n in names), "label names must not contain commas"
        argv += ["--label_names", ",".join(names)]
    return argv


def main():
    sys.argv = inject_defaults(sys.argv)
    base.make_loaders = make_loaders
    base.main()


if __name__ == "__main__":
    main()
