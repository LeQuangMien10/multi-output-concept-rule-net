"""
rebuild_icrl_cached.py — Dựng lại ICRL cho CUB từ logit S1 đã cache
====================================================================

Chạy NGUYÊN pipeline src/scripts/cub/train_icrl.py (tức pad_ufes20/train_icrl.py:
đo θ, build rule memory 3 lượt, quét n_min, train head, chấm accuracy rule trên
val, prune conf_min, dedupe, xuất rule), chỉ thay hai thứ:
  - load_system1 -> CachedS1: trả về logit đã cache theo chỉ số ảnh, thay vì
    chạy ResNet-50 trên ảnh;
  - make_loaders -> DataLoader trên chỉ số ảnh (train shuffle như bản gốc,
    val/test giữ thứ tự), dataset mang đúng concept_names/label_names.
Mục đích: quét θ trên CPU trong vài phút mỗi giá trị thay vì phải chạy lại S1.

Mọi tham số còn lại truyền thẳng cho train_icrl (vd --theta, --theta_merge,
--n_min_sweep, --exclude_label_slot, --output_dir).

Usage:
    python -m src.scripts.cub.rebuild_icrl_cached --cache outputs/cub_outputs/s1_cache.pt \\
        --output_dir outputs/cub_icrl_theta/theta_0.90 --theta 0.90 --theta_merge 0.94 \\
        --n_min_sweep 2,3,5,8,12 --exclude_label_slot --device cpu
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import src.scripts.pad_ufes20.train_icrl as base
from src.scripts.cub.train_icrl import inject_defaults
from src.scripts.cub.train_system1_baseline import _arg_value

SPLITS = ("train", "val", "test")


class CachedS1(nn.Module):
    """Input: LongTensor chỉ số toàn cục (train, rồi val, rồi test). Output: cùng
    dict {"concepts", "label"} logit như FitzpatrickSystem1."""

    def __init__(self, blob: dict):
        super().__init__()
        self.register_buffer("concepts", torch.cat([blob[s]["concepts"] for s in SPLITS]))
        self.register_buffer("label", torch.cat([blob[s]["label"] for s in SPLITS]))

    def forward(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"concepts": self.concepts[idx], "label": self.label[idx]}


class IndexDataset(Dataset):
    def __init__(self, blob: dict, split: str):
        self.offset = sum(len(blob[s]["y"]) for s in SPLITS[:SPLITS.index(split)])
        self.y = blob[split]["y"]
        self.concept_names = blob["concept_names"]
        self.label_names = blob["label_names"]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return torch.tensor(self.offset + i), {"label": self.y[i]}


def main():
    cache = _arg_value(sys.argv, "--cache")
    if cache is None:
        raise SystemExit("--cache is required")
    i = sys.argv.index("--cache") if "--cache" in sys.argv else None
    argv = sys.argv[:i] + sys.argv[i + 2:] if i is not None else [a for a in sys.argv if not a.startswith("--cache=")]
    blob = torch.load(cache, weights_only=False)

    def load_system1(_ckpt_path, device):
        return CachedS1(blob).to(device).eval(), blob["image_size"]

    def make_loaders(_data_dir, _img_dir, _image_size, batch_size, _num_workers):
        def _loader(split, shuffle):
            return DataLoader(IndexDataset(blob, split), batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return _loader("train", True), _loader("val", False), _loader("test", False)

    if _arg_value(argv, "--system1_ckpt") is None:
        argv += ["--system1_ckpt", cache]
    sys.argv = inject_defaults(argv)
    base.load_system1 = load_system1
    base.make_loaders = make_loaders
    base.main()


if __name__ == "__main__":
    main()
