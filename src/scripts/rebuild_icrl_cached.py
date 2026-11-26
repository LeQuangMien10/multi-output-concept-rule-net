"""
rebuild_icrl_cached.py - Dung lai ICRL tu logit S1 da cache, cho Fitzpatrick va CUB
===================================================================================

Chay NGUYEN pipeline train_icrl.py cua module duoc chon (do theta neu 'auto',
build rule memory, quet n_min, train head, cham accuracy rule tren val, prune
conf_min, dedupe, bo rule rong, c_r tren val), chi thay load_system1 bang
CachedS1 va make_loaders bang DataLoader tren chi so anh (giong
src/scripts/cub/rebuild_icrl_cached.py, ban nay them module fitzpatrick).

    --module fitzpatrick : src.scripts.fitzpatrick.train_icrl (can --label_names)
    --module cub         : src.scripts.cub.rebuild_icrl_cached (giu nguyen)

Usage:
    python -m src.scripts.rebuild_icrl_cached --module fitzpatrick \\
        --cache outputs/s1_cache/concept_only.pt --output_dir outputs/x \\
        --label_names benign,malignant,non-neoplastic --theta 0.95 --theta_merge 0.99 \\
        --n_min_sweep 2,3,5 --exclude_label_slot --device cpu
"""
from __future__ import annotations

import sys

import torch
from torch.utils.data import DataLoader

from src.scripts.cub.rebuild_icrl_cached import CachedS1, IndexDataset


def _pop(argv, name):
    if name not in argv:
        return argv, None
    i = argv.index(name)
    return argv[:i] + argv[i + 2:], argv[i + 1]


def main():
    argv, module = _pop(sys.argv, "--module")
    if module == "cub":
        sys.argv = argv
        from src.scripts.cub import rebuild_icrl_cached as cub
        return cub.main()
    if module != "fitzpatrick":
        raise SystemExit("--module must be fitzpatrick or cub")
    argv, cache = _pop(argv, "--cache")
    if cache is None:
        raise SystemExit("--cache is required")
    blob = torch.load(cache, weights_only=False)
    import src.scripts.fitzpatrick.train_icrl as base

    def load_system1(_ckpt_path, device):
        return CachedS1(blob).to(device).eval(), blob["image_size"]

    def make_loaders(_data_dir, _img_dir, _image_size, batch_size, _num_workers):
        def _loader(split, shuffle):
            return DataLoader(IndexDataset(blob, split), batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return _loader("train", True), _loader("val", False), _loader("test", False)

    for flag, val in (("--system1_ckpt", cache), ("--data_dir", "cached"), ("--img_dir", "cached")):
        if flag not in argv:
            argv += [flag, val]
    sys.argv = argv
    base.load_system1 = load_system1
    base.make_loaders = make_loaders
    base.main()


if __name__ == "__main__":
    main()
