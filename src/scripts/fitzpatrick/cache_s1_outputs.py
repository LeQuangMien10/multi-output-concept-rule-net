"""
cache_s1_outputs.py - Chay S1 Fitzpatrick mot lan tren train/val/test va luu logit
===================================================================================

Ban Fitzpatrick cua src/scripts/cub/cache_s1_outputs.py: transform sach giong het
make_loaders cua fitzpatrick/train_icrl.py, thu tu anh theo CSV (khong shuffle).
Output cung dinh dang voi cache CUB, dung cho src/scripts/rebuild_icrl_cached.py.

Usage:
    python -m src.scripts.fitzpatrick.cache_s1_outputs \\
        --system1_ckpt outputs/scope_rebuild/fulldata/system1/best_model.pt \\
        --data_dir data/fitzpatrick17k_prepared --out outputs/s1_cache/fulldata.pt \\
        --label_names benign,malignant,non-neoplastic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.fitzpatrick.fitzpatrick_dataset import FitzpatrickDataset, build_transforms
from src.scripts.fitzpatrick.train_icrl import load_system1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--system1_ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, default="data/fitzpatrick17k/data/finalfitz17k")
    p.add_argument("--label_names", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) \
        if args.device == "auto" else torch.device(args.device)
    system1, image_size = load_system1(Path(args.system1_ckpt), device)
    system1.eval()
    clean = build_transforms("val", image_size)
    blob = {"system1_ckpt": args.system1_ckpt, "image_size": image_size,
            "label_names": args.label_names.split(",")}
    for split in ("train", "val", "test"):
        ds = FitzpatrickDataset(Path(args.data_dir) / f"{split}.csv", args.img_dir, clean)
        blob["concept_names"] = ds.concept_names
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        cs, ls, ys = [], [], []
        for images, labels in tqdm(loader, desc=f"S1 {split}", mininterval=30):
            out = system1(images.to(device))
            cs.append(out["concepts"].cpu()); ls.append(out["label"].cpu()); ys.append(labels["label"])
        blob[split] = {"concepts": torch.cat(cs), "label": torch.cat(ls), "y": torch.cat(ys)}
        print(f"[{split}] {len(blob[split]['y'])} images", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
