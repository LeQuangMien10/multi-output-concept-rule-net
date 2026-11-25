"""
cache_s1_outputs.py — Chạy S1 một lần trên train/val/test của CUB và lưu logit
==============================================================================

Dùng transform sạch (không augment) giống hệt make_loaders của
src/scripts/cub/train_icrl.py, thứ tự ảnh giữ nguyên theo CSV (không shuffle).
Output dùng cho rebuild_icrl_cached.py: dựng lại ICRL nhiều lần (quét θ) mà
không phải chạy lại S1.

Lưu: {split: {"concepts": FloatTensor[N, C] logit, "label": FloatTensor[N, K]
logit, "y": LongTensor[N]}, "concept_names", "label_names", "system1_ckpt",
"image_size"}

Usage:
    python -m src.scripts.cub.cache_s1_outputs \\
        --system1_ckpt outputs/cub_outputs/outputs/scope_rebuild/cub/system1/best_model.pt \\
        --data_dir data/CUB_prepared --img_dir data/CUB/CUB_200_2011/images \\
        --out outputs/cub_outputs/s1_cache.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.cub.cub_dataset import CubDataset, build_cub_transforms
from src.scripts.cub.train_icrl import load_system1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--system1_ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="data/CUB_prepared")
    p.add_argument("--img_dir", type=str, default="data/CUB/CUB_200_2011/images")
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
    clean = build_cub_transforms("val", image_size)
    blob = {"system1_ckpt": args.system1_ckpt, "image_size": image_size}
    for split in ("train", "val", "test"):
        ds = CubDataset(Path(args.data_dir) / f"{split}.csv", args.img_dir, clean)
        blob["concept_names"], blob["label_names"] = ds.concept_names, ds.label_names
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        cs, ls, ys = [], [], []
        for images, labels in tqdm(loader, desc=f"S1 {split}"):
            out = system1(images.to(device))
            cs.append(out["concepts"].cpu()); ls.append(out["label"].cpu()); ys.append(labels["label"])
        blob[split] = {"concepts": torch.cat(cs), "label": torch.cat(ls), "y": torch.cat(ys)}
        acc = (blob[split]["label"].argmax(1) == blob[split]["y"]).float().mean().item()
        print(f"[{split}] {len(ds)} images, S1 label acc {acc:.4f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(f"[DONE] -> {args.out}")


if __name__ == "__main__":
    main()
