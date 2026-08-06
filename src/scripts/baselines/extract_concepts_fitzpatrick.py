"""
extract_concepts_fitzpatrick.py — Run a frozen Fitzpatrick System1 checkpoint
over train/val/test and dump (concept_vec, label) to .npz. See
extract_concepts_multiconcept.py for the rationale (identical pattern).

Uses the "val" transform (resize+center-crop, no augment) for ALL splits,
same as train_icrl.py — augmentation would make concept vectors non-
deterministic per image, which is fine for training S1 but not for
extracting a fixed feature table for baselines.

Usage:
    python -m src.scripts.baselines.extract_concepts_fitzpatrick \
        --data_dir data/fitzpatrick17k_prepared \
        --img_dir data/fitzpatrick17k/data/finalfitz17k \
        --system1_ckpt outputs/fitzpatrick_system1/best_model.pt \
        --output_dir outputs/baselines/fitzpatrick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.fitzpatrick.fitzpatrick_dataset import FitzpatrickDataset, build_transforms
from src.models.fitzpatrick.system1 import FitzpatrickSystem1, soft_concept_vector
from src.utils.fitzpatrick_concepts import (
    FULL_CONCEPT_KEYS, FULL_CONCEPT_DIMS, FULL_CV_DIM, LABEL_NAMES, NUM_LABELS,
    S1_LABEL_CONCEPT_KEY,
)


def expand_concept_names() -> list[str]:
    """One name per FLAT concept_vec column — see extract_concepts_multiconcept.py."""
    names = []
    for key in FULL_CONCEPT_KEYS:
        dim = FULL_CONCEPT_DIMS[key]
        if dim == 1:
            names.append(key)
        elif key == S1_LABEL_CONCEPT_KEY:
            names.extend(f"label={ln}" for ln in LABEL_NAMES)
        else:
            names.extend(f"{key}[{i}]" for i in range(dim))
    return names


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--system1_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def load_system1(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    model = FitzpatrickSystem1(
        backbone_name=saved_args.get("backbone", "resnet50"),
        pretrained=False,
        num_concepts=saved_args.get("num_concepts"),
        num_labels=saved_args.get("num_labels", NUM_LABELS),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    return model, saved_args.get("image_size", 224)


@torch.no_grad()
def extract_split(system1, loader, device) -> tuple[np.ndarray, np.ndarray]:
    cvs, ys = [], []
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        cv = soft_concept_vector(out)
        cvs.append(cv.cpu().numpy())
        ys.append(labels["label"].numpy())
    return np.concatenate(cvs, axis=0), np.concatenate(ys, axis=0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system1, image_size = load_system1(Path(args.system1_ckpt), device)
    print(f"[INFO] System1 loaded (frozen): {args.system1_ckpt}  image_size={image_size}  device={device}")

    clean_transform = build_transforms("val", image_size)

    for split in ("train", "val", "test"):
        ds = FitzpatrickDataset(data_dir / f"{split}.csv", img_dir, clean_transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        cv, y = extract_split(system1, loader, device)
        np.savez(out_dir / f"{split}_concepts.npz", concept_vec=cv, label=y)
        print(f"[INFO] {split}: concept_vec {cv.shape}, label {y.shape} -> {out_dir / f'{split}_concepts.npz'}")

    meta = {
        "concept_dim": FULL_CV_DIM,
        "concept_names": expand_concept_names(),
        "label_names": LABEL_NAMES,
        "num_classes": NUM_LABELS,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] meta.json written to {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
