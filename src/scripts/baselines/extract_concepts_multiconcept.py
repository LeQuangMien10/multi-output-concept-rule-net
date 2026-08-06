"""
extract_concepts_multiconcept.py — Run a frozen MultiConcept System1
checkpoint over train/val/test and dump (concept_vec, label) to .npz.

Pure inference (no training) — feeds every downstream baseline (CRL-RRL,
decision tree, logistic regression) with the SAME frozen concepts ICRL
already uses, so all System-2 mechanisms are compared on identical inputs.

Usage:
    python -m src.scripts.baselines.extract_concepts_multiconcept \
        --data_dir data/mnist_multiconcept_v1 \
        --system1_ckpt outputs/multiconcept_system1/best_model.pt \
        --output_dir outputs/baselines/multiconcept_v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.multiconcept.mnist_multiconcept_dataset import MNISTMultiConceptPTDataset
from src.models.multiconcept.system1 import MultiConceptSystem1, soft_concept_vector
from src.utils.multiconcept_concepts import (
    FULL_CONCEPT_KEYS, FULL_CONCEPT_DIMS, FULL_CV_DIM, LABEL_NAMES, NUM_CONCEPTS, NUM_LABELS,
    S1_LABEL_CONCEPT_KEY,
)


def expand_concept_names() -> list[str]:
    """One name per FLAT concept_vec column — multi-dim slots (the label slot,
    dim=NUM_LABELS) get one name per class, not one name for the whole slot."""
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
    p.add_argument("--system1_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=512)
    return p.parse_args()


def load_system1(ckpt_path: Path, device: torch.device) -> MultiConceptSystem1:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = MultiConceptSystem1(
        feature_dim=args.get("feature_dim", 256),
        num_concepts=args.get("num_concepts", NUM_CONCEPTS),
        num_labels=args.get("num_labels", NUM_LABELS),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    return model


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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system1 = load_system1(Path(args.system1_ckpt), device)
    print(f"[INFO] System1 loaded (frozen): {args.system1_ckpt}")

    for split, fnames in [("train", ["train.pt"]), ("val", ["val.pt", "valid.pt"]), ("test", ["test.pt"])]:
        ds_path = None
        for fname in fnames:
            candidate = data_dir / fname
            if candidate.exists():
                ds_path = candidate
                break
        if ds_path is None:
            raise FileNotFoundError(f"No {split} split found in {data_dir}")

        ds = MNISTMultiConceptPTDataset(ds_path)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        cv, y = extract_split(system1, loader, device)
        np.savez(out_dir / f"{split}_concepts.npz", concept_vec=cv, label=y)
        print(f"[INFO] {split}: concept_vec {cv.shape}, label {y.shape} -> {out_dir / f'{split}_concepts.npz'}")

    meta = {
        "concept_dim": FULL_CV_DIM,
        "concept_names": expand_concept_names(),
        "label_names": LABEL_NAMES,
        "num_classes": NUM_LABELS,
    }
    import json
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] meta.json written to {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
