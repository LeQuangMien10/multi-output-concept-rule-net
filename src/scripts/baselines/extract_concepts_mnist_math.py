"""
extract_concepts_mnist_math.py — Run a frozen MultiHeadSystem1 checkpoint
over train/val/test and dump (concept_vec, label) to .npz. See
extract_concepts_multiconcept.py for the general rationale.

Target = digit3 (10-way). Concept vector = the FULL 40-dim soft vector
(digit1+op1+digit2+op2+digit3), matching outputs/icrl_coherence's own
methodology (digit3 is treated as "a concept too", like s1_label_pred in
MultiConcept/Fitzpatrick) — NOT the 30-dim input-only vector the old
crl_system2.py used. Feeding the same noisy digit3-slot ICRL itself
consumes keeps the comparison apples-to-apples: all System-2 mechanisms get
the same (S1-predicted, often wrong) inputs and are judged on how well they
denoise them, not on whether they were given an easier task.

Usage:
    python -m src.scripts.baselines.extract_concepts_mnist_math \
        --data_dir data/mnist_math_v3 \
        --system1_ckpt outputs/system1/best_model.pt \
        --output_dir outputs/baselines/mnist_math_v3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.mnist_math_dataset import MNISTMathPTDataset
from src.models.multi_head_system1 import MultiHeadSystem1
from src.models.rule_memory import CONCEPT_KEYS_ORDERED, CONCEPT_DIMS, CONCEPT_TOTAL_DIM, soft_concept_vector
from src.utils.symbols import ID_TO_SYMBOL


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--system1_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=512)
    return p.parse_args()


def load_system1(ckpt_path: Path, device: torch.device) -> MultiHeadSystem1:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = MultiHeadSystem1(
        feature_dim=args.get("feature_dim", 256),
        num_slots=args.get("num_slots", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    return model


def expand_concept_names() -> list[str]:
    """One name per FLAT concept_vec column (digit slots get a name per digit
    value, op slots get the actual symbol) — see extract_concepts_multiconcept.py."""
    names = []
    for key in CONCEPT_KEYS_ORDERED:
        dim = CONCEPT_DIMS[key]
        if key in ("op1", "op2"):
            names.extend(f"{key}={ID_TO_SYMBOL.get(i, i)}" for i in range(dim))
        else:
            names.extend(f"{key}={i}" for i in range(dim))
    return names


@torch.no_grad()
def extract_split(system1, loader, device) -> tuple[np.ndarray, np.ndarray]:
    cvs, ys = [], []
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        cv = soft_concept_vector(out)
        cvs.append(cv.cpu().numpy())
        ys.append(labels["digit3"].numpy())
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

        ds = MNISTMathPTDataset(ds_path)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        cv, y = extract_split(system1, loader, device)
        np.savez(out_dir / f"{split}_concepts.npz", concept_vec=cv, label=y)
        print(f"[INFO] {split}: concept_vec {cv.shape}, label {y.shape} -> {out_dir / f'{split}_concepts.npz'}")

    meta = {
        "concept_dim": CONCEPT_TOTAL_DIM,
        "concept_names": expand_concept_names(),
        "label_names": [str(i) for i in range(10)],
        "num_classes": 10,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] meta.json written to {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
