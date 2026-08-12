"""
eval_confusion_categories.py - Dem so anh TP/TN/FP/FN (S1 tu doan vs S2 qua
ICRL rule) tren 1 split -- tai lap bang so sanh so luong loi trong report
(vd. reports/error_breakdown_and_match_contribution.md muc 1).

TP: S1 dung, S2 dung           (S2 giu nguyen dung)
TN: S1 sai,  S2 dung           (S2 sua duoc loi S1)
FP: S1 dung, S2 sai            (S2 lam hong)
FN: S1 sai,  S2 sai            (S2 khong cuu duoc)

Usage:
    python -m src.scripts.fitzpatrick.eval_confusion_categories \
        --data_dir data/fitzpatrick17k_prepared \
        --img_dir data/fitzpatrick17k/data/finalfitz17k \
        --system1_ckpt outputs/fitzpatrick_system1/best_model.pt \
        --icrl_dir outputs/fitzpatrick_icrl_calibrated \
        --split test
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders, build_full_concept_layout
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.utils.fitzpatrick_concepts import LABEL_NAMES as DEFAULT_LABEL_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Dem TP/TN/FP/FN (S1 vs S2 qua ICRL rule).")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--system1_ckpt", type=str, required=True)
    p.add_argument("--icrl_dir", type=str, required=True)
    p.add_argument("--label_names", type=str, default=None,
                    help="Comma-separated, override cho scope khac 3-lop mac dinh.")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def categorize(s1_correct: bool, s2_correct: bool) -> str:
    if s1_correct and s2_correct:
        return "TP"
    if not s1_correct and s2_correct:
        return "TN"
    if s1_correct and not s2_correct:
        return "FP"
    return "FN"


@torch.no_grad()
def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

    label_names = args.label_names.split(",") if args.label_names else list(DEFAULT_LABEL_NAMES)

    system1, image_size = load_system1(Path(args.system1_ckpt), device)
    train_loader, val_loader, test_loader = make_loaders(
        Path(args.data_dir), args.img_dir, image_size, args.batch_size, args.num_workers,
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]
    print(f"[INFO] {args.split} N={len(loader.dataset)}  labels={label_names}")

    memory = ICRLRuleMemory.load(Path(args.icrl_dir) / "icrl_rule_memory.pt", device=str(device))
    head = nn.Linear(memory.concept_dim, len(label_names)).to(device)
    head.load_state_dict(torch.load(Path(args.icrl_dir) / "prediction_head.pt",
                                     map_location=device, weights_only=False))
    head.eval()
    centroids = memory.get_centroids().to(device)
    print(f"[INFO] ICRL memory: {memory.num_rules} rules, cluster_dims={memory.cluster_dims}")

    counts = Counter()
    for images, labels in loader:
        images = images.to(device)
        s1_out = system1(images)
        cv = soft_concept_vector(s1_out)
        y = labels["label"].to(device)
        s1_pred = s1_out["label"].argmax(dim=1)
        rule_ids, _ = memory.match(cv)
        s2_pred = head(centroids[rule_ids]).argmax(dim=1)
        for i in range(len(y)):
            counts[categorize(bool(s1_pred[i] == y[i]), bool(s2_pred[i] == y[i]))] += 1

    total = sum(counts.values())
    print(f"\nTP={counts['TP']}  TN={counts['TN']}  FP={counts['FP']}  FN={counts['FN']}  (total={total})")
    print(f"S1 error rate (TN+FN)/{total} = {(counts['TN']+counts['FN'])/total*100:.2f}%")
    print(f"S2 error rate (FP+FN)/{total} = {(counts['FP']+counts['FN'])/total*100:.2f}%")
    print(f"Net corrections (TN-FP) = {counts['TN']-counts['FP']}")


if __name__ == "__main__":
    main()
