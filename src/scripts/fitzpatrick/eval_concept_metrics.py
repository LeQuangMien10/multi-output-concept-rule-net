"""
eval_concept_metrics.py - Concept accuracy/F1 dung CHINH XAC cong thuc CRL
that (obiyoag/crl/metrics.py::compute_concept_metric): threshold 0.5, tinh
rieng tung concept (accuracy_score / f1_score(average="macro")), bo qua
concept nao GT toan 0 trong split dang xet, roi trung binh cac concept con
lai (macro theo CONCEPT, khong phai macro theo anh).

Cung in them breakdown present/absent (recall tren GT=1, specificity tren
GT=0) va F1 positive-class-only (quy uoc rieng cua du an) -- vi accuracy/F1
macro-2-lop bi pha loang boi lop "vang mat" chiem da so trong du lieu
concept thua (xem reports/error_breakdown_and_match_contribution.md muc 5).

So sanh S1 (tu du doan concept) voi "S2 concept" = pattern cua rule ICRL da
match (threshold 0.5 tren centroid) neu --icrl_dir duoc truyen vao.

Usage:
    python -m src.scripts.fitzpatrick.eval_concept_metrics \
        --data_dir data/fitzpatrick17k_prepared \
        --img_dir data/fitzpatrick17k/data/finalfitz17k \
        --system1_ckpt outputs/fitzpatrick_system1/best_model.pt \
        --icrl_dir outputs/fitzpatrick_icrl_calibrated \
        --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory


def parse_args():
    p = argparse.ArgumentParser(description="Concept accuracy/F1 dung cong thuc CRL that.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--system1_ckpt", type=str, required=True)
    p.add_argument("--icrl_dir", type=str, default=None,
                    help="Neu truyen vao, tinh them 'S2 concept' = pattern cua rule da match.")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int) -> float:
    tp = int(((y_pred == pos_label) & (y_true == pos_label)).sum())
    fp = int(((y_pred == pos_label) & (y_true != pos_label)).sum())
    fn = int(((y_pred != pos_label) & (y_true == pos_label)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def f1_macro_both_classes(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """== sklearn.metrics.f1_score(y_true, y_pred, average='macro') cho nhan nhi phan."""
    return (binary_f1(y_true, y_pred, 0) + binary_f1(y_true, y_pred, 1)) / 2


def report(pred: np.ndarray, gt: np.ndarray, concept_names: list[str], name: str):
    accs, f1s, f1s_pos = [], [], []
    n_skipped = 0
    for i in range(len(concept_names)):
        true_vars, pred_vars = gt[:, i], pred[:, i]
        if true_vars.sum() == 0:
            n_skipped += 1
            continue
        accs.append(accuracy_score(true_vars, pred_vars))
        f1s.append(f1_macro_both_classes(true_vars, pred_vars))
        f1s_pos.append(binary_f1(true_vars, pred_vars, 1))

    present_mask = (gt == 1)
    absent_mask = (gt == 0)
    recall_present = (pred[present_mask] == 1).mean() if present_mask.any() else float("nan")
    spec_absent = (pred[absent_mask] == 0).mean() if absent_mask.any() else float("nan")

    print(f"\n{name}:")
    print(f"  concept bi bo qua (GT toan 0): {n_skipped}/{len(concept_names)}")
    print(f"  Concept accuracy (CRL formula, macro/concept) = {np.mean(accs)*100:.2f}%")
    print(f"  Concept F1 (CRL formula, macro 2 lop)          = {np.mean(f1s):.4f}")
    print(f"  F1 positive-class-only (quy uoc rieng du an)   = {np.mean(f1s_pos):.4f}")
    print(f"  Recall tren GT=1 (co mat that)                 = {recall_present*100:.2f}%")
    print(f"  Specificity tren GT=0 (vang mat)                = {spec_absent*100:.2f}%")


@torch.no_grad()
def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

    system1, image_size = load_system1(Path(args.system1_ckpt), device)
    train_loader, val_loader, test_loader = make_loaders(
        Path(args.data_dir), args.img_dir, image_size, args.batch_size, args.num_workers,
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]
    concept_names = loader.dataset.concept_names
    num_concepts = len(concept_names)
    print(f"[INFO] {args.split} N={len(loader.dataset)}  num_concepts={num_concepts}")

    memory = None
    centroids_hard = None
    if args.icrl_dir:
        memory = ICRLRuleMemory.load(Path(args.icrl_dir) / "icrl_rule_memory.pt", device=str(device))
        centroids_hard = (memory.get_centroids()[:, :num_concepts] >= 0.5).float().to(device)
        print(f"[INFO] ICRL memory: {memory.num_rules} rules, cluster_dims={memory.cluster_dims}")

    all_s1_pred, all_s2_pred, all_gt = [], [], []
    for images, labels in loader:
        images = images.to(device)
        s1_out = system1(images)
        concept_mask = labels["concept_mask"]
        concept_gt = labels["concepts"]
        s1_concept_hard = (torch.sigmoid(s1_out["concepts"]) > 0.5).float()

        keep = concept_mask.bool()
        all_s1_pred.append(s1_concept_hard[keep].cpu())
        all_gt.append(concept_gt[keep].cpu())

        if memory is not None:
            cv = soft_concept_vector(s1_out)
            rule_ids, _ = memory.match(cv)
            s2_concept_hard = centroids_hard[rule_ids]
            all_s2_pred.append(s2_concept_hard[keep].cpu())

    s1_pred = torch.cat(all_s1_pred).numpy().astype(np.int32)
    gt = torch.cat(all_gt).numpy().astype(np.int32)
    print(f"\n[INFO] Anh co concept_mask=1: {gt.shape[0]}")

    report(s1_pred, gt, concept_names, "System 1 (tu du doan)")
    if memory is not None:
        s2_pred = torch.cat(all_s2_pred).numpy().astype(np.int32)
        report(s2_pred, gt, concept_names, "ICRL (pattern cua rule da match)")


if __name__ == "__main__":
    main()
