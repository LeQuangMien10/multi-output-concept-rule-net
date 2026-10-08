"""
analyze_disagreement_patterns.py - Khi S1 va S2 BAT DONG (pred1 != pred2),
dung 1 trong 2 LUON dung (bai toan 2 lop). Script nay gop toan bo cac ca
bat dong tren TEST cua 5 fold CRL-matched (outputs/crlmatched_cv_gated_confmin03,
Wilson score, S1 da sua lazy collapse) thanh 2 nhom:
  group A = S1 sai, S2 dung  (80 ca, tong 5 fold)
  group B = S2 sai, S1 dung  (200 ca, tong 5 fold)
va so sanh phan phoi cua 5 dac trung co the dung de phan biet 2 nhom nay
TRUOC KHI quyet dinh tin ai:
  s1_entropy      : entropy chuan hoa cua S1 (sau temperature scaling)
  s1_maxprob      : max-softmax cua S1 (sau temperature scaling)
  s2_maxprob      : max-softmax cua S2 (sau temperature scaling)
  rule_conf       : Wilson confidence cua rule S2 da khop
  rule_n          : so mau validation dung de tinh Wilson confidence do
  rule_coherence  : coherence hinh hoc cua cluster (memory._compute_coherence)

Neu mot dac trung co AUROC lech ro khoi 0.5 (phan biet duoc group A vs B),
day la bang chung CO THE xay 1 chien luoc ket hop moi dung chinh dac trung
do, thay vi chi dua vao do tu tin trung binh cua ca he thong.

Usage (local, inference only):
    python -m src.scripts.fitzpatrick.analyze_disagreement_patterns \\
        --cv_root outputs/crlmatched_cv_gated_confmin03 \\
        --system1_root outputs/crlmatched_cv_gated \\
        --data_root data/fitzpatrick17k_crl_matched_cv_official \\
        --img_dir data/fitzpatrick17k/data/finalfitz17k
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
from src.models.ensemble_strategies import fit_temperature, apply_temperature, entropy_confidence


def parse_args():
    p = argparse.ArgumentParser(description="Phan tich pattern trong cac ca S1/S2 bat dong.")
    p.add_argument("--cv_root", type=str, required=True)
    p.add_argument("--system1_root", type=str, default=None)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--label_names", type=str, default="benign,malignant")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


@torch.no_grad()
def collect(loader, system1, memory, head, centroids, confidences, device):
    ys, s1_logits, s2_logits, rc, rule_ids_all = [], [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        cv = soft_concept_vector(out)
        rule_ids, _ = memory.match(cv)
        ys.append(labels["label"].to(device))
        s1_logits.append(out["label"])
        s2_logits.append(head(centroids[rule_ids]))
        rc.append(confidences[rule_ids])
        rule_ids_all.append(rule_ids)
    return (torch.cat(ys), torch.cat(s1_logits), torch.cat(s2_logits),
            torch.cat(rc), torch.cat(rule_ids_all))


def auroc(feature: np.ndarray, is_group_a: np.ndarray) -> float:
    """AUROC = P(feature cua mot mau group A > mot mau group B), tinh qua
    rank (Mann-Whitney U / n1*n2) -- khong can scipy."""
    order = np.argsort(feature)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(feature) + 1)
    n1, n2 = int(is_group_a.sum()), int((~is_group_a).sum())
    if n1 == 0 or n2 == 0:
        return float("nan")
    r1 = ranks[is_group_a].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    return float(u1 / (n1 * n2))


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    cv_root, data_root, img_dir = Path(args.cv_root), Path(args.data_root), Path(args.img_dir)
    system1_root = Path(args.system1_root) if args.system1_root else cv_root
    label_names = args.label_names.split(",")

    rows = []  # per-sample feature dict, pooled across folds
    for k in range(args.num_folds):
        fold_dir = cv_root / f"fold_{k}"
        system1, image_size = load_system1(system1_root / f"fold_{k}" / "system1" / "best_model.pt", device)
        _, val_loader, test_loader = make_loaders(
            data_root / f"fold_{k}", img_dir, image_size, args.batch_size, args.num_workers,
        )
        memory = ICRLRuleMemory.load(fold_dir / "icrl" / "icrl_rule_memory.pt", device=str(device))
        head = nn.Linear(memory.concept_dim, len(label_names)).to(device)
        head.load_state_dict(torch.load(fold_dir / "icrl" / "prediction_head.pt",
                                        map_location=device, weights_only=False))
        head.eval()
        centroids = memory.get_centroids().to(device)
        confidences = torch.tensor(memory.get_confidences(), device=device)

        y_val, s1_logit_val, s2_logit_val, _, _ = collect(
            val_loader, system1, memory, head, centroids, confidences, device)
        T1 = fit_temperature(s1_logit_val.cpu(), y_val.cpu())
        T2 = fit_temperature(s2_logit_val.cpu(), y_val.cpu())

        y_test, s1_logit_test, s2_logit_test, rc_test, rule_ids_test = collect(
            test_loader, system1, memory, head, centroids, confidences, device)
        p1 = apply_temperature(s1_logit_test, T1)
        p2 = apply_temperature(s2_logit_test, T2)

        pred1, pred2 = p1.argmax(1), p2.argmax(1)
        s1_ent = entropy_confidence(p1)
        s1_conf, s2_conf = p1.max(1).values, p2.max(1).values
        rule_n = torch.tensor([float(memory._n[r]) for r in rule_ids_test.tolist()])
        rule_coh = torch.tensor([float(memory._compute_coherence(r)) for r in rule_ids_test.tolist()])

        s1_ok, s2_ok = (pred1 == y_test), (pred2 == y_test)
        disagree = pred1 != pred2
        for i in range(len(y_test)):
            if not disagree[i]:
                continue
            group = "A" if (not s1_ok[i] and s2_ok[i]) else ("B" if (not s2_ok[i] and s1_ok[i]) else None)
            if group is None:
                continue
            rows.append({
                "fold": k, "group": group,
                "s1_entropy": float(s1_ent[i]), "s1_maxprob": float(s1_conf[i]),
                "s2_maxprob": float(s2_conf[i]), "rule_conf": float(rc_test[i]),
                "rule_n": float(rule_n[i]), "rule_coherence": float(rule_coh[i]),
            })

    groupA = [r for r in rows if r["group"] == "A"]
    groupB = [r for r in rows if r["group"] == "B"]
    print(f"\nTong so ca bat dong: {len(rows)}  (group A [S1 sai, S2 dung]={len(groupA)}, "
          f"group B [S2 sai, S1 dung]={len(groupB)})")

    features = ["s1_entropy", "s1_maxprob", "s2_maxprob", "rule_conf", "rule_n", "rule_coherence"]
    is_a = np.array([r["group"] == "A" for r in rows])
    print(f"\n{'feature':<16s}{'mean_A':>10s}{'mean_B':>10s}{'median_A':>10s}{'median_B':>10s}{'AUROC(A>B)':>12s}")
    for feat in features:
        vals = np.array([r[feat] for r in rows])
        a_vals, b_vals = vals[is_a], vals[~is_a]
        au = auroc(vals, is_a)
        print(f"{feat:<16s}{a_vals.mean():>10.4f}{b_vals.mean():>10.4f}"
              f"{np.median(a_vals):>10.4f}{np.median(b_vals):>10.4f}{au:>12.4f}")
    print("\n(AUROC gan 0.5 = khong phan biet duoc; cang xa 0.5 (ve 0 hoac 1) = cang phan biet duoc.)")


if __name__ == "__main__":
    main()
