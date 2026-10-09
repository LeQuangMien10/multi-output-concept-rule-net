"""
calibration_audit.py - Do do muc do tin cay (calibration) cua confidence cua
System~1, System~2 va rule confidence dang dung lam tin hieu gating, tren
cac fold 5-fold CRL-matched da co san (chi inference, khong train lai).

Muc dich: buoc chan doan truoc khi thu cac cach ket hop moi (entropy,
log-linear pooling, naive-Bayes). Neu confidence da calibrate tot, van de
nam o cach ket hop; neu khong, can sua calibration truoc.

Do:
- ECE (Expected Calibration Error, n_bins deu nhau) cho:
    s1_conf : max-softmax cua System~1 (du doan dung/sai cua S1)
    s2_conf : max-softmax cua System~2 (du doan dung/sai cua S2)
    rule_conf : rule confidence (Wilson) cua rule khop, dung lam tin hieu
                gating, du bao dung/sai cua S2
- Bang doi chieu dung/sai 2x2 (S1 dung/sai x S2 dung/sai) tren test: day
  la baseline cho bang chung sua loi hai chieu o buoc sau.

Usage (local, inference only):
    python -m src.scripts.fitzpatrick.calibration_audit \\
        --cv_root outputs/crlmatched_cv \\
        --data_root data/fitzpatrick17k_crl_matched_cv_official \\
        --img_dir data/fitzpatrick17k/data/finalfitz17k \\
        --output_json outputs/calibration_audit/calibration_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory


def parse_args():
    p = argparse.ArgumentParser(description="Calibration audit cho S1, S2 va rule confidence tren 5-fold.")
    p.add_argument("--cv_root", type=str, required=True, help="Thu muc chua fold_0..fold_{k-1}.")
    p.add_argument("--system1_root", type=str, default=None,
                    help="Thu muc chua fold_k/system1/ -- mac dinh = --cv_root. Truyen rieng khi "
                         "icrl duoc tai-ap dung conf_min khac (xem apply_conf_min_cv.py) va khong "
                         "copy lai system1.")
    p.add_argument("--data_root", type=str, required=True, help="Thu muc chua fold_0..fold_{k-1} CSV.")
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_json", type=str, required=True)
    return p.parse_args()


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    """Expected Calibration Error voi cac bin deu nhau tren [0, 1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total, err = len(conf), 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        # Bin cuoi dong ca bien phai, de conf=1.0 khong bi bo sot.
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        err += (n / total) * abs(float(correct[in_bin].mean()) - float(conf[in_bin].mean()))
    return float(err)


def reliability_table(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> list[dict]:
    """Thong ke tung bin de ve reliability diagram sau nay."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = int(in_bin.sum())
        rows.append({
            "bin_lo": float(lo), "bin_hi": float(hi), "n": n,
            "mean_conf": float(conf[in_bin].mean()) if n else None,
            "acc": float(correct[in_bin].mean()) if n else None,
        })
    return rows


@torch.no_grad()
def collect(loader, system1, memory, head, centroids, confidences, device):
    """Tra ve y, xac suat S1, xac suat S2, rule confidence cua rule khop."""
    ys, s1p, s2p, rc = [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        cv = soft_concept_vector(out)
        rule_ids, _ = memory.match(cv)
        ys.append(labels["label"].to(device))
        s1p.append(F.softmax(out["label"], dim=-1))
        s2p.append(F.softmax(head(centroids[rule_ids]), dim=-1))
        rc.append(confidences[rule_ids])
    return (torch.cat(ys).cpu().numpy(), torch.cat(s1p).cpu().numpy(),
            torch.cat(s2p).cpu().numpy(), torch.cat(rc).cpu().numpy())


def audit_split(y, s1p, s2p, rc, n_bins):
    s1_pred, s2_pred = s1p.argmax(1), s2p.argmax(1)
    s1_correct = (s1_pred == y).astype(float)
    s2_correct = (s2_pred == y).astype(float)
    s1_conf, s2_conf = s1p.max(1), s2p.max(1)
    return {
        "n": int(len(y)),
        "s1_acc": float(s1_correct.mean()),
        "s2_acc": float(s2_correct.mean()),
        "s1_mean_conf": float(s1_conf.mean()),
        "s2_mean_conf": float(s2_conf.mean()),
        "ece_s1": ece(s1_conf, s1_correct, n_bins),
        "ece_s2": ece(s2_conf, s2_correct, n_bins),
        "ece_rule_conf_for_s2": ece(rc, s2_correct, n_bins),
        "reliability_s1": reliability_table(s1_conf, s1_correct, n_bins),
        "reliability_s2": reliability_table(s2_conf, s2_correct, n_bins),
        "reliability_rule_conf": reliability_table(rc, s2_correct, n_bins),
        "contingency_test": {
            "both_correct": int(((s1_correct == 1) & (s2_correct == 1)).sum()),
            "s1_only_correct": int(((s1_correct == 1) & (s2_correct == 0)).sum()),
            "s2_only_correct": int(((s1_correct == 0) & (s2_correct == 1)).sum()),
            "both_wrong": int(((s1_correct == 0) & (s2_correct == 0)).sum()),
        },
    }


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    cv_root, data_root = Path(args.cv_root), Path(args.data_root)
    system1_root = Path(args.system1_root) if args.system1_root else cv_root

    results = {}
    for k in range(args.num_folds):
        fold_dir = cv_root / f"fold_{k}"
        system1, image_size = load_system1(system1_root / f"fold_{k}" / "system1" / "best_model.pt", device)
        _, val_loader, test_loader = make_loaders(
            data_root / f"fold_{k}", args.img_dir, image_size, args.batch_size, args.num_workers,
        )
        memory = ICRLRuleMemory.load(fold_dir / "icrl" / "icrl_rule_memory.pt", device=str(device))
        head = nn.Linear(memory.concept_dim, 2).to(device)
        head.load_state_dict(torch.load(fold_dir / "icrl" / "prediction_head.pt",
                                        map_location=device, weights_only=False))
        head.eval()
        centroids = memory.get_centroids().to(device)
        confidences = torch.tensor(memory.get_confidences(), device=device)

        val = collect(val_loader, system1, memory, head, centroids, confidences, device)
        test = collect(test_loader, system1, memory, head, centroids, confidences, device)
        results[f"fold_{k}"] = {
            "num_rules": memory.num_rules,
            "val": audit_split(*val, args.n_bins),
            "test": audit_split(*test, args.n_bins),
        }
        t = results[f"fold_{k}"]["test"]
        print(f"fold_{k}: rules={memory.num_rules} "
              f"S1 acc={t['s1_acc']:.4f} ECE={t['ece_s1']:.4f} | "
              f"S2 acc={t['s2_acc']:.4f} ECE={t['ece_s2']:.4f} | "
              f"rule_conf ECE(vs S2 correct)={t['ece_rule_conf_for_s2']:.4f} | "
              f"contingency={t['contingency_test']}")

    def mean_std(key):
        vals = [results[f"fold_{k}"]["test"][key] for k in range(args.num_folds)]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}

    summary = {key: mean_std(key) for key in ["ece_s1", "ece_s2", "ece_rule_conf_for_s2", "s1_acc", "s2_acc"]}
    print("\nTEST mean+-std over folds:")
    for key, v in summary.items():
        print(f"  {key}: {v['mean']:.4f} +- {v['std']:.4f}")

    out = {"per_fold": results, "summary_test": summary}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
