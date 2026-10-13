"""
rebuild_icrl_cv.py - Buoc 2 cua ke hoach S1/S2 coupling (xem memory
project_s1s2_coupling_plan.md), option (a): rebuild ICRL Stage 2/3 (clustering
+ prediction head) tren toan bo k fold CRL-matched CV, REUSE System~1
checkpoint da train san (khong train lai S1 -- S1 luon frozen trong Stage
2/3), chi khac run_crlmatched_cv.py o 2 diem:
  1. Bo qua buoc train_system1_baseline (system1_root truyen vao, tai lai
     checkpoint co san).
  2. --exclude_label_slot la tuy chon (mac dinh TAT), thay vi luon hardcode
     BAT nhu run_crlmatched_cv.py -- de so sanh truc tiep 2 cach cluster
     (co/khong co slot nhan S1 du doan trong vector dung de cluster Stage 2).

Day KHONG phai "inference-only" thuan tuy nhu apply_conf_min_cv.py: online
clustering (Stage 2) phai chay lai tu dau vi no phu thuoc cluster_dims (chinh
cai dang thay doi o day), va head (Stage 3) cung phai train lai (nhe, chi 1
linear layer, head_epochs=20) vi no khop voi tap rule moi. Van RE local,
khong can GPU/Kaggle, vi S1 chi forward frozen.

Usage (local):
    python -m src.scripts.fitzpatrick.rebuild_icrl_cv \\
        --system1_root outputs/crlmatched_cv_gated \\
        --data_root data/fitzpatrick17k_crl_matched_cv_official \\
        --img_dir data/fitzpatrick17k/data/finalfitz17k \\
        --output_root outputs/crlmatched_cv_nolabelslot \\
        --conf_min 0.3
    (them --exclude_label_slot de rebuild lai ban hien tai lam sanity check)
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Rebuild ICRL Stage 2/3 cho k fold, tai dung S1 checkpoint.")
    p.add_argument("--system1_root", type=str, required=True,
                    help="Thu muc chua fold_k/system1/best_model.pt da train san.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--folds", type=str, default=None)
    p.add_argument("--exclude_label_slot", action="store_true",
                    help="Mac dinh TAT (slot nhan S1 du doan duoc dung de cluster) -- "
                         "truyen co nay de bat lai giong run_crlmatched_cv.py (sanity check).")

    p.add_argument("--theta", type=str, default="auto")
    p.add_argument("--n_min", type=int, default=15)
    p.add_argument("--n_min_sweep", type=str, default="5,10,15,20,30")
    p.add_argument("--conf_min", type=float, default=0.3)
    p.add_argument("--label_names", type=str, default="benign,malignant")
    p.add_argument("--head_epochs", type=int, default=20)
    p.add_argument("--head_lr", type=float, default=0.001)
    p.add_argument("--head_steps_per_epoch", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    system1_root = Path(args.system1_root)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    fold_indices = ([int(x) for x in args.folds.split(",")] if args.folds
                     else list(range(args.k)))
    print(f"[INFO] Running folds: {fold_indices}  exclude_label_slot={args.exclude_label_slot}")

    py = sys.executable

    for i in fold_indices:
        fold_data_dir = data_root / f"fold_{i}"
        system1_ckpt = system1_root / f"fold_{i}" / "system1" / "best_model.pt"
        icrl_dir = output_root / f"fold_{i}" / "icrl"
        fold_json = output_root / f"crlmatched_cv_fold{i}.json"

        print(f"\n{'=' * 70}\n[FOLD {i}] data={fold_data_dir}\n{'=' * 70}")

        icrl_cmd = [py, "-m", "src.scripts.fitzpatrick.train_icrl",
             "--data_dir", str(fold_data_dir), "--img_dir", args.img_dir,
             "--system1_ckpt", str(system1_ckpt),
             "--output_dir", str(icrl_dir),
             "--theta", args.theta, "--n_min", str(args.n_min),
             "--n_min_sweep", args.n_min_sweep, "--conf_min", str(args.conf_min),
             "--label_names", args.label_names,
             "--head_epochs", str(args.head_epochs), "--head_lr", str(args.head_lr),
             "--head_steps_per_epoch", str(args.head_steps_per_epoch),
             "--seed", str(args.seed)]
        if args.exclude_label_slot:
            icrl_cmd.append("--exclude_label_slot")
        run(icrl_cmd)

        run([py, "-m", "src.scripts.fitzpatrick.eval_crlmatched_metrics",
             "--data_dir", str(fold_data_dir), "--img_dir", args.img_dir,
             "--system1_ckpt", str(system1_ckpt),
             "--icrl_dir", str(icrl_dir), "--label_names", args.label_names,
             "--seed", str(i),
             "--output_json", str(fold_json)])

    fold_jsons = sorted(output_root.glob("crlmatched_cv_fold*.json"))
    print(f"\n{'=' * 70}\n[SUMMARY] {len(fold_jsons)} fold result files found\n{'=' * 70}")
    if len(fold_jsons) < args.k:
        print(f"[WARN] Expected {args.k} fold results, found {len(fold_jsons)} -- summary below is partial.")

    results = [json.load(open(f)) for f in fold_jsons]
    if not results:
        print("[WARN] No fold result JSONs found -- nothing to summarize.")
        return

    for r in results:
        print(f"  fold={r['seed']:>2}  concept_acc={r['concept_acc']*100:.2f}%  "
              f"concept_f1={r['concept_f1']*100:.2f}%  diagnosis_acc={r['diagnosis_acc']*100:.2f}%  "
              f"diagnosis_f1={r['diagnosis_f1']*100:.2f}%")

    def fmt(values):
        mean = stats.mean(values)
        std = stats.pstdev(values) if len(values) > 1 else 0.0
        return f"{mean*100:.2f}+-{std*100:.2f}%"

    summary = {
        "n_folds": len(results),
        "concept_acc": fmt([r["concept_acc"] for r in results]),
        "concept_f1": fmt([r["concept_f1"] for r in results]),
        "diagnosis_acc": fmt([r["diagnosis_acc"] for r in results]),
        "diagnosis_f1": fmt([r["diagnosis_f1"] for r in results]),
    }
    print("\n[MEAN +- STD across folds]")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(output_root / "cv_summary.json", "w") as f:
        json.dump({"folds": results, "summary": summary}, f, indent=2)
    print(f"\n[DONE] Saved {output_root / 'cv_summary.json'}")


if __name__ == "__main__":
    main()
