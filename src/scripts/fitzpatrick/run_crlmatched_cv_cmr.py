"""
run_crlmatched_cv_cmr.py - Orchestrates all k folds of the CMR baseline
(CRL-matched scope) in a single Kaggle run, same pattern as
run_crlmatched_cv.py for ICRL. CMR is a single end-to-end script per fold
(no separate classifier-training stage like Basci et al.), so this just
loops baseline_cmr.py over folds and wires up paths -- the script itself
is unchanged.

Usage (Kaggle; `pip install lightning` first, as baseline_cmr.py requires):
    python -m src.scripts.fitzpatrick.run_crlmatched_cv_cmr \\
        --data_root /kaggle/input/datasets/lquangmin/fitzpatrick17k-crl-matched-cv-official \\
        --img_dir /kaggle/input/datasets/lquangmin/fitzpatrick17k/data/finalfitz17k \\
        --output_root /kaggle/working/outputs/crlmatched_cv_cmr \\
        --k 5
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run all k CRL-matched CV folds for the CMR baseline.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--folds", type=str, default=None,
                    help="Comma-separated subset of fold indices to run, e.g. '2,3'. Default: all 0..k-1.")

    # CMR hyperparameters -- defaults match baseline_cmr.py's own defaults,
    # i.e. the exact recipe already used for outputs/cmr_seed42.json.
    p.add_argument("--label_names", type=str, default="benign,malignant")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--n_rules", type=int, default=3)
    p.add_argument("--rule_emb_size", type=int, default=500)
    p.add_argument("--selector_input", type=str, default="embedding", choices=["embedding", "concepts"])
    p.add_argument("--w_c", type=float, default=1.0)
    p.add_argument("--w_y", type=float, default=30.0)
    p.add_argument("--w_yF", type=float, default=0.005)
    p.add_argument("--reset_selector_every_n_epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--backbone_lr_scale", type=float, default=100.0)
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=42,
                    help="Training seed, reused identically for every fold (matches CRL's "
                         "protocol of one seed per fold, varying only the data).")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    fold_indices = ([int(x) for x in args.folds.split(",")] if args.folds
                     else list(range(args.k)))
    print(f"[INFO] Running folds: {fold_indices}")

    py = sys.executable

    for i in fold_indices:
        fold_data_dir = data_root / f"fold_{i}"
        cmr_ckpt_dir = output_root / f"fold_{i}" / "cmr_ckpt"
        fold_json = output_root / f"cmr_cv_fold{i}.json"
        fold_rules_txt = output_root / f"cmr_cv_fold{i}_rules.txt"

        print(f"\n{'=' * 70}\n[FOLD {i}] data={fold_data_dir}\n{'=' * 70}")

        run([py, "-m", "src.scripts.fitzpatrick.baseline_cmr",
             "--data_dir", str(fold_data_dir), "--img_dir", args.img_dir,
             "--label_names", args.label_names, "--image_size", str(args.image_size),
             "--backbone", args.backbone, "--dropout", str(args.dropout),
             "--n_rules", str(args.n_rules), "--rule_emb_size", str(args.rule_emb_size),
             "--selector_input", args.selector_input,
             "--w_c", str(args.w_c), "--w_y", str(args.w_y), "--w_yF", str(args.w_yF),
             "--reset_selector_every_n_epochs", str(args.reset_selector_every_n_epochs),
             "--lr", str(args.lr), "--backbone_lr_scale", str(args.backbone_lr_scale),
             "--train_batch_size", str(args.train_batch_size), "--max_epochs", str(args.max_epochs),
             "--seed", str(args.seed),  # CMR actually resets its own RNG with this seed (unlike Basci/ICRL's --seed label)
             "--output_dir", str(cmr_ckpt_dir),
             "--output_json", str(fold_json),
             "--output_rules_txt", str(fold_rules_txt)])

    fold_jsons = sorted(output_root.glob("cmr_cv_fold*.json"))
    print(f"\n{'=' * 70}\n[SUMMARY] {len(fold_jsons)} fold result files found\n{'=' * 70}")
    if len(fold_jsons) < args.k:
        print(f"[WARN] Expected {args.k} fold results, found {len(fold_jsons)} -- "
              f"summary below is partial.")

    results = [json.load(open(f)) for f in fold_jsons]
    if not results:
        print("[WARN] No fold result JSONs found -- nothing to summarize.")
        return

    # Fold identity comes from the filename (cmr_cv_fold<i>.json): --seed is
    # fixed at the same training seed for every fold (varies only the data),
    # so the JSON's own "seed" field is the same value for every fold.
    for path, r in zip(fold_jsons, results):
        print(f"  {path.stem}  concept_acc={r['concept_acc']*100:.2f}%  "
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
