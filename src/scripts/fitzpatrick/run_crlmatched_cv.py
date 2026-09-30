"""
run_crlmatched_cv.py - Orchestrates all k folds of the CRL-matched CV
evaluation in a single Kaggle run, instead of copy-pasting 3 commands x k
folds by hand. For each fold it shells out to the exact same 3 scripts
used before (train_system1_baseline -> train_icrl -> eval_crlmatched_metrics),
in sequence, so nothing about those scripts changes or risks diverging --
this file only loops and wires up fold-specific paths. After all folds it
prints the mean+-std summary (same computation as aggregate_seed_results.py).

Usage (Kaggle, --protocol official data from prepare_dataset_crl_matched_cv.py):
    python -m src.scripts.fitzpatrick.run_crlmatched_cv \\
        --data_root /kaggle/input/datasets/lquangmin/fitzpatrick17k-crl-matched-cv-official \\
        --img_dir /kaggle/input/datasets/lquangmin/fitzpatrick17k/data/finalfitz17k \\
        --output_root /kaggle/working/outputs/crlmatched_cv \\
        --k 5

A single fold's failure stops the whole run immediately (raises
CalledProcessError) rather than silently skipping it -- re-run the same
command after fixing the issue; already-finished folds' checkpoints stay on
disk under --output_root and are not recomputed if you narrow --folds.
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run all k CRL-matched CV folds (S1 -> ICRL -> eval) in one call.")
    p.add_argument("--data_root", type=str, required=True,
                    help="Parent dir containing fold_0..fold_{k-1}, each with train/val/test.csv "
                         "(output of prepare_dataset_crl_matched_cv.py).")
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True,
                    help="Per-fold outputs go under <output_root>/fold_<i>/{system1,icrl}/ and "
                         "<output_root>/crlmatched_cv_fold<i>.json; the final summary is printed "
                         "and saved to <output_root>/cv_summary.json.")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--folds", type=str, default=None,
                    help="Comma-separated subset of fold indices to run, e.g. '2,3' to redo just "
                         "those after a failure. Default: all 0..k-1.")

    # System~1 hyperparameters -- defaults match the tuned CRL-matched recipe
    # already used for the fixed-split seed runs (epochs=150, lr=5e-6,
    # backbone_lr_scale=100, weight_decay=0.01, cosine, label_acc monitor).
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--backbone_lr_scale", type=float, default=100.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lr_schedule", type=str, default="cosine")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_concepts", type=int, default=48)
    p.add_argument("--num_labels", type=int, default=2)
    p.add_argument("--monitor", type=str, default="label_acc")
    p.add_argument("--seed", type=int, default=42,
                    help="Training seed, reused identically for every fold (matches CRL's "
                         "protocol of one seed per fold, varying only the data).")

    # ICRL Stage 2/3 hyperparameters
    p.add_argument("--theta", type=str, default="auto")
    p.add_argument("--n_min", type=int, default=15)
    p.add_argument("--n_min_sweep", type=str, default="5,10,15,20,30")
    p.add_argument("--conf_min", type=float, default=0.5)
    p.add_argument("--label_names", type=str, default="benign,malignant")
    p.add_argument("--head_epochs", type=int, default=20)
    p.add_argument("--head_lr", type=float, default=0.001)
    p.add_argument("--head_steps_per_epoch", type=int, default=250)
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
        system1_dir = output_root / f"fold_{i}" / "system1"
        icrl_dir = output_root / f"fold_{i}" / "icrl"
        fold_json = output_root / f"crlmatched_cv_fold{i}.json"

        print(f"\n{'=' * 70}\n[FOLD {i}] data={fold_data_dir}\n{'=' * 70}")

        run([py, "-m", "src.scripts.fitzpatrick.train_system1_baseline",
             "--data_dir", str(fold_data_dir), "--img_dir", args.img_dir,
             "--output_dir", str(system1_dir),
             "--epochs", str(args.epochs), "--lr", str(args.lr),
             "--backbone_lr_scale", str(args.backbone_lr_scale),
             "--weight_decay", str(args.weight_decay),
             "--lr_schedule", args.lr_schedule, "--batch_size", str(args.batch_size),
             "--num_concepts", str(args.num_concepts), "--num_labels", str(args.num_labels),
             "--monitor", args.monitor, "--seed", str(args.seed)])

        run([py, "-m", "src.scripts.fitzpatrick.train_icrl",
             "--data_dir", str(fold_data_dir), "--img_dir", args.img_dir,
             "--system1_ckpt", str(system1_dir / "best_model.pt"),
             "--output_dir", str(icrl_dir),
             "--theta", args.theta, "--n_min", str(args.n_min),
             "--n_min_sweep", args.n_min_sweep, "--conf_min", str(args.conf_min),
             "--exclude_label_slot", "--label_names", args.label_names,
             "--head_epochs", str(args.head_epochs), "--head_lr", str(args.head_lr),
             "--head_steps_per_epoch", str(args.head_steps_per_epoch),
             "--seed", str(args.seed)])

        run([py, "-m", "src.scripts.fitzpatrick.eval_crlmatched_metrics",
             "--data_dir", str(fold_data_dir), "--img_dir", args.img_dir,
             "--system1_ckpt", str(system1_dir / "best_model.pt"),
             "--icrl_dir", str(icrl_dir), "--label_names", args.label_names,
             "--seed", str(i),  # fold index, not the training seed -- see script docstring
             "--output_json", str(fold_json)])

    # -- Aggregate every fold JSON found under output_root (not just the
    #    ones run this call, so --folds subset re-runs still produce a
    #    full summary once all k are present) --
    fold_jsons = sorted(output_root.glob("crlmatched_cv_fold*.json"))
    print(f"\n{'=' * 70}\n[SUMMARY] {len(fold_jsons)} fold result files found\n{'=' * 70}")
    if len(fold_jsons) < args.k:
        print(f"[WARN] Expected {args.k} fold results, found {len(fold_jsons)} -- "
              f"summary below is partial.")

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
