"""
run_crlmatched_cv_basci.py - Orchestrates all k folds of the Basci et al.
baseline (CRL-matched scope) in a single Kaggle run, same pattern as
run_crlmatched_cv.py for ICRL. Per fold: (1) train a label-only classifier
via train_system1_baseline.py --concept_loss_weight 0.0 (STANDARD recipe,
not ICRL's CRL-matched-tuned recipe -- see baseline_neurosymbolic_rules.py's
own docstring for why), (2) extract rules from it with
baseline_neurosymbolic_rules.py. Shells out to both existing scripts
unchanged; this file only loops over folds and wires up paths.

--tree_max_depth defaults to 3, matching what is already reported in
Table 1 for the single-split run (outputs/basci_seed42_depth3.json), not
baseline_neurosymbolic_rules.py's own default of 5.

Usage (Kaggle):
    python -m src.scripts.fitzpatrick.run_crlmatched_cv_basci \\
        --data_root /kaggle/input/datasets/lquangmin/fitzpatrick17k-crl-matched-cv-official \\
        --img_dir /kaggle/input/datasets/lquangmin/fitzpatrick17k/data/finalfitz17k \\
        --output_root /kaggle/working/outputs/crlmatched_cv_basci \\
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
    p = argparse.ArgumentParser(description="Run all k CRL-matched CV folds for the Basci et al. baseline.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--folds", type=str, default=None,
                    help="Comma-separated subset of fold indices to run, e.g. '2,3'. Default: all 0..k-1.")

    # Classifier hyperparameters -- STANDARD recipe (see module docstring),
    # matching outputs/fitzpatrick_basci_classifier_seed42's own args exactly.
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr_scale", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lr_schedule", type=str, default="cosine")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_concepts", type=int, default=48)
    p.add_argument("--num_labels", type=int, default=2)
    p.add_argument("--monitor", type=str, default="label_acc")
    p.add_argument("--seed", type=int, default=42,
                    help="Training seed, reused identically for every fold (matches CRL's "
                         "protocol of one seed per fold, varying only the data).")

    # Rule-extraction hyperparameters
    p.add_argument("--label_names", type=str, default="benign,malignant")
    p.add_argument("--tree_max_depth", type=str, default="3",
                    help="'none' for unlimited depth. Default 3 matches Table 1's reported row.")
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
        classifier_dir = output_root / f"fold_{i}" / "classifier"
        fold_json = output_root / f"basci_cv_fold{i}.json"

        print(f"\n{'=' * 70}\n[FOLD {i}] data={fold_data_dir}\n{'=' * 70}")

        run([py, "-m", "src.scripts.fitzpatrick.train_system1_baseline",
             "--data_dir", str(fold_data_dir), "--img_dir", args.img_dir,
             "--output_dir", str(classifier_dir),
             "--epochs", str(args.epochs), "--lr", str(args.lr),
             "--backbone_lr_scale", str(args.backbone_lr_scale),
             "--weight_decay", str(args.weight_decay),
             "--lr_schedule", args.lr_schedule, "--batch_size", str(args.batch_size),
             "--num_concepts", str(args.num_concepts), "--num_labels", str(args.num_labels),
             "--monitor", args.monitor, "--concept_loss_weight", "0.0",
             "--seed", str(args.seed)])

        run([py, "-m", "src.scripts.fitzpatrick.baseline_neurosymbolic_rules",
             "--data_dir", str(fold_data_dir), "--img_dir", args.img_dir,
             "--classifier_ckpt", str(classifier_dir / "best_model.pt"),
             "--label_names", args.label_names,
             "--tree_max_depth", args.tree_max_depth,
             # NOTE: despite this script's own --help text ("chi de ghi vao
             # JSON, khong dat lai seed o day"), --seed is actually passed as
             # DecisionTreeClassifier's random_state (line ~239 of that
             # script) -- a real, if minor, effect on tie-breaking. We keep
             # it fixed at the training seed across folds (not the fold
             # index) so folds differ only in data, matching every other
             # stage's design; the fold identity is tracked by filename below.
             "--seed", str(args.seed),
             "--output_json", str(fold_json)])

    fold_jsons = sorted(output_root.glob("basci_cv_fold*.json"))
    print(f"\n{'=' * 70}\n[SUMMARY] {len(fold_jsons)} fold result files found\n{'=' * 70}")
    if len(fold_jsons) < args.k:
        print(f"[WARN] Expected {args.k} fold results, found {len(fold_jsons)} -- "
              f"summary below is partial.")

    results = [json.load(open(f)) for f in fold_jsons]
    if not results:
        print("[WARN] No fold result JSONs found -- nothing to summarize.")
        return

    # Fold identity comes from the filename (basci_cv_fold<i>.json), not the
    # JSON's own "seed" field, which is now fixed at the training seed for
    # every fold (see the --seed comment above) rather than the fold index.
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
