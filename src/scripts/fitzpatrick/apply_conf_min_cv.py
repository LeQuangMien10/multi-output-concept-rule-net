"""
apply_conf_min_cv.py - Ap dung mot gia tri conf_min moi (khac voi conf_min
da dung khi train tren Kaggle) cho toan bo k fold CRL-matched CV, CHI inference
lai tren checkpoint da co san (System~1 + prediction_head da train) -- khong
train lai gi ca. Day la buoc ke tiep sau conf_min_ablation.py: ablation do
xem conf_min=0.3 la diem can bang tot (so rule tang 7 lan, diagnosis acc
khong doi), script nay AP DUNG gia tri do va xuat ra bo file chinh thuc
(icrl_rule_memory.pt, icrl_rules.json, icrl_rules_summary.json, metrics.json)
+ goi eval_crlmatched_metrics.py THAT (khong tu viet lai gated-override) de
cv_summary.json moi dung dinh dang/logic giong het run_crlmatched_cv.py.

Ghi vao --output_root MOI (khong de len output_root cu) de giu lai ket qua
conf_min=0.5 lam baseline doi chieu.

Usage (local):
    python -m src.scripts.fitzpatrick.apply_conf_min_cv \\
        --cv_root outputs/crlmatched_cv_gated \\
        --data_root data/fitzpatrick17k_crl_matched_cv_official \\
        --img_dir data/fitzpatrick17k/data/finalfitz17k \\
        --conf_min 0.3 \\
        --output_root outputs/crlmatched_cv_gated_confmin03
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

from src.scripts.fitzpatrick.train_icrl import (
    load_system1, make_loaders, record_rule_accuracy, evaluate,
    build_full_concept_layout, export_rules,
)
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY


def parse_args():
    p = argparse.ArgumentParser(description="Ap dung conf_min moi cho k fold CRL-matched CV (inference only).")
    p.add_argument("--cv_root", type=str, required=True, help="Fold da train san (co base memory + head).")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--conf_min", type=float, required=True)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--label_names", type=str, default="benign,malignant")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_root", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    cv_root, data_root, output_root = Path(args.cv_root), Path(args.data_root), Path(args.output_root)
    label_names = args.label_names.split(",")
    output_root.mkdir(parents=True, exist_ok=True)

    for k in range(args.num_folds):
        fold_dir = cv_root / f"fold_{k}"
        out_icrl_dir = output_root / f"fold_{k}" / "icrl"
        out_icrl_dir.mkdir(parents=True, exist_ok=True)

        system1, image_size = load_system1(fold_dir / "system1" / "best_model.pt", device)
        _, val_loader, test_loader = make_loaders(
            data_root / f"fold_{k}", args.img_dir, image_size, args.batch_size, args.num_workers,
        )
        concept_names = val_loader.dataset.concept_names
        full_concept_keys, full_concept_offsets, full_concept_dims, full_cv_dim = \
            build_full_concept_layout(concept_names, label_names)

        sweep_summary = json.load(open(fold_dir / "icrl" / "n_min_sweep_summary.json"))
        rec_n_min = sweep_summary["recommended_n_min"]

        memory = ICRLRuleMemory.load(fold_dir / "icrl" / "icrl_rule_memory_base.pt", device=str(device))
        memory.n_min = rec_n_min
        memory.prune(verbose=False, conf_min_override=0.0)

        head = nn.Linear(memory.concept_dim, len(label_names)).to(device)
        head.load_state_dict(torch.load(fold_dir / "icrl" / "prediction_head.pt",
                                        map_location=device, weights_only=False))
        head.eval()

        record_rule_accuracy(system1, head, memory, val_loader, device)

        memory.conf_min = args.conf_min
        memory.prune(verbose=True)
        dedupe_stats = memory.dedupe_by_decoded_pattern(
            full_concept_keys, full_concept_offsets, full_concept_dims,
            exclude_keys={S1_LABEL_CONCEPT_KEY}, verbose=True,
        )
        print(f"fold_{k}: n_min={rec_n_min} conf_min={args.conf_min} -> {memory.num_rules} rules "
              f"(circular removed={dedupe_stats['removed_circular']})")

        torch.save(head.state_dict(), out_icrl_dir / "prediction_head.pt")
        memory.save(out_icrl_dir / "icrl_rule_memory.pt")

        rule_summary = export_rules(memory, out_icrl_dir, full_concept_keys, full_concept_offsets,
                                     full_concept_dims, label_names)

        val_metrics = evaluate(system1, head, val_loader, device, memory, "val")
        test_metrics = evaluate(system1, head, test_loader, device, memory, "test")
        confidences = memory.get_confidences()
        metrics = {
            "n_min": rec_n_min,
            "conf_min": args.conf_min,
            "val_accuracy": val_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "num_rules": memory.num_rules,
            "num_effective_rules": rule_summary["num_effective_rules"],
            "circular_rate": rule_summary["circular_rate"],
            "dedupe_stats": dedupe_stats,
            "rule_confidence_stats": {
                "mean": sum(confidences) / max(1, memory.num_rules),
                "min": min(confidences) if memory.num_rules else 0,
                "max": max(confidences) if memory.num_rules else 0,
            },
        }
        with open(out_icrl_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Goi eval_crlmatched_metrics.py THAT (khong tu viet lai gated-override),
        # de cv_summary.json dung logic chinh thuc giong het run_crlmatched_cv.py.
        fold_json = output_root / f"crlmatched_cv_fold{k}.json"
        subprocess.run([
            sys.executable, "-m", "src.scripts.fitzpatrick.eval_crlmatched_metrics",
            "--data_dir", str(data_root / f"fold_{k}"), "--img_dir", args.img_dir,
            "--system1_ckpt", str(fold_dir / "system1" / "best_model.pt"),
            "--icrl_dir", str(out_icrl_dir), "--label_names", args.label_names,
            "--seed", str(k), "--output_json", str(fold_json),
        ], check=True)

    fold_jsons = sorted(output_root.glob("crlmatched_cv_fold*.json"))
    results = [json.load(open(f)) for f in fold_jsons]

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
    for key, v in summary.items():
        print(f"  {key}: {v}")

    with open(output_root / "cv_summary.json", "w") as f:
        json.dump({"folds": results, "summary": summary}, f, indent=2)
    print(f"\n[DONE] Saved {output_root / 'cv_summary.json'}")


if __name__ == "__main__":
    main()
