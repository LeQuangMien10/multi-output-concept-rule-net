"""
apply_conf_min_scope.py - Ban tong quat cua
src/scripts/fitzpatrick/apply_conf_min_cv.py cho 1-split scope (fulldata,
concept_only, pad_ufes20) -- ap dung 1 gia tri conf_min MOI (khac gia tri
0.5 giu lai luc rebuild S1+ICRL, xem src/scripts/run_scope_rebuild.py va
memory project_s1s2_coupling_plan.md), CHI inference lai tren checkpoint da
co (System~1 + prediction_head da train cua n_min recommend) -- khong train
lai gi.

Ghi vao --output_dir MOI (khong de len icrl/ cu) de giu lai ket qua conf_min
goc lam doi chieu, giong dung quy uoc da dung cho CRL-matched.

Usage (local):
    python -m src.scripts.apply_conf_min_scope --scope concept_only \\
        --icrl_root outputs/scope_rebuild/concept_only \\
        --data_dir data/fitzpatrick17k_prepared_concept_only \\
        --img_dir data/fitzpatrick17k/data/finalfitz17k \\
        --conf_min 0.3 \\
        --output_dir outputs/scope_rebuild/concept_only/icrl_confmin03
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.scripts.run_scope_rebuild import SCOPE_CONFIGS
from src.models.icrl_rule_memory import ICRLRuleMemory


def parse_args():
    p = argparse.ArgumentParser(description="Ap dung conf_min moi cho 1 scope 1-split (inference only).")
    p.add_argument("--scope", type=str, required=True, choices=list(SCOPE_CONFIGS.keys()))
    p.add_argument("--icrl_root", type=str, required=True,
                    help="Thu muc chua system1/best_model.pt va icrl/ (da train san).")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--conf_min", type=float, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    cfg = SCOPE_CONFIGS[args.scope]
    module = cfg["module"]
    label_names = cfg["icrl_args"]["label_names"].split(",")

    if module == "fitzpatrick":
        from src.scripts.fitzpatrick.train_icrl import (
            load_system1, make_loaders, record_rule_accuracy, evaluate,
            build_full_concept_layout, export_rules,
        )
        from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY
    else:
        from src.scripts.pad_ufes20.train_icrl import (
            load_system1, make_loaders, record_rule_accuracy, evaluate,
            build_full_concept_layout, export_rules,
        )
        from src.utils.pad_ufes20_concepts import S1_LABEL_CONCEPT_KEY

    icrl_root = Path(args.icrl_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    system1, image_size = load_system1(icrl_root / "system1" / "best_model.pt", device)
    _, val_loader, test_loader = make_loaders(Path(args.data_dir), args.img_dir, image_size,
                                               args.batch_size, args.num_workers)
    concept_names = val_loader.dataset.concept_names
    full_concept_keys, full_concept_offsets, full_concept_dims, _ = build_full_concept_layout(
        concept_names, label_names)
    exclude_keys = {S1_LABEL_CONCEPT_KEY}

    sweep_summary = json.load(open(icrl_root / "icrl" / "n_min_sweep_summary.json"))
    rec_n_min = sweep_summary["recommended_n_min"]
    rec_dir = icrl_root / "icrl" / f"n_min_{rec_n_min}"

    memory = ICRLRuleMemory.load(icrl_root / "icrl" / "icrl_rule_memory_base.pt", device=str(device))
    memory.n_min = rec_n_min
    memory.prune(verbose=False, conf_min_override=0.0)

    head = nn.Linear(memory.concept_dim, len(label_names)).to(device)
    head.load_state_dict(torch.load(rec_dir / "prediction_head.pt",
                                     map_location=device, weights_only=False))
    head.eval()

    record_rule_accuracy(system1, head, memory, val_loader, device)

    memory.conf_min = args.conf_min
    memory.prune(verbose=True)
    dedupe_stats = memory.dedupe_by_decoded_pattern(
        full_concept_keys, full_concept_offsets, full_concept_dims,
        exclude_keys=exclude_keys, verbose=True,
    )
    print(f"{args.scope}: n_min={rec_n_min} conf_min={args.conf_min} -> {memory.num_rules} rules "
          f"(circular removed={dedupe_stats['removed_circular']})")

    torch.save(head.state_dict(), output_dir / "prediction_head.pt")
    memory.save(output_dir / "icrl_rule_memory.pt")

    rule_summary = export_rules(memory, output_dir, full_concept_keys, full_concept_offsets,
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
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[DONE] Saved -> {output_dir}")
    print(f"  val_accuracy  = {val_metrics['accuracy']:.4f}")
    print(f"  test_accuracy = {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
