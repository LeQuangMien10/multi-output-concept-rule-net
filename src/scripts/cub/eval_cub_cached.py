"""
eval_cub_cached.py — Đánh giá ensemble deployed (multi/soft/learned) và độ phủ
giải thích trên CUB, từ logit S1 đã cache
===============================================================================

Dùng lại đúng các hàm của src/scripts/eval_explanation_coverage.py (vote nhiều
rule, đặc trưng feat_agg5, learned_soft hiệu chỉnh trên val, phân loại 4 trường
hợp a/b/c/d), nên số liệu so sánh trực tiếp được với Fitzpatrick. Mỗi thư mục
ICRL truyền vào (vd một giá trị θ) cho một dòng kết quả.

Usage:
    python -m src.scripts.cub.eval_cub_cached --cache outputs/cub_outputs/s1_cache.pt \\
        --icrl_dirs outputs/cub_icrl_theta/theta_0.90,outputs/cub_icrl_theta/theta_0.95
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.scripts.cub.train_icrl import build_full_concept_layout
from src.scripts.eval_explanation_coverage import (
    CASES, assemble, classify_cases, empty_rule_mask, learned_soft_pred, vote_variant,
)
from src.utils.cub_concepts import S1_LABEL_CONCEPT_KEY

GATES = [0.5, None]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--icrl_dirs", type=str, required=True, help="Comma-separated ICRL output dirs.")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def split_inputs(blob, split):
    out = {"concepts": blob[split]["concepts"], "label": blob[split]["label"]}
    return blob[split]["y"], F.softmax(out["label"], dim=-1), soft_concept_vector(out)


@torch.no_grad()
def evaluate_dir(icrl_dir: Path, blob: dict, layout, num_classes: int) -> dict:
    memory = ICRLRuleMemory.load(icrl_dir / "icrl_rule_memory.pt", device="cpu")
    head = nn.Linear(memory.concept_dim, num_classes)
    head.load_state_dict(torch.load(icrl_dir / "prediction_head.pt", map_location="cpu", weights_only=False))
    head.eval()
    keys, offsets, dims, exclude = layout
    empty = empty_rule_mask(memory, [k for k in keys if k not in exclude], offsets, dims)
    centroids = memory.get_centroids()
    rule_labels = [collections.Counter(l).most_common(1)[0][0] for l in memory._labels]
    y_val, p1_val, cv_val = split_inputs(blob, "val")
    y_test, p1_test, cv_test = split_inputs(blob, "test")
    res = {"num_rules": memory.num_rules, "classes_with_rule": len(set(rule_labels)),
           "theta": float(memory.theta), "theta_merge": float(memory.theta_merge),
           "n_min": int(memory.n_min), "variants": {}}
    for gate in GATES:
        a_cal = assemble(y_val, p1_val, vote_variant(memory, head, centroids, cv_val, layout,
                                                     num_classes, gate, "head", empty))
        a_test = assemble(y_test, p1_test, vote_variant(memory, head, centroids, cv_test, layout,
                                                        num_classes, gate, "head", empty))
        final = learned_soft_pred(a_cal, a_test)
        case = classify_cases(a_test, final.numpy())
        fired = ~(a_test["low"] | a_test["vacuous"])
        res["variants"]["gate=off" if gate is None else f"gate={gate}"] = {
            "s1_acc": float((a_test["pred1"] == y_test).float().mean()) * 100,
            "s2_acc": float((a_test["pred2m"] == y_test).float().mean()) * 100,
            "ensemble_acc": float((final == y_test).float().mean()) * 100,
            "case_pct": {c: float((case == c).mean() * 100) for c in CASES},
            "case_acc": {c: (float((final.numpy()[case == c] == y_test.numpy()[case == c]).mean() * 100)
                             if (case == c).any() else None) for c in CASES},
            "eff_rules_median": float(np.median(a_test["eff_n"][fired].numpy())) if fired.any() else None,
        }
    return res


def main():
    args = parse_args()
    blob = torch.load(args.cache, weights_only=False)
    num_classes = len(blob["label_names"])
    keys, offsets, dims, _ = build_full_concept_layout(blob["concept_names"], blob["label_names"])
    layout = (keys, offsets, dims, {S1_LABEL_CONCEPT_KEY})
    results = {}
    print(f"{'icrl_dir':<44s} {'theta':>5s} {'rules':>5s} {'cls':>4s} {'gate':>8s} {'S1':>6s} {'S2':>6s} "
          f"{'ens':>6s} | " + " ".join(f"{c:>9s}" for c in CASES) + " | effN")
    for d in args.icrl_dirs.split(","):
        r = evaluate_dir(Path(d), blob, layout, num_classes)
        results[d] = r
        for g, v in r["variants"].items():
            cs = " ".join(f"{v['case_pct'][c]:9.1f}" for c in CASES)
            en = v["eff_rules_median"]
            print(f"{d[-44:]:<44s} {r['theta']:5.3f} {r['num_rules']:5d} {r['classes_with_rule']:4d} {g:>8s} "
                  f"{v['s1_acc']:6.2f} {v['s2_acc']:6.2f} {v['ensemble_acc']:6.2f} | {cs} | "
                  f"{'-' if en is None else f'{en:.1f}'}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
