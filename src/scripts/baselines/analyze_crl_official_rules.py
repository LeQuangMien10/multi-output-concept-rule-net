"""
analyze_crl_official_rules.py — Rule-quality metrics for a rules.txt file
produced by the REAL, official CRL repo (obiyoag/crl)'s ExtractRule
callback (callbacks.py::ExtractRule.on_predict_end), run by the user
directly on Kaggle with the actual paper code (not our port).

File format (tab-separated), reconstructed from the official source:
    RID\t{class0}(b=X.XXXX)\t{class1}(b=X.XXXX)[...]\tSupport\tRule
    <rid>\t<w0>\t<w1>[...]\t<support>\t<rule string>
    ...
    ############################################################

This is the honest, paper-faithful counterpart to our own CRL-RRL port's
`crl_rrl_rules.json` — use this one when comparing against ACTUAL CRL
numbers; use crl_rrl_rules.json only for the "same S1 input, isolate
System-2 mechanism" ablation-style comparison (see conversation notes: the
official Fitzpatrick/SkinCon task is 2-class benign/malignant, hardcoded in
their data/skincon.py — NOT directly accuracy-comparable to our 3-class
ICRL setup).

Usage:
    python -m src.scripts.baselines.analyze_crl_official_rules \
        --rules_txt path/to/skin_crl.txt \
        --output_dir outputs/baselines/fitzpatrick_official_crl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rules_txt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def parse_rules_txt(path: Path) -> tuple[list[str], list[dict]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    # header: ["RID", "class0(b=...)", "class1(b=...)", ..., "Support", "Rule"]
    class_names = [re.sub(r"\(b=.*\)", "", h) for h in header[1:-2]]
    n_classes = len(class_names)

    rules = []
    for line in lines[1:]:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        rid = int(parts[0])
        weights = [float(x) for x in parts[1:1 + n_classes]]
        support = float(parts[1 + n_classes])
        rule_str = parts[2 + n_classes] if len(parts) > 2 + n_classes else ""
        # num_conditions: count of literals (rule_str is "A & B & ~C" or "A | B")
        num_conditions = len(re.findall(r"[^&|]+", rule_str)) if rule_str else 0
        predicts_class = int(max(range(n_classes), key=lambda i: weights[i]))
        rules.append({
            "rule_id": rid,
            "class_weights": dict(zip(class_names, weights)),
            "support": support,
            "rule_string": rule_str,
            "num_conditions": num_conditions,
            "predicts_class": class_names[predicts_class],
            "weight_margin": max(weights) - sorted(weights)[-2] if n_classes > 1 else max(weights),
        })
    return class_names, rules


def main():
    args = parse_args()
    rules_path = Path(args.rules_txt)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names, rules = parse_rules_txt(rules_path)
    print(f"[INFO] Parsed {len(rules)} rules, classes={class_names}")

    if rules:
        supports = [r["support"] for r in rules]
        conds = [r["num_conditions"] for r in rules]
        margins = [r["weight_margin"] for r in rules]
        summary = {
            "num_rules": len(rules),
            "classes": class_names,
            "support_stats": {"mean": sum(supports) / len(supports), "min": min(supports), "max": max(supports)},
            "num_conditions_stats": {"mean": sum(conds) / len(conds), "min": min(conds), "max": max(conds)},
            "weight_margin_stats": {"mean": sum(margins) / len(margins), "min": min(margins), "max": max(margins)},
        }
    else:
        summary = {"num_rules": 0}

    rules.sort(key=lambda r: -r["support"])
    with open(out_dir / "official_crl_rules.json", "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)
    with open(out_dir / "official_crl_rule_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\n[INFO] Top 10 rules by support:")
    for r in rules[:10]:
        print(f"  [support={r['support']:.3f}] {r['num_conditions']} cond -> {r['predicts_class']}: {r['rule_string'][:80]}")
    print(f"\n[INFO] Written to {out_dir}")


if __name__ == "__main__":
    main()
