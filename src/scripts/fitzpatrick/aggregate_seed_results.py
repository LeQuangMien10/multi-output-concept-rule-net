"""
aggregate_seed_results.py - Gop nhieu file JSON tu eval_crlmatched_metrics.py
(1 file/seed) thanh mean+-std, dinh dang san sang dan vao Table 1 cua paper
(cung dinh dang % voi 2 chu so thap phan nhu bai CRL).

Usage:
    python -m src.scripts.fitzpatrick.aggregate_seed_results \\
        outputs/crlmatched_seed42.json outputs/crlmatched_seed1.json \\
        outputs/crlmatched_seed2.json outputs/crlmatched_seed3.json \\
        outputs/crlmatched_seed4.json
"""
from __future__ import annotations

import argparse
import json
import statistics as stats


def parse_args():
    p = argparse.ArgumentParser(description="Gop metrics nhieu seed thanh mean+-std.")
    p.add_argument("json_files", nargs="+", type=str)
    return p.parse_args()


def fmt(values: list[float], as_pct: bool) -> str:
    mean = stats.mean(values)
    std = stats.pstdev(values) if len(values) > 1 else 0.0
    if as_pct:
        return f"{mean*100:.2f}$\\pm${std*100:.2f}\\%"
    return f"{mean*100:.2f}$\\pm${std*100:.2f}\\%"


def main():
    args = parse_args()
    results = [json.load(open(f)) for f in args.json_files]
    seeds = [r["seed"] for r in results]
    print(f"[INFO] {len(results)} seeds: {seeds}")
    if len(set(seeds)) != len(seeds):
        print("[WARN] trung seed! kiem tra lai danh sach file.")

    concept_acc = [r["concept_acc"] for r in results]
    concept_f1 = [r["concept_f1"] for r in results]
    diagnosis_acc = [r["diagnosis_acc"] for r in results]
    diagnosis_f1 = [r["diagnosis_f1"] for r in results]

    print("\nRaw per-seed:")
    for r in results:
        print(f"  seed={r['seed']:>3}  concept_acc={r['concept_acc']*100:.2f}%  "
              f"concept_f1={r['concept_f1']*100:.2f}%  diagnosis_acc={r['diagnosis_acc']*100:.2f}%  "
              f"diagnosis_f1={r['diagnosis_f1']*100:.2f}%")

    print("\nLaTeX-ready (paste into Table 1's ICRL row):")
    print(f"  Concept acc.   = {fmt(concept_acc, True)}")
    print(f"  Concept F1     = {fmt(concept_f1, True)}")
    print(f"  Diagnosis acc. = {fmt(diagnosis_acc, True)}")
    print(f"  Diagnosis F1   = {fmt(diagnosis_f1, True)}")


if __name__ == "__main__":
    main()
