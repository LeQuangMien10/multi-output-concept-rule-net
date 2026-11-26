"""
select_theta_nmin.py - Quy tac chung chon (theta, n_min) cho ban ICRL moi nhat
===============================================================================

Mot luoi chung cho moi bo du lieu:
    theta in THETAS (theta_merge = min(theta + 0.04, 0.999), nhu theta auto)
    n_min in N_MINS, conf_min = 0
Moi theta: mot lan build day du tu logit S1 da cache (src/scripts/rebuild_icrl_cached.py),
quet n_min ben trong train_icrl. Moi ung vien (theta, n_min) duoc cham tren VAL
bang ba tieu chi:
    ce      cross-entropy cua P2' (phieu vote cua rule) tren val, clamp 1e-8
    s2_bal  balanced accuracy cua P2' tren val
    ens     accuracy cua ensemble tren val (g(x) fit tren val)
CRL-matched: trung binh tieu chi qua 5 fold, mot cap (theta, n_min) chung cho ca 5.
Hoa: chon ung vien it rule hon. So lieu test chi de bao cao.

Usage:
    python -m src.scripts.select_theta_nmin --phase build --workers 7
    python -m src.scripts.select_theta_nmin --phase select
Output: outputs/theta_nmin_grid/<setting>/theta_<t>/n_min_<v>/, selection.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import icrl_inference as ii
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.scripts.eval_icrl_latest import evaluate, merge
from src.scripts.fitzpatrick.train_icrl import build_full_concept_layout
from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY

ROOT = Path("outputs/theta_nmin_grid")
THETAS = [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
N_MINS = [2, 3, 5, 10, 15, 25]
FIT3 = "benign,malignant,non-neoplastic"
SETTINGS = {f"crlmatched_fold{k}": dict(module="fitzpatrick", cache=f"outputs/s1_cache/crlmatched_fold{k}.pt",
                                        labels="benign,malignant") for k in range(5)}
SETTINGS.update({
    "fulldata": dict(module="fitzpatrick", cache="outputs/s1_cache/fulldata.pt", labels=FIT3),
    "concept_only": dict(module="fitzpatrick", cache="outputs/s1_cache/concept_only.pt", labels=FIT3),
    "cub_class": dict(module="cub", cache="outputs/cub_outputs/s1_cache.pt", labels=None),
})
DATASETS = {"crlmatched": [f"crlmatched_fold{k}" for k in range(5)], "fulldata": ["fulldata"],
            "concept_only": ["concept_only"], "cub_class": ["cub_class"]}
CRITERIA = {"ce": -1, "s2_bal": 1, "ens": 1}   # -1: nho hon la tot


def tdir(name, theta):
    return ROOT / name / f"theta_{theta:.2f}"


def build_one(name, theta):
    out = tdir(name, theta)
    if (out / "n_min_sweep_summary.json").exists():
        return name, theta, "cached"
    s = SETTINGS[name]
    cmd = [sys.executable, "-m", "src.scripts.rebuild_icrl_cached", "--module", s["module"], "--cache", s["cache"],
           "--output_dir", str(out), "--theta", f"{theta}", "--theta_merge", f"{min(theta + 0.04, 0.999)}",
           "--n_min_sweep", ",".join(map(str, N_MINS)), "--conf_min", "0", "--exclude_label_slot",
           "--device", "cpu"]
    if s["labels"]:
        cmd += ["--label_names", s["labels"]]
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TQDM_DISABLE="1")
    with open(out / "build.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    return name, theta, "ok" if rc == 0 else f"FAILED rc={rc}"


def load_cache(name):
    b = torch.load(SETTINGS[name]["cache"], weights_only=False)

    def split(s):
        out = {"concepts": b[s]["concepts"], "label": b[s]["label"]}
        return {"y": b[s]["y"], "p1": F.softmax(out["label"], dim=-1), "cv": soft_concept_vector(out)}

    keys, offsets, dims, _ = build_full_concept_layout(b["concept_names"], b["label_names"])
    return split("val"), split("test"), (keys, offsets, dims, {S1_LABEL_CONCEPT_KEY}), len(b["label_names"])


@torch.no_grad()
def score(cand_dir, val, test, layout, num_classes):
    mem_path = cand_dir / "icrl_rule_memory.pt"
    if not mem_path.exists():
        return None
    mem = ICRLRuleMemory.load(mem_path, device="cpu")
    if mem.num_rules == 0 or mem._coverage is None:
        return None
    head = nn.Linear(mem.concept_dim, num_classes)
    head.load_state_dict(torch.load(cand_dir / "prediction_head.pt", map_location="cpu", weights_only=False))
    head.eval()
    vv = ii.s2_vote(mem, head, val["cv"], layout, num_classes)
    y = val["y"]
    p_true = vv["p2"][torch.arange(len(y)), y].clamp(min=1e-8)
    pred2 = vv["p2"].argmax(1)
    rec = [float(((pred2 == y) & (y == k)).sum() / (y == k).sum()) for k in range(num_classes) if (y == k).any()]
    out = evaluate(mem, head, val, test, layout, num_classes)
    out.update(val_ce=float(-p_true.log().mean()), val_s2_bal=float(np.mean(rec)))
    return out


def select():
    caches = {n: load_cache(n) for n in SETTINGS}
    report = {}
    for ds, names in DATASETS.items():
        grid = {}
        for theta in THETAS:
            for n_min in N_MINS:
                parts = [score(tdir(n, theta) / f"n_min_{n_min}", *caches[n]) for n in names]
                if any(p is None for p in parts):
                    continue
                m = merge(parts)
                m.update(val_ce=float(np.mean([p["val_ce"] for p in parts])),
                         val_s2_bal=100 * float(np.mean([p["val_s2_bal"] for p in parts])))
                grid[(theta, n_min)] = m
        picks = {}
        for crit, sign in CRITERIA.items():
            key = {"ce": "val_ce", "s2_bal": "val_s2_bal", "ens": "val_acc_ens"}[crit]
            best = max(grid, key=lambda c: (sign * round(grid[c][key], 9), -sum(grid[c]["n_rules"])))
            picks[crit] = best
        report[ds] = dict(picks={c: list(v) for c, v in picks.items()},
                          grid={f"{t:.2f}/{n}": v for (t, n), v in grid.items()})
        print(f"\n=== {ds} ({len(grid)} valid candidates) ===")
        print(f"{'theta/n_min':<12s} {'rules':>6s} {'valCE':>6s} {'valS2bal':>8s} {'valEns':>7s} | "
              f"{'S1':>6s} {'S2':>6s} {'ens':>6s} {'balS2':>6s} {'balEns':>6s} {'short':>6s} {'effN':>5s} {'(a)':>5s}")
        for (t, n), m in sorted(grid.items()):
            tag = "".join(c[0].upper() for c, v in picks.items() if v == (t, n))
            print(f"{t:.2f}/{n:<7d} {sum(m['n_rules']):>6d} {m['val_ce']:6.3f} {m['val_s2_bal']:8.1f} "
                  f"{m['val_acc_ens']:7.2f} | {m['s1']:6.2f} {m['s2']:6.2f} {m['ens']:6.2f} {m['bal_s2']:6.1f} "
                  f"{m['bal_ens']:6.1f} {m['short_w']:5.1f}% {m['eff_n_median']:5.1f} {m['cases_pct']['a']:5.1f} {tag}")
    with open(ROOT / "selection.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {ROOT / 'selection.json'}  (picks: C=ce, S=s2_bal, E=ens)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["build", "select"], required=True)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--only", type=str, default=None, help="chi build cac setting nay (phay)")
    args = ap.parse_args()
    if args.phase == "select":
        return select()
    names = args.only.split(",") if args.only else list(SETTINGS)
    # theta cao tao nhieu rule nhat (Stage 2 cham nhat): chay truoc de can bang tai.
    jobs = [(n, t) for t in sorted(THETAS, reverse=True) for n in names]
    with ThreadPoolExecutor(args.workers) as ex:
        for name, theta, status in ex.map(lambda j: build_one(*j), jobs):
            print(f"{name:<18s} theta={theta:.2f}  {status}", flush=True)


if __name__ == "__main__":
    main()
