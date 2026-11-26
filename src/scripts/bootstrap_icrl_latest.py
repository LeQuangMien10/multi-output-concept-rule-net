"""
bootstrap_icrl_latest.py - R4: khoang tin cay bootstrap cho ban ICRL moi nhat.

Paired bootstrap tren anh test (resample co hoan lai, B lan, seed 0) cho
accuracy va balanced accuracy cua S1, S2, ensemble va chenh lech ensemble - S1.
CRL-matched: resample rieng trong tung fold roi lay trung binh 5 fold (khop
cach bao cao trung binh fold nhu CRL). Artifact: outputs/icrl_latest/.

Usage:
    python -m src.scripts.bootstrap_icrl_latest --n_boot 10000
Output: outputs/icrl_latest/bootstrap.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models import icrl_inference as ii
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.scripts.eval_icrl_latest import DATASETS, LATEST

OUT = LATEST / "bootstrap.json"


@torch.no_grad()
def predictions(part):
    _, _, val, test, layout, num_classes = part["load"]()
    d = LATEST / part["tag"]
    mem = ICRLRuleMemory.load(d / "icrl_rule_memory.pt", device="cpu")
    head = nn.Linear(mem.concept_dim, num_classes)
    head.load_state_dict(torch.load(d / "prediction_head.pt", map_location="cpu", weights_only=False))
    head.eval()
    a_val = ii.assemble(val["y"], val["p1"], ii.s2_vote(mem, head, val["cv"], layout, num_classes))
    a_test = ii.assemble(test["y"], test["p1"], ii.s2_vote(mem, head, test["cv"], layout, num_classes))
    final, _ = ii.Combiner().fit(a_val).predict(a_test)
    return dict(y=test["y"].numpy(), s1=a_test["pred1"].numpy(), s2=a_test["pred2"].numpy(),
                ens=final.numpy(), k=num_classes)


def acc(p, y):
    return float((p == y).mean())


def bal(p, y, k):
    n = np.bincount(y, minlength=k)
    hit = np.bincount(y[p == y], minlength=k)
    present = n > 0
    return float(np.mean(hit[present] / n[present]))


def stats(folds, idx_list):
    """Trung binh qua fold cua cac chi so, voi chi so anh idx trong tung fold."""
    out = {}
    for name in ("s1", "s2", "ens"):
        out[f"acc_{name}"] = np.mean([acc(f[name][i], f["y"][i]) for f, i in zip(folds, idx_list)])
        out[f"bal_{name}"] = np.mean([bal(f[name][i], f["y"][i], f["k"]) for f, i in zip(folds, idx_list)])
    out["acc_ens_minus_s1"] = out["acc_ens"] - out["acc_s1"]
    out["bal_ens_minus_s1"] = out["bal_ens"] - out["bal_s1"]
    out["acc_s2_minus_s1"] = out["acc_s2"] - out["acc_s1"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    results = {}
    for ds, cfg in DATASETS.items():
        folds = [predictions(p) for p in cfg["parts"]]
        point = stats(folds, [np.arange(len(f["y"])) for f in folds])
        boots = {k: [] for k in point}
        for _ in range(args.n_boot):
            s = stats(folds, [rng.integers(0, len(f["y"]), len(f["y"])) for f in folds])
            for k, v in s.items():
                boots[k].append(v)
        res = {}
        for k, v in point.items():
            lo, hi = np.percentile(boots[k], [2.5, 97.5])
            res[k] = dict(point=100 * v, ci95=[100 * lo, 100 * hi])
            if k.endswith("_minus_s1"):
                res[k]["p_le_0"] = float(np.mean(np.array(boots[k]) <= 0))
        results[ds] = res
        r = res
        print(f"{ds:<14s} ens-S1 acc {r['acc_ens_minus_s1']['point']:+.2f} "
              f"[{r['acc_ens_minus_s1']['ci95'][0]:+.2f}, {r['acc_ens_minus_s1']['ci95'][1]:+.2f}] "
              f"P(<=0)={r['acc_ens_minus_s1']['p_le_0']:.3f} | bal {r['bal_ens_minus_s1']['point']:+.2f} "
              f"[{r['bal_ens_minus_s1']['ci95'][0]:+.2f}, {r['bal_ens_minus_s1']['ci95'][1]:+.2f}] | "
              f"ens {r['acc_ens']['point']:.2f} [{r['acc_ens']['ci95'][0]:.2f}, {r['acc_ens']['ci95'][1]:.2f}]",
              flush=True)
    with open(OUT, "w") as f:
        json.dump(dict(n_boot=args.n_boot, seed=args.seed, results=results), f, indent=2)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
