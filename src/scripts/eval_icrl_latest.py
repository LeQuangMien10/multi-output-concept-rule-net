"""
eval_icrl_latest.py - R2 + R3 cho ban ICRL moi nhat (chot 2026-11-26).

Dung lai buoc prune cua run_stage3_onward() tu base memory (truoc prune) cua
tung thiet lap, voi head da train (head train truoc buoc conf_min nen dung
chinh xac cho moi gia tri conf_min), roi them buoc cuoi cua ban moi nhat:
    base -> prune(n_min, conf 0) -> rule accuracy tren val (head)
         -> prune(conf_min) -> dedupe -> bo rule rong -> c_r tren val

R2 (mac dinh): danh gia artifact cua ban moi nhat trong outputs/icrl_latest/
    (build_icrl_latest.py, conf_min = 0 cho moi bo).
R3 (--conf_sweep, lich su): quet conf_min in {0, 0.3, 0.5} voi n_min giu nguyen
    nhu lan build goc; ket qua dan den quyet dinh conf_min = 0 (2026-11-26).

Usage (local, dung cache S1 co san):
    python -m src.scripts.eval_icrl_latest
Output: outputs/icrl_latest/eval_results.json
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models import icrl_inference as ii
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.scripts.eval_coverage_normalized import CUB, FITZ_SCOPES, cub_inputs, fitz_inputs

LATEST = Path("outputs/icrl_latest")
OUT = LATEST / "eval_results.json"
SWEEP_OUT = LATEST / "conf_min_sweep_results.json"
CONF_GRID = [0.0, 0.3, 0.5]
SHORT_LEN = 1

DATASETS = {
    "crlmatched": dict(parts=[
        dict(tag=f"crlmatched_fold{k}", load=lambda k=k: fitz_inputs("crlmatched", k),
             build=Path(f"outputs/crlmatched_cv_gated/fold_{k}/icrl")) for k in range(5)]),
    "fulldata": dict(parts=[
        dict(tag="fulldata", load=lambda: fitz_inputs("fulldata", None),
             build=FITZ_SCOPES["fulldata"]["icrl"](None))]),
    "concept_only": dict(parts=[
        dict(tag="concept_only", load=lambda: fitz_inputs("concept_only", None),
             build=FITZ_SCOPES["concept_only"]["icrl"](None))]),
    "cub_class": dict(parts=[
        dict(tag="cub_class", load=lambda: cub_inputs("cub_class"), build=Path(CUB["cub_class"]["icrl"]))]),
    "cub_instance": dict(parts=[
        dict(tag="cub_instance", load=lambda: cub_inputs("cub_instance"),
             build=Path(CUB["cub_instance"]["icrl"]))]),
}


def recommended_n_min(build_dir: Path) -> int:
    return json.load(open(build_dir / "n_min_sweep_summary.json"))["recommended_n_min"]


@torch.no_grad()
def replay(build_dir, head, val, layout, conf_min):
    keys, offsets, dims, exclude = layout
    mem = ICRLRuleMemory.load(build_dir / "icrl_rule_memory_base.pt", device="cpu")
    mem.n_min, mem.conf_min = recommended_n_min(build_dir), 0.0
    mem.prune(verbose=False)
    rule_ids, _ = mem.match(val["cv"])
    mem.update_accuracy(val["cv"], val["y"], head(mem.get_centroids()[rule_ids]).argmax(dim=1))
    mem.conf_min = conf_min
    mem.prune(verbose=False)
    mem.dedupe_by_decoded_pattern(keys, offsets, dims, exclude_keys=exclude, verbose=False)
    mem.remove_empty_rules(keys, offsets, dims, exclude, verbose=False)
    if mem.num_rules:
        mem.fit_coverage(val["cv"], keys, offsets, dims, exclude)
    return mem


def same_memory(a, b):
    return (a.num_rules == b.num_rules and a._labels == b._labels and a._n == b._n
            and all(torch.allclose(x, y, atol=1e-6) for x, y in zip(a._mu, b._mu))
            and np.allclose(a._coverage, b._coverage, rtol=1e-6))


def per_class_recall(pred, y, num_classes):
    return [float(((pred == y) & (y == k)).sum() / (y == k).sum()) for k in range(num_classes) if (y == k).any()]


@torch.no_grad()
def evaluate(mem, head, val, test, layout, num_classes):
    keys, offsets, dims, exclude = layout
    if mem.num_rules == 0:
        return None
    length = mem.present_mask(keys, offsets, dims, exclude).sum(dim=1)
    a_val = ii.assemble(val["y"], val["p1"], ii.s2_vote(mem, head, val["cv"], layout, num_classes))
    vt = ii.s2_vote(mem, head, test["cv"], layout, num_classes)
    a_test = ii.assemble(test["y"], test["p1"], vt)
    comb = ii.Combiner().fit(a_val)
    final, w2 = comb.predict(a_test)
    final_val, _ = comb.predict(a_val)
    y = test["y"]
    case = ii.explanation_cases(a_test, final.numpy())
    correct = (final == y).numpy()
    fired = ~vt["no_rule"]
    post = vt["rule_post"][fired]
    dis = (a_test["pred1"] != a_test["pred2"])
    labels = ii.rule_labels(mem)
    return dict(
        n=len(y), n_rules=mem.num_rules,
        rule_label_hist=[int((labels == k).sum()) for k in range(num_classes)],
        rule_len_median=float(length.median()), rule_len_mean=float(length.mean()),
        correct_s1=int((a_test["pred1"] == y).sum()), correct_s2=int((a_test["pred2"] == y).sum()),
        correct_ens=int(correct.sum()),
        bal_s1=float(np.mean(per_class_recall(a_test["pred1"], y, num_classes))),
        bal_s2=float(np.mean(per_class_recall(a_test["pred2"], y, num_classes))),
        bal_ens=float(np.mean(per_class_recall(final, y, num_classes))),
        val_acc_ens=float((final_val == val["y"]).float().mean()),
        val_acc_s2=float((a_val["pred2"] == val["y"]).float().mean()),
        case_counts={c: int((case == c).sum()) for c in ii.CASES},
        case_correct={c: int(correct[case == c].sum()) for c in ii.CASES},
        n_fired_imgs=int(fired.sum()),
        len_w_sum=float((post * length.unsqueeze(0)).sum()),
        short_w_sum=float((post * (length <= SHORT_LEN).float().unsqueeze(0)).sum()),
        top1_share_sum=float(post.max(dim=1).values.sum()),
        eff_n=vt["eff_n"][fired].tolist(),
        w2_disagree_mean=float(w2[dis].mean()) if dis.any() else None,
    )


def merge(parts):
    n = sum(p["n"] for p in parts)
    nf = max(1, sum(p["n_fired_imgs"] for p in parts))
    fold = lambda k: [p[k] / p["n"] for p in parts]
    out = dict(
        n_test=n, n_rules=[p["n_rules"] for p in parts], rule_label_hist=[p["rule_label_hist"] for p in parts],
        rule_len_median=[p["rule_len_median"] for p in parts],
        s1=100 * float(np.mean(fold("correct_s1"))), s2=100 * float(np.mean(fold("correct_s2"))),
        ens=100 * float(np.mean(fold("correct_ens"))), ens_std=100 * float(np.std(fold("correct_ens"))),
        s1_std=100 * float(np.std(fold("correct_s1"))),
        ens_pooled=100 * sum(p["correct_ens"] for p in parts) / n,
        bal_s1=100 * float(np.mean([p["bal_s1"] for p in parts])),
        bal_s2=100 * float(np.mean([p["bal_s2"] for p in parts])),
        bal_ens=100 * float(np.mean([p["bal_ens"] for p in parts])),
        val_acc_ens=100 * float(np.mean([p["val_acc_ens"] for p in parts])),
        val_acc_s2=100 * float(np.mean([p["val_acc_s2"] for p in parts])),
        cases_pct={c: 100 * sum(p["case_counts"][c] for p in parts) / n for c in ii.CASES},
        case_acc={c: (100 * sum(p["case_correct"][c] for p in parts) / max(1, sum(p["case_counts"][c] for p in parts))
                      if sum(p["case_counts"][c] for p in parts) else None) for c in ii.CASES},
        len_w=sum(p["len_w_sum"] for p in parts) / nf,
        short_w=100 * sum(p["short_w_sum"] for p in parts) / nf,
        top1_share=100 * sum(p["top1_share_sum"] for p in parts) / nf,
        eff_n_median=float(np.median(sum((p["eff_n"] for p in parts), []))),
        w2_disagree_mean=float(np.mean([p["w2_disagree_mean"] for p in parts if p["w2_disagree_mean"] is not None])),
    )
    return out


def fmt(name, r):
    cs = " ".join(f"{c}:{r['cases_pct'][c]:5.1f}" for c in ii.CASES)
    return (f"{name:<26s} rules {sum(r['n_rules']):>5d} | S1 {r['s1']:6.2f} S2 {r['s2']:6.2f} ens {r['ens']:6.2f} "
            f"(val {r['val_acc_ens']:6.2f}) | bal S2 {r['bal_s2']:5.1f} ens {r['bal_ens']:5.1f} | len_w {r['len_w']:5.2f} "
            f"short {r['short_w']:5.1f}% effN {r['eff_n_median']:4.1f} | {cs}")


def evaluate_artifacts():
    """R2: danh gia artifact cua ban moi nhat trong outputs/icrl_latest/
    (build_icrl_latest.py: conf_min = 0, n_min chon lai theo quy tac sweep)."""
    results = {}
    for ds, cfg in DATASETS.items():
        parts = []
        for p in cfg["parts"]:
            _, _, val, test, layout, num_classes = p["load"]()
            d = LATEST / p["tag"]
            mem = ICRLRuleMemory.load(d / "icrl_rule_memory.pt", device="cpu")
            head = nn.Linear(mem.concept_dim, num_classes)
            head.load_state_dict(torch.load(d / "prediction_head.pt", map_location="cpu", weights_only=False))
            head.eval()
            parts.append(evaluate(mem, head, val, test, layout, num_classes))
        results[ds] = merge(parts)
        results[ds]["n_min"] = [json.load(open(LATEST / p["tag"] / "build_info.json"))["n_min"] for p in cfg["parts"]]
        print(fmt(ds, results[ds]), flush=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {OUT}")


def conf_min_sweep():
    """R3 (lich su, 2026-11-26): quet conf_min voi n_min giu nguyen nhu lan build goc."""
    results = {}
    for ds, cfg in DATASETS.items():
        inputs = [(p, p["load"]()) for p in cfg["parts"]]
        grid = {}
        for conf_min in CONF_GRID:
            parts = [evaluate(replay(p["build"], head, val, layout, conf_min), head, val, test, layout, num_classes)
                     for p, (_, head, val, test, layout, num_classes) in inputs]
            if any(x is None for x in parts):
                continue
            grid[conf_min] = merge(parts)
            print(fmt(f"{ds} conf_min={conf_min}", grid[conf_min]), flush=True)
        best = max(grid, key=lambda c: (round(grid[c]["val_acc_ens"], 9), -c))
        results[ds] = dict(selected_conf_min=best, grid={str(c): v for c, v in grid.items()})
    with open(SWEEP_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {SWEEP_OUT}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf_sweep", action="store_true", help="chay lai R3 (quet conf_min) thay vi R2")
    if ap.parse_args().conf_sweep:
        conf_min_sweep()
    else:
        evaluate_artifacts()


if __name__ == "__main__":
    main()
