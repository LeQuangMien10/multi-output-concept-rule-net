"""
rebuild_prune_balanced.py - Dung lai buoc prune cua train_icrl.py (sau Stage 2)
tren rule memory GOC truoc prune (icrl_rule_memory_base.pt), de (1) dem so rule
theo nhan bi loai o tung buoc va (2) thu cac cach prune khong loai bo lop thieu
so. Xem notes/contribution_framing_explainability.md.

Khong train lai gi: dung lai prediction_head.pt DEPLOYED (head tuyen tinh tren
vector concept, khong phu thuoc chi so rule) de cham accuracy rule tren val,
dung cache vector concept cua S1 tu eval_explanation_coverage.py. Bien the
"deployed" duoc assert la tai tao CHINH XAC icrl_rule_memory.pt dang dung.

Thu tu buoc, khop run_stage3_onward():
  base -> prune(n_min, conf 0) [loc n_min + gop theta_merge]
       -> record_rule_accuracy tren val (head deployed)
       -> prune(conf_min) [loc conf + gop theta_merge lan nua]
       -> dedupe_by_decoded_pattern

Usage (local, inference only):
    python -m src.scripts.rebuild_prune_balanced
"""
from __future__ import annotations

import collections
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.icrl_rule_memory import ICRLRuleMemory
from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY
from src.scripts.fitzpatrick.train_icrl import build_full_concept_layout
from src.scripts.eval_ablation_crlmatched import strip_empty_rules
from src.scripts.eval_explanation_coverage import (
    vote_variant, assemble, learned_soft_pred, classify_cases, empty_rule_mask, CASES,
)

OUT_DIR = Path("outputs/explanation_coverage")
FIT_NAMES = ["benign", "malignant", "non-neoplastic"]
SCOPES = {
    "fulldata": dict(dir=Path("outputs/scope_rebuild/fulldata/icrl_nolabelslot"), n_min=25,
                     cache=OUT_DIR / "cache_fulldata.pt"),
    "concept_only": dict(dir=Path("outputs/scope_rebuild/concept_only/icrl_nolabelslot"), n_min=10,
                         cache=OUT_DIR / "cache_concept_only.pt"),
}
CONF_MIN = 0.5

# (ten, conf_mode, keep_top_k_per_class, n_min_minority)
PRUNE_VARIANTS = [
    ("deployed",               "val_head", 0, None),
    ("top3/class",             "val_head", 3, None),
    ("top5/class",             "val_head", 5, None),
    ("purity",                 "purity",   0, None),
    ("purity+nmin_minor5",     "purity",   0, 5),
    ("top3/class+nmin_minor5", "val_head", 3, 5),
]
# (ten, strip_empty, class_prior, gate)
EVAL_CONFIGS = [
    ("as-built",                 False, False, 0.5),
    ("strip+prior, gate 0.5",    True,  True,  0.5),
    ("strip+prior, gate off",    True,  True,  None),
]


def majority_labels(mem):
    return [collections.Counter(l).most_common(1)[0][0] if l else -1 for l in mem._labels]


def label_hist(mem, num_classes):
    c = collections.Counter(majority_labels(mem))
    return [c.get(k, 0) for k in range(num_classes)]


def filter_rules(mem, keep):
    mem._compact([i for i, k in enumerate(keep) if k])


def rebuild(base_path, head, val_cv, val_y, n_min, layout, num_classes,
            conf_mode, keep_k, n_min_minor, major_class):
    keys, offsets, dims, exclude = layout
    mem = ICRLRuleMemory.load(base_path, device="cpu")
    stages = {"base": label_hist(mem, num_classes)}

    # 1. Loc n_min (co the rieng cho lop thieu so) + gop theta_merge, conf chua do.
    if n_min_minor is None:
        mem.n_min = n_min
    else:
        labs = majority_labels(mem)
        filter_rules(mem, [mem._n[r] >= (n_min if labs[r] == major_class else n_min_minor)
                           for r in range(mem.num_rules)])
        mem.n_min = 1
    mem.conf_min = 0.0
    mem.prune(verbose=False)
    stages["after_nmin"] = label_hist(mem, num_classes)

    # 2. Cham accuracy rule tren val bang head deployed (record_rule_accuracy).
    with torch.no_grad():
        centroids = mem.get_centroids()
        rule_ids, _ = mem.match(val_cv)
        preds = head(centroids[rule_ids]).argmax(dim=1)
    mem.update_accuracy(val_cv, val_y, preds)
    if conf_mode == "purity":
        # Accuracy = do thuan nhan tren chinh anh train cua rule -> Wilson LB cua purity.
        for r in range(mem.num_rules):
            c = collections.Counter(mem._labels[r])
            mem._correct[r] = c.most_common(1)[0][1]
            mem._total_pred[r] = sum(c.values())

    # 3. Prune theo conf (+ giu top-k moi lop neu co) roi gop theta_merge.
    if keep_k == 0:
        mem.conf_min = CONF_MIN
        mem.prune(verbose=False)
    else:
        confs, labs = mem.get_confidences(), majority_labels(mem)
        keep = [c >= CONF_MIN for c in confs]
        for k in range(num_classes):
            idx = sorted([r for r in range(mem.num_rules) if labs[r] == k], key=lambda r: -confs[r])
            for r in idx[:keep_k]:
                keep[r] = True
        filter_rules(mem, keep)
        mem.conf_min = 0.0
        mem.prune(verbose=False)
    stages["after_conf"] = label_hist(mem, num_classes)

    # 4. Dedupe theo pattern hien thi.
    dd = mem.dedupe_by_decoded_pattern(keys, offsets, dims, exclude_keys=exclude, verbose=False)
    stages["after_dedupe"] = label_hist(mem, num_classes)
    stages["circular_removed"] = dd["removed_circular"]
    return mem, stages


def same_memory(a, b):
    if a.num_rules != b.num_rules:
        return False
    return all(torch.allclose(x, y, atol=1e-6) for x, y in zip(a._mu, b._mu)) and a._n == b._n


def evaluate(mem, head, blob, layout, num_classes, strip, use_prior, gate):
    mem = copy.deepcopy(mem)
    keys, offsets, dims, exclude = layout
    if strip:
        strip_empty_rules(mem, keys, offsets, dims, exclude, verbose=False)
    if mem.num_rules == 0:
        return None
    centroids = mem.get_centroids()
    keys_used = [k for k in keys if k not in exclude]
    empty = empty_rule_mask(mem, keys_used, offsets, dims)
    y_val = blob["val"]["y"]
    prior = (torch.bincount(y_val, minlength=num_classes).float() / len(y_val)) if use_prior else None
    with torch.no_grad():
        a_cal = assemble(y_val, blob["val"]["p1"],
                         vote_variant(mem, head, centroids, blob["val"]["cv"], layout,
                                      num_classes, gate, "head", empty, prior))
        a_test = assemble(blob["test"]["y"], blob["test"]["p1"],
                          vote_variant(mem, head, centroids, blob["test"]["cv"], layout,
                                       num_classes, gate, "head", empty, prior))
    final = learned_soft_pred(a_cal, a_test)
    yt, p2 = a_test["y"], a_test["pred2m"]
    recall = [float(((p2 == yt) & (yt == k)).sum() / max(1, int((yt == k).sum()))) * 100
              for k in range(num_classes)]
    case = classify_cases(a_test, final.numpy())
    return dict(
        acc=float((final == yt).float().mean()) * 100,
        s1_acc=float((a_test["pred1"] == yt).float().mean()) * 100,
        s2_acc=float((p2 == yt).float().mean()) * 100,
        s2_bal=float(np.mean(recall)),
        s2_recall=recall,
        s2_pred_pct=(torch.bincount(p2, minlength=num_classes).float() / len(p2) * 100).tolist(),
        case_pct={c: float((case == c).mean() * 100) for c in CASES},
        n_rules=mem.num_rules,
    )


def main():
    results = {}
    for scope, cfg in SCOPES.items():
        num_classes = len(FIT_NAMES)
        blob = torch.load(cfg["cache"], weights_only=False)
        keys, offsets, dims, _ = build_full_concept_layout(blob["concept_names"], FIT_NAMES)
        layout = (keys, offsets, dims, {S1_LABEL_CONCEPT_KEY})
        deployed = ICRLRuleMemory.load(cfg["dir"] / "icrl_rule_memory.pt", device="cpu")
        head = nn.Linear(deployed.concept_dim, num_classes)
        head.load_state_dict(torch.load(cfg["dir"] / "prediction_head.pt", map_location="cpu",
                                        weights_only=False))
        head.eval()
        base_path = cfg["dir"] / "icrl_rule_memory_base.pt"
        base = ICRLRuleMemory.load(base_path, device="cpu")
        all_labels = [y for l in base._labels for y in l]
        major = collections.Counter(all_labels).most_common(1)[0][0]

        print(f"\n=== {scope} (n_min={cfg['n_min']}, conf_min={CONF_MIN}, lop da so={FIT_NAMES[major]}) ===")
        print("Rule theo nhan (benign/malignant/nn) qua tung buoc:")
        results[scope] = {}
        for name, conf_mode, keep_k, nmm in PRUNE_VARIANTS:
            mem, stages = rebuild(base_path, head, blob["val"]["cv"], blob["val"]["y"], cfg["n_min"],
                                  layout, num_classes, conf_mode, keep_k, nmm, major)
            if name == "deployed":
                assert same_memory(mem, deployed), f"{scope}: rebuilt deployed memory != icrl_rule_memory.pt"
            fmt = lambda h: "/".join(str(x) for x in h)
            print(f"  {name:<24s} base {fmt(stages['base'])} -> n_min {fmt(stages['after_nmin'])} -> "
                  f"conf {fmt(stages['after_conf'])} -> dedupe {fmt(stages['after_dedupe'])} "
                  f"(circular removed {stages['circular_removed']})")
            evals = {}
            for ename, strip, use_prior, gate in EVAL_CONFIGS:
                evals[ename] = evaluate(mem, head, blob, layout, num_classes, strip, use_prior, gate)
            results[scope][name] = dict(stages=stages, eval=evals)
        print(f"  [deployed rebuild == icrl_rule_memory.pt: OK]")

        print(f"\nDanh gia (S1 alone = {results[scope]['deployed']['eval']['as-built']['s1_acc']:.2f}%):")
        print(f"  {'prune':<24s} {'eval':<24s} {'rules':>5s} {'ens':>6s} {'S2':>6s} {'S2bal':>6s} | "
              f"S2 recall b/m/nn | " + " ".join(f"{c:>6s}" for c in CASES))
        for name, r in results[scope].items():
            for ename, e in r["eval"].items():
                if e is None:
                    print(f"  {name:<24s} {ename:<24s}  (memory rong)")
                    continue
                rc = "/".join(f"{x:.0f}" for x in e["s2_recall"])
                cs = " ".join(f"{e['case_pct'][c]:6.1f}" for c in CASES)
                print(f"  {name:<24s} {ename:<24s} {e['n_rules']:5d} {e['acc']:6.2f} {e['s2_acc']:6.2f} "
                      f"{e['s2_bal']:6.2f} | {rc:>16s} | {cs}")

    with open(OUT_DIR / "rebuild_prune_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_DIR / 'rebuild_prune_results.json'}")


if __name__ == "__main__":
    main()
