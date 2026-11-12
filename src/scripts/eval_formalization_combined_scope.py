"""
eval_formalization_combined_scope.py - Ban tong quat 1-split cua
eval_formalization_combined.py (Idea 1a/1b, ca 2 bien the P2) cho
fulldata/concept_only/pad_ufes20.

Usage (local, inference only):
    python -m src.scripts.eval_formalization_combined_scope --scope fulldata \\
        --icrl_root outputs/scope_rebuild/fulldata \\
        --icrl_dir outputs/scope_rebuild/fulldata/icrl_nolabelslot \\
        --data_dir data/fitzpatrick17k_prepared \\
        --img_dir data/fitzpatrick17k/data/finalfitz17k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.scripts.run_scope_rebuild import SCOPE_CONFIGS
from src.scripts.concept_rule_validity_table import build_valid_pattern_set
from src.scripts.greedy_constrained_match import greedy_repair
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.ensemble_strategies import entropy_confidence
from src.scripts.fitzpatrick.meta_classifier_ensemble import fit_pool_calibration, continuous_pool_pred_calibrated


def parse_args():
    p = argparse.ArgumentParser(description="Idea 1a/1b, ca 2 bien the P2, 1-split, scope bat ky.")
    p.add_argument("--scope", type=str, required=True, choices=list(SCOPE_CONFIGS.keys()))
    p.add_argument("--icrl_root", type=str, required=True)
    p.add_argument("--icrl_dir", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--max_flips", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


@torch.no_grad()
def collect(loader, system1, memory, head, centroids, confidences, rule_n, rule_coh,
            num_classes, exclude_keys, keys, concept_keys, concept_offsets, concept_dims,
            valid_patterns, max_flips, device):
    rows = {k2: [] for k2 in ["y", "p1", "p2_single", "rc", "rule_n", "rule_coh", "sim",
                               "repair_dist", "class_scores", "total_weight", "n_fired"]}
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        cv = soft_concept_vector(out)
        y = labels["label"].to(device)
        p1 = F.softmax(out["label"], dim=-1)

        rule_ids, sims = memory.match(cv, return_scores=True)
        p2_single = F.softmax(head(centroids[rule_ids]), dim=-1)
        sim_best = sims.gather(1, rule_ids.unsqueeze(1)).squeeze(1)

        vote = memory.predict_weighted_activation(
            cv, concept_keys, concept_offsets, concept_dims, num_classes=num_classes,
            exclude_keys=exclude_keys)
        cs = vote["class_scores"]
        tw = cs.sum(dim=1)

        repair_d = []
        for i in range(cv.shape[0]):
            _, n_flips, success = greedy_repair(cv[i], valid_patterns, keys, concept_offsets,
                                                  concept_dims, max_flips)
            repair_d.append(float(n_flips) if success or n_flips == 0 else float(max_flips + 1))

        rows["y"].append(y); rows["p1"].append(p1); rows["p2_single"].append(p2_single)
        rows["rc"].append(confidences[rule_ids]); rows["rule_n"].append(rule_n[rule_ids])
        rows["rule_coh"].append(rule_coh[rule_ids]); rows["sim"].append(sim_best)
        rows["repair_dist"].append(torch.tensor(repair_d, device=device))
        rows["class_scores"].append(cs); rows["total_weight"].append(tw)
        rows["n_fired"].append(vote["n_fired"].float())
    return {k2: torch.cat(v) for k2, v in rows.items()}


def build_p2_multi(d):
    cs, tw, fb = d["class_scores"], d["total_weight"], d["p2_single"]
    p2 = cs / tw.clamp(min=1e-8).unsqueeze(-1)
    low = tw < 1e-6
    p2[low] = fb[low]
    return p2


def assemble(d):
    p1, p2s = d["p1"], d["p2_single"]
    p2m = build_p2_multi(d)
    pred1 = p1.argmax(1)
    pred2s, pred2m = p2s.argmax(1), p2m.argmax(1)
    y = d["y"]

    def margin(p, pred_a, pred_b):
        logp = p.clamp(min=1e-8).log()
        idx = torch.arange(len(p))
        return logp[idx, pred_a] - logp[idx, pred_b]

    margin1_single = margin(p1, pred1, pred2s)
    margin1_multi = margin(p1, pred1, pred2m)
    margin2_single = margin(p2s, pred2s, pred1)
    margin2_multi = margin(p2m, pred2m, pred1)
    s1_ent = entropy_confidence(p1)

    feat6 = torch.stack([margin1_single, margin2_single, d["rc"], torch.log1p(d["rule_n"]),
                          d["rule_coh"], s1_ent], dim=1)
    feat8 = torch.cat([feat6, d["sim"].unsqueeze(1), d["repair_dist"].unsqueeze(1)], dim=1)
    feat_agg5 = torch.stack([margin1_multi, margin2_multi, d["total_weight"], d["n_fired"], s1_ent], dim=1)
    feat_agg7 = torch.cat([feat_agg5, d["sim"].unsqueeze(1), d["repair_dist"].unsqueeze(1)], dim=1)

    return dict(y=y, p1=p1, p2s=p2s, p2m=p2m, pred1=pred1, pred2s=pred2s, pred2m=pred2m,
                feat6=feat6, feat8=feat8, feat_agg5=feat_agg5, feat_agg7=feat_agg7,
                dis_s=(pred1 != pred2s), dis_m=(pred1 != pred2m),
                label_s=(pred2s == y).long(), label_m=(pred2m == y).long())


def run_combo_1split(a_val, a_test, feat_key, p2_key, dis_key, label_key):
    dis_val = a_val[dis_key].numpy()
    n_dis = int(dis_val.sum())
    if n_dis < 10 or len(np.unique(a_val[label_key].numpy()[dis_val])) < 2:
        return None, None, n_dis

    X_train = a_val[feat_key][dis_val].numpy()
    y_train = a_val[label_key].numpy()[dis_val]
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(X_train), y_train)

    proba_train = clf.predict_proba(X_train if False else scaler.transform(X_train))[:, 1]
    best_t, best_acc = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        acc_t = float(((proba_train >= t).astype(int) == y_train).mean())
        if acc_t > best_acc:
            best_acc, best_t = acc_t, float(t)

    dis_test = a_test[dis_key].numpy()
    pred1_test = a_test["pred1"].numpy()
    pred2_test = a_test["pred2s" if p2_key == "p2s" else "pred2m"].numpy()
    y_test = a_test["y"].numpy()
    pred_hard = pred1_test.copy()
    if dis_test.sum() > 0:
        proba_test = clf.predict_proba(scaler.transform(a_test[feat_key][dis_test].numpy()))[:, 1]
        idx_dis = np.where(dis_test)[0]
        override = proba_test >= best_t
        pred_hard[idx_dis[override]] = pred2_test[idx_dis[override]]
    acc_hard = float((pred_hard == y_test).mean())

    pool_a, pool_b, _ = fit_pool_calibration(clf, scaler, X_train, a_val["p1"][dis_val],
                                              a_val[p2_key][dis_val], y_train)
    dd = {"dis_test": dis_test, "pred1_test": pred1_test, "p1_test": a_test["p1"],
          "p2_test": a_test[p2_key], "X_test": a_test[feat_key]}
    pred_pool = continuous_pool_pred_calibrated(clf, scaler, pool_a, pool_b, dd)
    acc_pool = float((pred_pool == y_test).mean())
    return acc_hard, acc_pool, n_dis


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    cfg = SCOPE_CONFIGS[args.scope]
    module = cfg["module"]
    label_names = cfg["icrl_args"]["label_names"].split(",")
    num_classes = len(label_names)

    if module == "fitzpatrick":
        from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders, build_full_concept_layout
        from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY
    else:
        from src.scripts.pad_ufes20.train_icrl import load_system1, make_loaders, build_full_concept_layout
        from src.utils.pad_ufes20_concepts import S1_LABEL_CONCEPT_KEY

    icrl_root, icrl_dir = Path(args.icrl_root), Path(args.icrl_dir)
    system1, image_size = load_system1(icrl_root / "system1" / "best_model.pt", device)
    _, val_loader, test_loader = make_loaders(Path(args.data_dir), args.img_dir, image_size,
                                               args.batch_size, args.num_workers)
    concept_names = test_loader.dataset.concept_names
    concept_keys, concept_offsets, concept_dims, _ = build_full_concept_layout(concept_names, label_names)
    exclude_keys = {S1_LABEL_CONCEPT_KEY}
    keys = [kk for kk in concept_keys if kk not in exclude_keys]

    memory = ICRLRuleMemory.load(icrl_dir / "icrl_rule_memory.pt", device=str(device))
    head = nn.Linear(memory.concept_dim, num_classes).to(device)
    head.load_state_dict(torch.load(icrl_dir / "prediction_head.pt",
                                     map_location=device, weights_only=False))
    head.eval()
    centroids = memory.get_centroids().to(device)
    confidences = torch.tensor(memory.get_confidences(), device=device)
    rule_n = torch.tensor([float(memory._n[r]) for r in range(memory.num_rules)], device=device)
    rule_coh = torch.tensor([float(memory._compute_coherence(r)) for r in range(memory.num_rules)], device=device)
    valid_patterns = build_valid_pattern_set(memory, concept_keys, concept_offsets, concept_dims, exclude_keys)

    print(f"[{args.scope}] thu thap du lieu...")
    d_val = collect(val_loader, system1, memory, head, centroids, confidences, rule_n, rule_coh,
                     num_classes, exclude_keys, keys, concept_keys, concept_offsets, concept_dims,
                     valid_patterns, args.max_flips, device)
    d_test = collect(test_loader, system1, memory, head, centroids, confidences, rule_n, rule_coh,
                      num_classes, exclude_keys, keys, concept_keys, concept_offsets, concept_dims,
                      valid_patterns, args.max_flips, device)
    a_val, a_test = assemble(d_val), assemble(d_test)

    combos = [
        ("A: P2 don, 6 dac trung (goc)", "feat6", "p2s", "dis_s", "label_s"),
        ("A: P2 don, 8 dac trung (+sim+repair_dist)", "feat8", "p2s", "dis_s", "label_s"),
        ("B: P2 da lam giau, 5 dac trung (goc)", "feat_agg5", "p2m", "dis_m", "label_m"),
        ("B: P2 da lam giau, 7 dac trung (+sim+repair_dist)", "feat_agg7", "p2m", "dis_m", "label_m"),
    ]
    results = {}
    print(f"\n{'ket hop':<50s}{'chon cung':>12s}{'pooling':>12s}{'n_dis':>8s}")
    for name, feat_key, p2_key, dis_key, label_key in combos:
        acc_hard, acc_pool, n_dis = run_combo_1split(a_val, a_test, feat_key, p2_key, dis_key, label_key)
        results[name] = {"acc_hard": acc_hard, "acc_pool": acc_pool, "n_dis": n_dis}
        if acc_hard is None:
            print(f"{name:<50s}{'(qua it mau)':>12s}{'':>12s}{n_dis:>8d}")
        else:
            print(f"{name:<50s}{acc_hard*100:>11.2f}%{acc_pool*100:>11.2f}%{n_dis:>8d}")

    out_path = Path(f"outputs/concept_rule_balancing/formalization_combined_{args.scope}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
