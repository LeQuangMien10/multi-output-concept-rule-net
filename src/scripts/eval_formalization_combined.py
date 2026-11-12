"""
eval_formalization_combined.py - Trien khai Idea 1a/1b CHO CA 2 truong hop
P2 (notes/s1s2_formalization.md), de khong bo sot huong nao:

  (A) P2 CHUA lam giau  -- 1-nearest match() cu, dung dung 6 dac trung
      trong meta_classifier_ensemble.py, CONG THEM sim(x) [Idea 1a] va
      repair_dist(x) [Idea 1b]  ->  8 dac trung.
  (B) P2 DA lam giau    -- predict_weighted_activation() (multi-rule vote,
      eval_multirule_vote.py), dung bo dac trung "tong hop" tuong duong
      (margin1, margin2' theo P2', total_weight, n_fired, s1_entropy) =
      5 dac trung, CONG THEM sim(x), repair_dist(x) -> 7 dac trung.
      (sim(x)/repair_dist(x) la thuoc tinh cua x so VOI TOAN BO R, khong
      phu thuoc P2 nao dang dung de du doan -- ap dung duoc ca 2 truong
      hop.)

Voi moi trong 4 to hop, hoc g(x) (logistic regression, leave-one-fold-out,
giong meta_classifier_ensemble.py) roi danh gia CA 2 cach dung trong so:
chon cung (threshold) va pooling lien tuc (a,b hieu chinh rieng, tai dung
fit_pool_calibration()/continuous_pool_pred_calibrated()).

scope: crlmatched (5-fold) -- cung checkpoint voi 2 thu nghiem truoc, de
so sanh truc tiep voi 78.66%/78.55% (6-dac-trung, P2 don) va 79.21%
(total_weight don, P2 da lam giau).

Usage (local, inference only):
    python -m src.scripts.eval_formalization_combined --device cpu
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.scripts.measure_contingency_multiscope import DEFAULT_IMG_DIR
from src.scripts.concept_rule_validity_table import build_valid_pattern_set
from src.scripts.greedy_constrained_match import greedy_repair
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.ensemble_strategies import entropy_confidence
from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY
from src.scripts.fitzpatrick.meta_classifier_ensemble import fit_pool_calibration, continuous_pool_pred_calibrated


@torch.no_grad()
def collect_fold(k, cv_root, system1_root, data_root, img_dir, device,
                  max_flips=5, batch_size=64, num_workers=0):
    from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders, build_full_concept_layout

    fold_dir = cv_root / f"fold_{k}"
    system1, image_size = load_system1(system1_root / f"fold_{k}" / "system1" / "best_model.pt", device)
    _, val_loader, test_loader = make_loaders(data_root / f"fold_{k}", img_dir, image_size, batch_size, num_workers)
    concept_names = test_loader.dataset.concept_names
    concept_keys, concept_offsets, concept_dims, _ = build_full_concept_layout(
        concept_names, ["label_0", "label_1"])
    exclude_keys = {S1_LABEL_CONCEPT_KEY}
    keys = [kk for kk in concept_keys if kk not in exclude_keys]

    memory = ICRLRuleMemory.load(fold_dir / "icrl" / "icrl_rule_memory.pt", device=str(device))
    head = nn.Linear(memory.concept_dim, 2).to(device)
    head.load_state_dict(torch.load(fold_dir / "icrl" / "prediction_head.pt",
                                     map_location=device, weights_only=False))
    head.eval()
    centroids = memory.get_centroids().to(device)
    confidences = torch.tensor(memory.get_confidences(), device=device)
    rule_n = torch.tensor([float(memory._n[r]) for r in range(memory.num_rules)], device=device)
    rule_coh = torch.tensor([float(memory._compute_coherence(r)) for r in range(memory.num_rules)], device=device)

    valid_patterns = build_valid_pattern_set(memory, concept_keys, concept_offsets, concept_dims, exclude_keys)
    pattern_to_label = {}
    for r in range(memory.num_rules):
        d = memory.decode_rule(r, concept_keys, concept_offsets, concept_dims)
        pat = tuple(kk for kk in keys if d["slots"][kk]["value"] == "present")
        pattern_to_label[pat] = d["label"]

    def run(loader):
        rows = {k2: [] for k2 in ["y", "p1", "p2_single", "rc", "rule_n", "rule_coh", "sim",
                                   "repair_dist", "class_scores", "total_weight", "n_fired", "p2_fallback"]}
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
                cv, concept_keys, concept_offsets, concept_dims, num_classes=2, exclude_keys=exclude_keys)
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
            rows["p2_fallback"].append(p2_single)  # dung khi total_weight ~ 0
        return {k2: torch.cat(v) for k2, v in rows.items()}

    return {"val": run(val_loader), "test": run(test_loader)}


def build_p2_multi(d):
    cs, tw, fb = d["class_scores"], d["total_weight"], d["p2_fallback"]
    p2 = cs / tw.clamp(min=1e-8).unsqueeze(-1)
    low = tw < 1e-6
    p2[low] = fb[low]
    return p2


def assemble(d):
    """Tra ve dict cac tensor dung chung cho ca 4 to hop, tu 1 ban `d` (val hoac test)."""
    p1, p2s = d["p1"], d["p2_single"]
    p2m = build_p2_multi(d)
    pred1 = p1.argmax(1)
    pred2s, pred2m = p2s.argmax(1), p2m.argmax(1)
    y = d["y"]

    def margin(p, pred_a, pred_b):
        logp = p.clamp(min=1e-8).log()
        idx = torch.arange(len(p))
        return logp[idx, pred_a] - logp[idx, pred_b]

    # margin1 phai tinh THEO DUNG doi thu cua tung bien the P2 -- voi bai
    # toan nhi phan, pred1==pred2 lam margin suy bien ve 0, nen dung sai
    # doi thu se lam feature sai hoan toan tren chinh tap bat dong dang xet.
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


def run_combo(fold_assembled, feat_key, p2_key, dis_key, label_key, num_folds=5):
    accs_hard, accs_pool = [], []
    for k in range(num_folds):
        X_train = np.concatenate([fold_assembled[j]["val"][feat_key][fold_assembled[j]["val"][dis_key]].numpy()
                                   for j in range(num_folds) if j != k], axis=0)
        y_train = np.concatenate([fold_assembled[j]["val"][label_key][fold_assembled[j]["val"][dis_key]].numpy()
                                   for j in range(num_folds) if j != k], axis=0)
        if len(np.unique(y_train)) < 2 or len(X_train) < 10:
            accs_hard.append(None); accs_pool.append(None); continue

        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(X_train), y_train)

        d_test = fold_assembled[k]["test"]
        dis_test = d_test[dis_key].numpy()
        pred1_test = d_test["pred1"].numpy()
        pred2_test = d_test["pred2s" if p2_key == "p2s" else "pred2m"].numpy()
        y_test = d_test["y"].numpy()

        # chon cung: nguong do tren VAL CUA CHINH fold k (khong dung de
        # train clf) -- DUNG Y HET protocol goc trong meta_classifier_ensemble.py,
        # KHONG phai tren 4 fold dung de train clf (2 viec khac nhau).
        d_val_k = fold_assembled[k]["val"]
        dis_val_k = d_val_k[dis_key].numpy()
        label_val_k = d_val_k[label_key].numpy()[dis_val_k]
        proba_val_k = (clf.predict_proba(scaler.transform(d_val_k[feat_key][dis_val_k].numpy()))[:, 1]
                       if dis_val_k.sum() > 0 else np.array([]))
        best_t, best_acc = 0.5, -1.0
        if len(proba_val_k) > 0:
            for t in np.linspace(0.05, 0.95, 19):
                a = float(((proba_val_k >= t).astype(int) == label_val_k).mean())
                if a > best_acc:
                    best_acc, best_t = a, float(t)

        pred_hard = pred1_test.copy()
        if dis_test.sum() > 0:
            X_test_dis = d_test[feat_key][dis_test].numpy()
            proba_test = clf.predict_proba(scaler.transform(X_test_dis))[:, 1]
            idx_dis = np.where(dis_test)[0]
            override = proba_test >= best_t
            pred_hard[idx_dis[override]] = pred2_test[idx_dis[override]]
        acc_hard = float((pred_hard == y_test).mean())

        # pooling: hieu chinh (a,b) rieng tren CHINH tap train (4 fold kia)
        d_val_dis_list = []
        for j in range(num_folds):
            if j == k:
                continue
            dv = fold_assembled[j]["val"]
            m = dv[dis_key]
            d_val_dis_list.append((dv["p1"][m], dv[p2_key][m], dv["y"][m]))
        p1_train_dis = torch.cat([x[0] for x in d_val_dis_list])
        p2_train_dis = torch.cat([x[1] for x in d_val_dis_list])
        y_train_dis = torch.cat([x[2] for x in d_val_dis_list]).numpy()
        pool_a, pool_b, _ = fit_pool_calibration(clf, scaler, X_train, p1_train_dis, p2_train_dis, y_train_dis)

        dd = {"dis_test": dis_test, "pred1_test": pred1_test, "p1_test": d_test["p1"],
              "p2_test": d_test[p2_key], "X_test": d_test[feat_key]}
        pred_pool = continuous_pool_pred_calibrated(clf, scaler, pool_a, pool_b, dd)
        acc_pool = float((pred_pool == y_test).mean())

        accs_hard.append(acc_hard); accs_pool.append(acc_pool)
    valid_h = [a for a in accs_hard if a is not None]
    valid_p = [a for a in accs_pool if a is not None]
    return (sum(valid_h) / len(valid_h), sum(valid_p) / len(valid_p), accs_hard, accs_pool)


def main():
    device = torch.device("cpu")
    cv_root = Path("outputs/crlmatched_cv_gated_confmin03")
    system1_root = Path("outputs/crlmatched_cv_gated")
    data_root = Path("data/fitzpatrick17k_crl_matched_cv_official")

    print("Thu thap du lieu 5 fold...")
    fold_raw = [collect_fold(k, cv_root, system1_root, data_root, DEFAULT_IMG_DIR, device) for k in range(5)]
    fold_assembled = [{"val": assemble(fr["val"]), "test": assemble(fr["test"])} for fr in fold_raw]

    combos = [
        ("A: P2 don, 6 dac trung (goc)", "feat6", "p2s", "dis_s", "label_s"),
        ("A: P2 don, 8 dac trung (+sim+repair_dist)", "feat8", "p2s", "dis_s", "label_s"),
        ("B: P2 da lam giau, 5 dac trung (goc)", "feat_agg5", "p2m", "dis_m", "label_m"),
        ("B: P2 da lam giau, 7 dac trung (+sim+repair_dist)", "feat_agg7", "p2m", "dis_m", "label_m"),
    ]

    results = {}
    print(f"\n{'ket hop':<50s}{'chon cung':>12s}{'pooling':>12s}")
    for name, feat_key, p2_key, dis_key, label_key in combos:
        acc_hard, acc_pool, per_fold_hard, per_fold_pool = run_combo(
            fold_assembled, feat_key, p2_key, dis_key, label_key)
        results[name] = {"acc_hard": acc_hard, "acc_pool": acc_pool,
                          "per_fold_hard": per_fold_hard, "per_fold_pool": per_fold_pool}
        print(f"{name:<50s}{acc_hard*100:>11.2f}%{acc_pool*100:>11.2f}%")

    out_path = Path("outputs/concept_rule_balancing/formalization_combined.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
