"""
eval_ensemble_infotheory_scope.py - Ban tong quat cua
src/scripts/fitzpatrick/eval_ensemble_infotheory.py cho 1-split scope
(fulldata, concept_only, pad_ufes20) thay vi 5-fold CRL-matched, VA tong
quat cho so lop bat ky (khong chi 2 lop) -- dung sau khi rebuild S1+ICRL
cho 3 scope nay voi code hien tai (xem run_scope_rebuild.py, memory
project_s1s2_coupling_plan.md).

Boi canh: oracle ceiling (gioi han tren ly thuyet cua MOI cach ket hop) da
tinh truoc cho thay headroom tren s1_acc rieng la +8.9pp (crlmatched),
+4.5pp (pad_ufes20), +1.7pp (fulldata), +1.3pp (concept_only) -- nen
concept_only/fulldata it kha nang co chien luoc nao thang duoc baseline,
chay o day la DE XAC NHAN bang so thuc chu khong ky vong tim duoc gi moi.

Cung 8 chien luoc nhu ban CRL-matched (baseline, weighted_avg,
confidence_pick, gated_override, entropy_gated, log_linear, naive_bayes,
rule_conf_override, rc_weighted_pool), KHONG co bootstrap/meta-classifier o
day (disagreement bucket qua nho de train/test tach biet dang tin cho ca 3
scope nay, khac CRL-matched co 5 fold de lam leave-one-fold-out).

Usage (local, inference only):
    python -m src.scripts.eval_ensemble_infotheory_scope --scope pad_ufes20 \\
        --icrl_root outputs/scope_rebuild/pad_ufes20 \\
        --icrl_dir outputs/scope_rebuild/pad_ufes20/icrl \\
        --data_dir data/PAD-UFES-20_prepared --img_dir data/PAD-UFES-20/images \\
        --output_json outputs/ensemble_infotheory/pad_ufes20_ensemble.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scripts.run_scope_rebuild import SCOPE_CONFIGS
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.ensemble_strategies import (
    fit_temperature, apply_temperature, entropy_confidence,
    log_linear_pool, fit_log_linear_weights,
    fit_confusion_likelihood, fit_class_prior, naive_bayes_fusion,
    rule_conf_override, fit_rule_conf_override_thresh,
    rc_weighted_pool, fit_rc_weighted_scale,
    bidirectional_correction_counts,
)


def parse_args():
    p = argparse.ArgumentParser(description="8 chien luoc ensemble S1+S2 cho 1 scope 1-split, so lop bat ky.")
    p.add_argument("--scope", type=str, required=True, choices=list(SCOPE_CONFIGS.keys()))
    p.add_argument("--icrl_root", type=str, required=True, help="Thu muc chua system1/best_model.pt.")
    p.add_argument("--icrl_dir", type=str, required=True,
                    help="Thu muc chua icrl_rule_memory.pt + prediction_head.pt (co the khac icrl_root/icrl "
                         "neu dung ban da ap conf_min rieng, vd icrl_confmin03).")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_json", type=str, required=True)
    return p.parse_args()


def accuracy_score(y_true, y_pred):
    return float((y_true == y_pred).float().mean())


def binary_f1(y_true, y_pred, pos_label):
    tp = int(((y_pred == pos_label) & (y_true == pos_label)).sum())
    fp = int(((y_pred == pos_label) & (y_true != pos_label)).sum())
    fn = int(((y_pred != pos_label) & (y_true == pos_label)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def f1_macro_nclass(y_true, y_pred, num_classes):
    """Tong quat cua f1_macro_2class cho so lop bat ky."""
    return float(np.mean([binary_f1(y_true, y_pred, c) for c in range(num_classes)]))


@torch.no_grad()
def collect_logits(loader, system1, memory, head, centroids, confidences, device):
    ys, s1_logits, s2_logits, rc = [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        cv = soft_concept_vector(out)
        rule_ids, _ = memory.match(cv)
        ys.append(labels["label"].to(device))
        s1_logits.append(out["label"])
        s2_logits.append(head(centroids[rule_ids]))
        rc.append(confidences[rule_ids])
    return (torch.cat(ys), torch.cat(s1_logits), torch.cat(s2_logits), torch.cat(rc))


def gated_override_grid(p1_val, rc_val, y_val, p1_test, rc_test, y_test,
                         s2_val_argmax, s2_test_argmax, conf_fn_val, conf_fn_test):
    conf_val, conf_test = conf_fn_val, conf_fn_test
    best_combo, best_val_acc = None, -1.0
    thresh_grid = np.linspace(conf_val.min().item(), conf_val.max().item(), 9)
    for t1 in thresh_grid:
        for t2 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            override = (conf_val < t1) & (rc_val > t2)
            pred = p1_val.argmax(1).clone()
            pred[override] = s2_val_argmax[override]
            a = accuracy_score(y_val, pred)
            if a > best_val_acc:
                best_val_acc, best_combo = a, (t1, t2)
    t1, t2 = best_combo
    override_test = (conf_test < t1) & (rc_test > t2)
    pred_test = p1_test.argmax(1).clone()
    pred_test[override_test] = s2_test_argmax[override_test]
    return pred_test, best_combo


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    cfg = SCOPE_CONFIGS[args.scope]
    module = cfg["module"]
    label_names = cfg["icrl_args"]["label_names"].split(",")
    num_classes = len(label_names)

    if module == "fitzpatrick":
        from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders
    else:
        from src.scripts.pad_ufes20.train_icrl import load_system1, make_loaders

    icrl_root, icrl_dir = Path(args.icrl_root), Path(args.icrl_dir)
    system1, image_size = load_system1(icrl_root / "system1" / "best_model.pt", device)
    _, val_loader, test_loader = make_loaders(Path(args.data_dir), args.img_dir, image_size,
                                               args.batch_size, args.num_workers)

    memory = ICRLRuleMemory.load(icrl_dir / "icrl_rule_memory.pt", device=str(device))
    head = nn.Linear(memory.concept_dim, num_classes).to(device)
    head.load_state_dict(torch.load(icrl_dir / "prediction_head.pt",
                                     map_location=device, weights_only=False))
    head.eval()
    centroids = memory.get_centroids().to(device)
    confidences = torch.tensor(memory.get_confidences(), device=device)

    y_val, s1_logit_val, s2_logit_val, rc_val = collect_logits(
        val_loader, system1, memory, head, centroids, confidences, device)
    y_test, s1_logit_test, s2_logit_test, rc_test = collect_logits(
        test_loader, system1, memory, head, centroids, confidences, device)

    p1_val_raw, p2_val_raw = F.softmax(s1_logit_val, -1), F.softmax(s2_logit_val, -1)
    p1_test_raw, p2_test_raw = F.softmax(s1_logit_test, -1), F.softmax(s2_logit_test, -1)
    pred1_raw, pred2_raw = p1_test_raw.argmax(1), p2_test_raw.argmax(1)
    pred_baseline, combo_baseline = gated_override_grid(
        p1_val_raw, rc_val, y_val, p1_test_raw, rc_test, y_test,
        p2_val_raw.argmax(1), p2_test_raw.argmax(1),
        p1_val_raw.max(1).values, p1_test_raw.max(1).values)

    T1 = fit_temperature(s1_logit_val.cpu(), y_val.cpu())
    T2 = fit_temperature(s2_logit_val.cpu(), y_val.cpu())
    p1_val, p2_val = apply_temperature(s1_logit_val, T1), apply_temperature(s2_logit_val, T2)
    p1_test, p2_test = apply_temperature(s1_logit_test, T1), apply_temperature(s2_logit_test, T2)
    pred1, pred2 = p1_test.argmax(1), p2_test.argmax(1)
    assert torch.equal(pred1, pred1_raw) and torch.equal(pred2, pred2_raw), \
        "Temperature scaling khong duoc doi argmax -- loi logic neu assert nay fail."

    strategies = {}

    best_alpha, best_acc = 0.5, -1.0
    for alpha in np.linspace(0, 1, 21):
        pred = (alpha * p1_val + (1 - alpha) * p2_val).argmax(1)
        a = accuracy_score(y_val, pred)
        if a > best_acc:
            best_acc, best_alpha = a, alpha
    strategies["weighted_avg"] = (best_alpha * p1_test + (1 - best_alpha) * p2_test).argmax(1)

    pick_s1 = p1_test.max(1).values >= p2_test.max(1).values
    strategies["confidence_pick"] = torch.where(pick_s1, pred1, pred2)

    pred_go, combo_go = gated_override_grid(
        p1_val, rc_val, y_val, p1_test, rc_test, y_test, pred2, pred2,
        p1_val.max(1).values, p1_test.max(1).values)
    strategies["gated_override"] = pred_go

    ent_val, ent_test = entropy_confidence(p1_val), entropy_confidence(p1_test)
    pred_eg, combo_eg = gated_override_grid(
        p1_val, rc_val, y_val, p1_test, rc_test, y_test, pred2, pred2, ent_val, ent_test)
    strategies["entropy_gated"] = pred_eg

    w1, w2 = fit_log_linear_weights(p1_val, p2_val, y_val)
    strategies["log_linear"] = log_linear_pool(p1_test, p2_test, w1, w2).argmax(1)

    prior = fit_class_prior(y_val, num_classes)
    pred1_val, pred2_val = p1_val.argmax(1), p2_val.argmax(1)
    L1 = fit_confusion_likelihood(pred1_val, y_val, num_classes)
    L2 = fit_confusion_likelihood(pred2_val, y_val, num_classes)
    post = naive_bayes_fusion(pred1, pred2, prior, L1, L2)
    strategies["naive_bayes"] = post.argmax(1)

    rc_thresh = fit_rule_conf_override_thresh(p1_val, p2_val, rc_val, y_val)
    strategies["rule_conf_override"] = rule_conf_override(p1_test, p2_test, rc_test, rc_thresh)

    rc_scale = fit_rc_weighted_scale(p1_val, p2_val, rc_val, y_val)
    strategies["rc_weighted_pool"] = rc_weighted_pool(p1_test, p2_test, rc_test, rc_scale).argmax(1)

    results = {}
    for name, pred in [("baseline", pred_baseline)] + list(strategies.items()):
        acc = accuracy_score(y_test, pred)
        f1 = f1_macro_nclass(y_test.numpy(), pred.numpy(), num_classes)
        bidir = bidirectional_correction_counts(pred1, pred2, pred, y_test)
        results[name] = {"diagnosis_acc": acc, "diagnosis_f1": f1, **bidir}

    print(f"\n[{args.scope}] n_test={len(y_test)}  T1={T1:.3f}  T2={T2:.3f}")
    for name, r in results.items():
        print(f"  {name:<18s} acc={r['diagnosis_acc']*100:.2f}%  f1={r['diagnosis_f1']:.4f}  "
              f"s2_fixes_s1={r['s2_fixes_s1_count']}/{r['s1_wrong_s2_right_count']}  "
              f"s1_fixes_s2={r['s1_fixes_s2_count']}/{r['s2_wrong_s1_right_count']}")

    out = {
        "scope": args.scope, "n_test": len(y_test), "T1": T1, "T2": T2,
        "weighted_avg_alpha": float(best_alpha), "log_linear_w1": float(w1),
        "rule_conf_override_thresh": float(rc_thresh), "rc_weighted_scale": float(rc_scale),
        "gated_override_combo": combo_go, "entropy_gated_combo": combo_eg,
        "baseline_combo": combo_baseline,
        "strategies": results,
        "y_test": y_test.numpy().tolist(),
        "pred_test": {name: pred.numpy().tolist() for name, pred in
                      [("baseline", pred_baseline)] + list(strategies.items())},
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
