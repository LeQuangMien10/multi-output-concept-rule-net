"""
eval_ensemble_infotheory.py - So sanh 6 chien luoc ket hop System~1 + System~2
tren 5-fold CRL-matched (checkpoint da conf_min=0.3, xem
outputs/crlmatched_cv_gated_confmin03/), CHI inference lai, khong train lai:

  baseline        : gated-override tren xac suat THO (dung lai dung cong
                    thuc eval_crlmatched_metrics.py, de doi chieu voi so da
                    bao cao 78.99%).
  weighted_avg    : alpha*p1 + (1-alpha)*p2, alpha fit tren val -- tren xac
                    suat DA temperature-scale.
  confidence_pick : chon theo max-softmax cao hon -- tren xac suat da scale.
  gated_override  : giong baseline nhung tren xac suat da scale (tach rieng
                    hieu ung calibration voi hieu ung cong thuc ket hop).
  entropy_gated   : giong gated_override nhung tin hieu tin cay cua S1 la
                    entropy chuan hoa (1 - H(p)/logK) thay cho max-softmax.
  log_linear      : P(y) ~ P1(y)^w1 * P2(y)^w2, w1 fit tren val.
  naive_bayes     : P(y|S1,S2) ~ P(y)*P(S1|y)*P(S2|y), likelihood la
                    confusion matrix cua tung system do tren val.

Voi MOI chien luoc (ke ca baseline), dem bidirectional correction tren test
(so ca S2 sua S1 sai, S1 sua S2 sai) -- xem src/models/ensemble_strategies.py.

Temperature T1 (S1), T2 (S2) fit rieng tren val moi fold (Guo et al. 2017).
Luu y: temperature scaling KHONG doi argmax cua tung system rieng le (scale
duong khong doi thu tu logit) -- chi doi cac dai luong lien tuc (max-prob,
entropy, trong so pha tron) dung de QUYET DINH ket hop. Vi vay pred1/pred2
(va do do s1_only/s2_only correct) giong het nhau o moi chien luoc; chi
pred_combined thay doi.

Usage (local, inference only):
    python -m src.scripts.fitzpatrick.eval_ensemble_infotheory \\
        --cv_root outputs/crlmatched_cv_gated_confmin03 \\
        --data_root data/fitzpatrick17k_crl_matched_cv_official \\
        --img_dir data/fitzpatrick17k/data/finalfitz17k \\
        --output_json outputs/ensemble_infotheory/ensemble_infotheory.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.ensemble_strategies import (
    fit_temperature, apply_temperature, entropy_confidence,
    log_linear_pool, fit_log_linear_weights,
    fit_confusion_likelihood, fit_class_prior, naive_bayes_fusion,
    rule_conf_override, fit_rule_conf_override_thresh,
    rc_weighted_pool, fit_rc_weighted_scale,
    bidirectional_correction_counts,
)


def parse_args():
    p = argparse.ArgumentParser(description="So sanh 6 chien luoc ensemble S1+S2, co bidirectional correction.")
    p.add_argument("--cv_root", type=str, required=True,
                    help="Thu muc chua fold_k/icrl/ (icrl_rule_memory.pt, prediction_head.pt).")
    p.add_argument("--system1_root", type=str, default=None,
                    help="Thu muc chua fold_k/system1/best_model.pt -- mac dinh = --cv_root, "
                         "truyen rieng khi icrl duoc tai-ap dung conf_min khac (xem apply_conf_min_cv.py) "
                         "va khong copy lai system1 (System~1 khong doi khi chi doi conf_min).")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--label_names", type=str, default="benign,malignant")
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


def f1_macro_2class(y_true, y_pred):
    return (binary_f1(y_true, y_pred, 0) + binary_f1(y_true, y_pred, 1)) / 2


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total, err = len(conf), 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        err += (n / total) * abs(float(correct[in_bin].mean()) - float(conf[in_bin].mean()))
    return float(err)


@torch.no_grad()
def collect_logits(loader, system1, memory, head, centroids, confidences, device):
    """Tra ve logits THO (chua softmax) de fit temperature, cung rule
    confidence (rc) cua rule da khop -- khong lien quan toi temperature."""
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
    """Grid-search (conf_thresh, rule_conf_thresh) tren val, dung conf_fn de
    tinh tin hieu tin cay cua S1 (max-softmax hoac entropy, tuy chien luoc)."""
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


def run_fold(k, cv_root, system1_root, data_root, img_dir, label_names, batch_size, num_workers, device):
    fold_dir = cv_root / f"fold_{k}"
    system1, image_size = load_system1(system1_root / f"fold_{k}" / "system1" / "best_model.pt", device)
    _, val_loader, test_loader = make_loaders(
        data_root / f"fold_{k}", img_dir, image_size, batch_size, num_workers,
    )
    memory = ICRLRuleMemory.load(fold_dir / "icrl" / "icrl_rule_memory.pt", device=str(device))
    head = nn.Linear(memory.concept_dim, len(label_names)).to(device)
    head.load_state_dict(torch.load(fold_dir / "icrl" / "prediction_head.pt",
                                    map_location=device, weights_only=False))
    head.eval()
    centroids = memory.get_centroids().to(device)
    confidences = torch.tensor(memory.get_confidences(), device=device)

    y_val, s1_logit_val, s2_logit_val, rc_val = collect_logits(
        val_loader, system1, memory, head, centroids, confidences, device)
    y_test, s1_logit_test, s2_logit_test, rc_test = collect_logits(
        test_loader, system1, memory, head, centroids, confidences, device)

    # ---- baseline: gated-override tren xac suat THO (doi chieu voi so da bao cao) ----
    p1_val_raw, p2_val_raw = F.softmax(s1_logit_val, -1), F.softmax(s2_logit_val, -1)
    p1_test_raw, p2_test_raw = F.softmax(s1_logit_test, -1), F.softmax(s2_logit_test, -1)
    pred1_raw, pred2_raw = p1_test_raw.argmax(1), p2_test_raw.argmax(1)
    pred_baseline, combo_baseline = gated_override_grid(
        p1_val_raw, rc_val, y_val, p1_test_raw, rc_test, y_test,
        p2_val_raw.argmax(1), p2_test_raw.argmax(1),
        p1_val_raw.max(1).values, p1_test_raw.max(1).values)

    # ---- temperature scaling (fit tren val, S1 va S2 rieng) ----
    T1 = fit_temperature(s1_logit_val.cpu(), y_val.cpu())
    T2 = fit_temperature(s2_logit_val.cpu(), y_val.cpu())
    p1_val, p2_val = apply_temperature(s1_logit_val, T1), apply_temperature(s2_logit_val, T2)
    p1_test, p2_test = apply_temperature(s1_logit_test, T1), apply_temperature(s2_logit_test, T2)
    pred1, pred2 = p1_test.argmax(1), p2_test.argmax(1)
    assert torch.equal(pred1, pred1_raw) and torch.equal(pred2, pred2_raw), \
        "Temperature scaling khong duoc doi argmax -- loi logic neu assert nay fail."

    ece_s1_before = ece(p1_test_raw.max(1).values.numpy(), (pred1_raw == y_test).numpy().astype(float))
    ece_s1_after = ece(p1_test.max(1).values.numpy(), (pred1 == y_test).numpy().astype(float))
    ece_s2_before = ece(p2_test_raw.max(1).values.numpy(), (pred2_raw == y_test).numpy().astype(float))
    ece_s2_after = ece(p2_test.max(1).values.numpy(), (pred2 == y_test).numpy().astype(float))

    strategies = {}

    # ---- weighted_avg ----
    best_alpha, best_acc = 0.5, -1.0
    for alpha in np.linspace(0, 1, 21):
        pred = (alpha * p1_val + (1 - alpha) * p2_val).argmax(1)
        a = accuracy_score(y_val, pred)
        if a > best_acc:
            best_acc, best_alpha = a, alpha
    strategies["weighted_avg"] = (best_alpha * p1_test + (1 - best_alpha) * p2_test).argmax(1)

    # ---- confidence_pick ----
    pick_s1 = p1_test.max(1).values >= p2_test.max(1).values
    pred_cp = torch.where(pick_s1, pred1, pred2)
    strategies["confidence_pick"] = pred_cp

    # ---- gated_override (tren xac suat da scale) ----
    pred_go, combo_go = gated_override_grid(
        p1_val, rc_val, y_val, p1_test, rc_test, y_test, pred2, pred2,
        p1_val.max(1).values, p1_test.max(1).values)
    strategies["gated_override"] = pred_go

    # ---- entropy_gated ----
    ent_val, ent_test = entropy_confidence(p1_val), entropy_confidence(p1_test)
    pred_eg, combo_eg = gated_override_grid(
        p1_val, rc_val, y_val, p1_test, rc_test, y_test, pred2, pred2,
        ent_val, ent_test)
    strategies["entropy_gated"] = pred_eg

    # ---- log_linear ----
    w1, w2 = fit_log_linear_weights(p1_val, p2_val, y_val)
    strategies["log_linear"] = log_linear_pool(p1_test, p2_test, w1, w2).argmax(1)

    # ---- naive_bayes ----
    prior = fit_class_prior(y_val, len(label_names))
    pred1_val, pred2_val = p1_val.argmax(1), p2_val.argmax(1)
    L1 = fit_confusion_likelihood(pred1_val, y_val, len(label_names))
    L2 = fit_confusion_likelihood(pred2_val, y_val, len(label_names))
    post = naive_bayes_fusion(pred1, pred2, prior, L1, L2)
    strategies["naive_bayes"] = post.argmax(1)

    # ---- rule_conf_override: suy ra tu pattern (s1 confidence nguoc huong,
    # chi rule_conf dung huong) -- bo han dieu kien "S1 khong tu tin" ----
    rc_thresh = fit_rule_conf_override_thresh(p1_val, p2_val, rc_val, y_val)
    strategies["rule_conf_override"] = rule_conf_override(p1_test, p2_test, rc_test, rc_thresh)

    # ---- rc_weighted_pool: trong so S2 theo TUNG MAU, ti le voi rule_conf ----
    rc_scale = fit_rc_weighted_scale(p1_val, p2_val, rc_val, y_val)
    strategies["rc_weighted_pool"] = rc_weighted_pool(p1_test, p2_test, rc_test, rc_scale).argmax(1)

    results = {}
    for name, pred in [("baseline", pred_baseline)] + list(strategies.items()):
        acc = accuracy_score(y_test, pred)
        f1 = f1_macro_2class(y_test.numpy(), pred.numpy())
        bidir = bidirectional_correction_counts(pred1, pred2, pred, y_test)
        results[name] = {"diagnosis_acc": acc, "diagnosis_f1": f1, **bidir}

    fold_summary = {
        "T1": T1, "T2": T2,
        "ece_s1_before": ece_s1_before, "ece_s1_after": ece_s1_after,
        "ece_s2_before": ece_s2_before, "ece_s2_after": ece_s2_after,
        "weighted_avg_alpha": float(best_alpha),
        "log_linear_w1": float(w1),
        "rule_conf_override_thresh": float(rc_thresh),
        "rc_weighted_scale": float(rc_scale),
        "gated_override_combo": combo_go,
        "entropy_gated_combo": combo_eg,
        "baseline_combo": combo_baseline,
        "strategies": results,
    }
    return fold_summary


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    cv_root, data_root, img_dir = Path(args.cv_root), Path(args.data_root), Path(args.img_dir)
    system1_root = Path(args.system1_root) if args.system1_root else cv_root
    label_names = args.label_names.split(",")

    all_folds = {}
    for k in range(args.num_folds):
        print(f"\n{'='*70}\n[FOLD {k}]\n{'='*70}")
        fold_summary = run_fold(k, cv_root, system1_root, data_root, img_dir, label_names,
                                 args.batch_size, args.num_workers, device)
        all_folds[f"fold_{k}"] = fold_summary
        print(f"  T1={fold_summary['T1']:.3f} T2={fold_summary['T2']:.3f}  "
              f"ECE_S1 {fold_summary['ece_s1_before']:.4f}->{fold_summary['ece_s1_after']:.4f}  "
              f"ECE_S2 {fold_summary['ece_s2_before']:.4f}->{fold_summary['ece_s2_after']:.4f}")
        for name, r in fold_summary["strategies"].items():
            print(f"  {name:<16s} acc={r['diagnosis_acc']:.4f}  f1={r['diagnosis_f1']:.4f}  "
                  f"s2_fixes_s1={r['s2_fixes_s1_count']}/{r['s1_wrong_s2_right_count']}  "
                  f"s1_fixes_s2={r['s1_fixes_s2_count']}/{r['s2_wrong_s1_right_count']}")

    strategy_names = list(next(iter(all_folds.values()))["strategies"].keys())
    summary = {}
    for name in strategy_names:
        accs = [all_folds[f"fold_{k}"]["strategies"][name]["diagnosis_acc"] for k in range(args.num_folds)]
        f1s = [all_folds[f"fold_{k}"]["strategies"][name]["diagnosis_f1"] for k in range(args.num_folds)]
        s2_fix = sum(all_folds[f"fold_{k}"]["strategies"][name]["s2_fixes_s1_count"] for k in range(args.num_folds))
        s1_wrong_s2_right = sum(all_folds[f"fold_{k}"]["strategies"][name]["s1_wrong_s2_right_count"] for k in range(args.num_folds))
        s1_fix = sum(all_folds[f"fold_{k}"]["strategies"][name]["s1_fixes_s2_count"] for k in range(args.num_folds))
        s2_wrong_s1_right = sum(all_folds[f"fold_{k}"]["strategies"][name]["s2_wrong_s1_right_count"] for k in range(args.num_folds))
        summary[name] = {
            "diagnosis_acc_mean": float(np.mean(accs)), "diagnosis_acc_std": float(np.std(accs, ddof=1)),
            "diagnosis_f1_mean": float(np.mean(f1s)), "diagnosis_f1_std": float(np.std(f1s, ddof=1)),
            "s2_fixes_s1_total": f"{s2_fix}/{s1_wrong_s2_right}",
            "s1_fixes_s2_total": f"{s1_fix}/{s2_wrong_s1_right}",
        }

    print(f"\n{'='*70}\n[SUMMARY across {args.num_folds} folds]\n{'='*70}")
    for name, s in summary.items():
        print(f"  {name:<16s} acc={s['diagnosis_acc_mean']*100:.2f}+-{s['diagnosis_acc_std']*100:.2f}%  "
              f"f1={s['diagnosis_f1_mean']*100:.2f}+-{s['diagnosis_f1_std']*100:.2f}%  "
              f"S2->S1 fix={s['s2_fixes_s1_total']}  S1->S2 fix={s['s1_fixes_s2_total']}")

    out = {"per_fold": all_folds, "summary": summary}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
