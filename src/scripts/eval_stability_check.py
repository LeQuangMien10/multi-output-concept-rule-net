"""
eval_stability_check.py - Kiem tra do on dinh cua ket qua cao nhat moi
(multi/soft/learned/per_fold = 80.53%, eval_ablation_crlmatched.py) so voi
baseline (single/hard/simple/per_fold = 79.43%), bang PAIRED BOOTSTRAP tren
TOAN BO ~909 anh test gop ca 5 fold (moi anh xuat hien dung 1 lan trong
CV chinh thuc, nen gop lai hop le).

Voi moi anh, co 1 cap (dung/sai theo baseline, dung/sai theo phuong an
moi) -- resample CO HOAN LAI tap cap nay nhieu lan, tinh lai do chenh
lech accuracy moi lan, de xem do chenh lech +1.1pp co von dinh (nam ngoai
khoang tin cay 95%) hay chi la nhieu ngau nhien tu mau nho.

Usage (local, inference only):
    python -m src.scripts.eval_stability_check --device cpu --n_boot 3000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.scripts.eval_ablation_crlmatched import (
    collect_fold, assemble, simple_hard, learned_soft, NUM_FOLDS,
)
from src.scripts.measure_contingency_multiscope import DEFAULT_IMG_DIR
import torch


def simple_hard_correct(a_cal, a_test, p2_key, cal_signal_key):
    """Nhu simple_hard() nhung tra ve mang dung/sai TUNG MAU thay vi accuracy gop."""
    s1c_cal, sig_cal, y_cal = a_cal["s1_conf"], a_cal[cal_signal_key], a_cal["y"]
    pred1_cal, pred2_cal = a_cal["pred1"], a_cal["pred2s" if p2_key == "p2s" else "pred2m"]
    thresh_grid = np.linspace(float(s1c_cal.min()), float(s1c_cal.max()), 9)
    sig_grid = np.linspace(float(sig_cal.min()), float(sig_cal.max()), 6)
    best_combo, best_acc = None, -1.0
    for t1 in thresh_grid:
        for t2 in sig_grid:
            override = (s1c_cal < t1) & (sig_cal > t2)
            pred = pred1_cal.clone(); pred[override] = pred2_cal[override]
            a = float((pred == y_cal).float().mean())
            if a > best_acc:
                best_acc, best_combo = a, (t1, t2)
    t1, t2 = best_combo
    s1c_t, sig_t = a_test["s1_conf"], a_test[cal_signal_key]
    pred1_t, pred2_t = a_test["pred1"], a_test["pred2s" if p2_key == "p2s" else "pred2m"]
    override_t = (s1c_t < t1) & (sig_t > t2)
    pred = pred1_t.clone(); pred[override_t] = pred2_t[override_t]
    return (pred == a_test["y"]).numpy()


def learned_soft_correct(a_cal, a_test, p2_key, feat_key):
    """Nhu learned_soft() nhung tra ve mang dung/sai TUNG MAU."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    pred2_key = "pred2s" if p2_key == "p2s" else "pred2m"
    dis_cal = (a_cal["pred1"] != a_cal[pred2_key]).numpy()
    X_cal = a_cal[feat_key][dis_cal].numpy()
    y_cal_label = (a_cal[pred2_key][dis_cal] == a_cal["y"][dis_cal]).long().numpy()
    pred1_test = a_test["pred1"].clone()
    if len(np.unique(y_cal_label)) < 2 or len(X_cal) < 6:
        return (pred1_test == a_test["y"]).numpy()
    scaler = StandardScaler().fit(X_cal)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(X_cal), y_cal_label)

    z_cal = torch.tensor(clf.decision_function(scaler.transform(X_cal)), dtype=torch.float32)
    p1_cal_dis, p2_cal_dis = a_cal["p1"][dis_cal], a_cal[p2_key][dis_cal]
    y_cal_dis = a_cal["y"][dis_cal]
    log_p1, log_p2 = p1_cal_dis.clamp(min=1e-8).log(), p2_cal_dis.clamp(min=1e-8).log()
    best_a, best_b, best_acc = 1.0, 0.0, -1.0
    for aa in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        for bb in (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0):
            w2 = torch.sigmoid(aa * z_cal + bb).unsqueeze(-1)
            w1 = 1.0 - w2
            pred = (w1 * log_p1 + w2 * log_p2).argmax(dim=-1)
            a = float((pred == y_cal_dis).float().mean())
            if a > best_acc:
                best_acc, best_a, best_b = a, aa, bb

    dis_test = (a_test["pred1"] != a_test[pred2_key]).numpy()
    if dis_test.sum() > 0:
        z_t = torch.tensor(clf.decision_function(scaler.transform(a_test[feat_key][dis_test].numpy())),
                            dtype=torch.float32)
        w2 = torch.sigmoid(best_a * z_t + best_b).unsqueeze(-1)
        w1 = 1.0 - w2
        p1_t = a_test["p1"][torch.from_numpy(dis_test)]
        p2_t = a_test[p2_key][torch.from_numpy(dis_test)]
        pred_dis = (w1 * p1_t.clamp(min=1e-8).log() + w2 * p2_t.clamp(min=1e-8).log()).argmax(dim=-1)
        idx = np.where(dis_test)[0]
        pred1_test[idx] = pred_dis
    return (pred1_test == a_test["y"]).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n_boot", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = torch.device(args.device)

    cv_root = Path("outputs/crlmatched_cv_gated_confmin03")
    system1_root = Path("outputs/crlmatched_cv_gated")
    data_root = Path("data/fitzpatrick17k_crl_matched_cv_official")

    print("Thu thap du lieu 5 fold...")
    fold_raw = [collect_fold(k, cv_root, system1_root, data_root, DEFAULT_IMG_DIR, device) for k in range(NUM_FOLDS)]
    fold_a = [{"val": assemble(fr["val"]), "test": assemble(fr["test"])} for fr in fold_raw]

    baseline_correct, winner_correct = [], []
    for k in range(NUM_FOLDS):
        a_cal, a_test = fold_a[k]["val"], fold_a[k]["test"]
        bc = simple_hard_correct(a_cal, a_test, "p2s", "rc")
        wc = learned_soft_correct(a_cal, a_test, "p2m", "feat_agg5")
        baseline_correct.append(bc); winner_correct.append(wc)
        print(f"  fold_{k}: n={len(bc)}  baseline_acc={bc.mean()*100:.2f}%  winner_acc={wc.mean()*100:.2f}%")

    baseline_correct = np.concatenate(baseline_correct)
    winner_correct = np.concatenate(winner_correct)
    n = len(baseline_correct)
    print(f"\nPooled n={n}  baseline_acc={baseline_correct.mean()*100:.2f}%  "
          f"winner_acc={winner_correct.mean()*100:.2f}%  diff={1e2*(winner_correct.mean()-baseline_correct.mean()):.2f}pp")

    rng = np.random.default_rng(args.seed)
    diffs, baseline_accs, winner_accs = [], [], []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        b_acc = baseline_correct[idx].mean()
        w_acc = winner_correct[idx].mean()
        baseline_accs.append(b_acc); winner_accs.append(w_acc); diffs.append(w_acc - b_acc)
    diffs = np.array(diffs); baseline_accs = np.array(baseline_accs); winner_accs = np.array(winner_accs)

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_diff_le_0 = float((diffs <= 0).mean())
    print(f"\n[Paired bootstrap, n_boot={args.n_boot}]")
    print(f"  baseline acc 95% CI: [{np.percentile(baseline_accs,2.5)*100:.2f}%, {np.percentile(baseline_accs,97.5)*100:.2f}%]")
    print(f"  winner   acc 95% CI: [{np.percentile(winner_accs,2.5)*100:.2f}%, {np.percentile(winner_accs,97.5)*100:.2f}%]")
    print(f"  diff (winner-baseline) 95% CI: [{lo*100:.2f}pp, {hi*100:.2f}pp]")
    print(f"  P(diff <= 0) qua bootstrap = {p_diff_le_0*100:.1f}%  (xap xi 1 phia p-value cho H0: khong co su khac biet)")


if __name__ == "__main__":
    main()
