"""
bootstrap_ensemble_strategies.py - Paired bootstrap tren ket qua da co san
cua eval_ensemble_infotheory.py (outputs/ensemble_infotheory/ensemble_infotheory.json,
phai chua "y_test"/"pred_test" moi fold -- chay lai eval_ensemble_infotheory.py
mot lan de co truong nay neu file cu chua co).

Muc dich: kiem tra xem khoang cach nho (0.1-0.8pp) giua cac chien luoc co
phai chi la NHIEU DO VIEC CHON MAU test cu the (khong phai do cach chia
fold) hay khong. Resample CO LAP lai anh test TRONG TUNG FOLD (giu nguyen
cau truc 5-fold), dung CUNG MOT BO CHI SO resample cho MOI chien luoc trong
1 lan lap -- la paired bootstrap, loai bo duoc phan nhieu chung giua cac
chien luoc (vi chung du doan tren CUNG anh).

GIOI HAN QUAN TRONG (can nhac ro khi doc ket qua): day CHI danh gia nhieu
do viec LAY MAU ANH TEST trong 1 cach chia fold DUY NHAT da co. No KHONG
tra loi duoc "neu chia fold khac thi sao" -- muon biet dieu do phai tao
mot cach chia 5-fold khac va TRAIN LAI System~1 tu dau (Kaggle), vi
anh nao thuoc fold nao se khac, keo theo S1 duoc train tren du lieu khac.
Bootstrap o day la buoc re, lam truoc de quyet dinh co dang chi Kaggle cho
viec do khong.

Usage (local, doc lai file JSON da co, khong infer lai):
    python -m src.scripts.fitzpatrick.bootstrap_ensemble_strategies \\
        --input_json outputs/ensemble_infotheory/ensemble_infotheory.json \\
        --n_boot 2000
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Paired bootstrap tren ket qua ensemble da co.")
    p.add_argument("--input_json", type=str, required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--baseline_name", type=str, default="baseline")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data = json.load(open(args.input_json))
    per_fold = data["per_fold"]
    fold_names = sorted(per_fold.keys(), key=lambda s: int(s.split("_")[1]))
    strategy_names = list(per_fold[fold_names[0]]["pred_test"].keys())

    # Chuan bi numpy array moi fold: y, va pred cua tung chien luoc.
    y_by_fold = [np.array(per_fold[f]["y_test"]) for f in fold_names]
    pred_by_fold = [{name: np.array(per_fold[f]["pred_test"][name]) for name in strategy_names}
                     for f in fold_names]
    n_by_fold = [len(y) for y in y_by_fold]

    boot_mean_acc = {name: np.zeros(args.n_boot) for name in strategy_names}
    for b in range(args.n_boot):
        fold_accs = {name: [] for name in strategy_names}
        for fi, n in enumerate(n_by_fold):
            idx = rng.integers(0, n, size=n)  # resample co lap, CUNG idx cho moi chien luoc
            y_r = y_by_fold[fi][idx]
            for name in strategy_names:
                pred_r = pred_by_fold[fi][name][idx]
                fold_accs[name].append(float((pred_r == y_r).mean()))
        for name in strategy_names:
            boot_mean_acc[name][b] = float(np.mean(fold_accs[name]))

    base = boot_mean_acc[args.baseline_name]
    print(f"{'strategy':<20s}{'boot_mean':>10s}{'CI_lo':>8s}{'CI_hi':>8s}"
          f"{'diff_vs_base':>14s}{'diff_CI_lo':>12s}{'diff_CI_hi':>12s}{'P(better)':>10s}")
    for name in strategy_names:
        acc = boot_mean_acc[name]
        lo, hi = np.percentile(acc, [2.5, 97.5])
        diff = acc - base  # PAIRED (cung iteration, cung resampled test set)
        dlo, dhi = np.percentile(diff, [2.5, 97.5])
        p_better = float((diff > 0).mean())
        print(f"{name:<20s}{acc.mean()*100:>9.2f}%{lo*100:>7.2f}%{hi*100:>7.2f}%"
              f"{diff.mean()*100:>+13.2f}%{dlo*100:>+11.2f}%{dhi*100:>+11.2f}%{p_better:>10.3f}")

    print("\n(CI = khoang tin cay 95% cua accuracy trung binh 5-fold qua bootstrap.")
    print(" diff_CI chua 0 => khac biet voi baseline KHONG co y nghia thong ke duoi nhieu lay mau nay.")
    print(" P(better) = ti le lan bootstrap chien luoc nay > baseline, paired.)")
    print("\nNHAC LAI GIOI HAN: day la nhieu do LAY MAU ANH TEST trong 1 cach chia fold")
    print("DUY NHAT -- khong phai nhieu do CACH CHIA FOLD. Xem docstring dau file.")


if __name__ == "__main__":
    main()
