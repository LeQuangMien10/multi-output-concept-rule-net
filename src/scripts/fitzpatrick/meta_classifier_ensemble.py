"""
meta_classifier_ensemble.py - Coi viec "tin S1 hay S2" nhu bai toan phan
loai nhi phan: fit mot logistic regression nho, tren cac dac trung cua MOI
MAU (khong phai 1 nguong/trong so toan cuc), de hoc CACH KET HOP tot nhat
tu du lieu, roi doc he so (coefficient) de biet dac trung nao thuc su co
anh huong -- thay cho tu thiet ke cong thuc (log-linear, naive-Bayes) va
hy vong no dung.

Nhan: label=1 (tin S2) neu S2 dung con S1 sai, label=0 (tin S1) neu S1
dung con S2 sai -- CHI dinh nghia duoc tren cac mau S1/S2 BAT DONG (pred1
!= pred2), vi khi 2 he thong dong y thi chon ai cung nhu nhau.

Dac trung (6 cai, 2 cai dau la "log-likelihood-ratio margin" -- chinh la
dai luong log-linear pooling dang dung, nhung o day duoc HOC trong so thay
vi dat tay):
  margin1 = log P1(pred1) - log P1(pred2)   -- S1 "tin" pred1 hon pred2 bao nhieu
  margin2 = log P2(pred2) - log P2(pred1)   -- S2 "tin" pred2 hon pred1 bao nhieu
  rule_conf, log1p(rule_n), rule_coherence, s1_entropy

Leave-one-fold-out: voi fold k, train logistic regression tren cac mau
BAT DONG trong VAL cua 4 fold CON LAI (gop lai duoc ~60-70 mau, du hon han
fit tren 1 fold rieng ~12-20 mau), roi du doan tren TEST cua fold k -- khong
leak, vi anh cua fold k chi xuat hien trong val/test cua chinh fold k (giao
thuc official round-robin, moi anh thuoc dung 1 fold).

Usage (local, inference only):
    python -m src.scripts.fitzpatrick.meta_classifier_ensemble \\
        --cv_root outputs/crlmatched_cv_gated_confmin03 \\
        --system1_root outputs/crlmatched_cv_gated \\
        --data_root data/fitzpatrick17k_crl_matched_cv_official \\
        --img_dir data/fitzpatrick17k/data/finalfitz17k
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.ensemble_strategies import (
    fit_temperature, apply_temperature, entropy_confidence,
    bidirectional_correction_counts,
)

FEATURE_NAMES = ["margin1", "margin2", "rule_conf", "log1p_rule_n", "rule_coherence", "s1_entropy"]


def parse_args():
    p = argparse.ArgumentParser(description="Meta-classifier (logistic regression) cho quyet dinh S1 vs S2.")
    p.add_argument("--cv_root", type=str, required=True)
    p.add_argument("--system1_root", type=str, default=None)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--label_names", type=str, default="benign,malignant")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


@torch.no_grad()
def collect(loader, system1, memory, head, centroids, confidences, device):
    ys, s1_logits, s2_logits, rc, rule_ids_all = [], [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        cv = soft_concept_vector(out)
        rule_ids, _ = memory.match(cv)
        ys.append(labels["label"].to(device))
        s1_logits.append(out["label"])
        s2_logits.append(head(centroids[rule_ids]))
        rc.append(confidences[rule_ids])
        rule_ids_all.append(rule_ids)
    return (torch.cat(ys), torch.cat(s1_logits), torch.cat(s2_logits),
            torch.cat(rc), torch.cat(rule_ids_all))


def build_features(y, p1, p2, rc, rule_ids, memory):
    """Tra ve (X [N,6], disagree_mask, pred1, pred2) cho toan bo split."""
    pred1, pred2 = p1.argmax(1), p2.argmax(1)
    N = len(y)
    idx = torch.arange(N)
    log_p1, log_p2 = p1.clamp(min=1e-8).log(), p2.clamp(min=1e-8).log()
    margin1 = log_p1[idx, pred1] - log_p1[idx, pred2]
    margin2 = log_p2[idx, pred2] - log_p2[idx, pred1]
    rule_n = torch.tensor([float(memory._n[r]) for r in rule_ids.tolist()])
    rule_coh = torch.tensor([float(memory._compute_coherence(r)) for r in rule_ids.tolist()])
    s1_ent = entropy_confidence(p1)
    X = torch.stack([margin1, margin2, rc, torch.log1p(rule_n), rule_coh, s1_ent], dim=1)
    disagree = pred1 != pred2
    return X.numpy(), disagree.numpy(), pred1, pred2


def run_fold_data(k, cv_root, system1_root, data_root, img_dir, label_names,
                   batch_size, num_workers, device):
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

    y_val, s1_logit_val, s2_logit_val, rc_val, rid_val = collect(
        val_loader, system1, memory, head, centroids, confidences, device)
    T1 = fit_temperature(s1_logit_val.cpu(), y_val.cpu())
    T2 = fit_temperature(s2_logit_val.cpu(), y_val.cpu())
    p1_val, p2_val = apply_temperature(s1_logit_val, T1), apply_temperature(s2_logit_val, T2)

    y_test, s1_logit_test, s2_logit_test, rc_test, rid_test = collect(
        test_loader, system1, memory, head, centroids, confidences, device)
    p1_test, p2_test = apply_temperature(s1_logit_test, T1), apply_temperature(s2_logit_test, T2)

    X_val, dis_val, pred1_val, pred2_val = build_features(y_val, p1_val, p2_val, rc_val, rid_val, memory)
    X_test, dis_test, pred1_test, pred2_test = build_features(y_test, p1_test, p2_test, rc_test, rid_test, memory)

    label_val = (pred2_val.numpy() == y_val.numpy()).astype(int)  # 1 = tin S2 dung
    label_test = (pred2_test.numpy() == y_test.numpy()).astype(int)

    return {
        "X_val": X_val, "dis_val": dis_val, "label_val": label_val,
        "X_test": X_test, "dis_test": dis_test, "label_test": label_test,
        "y_test": y_test.numpy(), "pred1_test": pred1_test.numpy(), "pred2_test": pred2_test.numpy(),
    }


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    cv_root, data_root, img_dir = Path(args.cv_root), Path(args.data_root), Path(args.img_dir)
    system1_root = Path(args.system1_root) if args.system1_root else cv_root
    label_names = args.label_names.split(",")

    fold_data = [run_fold_data(k, cv_root, system1_root, data_root, img_dir, label_names,
                                args.batch_size, args.num_workers, device)
                 for k in range(args.num_folds)]

    all_coefs = []
    accs, f1s = [], []
    total_s2_fix, total_s1_wrong_s2_right = 0, 0
    total_s1_fix, total_s2_wrong_s1_right = 0, 0

    for k in range(args.num_folds):
        # Gop val-disagree cua 4 fold KHAC fold k de train
        X_train = np.concatenate([fold_data[j]["X_val"][fold_data[j]["dis_val"]]
                                   for j in range(args.num_folds) if j != k], axis=0)
        y_train = np.concatenate([fold_data[j]["label_val"][fold_data[j]["dis_val"]]
                                   for j in range(args.num_folds) if j != k], axis=0)

        scaler = StandardScaler().fit(X_train)
        # class_weight="balanced": nhom "tin S2" chi ~80/280=29% so mau bat dong --
        # khong can bang se khien logistic regression ngham dinh thien ve nhom da so.
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            scaler.transform(X_train), y_train)
        all_coefs.append(clf.coef_[0])

        d = fold_data[k]

        # Do ngUONG tren VAL CUA CHINH fold k (khong dung de train model, vi
        # train chi dung 4 fold KHAC) -- tim diem van hanh tot nhat co the co
        # cua model nay, cong bang hon so voi mac dinh threshold=0.5.
        dis_val_k = d["dis_val"]
        proba_val_k = clf.predict_proba(scaler.transform(d["X_val"][dis_val_k]))[:, 1]
        label_val_k = d["label_val"][dis_val_k]
        best_t, best_val_acc = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 19):
            acc_t = float(((proba_val_k >= t).astype(int) == label_val_k).mean())
            if acc_t > best_val_acc:
                best_val_acc, best_t = acc_t, float(t)

        dis_test = d["dis_test"]
        proba_test = clf.predict_proba(scaler.transform(d["X_test"][dis_test]))[:, 1]
        meta_pred_disagree = (proba_test >= best_t).astype(int)

        pred_final = d["pred1_test"].copy()
        # Tren cac mau bat dong: chon S2 neu classifier noi tin S2 (label=1)
        idx_dis = np.where(dis_test)[0]
        pred_final[idx_dis[meta_pred_disagree == 1]] = d["pred2_test"][idx_dis[meta_pred_disagree == 1]]

        y_test, pred1_test, pred2_test = d["y_test"], d["pred1_test"], d["pred2_test"]
        acc = float((pred_final == y_test).mean())
        tp = int(((pred_final == 1) & (y_test == 1)).sum()); fp = int(((pred_final == 1) & (y_test == 0)).sum())
        fn = int(((pred_final == 0) & (y_test == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1_pos = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        tp0 = int(((pred_final == 0) & (y_test == 0)).sum()); fp0 = int(((pred_final == 0) & (y_test == 1)).sum())
        fn0 = int(((pred_final == 1) & (y_test == 0)).sum())
        prec0 = tp0 / (tp0 + fp0) if (tp0 + fp0) else 0.0
        rec0 = tp0 / (tp0 + fn0) if (tp0 + fn0) else 0.0
        f1_neg = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) else 0.0
        f1 = (f1_pos + f1_neg) / 2

        bidir = bidirectional_correction_counts(
            torch.tensor(pred1_test), torch.tensor(pred2_test), torch.tensor(pred_final), torch.tensor(y_test))
        accs.append(acc); f1s.append(f1)
        total_s2_fix += bidir["s2_fixes_s1_count"]; total_s1_wrong_s2_right += bidir["s1_wrong_s2_right_count"]
        total_s1_fix += bidir["s1_fixes_s2_count"]; total_s2_wrong_s1_right += bidir["s2_wrong_s1_right_count"]
        print(f"fold_{k}: acc={acc:.4f}  f1={f1:.4f}  thresh={best_t:.2f} (val_acc={best_val_acc:.4f})  "
              f"s2_fixes_s1={bidir['s2_fixes_s1_count']}/{bidir['s1_wrong_s2_right_count']}  "
              f"s1_fixes_s2={bidir['s1_fixes_s2_count']}/{bidir['s2_wrong_s1_right_count']}  "
              f"(train n={len(y_train)}, test disagree n={int(dis_test.sum())})")

    print(f"\n[SUMMARY] meta_classifier: acc={np.mean(accs)*100:.2f}+-{np.std(accs, ddof=1)*100:.2f}%  "
          f"f1={np.mean(f1s)*100:.2f}+-{np.std(f1s, ddof=1)*100:.2f}%  "
          f"S2->S1 fix={total_s2_fix}/{total_s1_wrong_s2_right}  "
          f"S1->S2 fix={total_s1_fix}/{total_s2_wrong_s1_right}")

    coefs = np.array(all_coefs)  # [5, 6]
    print(f"\n[FEATURE IMPORTANCE] he so logistic regression (da standardize), "
          f"trung binh +- std qua 5 model leave-one-fold-out:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:<16s}: {coefs[:, i].mean():+.3f} +- {coefs[:, i].std():.3f}   "
              f"(tung fold: {[round(c, 2) for c in coefs[:, i]]})")


if __name__ == "__main__":
    main()
