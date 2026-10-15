"""
eval_concept_metrics_scope.py - So sanh concept-level accuracy/F1 giua
System~1 (tu du doan concept) va System~2 (pattern cua rule ICRL da khop),
tong quat cho ca 4 scope (crlmatched 5-fold CV, fulldata/concept_only/
pad_ufes20 1-split) bang checkpoint hien tai (Wilson score + lazy-collapse-
fixed S1) -- tra loi cau hoi: System~2 co dong gop gi ve CONCEPT (khong
phai nhan chan doan) khong, va co "tot hon" S1 khong.

Cong thuc giong eval_crlmatched_metrics.py/eval_concept_metrics.py: tung
concept rieng (accuracy_score + f1_macro 2 lop), BO QUA concept nao GT toan
0 trong tap dang xet, roi trung binh qua cac concept con lai (macro theo
CONCEPT). S2's concept prediction = pattern (threshold 0.5) cua CENTROID
cua rule da match -- khac S1's rieng cho tung anh.

Usage (local, inference only):
    python -m src.scripts.eval_concept_metrics_scope --scope crlmatched \\
        --cv_root outputs/crlmatched_cv_gated_confmin03 \\
        --system1_root outputs/crlmatched_cv_gated \\
        --data_root data/fitzpatrick17k_crl_matched_cv_official \\
        --img_dir data/fitzpatrick17k/data/finalfitz17k \\
        --output_json outputs/concept_metrics_scope/crlmatched.json

    python -m src.scripts.eval_concept_metrics_scope --scope pad_ufes20 \\
        --icrl_root outputs/scope_rebuild/pad_ufes20 \\
        --icrl_dir outputs/scope_rebuild/pad_ufes20/icrl \\
        --data_dir data/PAD-UFES-20_prepared --img_dir data/PAD-UFES-20/images \\
        --output_json outputs/concept_metrics_scope/pad_ufes20.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.scripts.run_scope_rebuild import SCOPE_CONFIGS
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.fitzpatrick.system1 import soft_concept_vector


def parse_args():
    p = argparse.ArgumentParser(description="So sanh concept metrics S1 vs S2, tong quat 4 scope.")
    p.add_argument("--scope", type=str, required=True,
                    choices=["crlmatched", "fulldata", "concept_only", "pad_ufes20"])
    p.add_argument("--cv_root", type=str, default=None, help="Chi dung cho scope=crlmatched.")
    p.add_argument("--system1_root", type=str, default=None, help="Chi dung cho scope=crlmatched.")
    p.add_argument("--data_root", type=str, default=None, help="Chi dung cho scope=crlmatched.")
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--icrl_root", type=str, default=None, help="Chi dung cho 1-split scope.")
    p.add_argument("--icrl_dir", type=str, default=None, help="Chi dung cho 1-split scope.")
    p.add_argument("--data_dir", type=str, default=None, help="Chi dung cho 1-split scope.")
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_json", type=str, required=True)
    return p.parse_args()


def accuracy_score(y_true, y_pred):
    return float((y_true == y_pred).mean())


def binary_f1(y_true, y_pred, pos_label):
    tp = int(((y_pred == pos_label) & (y_true == pos_label)).sum())
    fp = int(((y_pred == pos_label) & (y_true != pos_label)).sum())
    fn = int(((y_pred != pos_label) & (y_true == pos_label)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def f1_macro_both_classes(y_true, y_pred):
    return (binary_f1(y_true, y_pred, 0) + binary_f1(y_true, y_pred, 1)) / 2


def report_dict(pred: np.ndarray, gt: np.ndarray, concept_names: list[str]) -> dict:
    """Tung concept rieng, bo qua GT toan 0, roi trung binh -- giong eval_concept_metrics.py."""
    accs, f1s, f1s_pos = [], [], []
    n_skipped = 0
    for i in range(len(concept_names)):
        true_vals, pred_vals = gt[:, i], pred[:, i]
        if true_vals.sum() == 0:
            n_skipped += 1
            continue
        accs.append(accuracy_score(true_vals, pred_vals))
        f1s.append(f1_macro_both_classes(true_vals, pred_vals))
        f1s_pos.append(binary_f1(true_vals, pred_vals, 1))
    present_mask, absent_mask = (gt == 1), (gt == 0)
    recall_present = float((pred[present_mask] == 1).mean()) if present_mask.any() else float("nan")
    spec_absent = float((pred[absent_mask] == 0).mean()) if absent_mask.any() else float("nan")
    return {
        "n_concepts_total": len(concept_names), "n_concepts_skipped_allzero": n_skipped,
        "concept_acc": float(np.mean(accs)) if accs else 0.0,
        "concept_f1": float(np.mean(f1s)) if f1s else 0.0,
        "concept_f1_pos_only": float(np.mean(f1s_pos)) if f1s_pos else 0.0,
        "recall_present": recall_present, "specificity_absent": spec_absent,
    }


@torch.no_grad()
def collect_fitzpatrick(loader, system1, memory, centroids_hard, device):
    """concept_mask la SCALAR/anh (broadcast qua moi concept cua anh do)."""
    all_s1, all_s2, all_gt = [], [], []
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        concept_mask = labels["concept_mask"]
        concept_gt = labels["concepts"]
        s1_hard = (torch.sigmoid(out["concepts"]) > 0.5).float()
        keep = concept_mask.bool()
        all_s1.append(s1_hard[keep].cpu())
        all_gt.append(concept_gt[keep].cpu())
        cv = soft_concept_vector(out)
        rule_ids, _ = memory.match(cv)
        s2_hard = centroids_hard[rule_ids]
        all_s2.append(s2_hard[keep].cpu())
    return (torch.cat(all_s1).numpy().astype(np.int32),
            torch.cat(all_s2).numpy().astype(np.int32),
            torch.cat(all_gt).numpy().astype(np.int32))


@torch.no_grad()
def collect_pad_ufes20(loader, system1, memory, centroids_hard, device):
    """concept_mask la VECTOR [B, num_concepts] (UNK theo tung field rieng)."""
    all_s1, all_s2, all_gt, all_mask = [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        concept_mask = labels["concept_mask"]
        concept_gt = labels["concepts"]
        s1_hard = (torch.sigmoid(out["concepts"]) > 0.5).float()
        all_s1.append(s1_hard.cpu())
        all_gt.append(concept_gt.cpu())
        all_mask.append(concept_mask.cpu())
        cv = soft_concept_vector(out)
        rule_ids, _ = memory.match(cv)
        all_s2.append(centroids_hard[rule_ids].cpu())
    s1 = torch.cat(all_s1).numpy().astype(np.int32)
    s2 = torch.cat(all_s2).numpy().astype(np.int32)
    gt = torch.cat(all_gt).numpy().astype(np.int32)
    mask = torch.cat(all_mask).numpy().astype(bool)
    return s1, s2, gt, mask


def report_dict_masked_per_column(pred, gt, mask, concept_names):
    """Ban report_dict() cho mask THEO TUNG CONCEPT (pad_ufes20) -- loc theo
    cot truoc khi tinh, thay vi loc theo hang nhu ban fitzpatrick."""
    accs, f1s, f1s_pos = [], [], []
    n_skipped = 0
    recall_num, recall_den, spec_num, spec_den = 0, 0, 0, 0
    for i in range(len(concept_names)):
        keep = mask[:, i]
        true_vals, pred_vals = gt[keep, i], pred[keep, i]
        if true_vals.sum() == 0:
            n_skipped += 1
            continue
        accs.append(accuracy_score(true_vals, pred_vals))
        f1s.append(f1_macro_both_classes(true_vals, pred_vals))
        f1s_pos.append(binary_f1(true_vals, pred_vals, 1))
        present, absent = (true_vals == 1), (true_vals == 0)
        recall_num += int((pred_vals[present] == 1).sum()); recall_den += int(present.sum())
        spec_num += int((pred_vals[absent] == 0).sum()); spec_den += int(absent.sum())
    return {
        "n_concepts_total": len(concept_names), "n_concepts_skipped_allzero": n_skipped,
        "concept_acc": float(np.mean(accs)) if accs else 0.0,
        "concept_f1": float(np.mean(f1s)) if f1s else 0.0,
        "concept_f1_pos_only": float(np.mean(f1s_pos)) if f1s_pos else 0.0,
        "recall_present": recall_num / recall_den if recall_den else float("nan"),
        "specificity_absent": spec_num / spec_den if spec_den else float("nan"),
    }


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

    if args.scope == "crlmatched":
        from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders
        cv_root, system1_root, data_root = Path(args.cv_root), Path(args.system1_root), Path(args.data_root)
        all_s1, all_s2, all_gt = [], [], []
        per_fold = {}
        for k in range(args.num_folds):
            fold_dir = cv_root / f"fold_{k}"
            system1, image_size = load_system1(system1_root / f"fold_{k}" / "system1" / "best_model.pt", device)
            train_loader, val_loader, test_loader = make_loaders(
                data_root / f"fold_{k}", args.img_dir, image_size, args.batch_size, args.num_workers)
            loader = val_loader if args.split == "val" else test_loader
            concept_names = loader.dataset.concept_names
            memory = ICRLRuleMemory.load(fold_dir / "icrl" / "icrl_rule_memory.pt", device=str(device))
            centroids_hard = (memory.get_centroids()[:, :len(concept_names)] >= 0.5).float().to(device)
            s1, s2, gt = collect_fitzpatrick(loader, system1, memory, centroids_hard, device)
            per_fold[f"fold_{k}"] = {"s1": report_dict(s1, gt, concept_names),
                                       "s2": report_dict(s2, gt, concept_names)}
            all_s1.append(s1); all_s2.append(s2); all_gt.append(gt)
            print(f"[fold_{k}] S1 concept_acc={per_fold[f'fold_{k}']['s1']['concept_acc']*100:.2f}%  "
                  f"S2 concept_acc={per_fold[f'fold_{k}']['s2']['concept_acc']*100:.2f}%")
        s1_all, s2_all, gt_all = np.concatenate(all_s1), np.concatenate(all_s2), np.concatenate(all_gt)
        pooled = {"s1": report_dict(s1_all, gt_all, concept_names), "s2": report_dict(s2_all, gt_all, concept_names)}
        out = {"scope": "crlmatched", "per_fold": per_fold, "pooled": pooled}

    else:
        cfg = SCOPE_CONFIGS[args.scope]
        module = cfg["module"]
        if module == "fitzpatrick":
            from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders
        else:
            from src.scripts.pad_ufes20.train_icrl import load_system1, make_loaders

        icrl_root, icrl_dir = Path(args.icrl_root), Path(args.icrl_dir)
        system1, image_size = load_system1(icrl_root / "system1" / "best_model.pt", device)
        train_loader, val_loader, test_loader = make_loaders(Path(args.data_dir), args.img_dir, image_size,
                                                               args.batch_size, args.num_workers)
        loader = val_loader if args.split == "val" else test_loader
        concept_names = loader.dataset.concept_names
        memory = ICRLRuleMemory.load(icrl_dir / "icrl_rule_memory.pt", device=str(device))
        centroids_hard = (memory.get_centroids()[:, :len(concept_names)] >= 0.5).float().to(device)

        if module == "fitzpatrick":
            s1, s2, gt = collect_fitzpatrick(loader, system1, memory, centroids_hard, device)
            out = {"scope": args.scope, "s1": report_dict(s1, gt, concept_names),
                   "s2": report_dict(s2, gt, concept_names)}
        else:
            s1, s2, gt, mask = collect_pad_ufes20(loader, system1, memory, centroids_hard, device)
            out = {"scope": args.scope,
                   "s1": report_dict_masked_per_column(s1, gt, mask, concept_names),
                   "s2": report_dict_masked_per_column(s2, gt, mask, concept_names)}
        print(f"[{args.scope}] S1 concept_acc={out['s1']['concept_acc']*100:.2f}% f1={out['s1']['concept_f1']:.4f}  "
              f"S2 concept_acc={out['s2']['concept_acc']*100:.2f}% f1={out['s2']['concept_f1']:.4f}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
