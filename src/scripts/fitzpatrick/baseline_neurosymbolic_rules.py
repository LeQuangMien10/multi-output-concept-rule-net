"""
baseline_neurosymbolic_rules.py - Baseline Basci et al. (CEUR/TRUST-AI 2025,
"Explaining Concept Drift via Neuro-Symbolic Rules") cho Fitzpatrick17k,
scope CRL-matched (48 concept SkinCon, 2 lop benign/malignant).
==============================================================================

Khac CRL/ICRL: KHONG train mot concept-supervised model. Basci et al. trich
concept POST-HOC tu kernel activation cua mot classifier da train xong CHI
bang nhan (khong concept supervision) -- classifier dau vao phai la checkpoint
tu train_system1_baseline.py voi --concept_loss_weight 0.0 (xem script do).
Dung recipe CHUAN (khong can recipe rieng cho CRL-matched cua ICRL) vi recipe
do duoc tinh chinh de tranh "lazy collapse" cua concept BCE loss mat can bang
-- khong lien quan khi concept_loss_weight=0 (khong co concept loss nao ca).

Bien the correlation-based (uu tien vi la dong gop chinh cua bai, khac biet
that voi System~1 -- bien the MLP-based thuc chat la train them 1 concept
head, qua giong ICRL, khong con la baseline doc lap):
  1. Kernel activation = L1-norm (trung binh |activation| tren khong gian)
     cua layer conv cuoi cung (backbone.layer4, TRUOC avgpool), tren tap
     train.
  2. Voi moi concept: tinh point-biserial correlation (= Pearson voi bien
     nhi phan) giua MOI kernel va nhan concept that (Oracle = SkinCon ground
     truth), chi tren anh co concept_mask=1. Chon kernel |corr| lon nhat.
  3. Nhi phan hoa activation cua kernel do bang percentile threshold, calib
     theo ty le duong tinh THAT cua concept do trong tap con nay (vd concept
     hiem 5% -> chi 5% anh co activation cao nhat duoc danh dau duong tinh).
  4. Fit mot decision tree (sklearn, giong tinh than C4.5 ho dung, co
     max_depth de kiem soat do phuc tap) tren bang concept nhi phan (toan bo
     train) de XAP XI DU DOAN CUA CHINH CLASSIFIER (fidelity-based rule
     extraction -- dung phuong phap goc, khong fit vao ground truth).

Metric xuat ra cung dinh dang voi eval_crlmatched_metrics.py de dien thang
vao Table 1 (them 1 hang baseline):
  - concept_acc/concept_f1: cong thuc CRL that (macro theo concept, bo qua
    concept GT toan 0), tren TEST, chi anh co concept_mask=1.
  - diagnosis_acc/diagnosis_f1: decision tree du doan tren TEST so voi NHAN
    THAT (de so sanh cong bang voi cac baseline khac trong Table 1, khac
    "fidelity" ma bai goc nhan manh -- ghi them fidelity_to_classifier de
    tham khao).

Usage (Kaggle):
    python -m src.scripts.fitzpatrick.baseline_neurosymbolic_rules \\
        --data_dir /kaggle/input/datasets/lquangmin/fitzpatrick17k-crl-matched \\
        --img_dir /kaggle/input/datasets/lquangmin/fitzpatrick17k/data/finalfitz17k \\
        --classifier_ckpt /kaggle/working/outputs/fitzpatrick_basci_classifier_seed{SEED}/best_model.pt \\
        --seed {SEED} \\
        --output_json /kaggle/working/outputs/basci_seed{SEED}.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders


def _tree_max_depth_type(s: str):
    return None if s.lower() == "none" else int(s)


def parse_args():
    p = argparse.ArgumentParser(description="Basci et al. neuro-symbolic rule baseline, CRL-matched scope.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--classifier_ckpt", type=str, required=True,
                    help="Checkpoint tu train_system1_baseline.py --concept_loss_weight 0.0.")
    p.add_argument("--label_names", type=str, default="benign,malignant")
    p.add_argument("--seed", type=int, required=True, help="Chi de ghi vao JSON, khong dat lai seed o day.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--tree_max_depth", type=_tree_max_depth_type, default=5,
                    help="Do sau toi da decision tree (giu rule ngan gon/doc duoc, giong C4.5 co pruning). "
                         "Dat 'none' de khong gioi han (sklearn max_depth=None) -- dung khi can do "
                         "trade-off accuracy-vs-interpretability.")
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--output_tree_txt", type=str, default=None,
                    help="Neu dat, ghi rule text (sklearn export_text) ra file nay de doc thu cong.")
    return p.parse_args()


# ----- cong thuc metric giong het eval_crlmatched_metrics.py (giu file doc lap, khong import cheo) -----

def accuracy_score(y_true, y_pred):
    return float((y_true == y_pred).mean())


def binary_f1(y_true, y_pred, pos_label):
    tp = int(((y_pred == pos_label) & (y_true == pos_label)).sum())
    fp = int(((y_pred == pos_label) & (y_true != pos_label)).sum())
    fn = int(((y_pred != pos_label) & (y_true == pos_label)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def f1_macro_2class(y_true, y_pred):
    return (binary_f1(y_true, y_pred, 0) + binary_f1(y_true, y_pred, 1)) / 2


def concept_metrics(pred, gt, concept_names):
    """Cong thuc CRL that: macro theo CONCEPT, bo qua concept GT toan 0."""
    accs, f1s = [], []
    for i in range(len(concept_names)):
        true_vars, pred_vars = gt[:, i], pred[:, i]
        if true_vars.sum() == 0:
            continue
        accs.append(accuracy_score(true_vars, pred_vars))
        f1s.append(f1_macro_2class(true_vars, pred_vars))
    return float(sum(accs) / len(accs)), float(sum(f1s) / len(f1s))


# ----- Basci et al. core method -----

class KernelActivationHook:
    """Bat output cua backbone.layer4 (feature map TRUOC avgpool, [B,C,H,W]) --
    backbone(x) cua FitzpatrickSystem1 tra ve vector DA POOL (fc=Identity),
    nen khong dung truc tiep duoc, phai hook vao layer conv cuoi cung."""

    def __init__(self, layer: torch.nn.Module):
        self.activation = None
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.activation = output.detach()

    def l1_kernel_activation(self) -> torch.Tensor:
        """[B, C, H, W] -> [B, C] (trung binh |activation| tren khong gian)."""
        return self.activation.abs().mean(dim=(2, 3))

    def remove(self):
        self._handle.remove()


@torch.no_grad()
def collect_activations_and_targets(loader, classifier, hook, device):
    """1 pass qua loader -> kernel L1-activation [N,C], concept GT [N,num_concepts],
    concept_mask [N], nhan that [N], nhan classifier tu du doan [N]."""
    all_act, all_concepts, all_mask, all_y, all_pred = [], [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        out = classifier(images)
        act = hook.l1_kernel_activation()
        all_act.append(act.cpu())
        all_concepts.append(labels["concepts"])
        all_mask.append(labels["concept_mask"])
        all_y.append(labels["label"])
        all_pred.append(out["label"].argmax(dim=1).cpu())
    return (torch.cat(all_act).numpy(), torch.cat(all_concepts).numpy(),
            torch.cat(all_mask).numpy(), torch.cat(all_y).numpy(),
            torch.cat(all_pred).numpy())


def batch_point_biserial(activations: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """corr[k] = Pearson(activations[:,k], targets) -- tuong duong point-biserial
    khi targets nhi phan. Kernel co activation hang so (std=0) -> corr=0."""
    act_c = activations - activations.mean(axis=0, keepdims=True)
    act_std = activations.std(axis=0)
    tgt_c = targets - targets.mean()
    tgt_std = targets.std()
    denom = act_std * tgt_std
    cov = (act_c * tgt_c[:, None]).mean(axis=0)
    return np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-8)


def fit_concept_extractors(train_act, train_concepts, train_mask, concept_names):
    """Voi moi concept: chon kernel |corr| lon nhat (chi tren anh concept_mask=1),
    threshold = percentile ung voi ty le duong tinh THAT cua concept do trong
    cung tap con nay (percentile-based binarization)."""
    keep = train_mask.astype(bool)
    extractors = []
    for i, name in enumerate(concept_names):
        target = train_concepts[keep, i]
        pos_rate = target.mean()
        if target.sum() == 0 or target.sum() == len(target):
            # Concept toan 0 hoac toan 1 trong tap con nay -- khong co tin
            # hieu de hoc kernel nao, luon du doan theo da so.
            extractors.append({"concept": name, "kernel": -1, "sign": 1.0,
                                "threshold": 0.0, "constant_pred": float(pos_rate > 0.5)})
            continue
        corr = batch_point_biserial(train_act[keep], target)
        k = int(np.argmax(np.abs(corr)))
        sign = 1.0 if corr[k] >= 0 else -1.0
        signed_act = train_act[keep, k] * sign
        threshold = float(np.percentile(signed_act, 100.0 * (1.0 - pos_rate)))
        extractors.append({"concept": name, "kernel": k, "sign": sign,
                            "threshold": threshold, "constant_pred": None})
    return extractors


def apply_concept_extractors(activations: np.ndarray, extractors: list[dict]) -> np.ndarray:
    n = activations.shape[0]
    out = np.zeros((n, len(extractors)), dtype=int)
    for i, ext in enumerate(extractors):
        if ext["constant_pred"] is not None:
            out[:, i] = int(ext["constant_pred"])
            continue
        signed_act = activations[:, ext["kernel"]] * ext["sign"]
        out[:, i] = (signed_act > ext["threshold"]).astype(int)
    return out


@torch.no_grad()
def main():
    from sklearn.tree import DecisionTreeClassifier, export_text  # import cuc bo: chi Kaggle/may chay that can

    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    label_names = args.label_names.split(",")

    classifier, image_size = load_system1(Path(args.classifier_ckpt), device)
    train_loader, _val_loader, test_loader = make_loaders(
        Path(args.data_dir), args.img_dir, image_size, args.batch_size, args.num_workers,
    )
    concept_names = test_loader.dataset.concept_names

    hook = KernelActivationHook(classifier.backbone.layer4)
    train_act, train_concepts, train_mask, _train_y, train_pred = collect_activations_and_targets(
        train_loader, classifier, hook, device)
    test_act, test_concepts, test_mask, test_y, test_pred = collect_activations_and_targets(
        test_loader, classifier, hook, device)
    hook.remove()

    # --- Concept extraction: correlation-based, calib tren train, ap dung sang test ---
    extractors = fit_concept_extractors(train_act, train_concepts, train_mask, concept_names)
    train_concept_pred = apply_concept_extractors(train_act, extractors)
    test_concept_pred = apply_concept_extractors(test_act, extractors)

    test_keep = test_mask.astype(bool)
    concept_acc, concept_f1 = concept_metrics(
        test_concept_pred[test_keep], test_concepts[test_keep].astype(int), concept_names)

    # --- Rule extraction: decision tree xap xi DU DOAN CUA CHINH CLASSIFIER (fidelity) ---
    tree = DecisionTreeClassifier(max_depth=args.tree_max_depth, random_state=args.seed)
    tree.fit(train_concept_pred, train_pred)

    test_tree_pred = tree.predict(test_concept_pred)
    diagnosis_acc = accuracy_score(test_y, test_tree_pred)
    diagnosis_f1 = f1_macro_2class(test_y, test_tree_pred)
    fidelity = accuracy_score(test_pred, test_tree_pred)
    classifier_own_acc = accuracy_score(test_y, test_pred)

    result = {
        "seed": args.seed,
        "n_concept_eval": int(test_keep.sum()),
        "concept_acc": concept_acc,
        "concept_f1": concept_f1,
        "diagnosis_acc": diagnosis_acc,
        "diagnosis_f1": diagnosis_f1,
        "fidelity_to_classifier": fidelity,
        "classifier_own_diagnosis_acc": classifier_own_acc,
        "tree_max_depth": args.tree_max_depth,
        "tree_actual_depth": int(tree.get_depth()),
        "tree_n_leaves": int(tree.get_n_leaves()),
    }
    print(json.dumps(result, indent=2))
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    if args.output_tree_txt:
        rule_text = export_text(tree, feature_names=concept_names,
                                 class_names=[label_names[c] for c in tree.classes_])
        Path(args.output_tree_txt).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_tree_txt, "w", encoding="utf-8") as f:
            f.write(rule_text)


if __name__ == "__main__":
    main()
