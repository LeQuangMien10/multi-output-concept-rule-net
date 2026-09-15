"""
baseline_neurosymbolic_rules.py - Baseline Basci et al. (CEUR/TRUST-AI 2025,
"Explaining Concept Drift via Neuro-Symbolic Rules") cho PAD-UFES-20
(6 concept trieu chung, 6 lop diagnostic).
==============================================================================

Cung phuong phap voi ban Fitzpatrick (src/scripts/fitzpatrick/
baseline_neurosymbolic_rules.py): trich concept POST-HOC tu kernel activation
cua mot classifier da train xong CHI bang nhan (--concept_loss_weight 0.0),
roi fit 1 decision tree fidelity-based tren bang concept nhi phan.

2 diem SUA LAI so voi ban Fitzpatrick (khong phai copy nguyen, xem review truoc
khi trien khai):
  1. concept_mask o day la VECTOR theo tung concept (PadUfes20Dataset), khac
     Fitzpatrick (1 scalar/anh) -- concept_metrics()/fit_concept_extractors()
     phai loc theo TUNG COT concept rieng, khong dung 1 mask hang chung cho
     ca bang (ban Fitzpatrick lam vay se loi shape hoac sai ket qua o day).
  2. Nhan chan doan la 6 lop (khong phai 2 lop benign/malignant) -- ham F1
     macro cho nhan phai tinh tren ca 6 lop (f1_macro_nclass), khong dung lai
     f1_macro_2class (van giu nguyen cho F1 cua TUNG CONCEPT, vi moi concept
     van la nhi phan present/absent).

Usage (Kaggle, sau khi da train classifier --concept_loss_weight 0.0):
    python -m src.scripts.pad_ufes20.baseline_neurosymbolic_rules \\
        --data_dir /kaggle/working/PAD-UFES-20_prepared \\
        --img_dir /kaggle/input/datasets/lquangmin/pad-ufes-20/images \\
        --classifier_ckpt /kaggle/working/outputs/pad_ufes20_basci_classifier/best_model.pt \\
        --seed 42 \\
        --output_json /kaggle/working/outputs/pad_ufes20_basci_seed42.json \\
        --output_tree_txt /kaggle/working/outputs/pad_ufes20_basci_seed42_rules.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.scripts.pad_ufes20.train_icrl import load_system1, make_loaders
from src.utils.pad_ufes20_concepts import LABEL_NAMES as DEFAULT_LABEL_NAMES


def _tree_max_depth_type(s: str):
    return None if s.lower() == "none" else int(s)


def parse_args():
    p = argparse.ArgumentParser(description="Basci et al. neuro-symbolic rule baseline, PAD-UFES-20.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--classifier_ckpt", type=str, required=True,
                    help="Checkpoint tu train_system1_baseline.py --concept_loss_weight 0.0.")
    p.add_argument("--label_names", type=str, default=",".join(DEFAULT_LABEL_NAMES))
    p.add_argument("--seed", type=int, required=True, help="Chi de ghi vao JSON, khong dat lai seed o day.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--tree_max_depth", type=_tree_max_depth_type, default=5,
                    help="Do sau toi da decision tree. Dat 'none' de khong gioi han.")
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--output_tree_txt", type=str, default=None,
                    help="Neu dat, ghi rule text (sklearn export_text) ra file nay de doc thu cong.")
    return p.parse_args()


# ----- cong thuc metric giong eval_crlmatched_metrics.py, tru diagnosis F1 (N lop) -----

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
    """F1 macro cho 1 CONCEPT nhi phan (present/absent) -- moi concept van
    nhi phan bat ke dataset co bao nhieu lop chan doan, nen KHONG doi ham nay."""
    return (binary_f1(y_true, y_pred, 0) + binary_f1(y_true, y_pred, 1)) / 2


def f1_macro_nclass(y_true, y_pred, num_classes):
    """F1 macro cho NHAN CHAN DOAN -- PAD-UFES-20 co 6 lop (khac 2 lop
    benign/malignant cua Fitzpatrick CRL-matched), phai average tren ca 6 lop
    thay vi ham f1_macro_2class cu the cho 2 lop."""
    f1s = [binary_f1(y_true, y_pred, c) for c in range(num_classes)]
    return sum(f1s) / len(f1s)


def concept_metrics(pred, gt, mask, concept_names):
    """Cong thuc CRL that: macro theo CONCEPT, bo qua concept GT toan 0 (trong
    tap con da biet). mask: [N, num_concepts] THEO TUNG CONCEPT (khac ban
    Fitzpatrick dung 1 scalar mask/anh) -- UNK xay ra theo tung field rieng o
    PAD-UFES-20, nen phai loc theo dung cot concept, khong loc theo hang."""
    accs, f1s = [], []
    for i in range(len(concept_names)):
        keep_i = mask[:, i].astype(bool)
        true_vars, pred_vars = gt[keep_i, i], pred[keep_i, i]
        if true_vars.sum() == 0:
            continue
        accs.append(accuracy_score(true_vars, pred_vars))
        f1s.append(f1_macro_2class(true_vars, pred_vars))
    return float(sum(accs) / len(accs)), float(sum(f1s) / len(f1s))


# ----- Basci et al. core method (khong doi so voi ban Fitzpatrick) -----

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
    concept_mask [N,num_concepts] (vector, khong phai scalar), nhan that [N],
    nhan classifier tu du doan [N]."""
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
    """Voi moi concept: chon kernel |corr| lon nhat, threshold = percentile
    ung voi ty le duong tinh THAT cua concept do -- tat ca CHI tren anh biet
    dung concept do (train_mask[:, i]=1), khac ban Fitzpatrick dung 1 `keep`
    chung cho moi concept (dung khi mask luon dong nhat ca hang, sai o day)."""
    extractors = []
    for i, name in enumerate(concept_names):
        keep_i = train_mask[:, i].astype(bool)
        target = train_concepts[keep_i, i]
        pos_rate = target.mean()
        if target.sum() == 0 or target.sum() == len(target):
            # Concept toan 0 hoac toan 1 trong tap con nay -- khong co tin
            # hieu de hoc kernel nao, luon du doan theo da so.
            extractors.append({"concept": name, "kernel": -1, "sign": 1.0,
                                "threshold": 0.0, "constant_pred": float(pos_rate > 0.5)})
            continue
        corr = batch_point_biserial(train_act[keep_i], target)
        k = int(np.argmax(np.abs(corr)))
        sign = 1.0 if corr[k] >= 0 else -1.0
        signed_act = train_act[keep_i, k] * sign
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
    num_classes = len(label_names)

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

    concept_acc, concept_f1 = concept_metrics(
        test_concept_pred, test_concepts.astype(int), test_mask, concept_names)

    # --- Rule extraction: decision tree xap xi DU DOAN CUA CHINH CLASSIFIER (fidelity) ---
    tree = DecisionTreeClassifier(max_depth=args.tree_max_depth, random_state=args.seed)
    tree.fit(train_concept_pred, train_pred)

    test_tree_pred = tree.predict(test_concept_pred)
    diagnosis_acc = accuracy_score(test_y, test_tree_pred)
    diagnosis_f1 = f1_macro_nclass(test_y, test_tree_pred, num_classes)
    fidelity = accuracy_score(test_pred, test_tree_pred)
    classifier_own_acc = accuracy_score(test_y, test_pred)

    result = {
        "seed": args.seed,
        "n_concept_eval": int(test_mask.sum()),
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
