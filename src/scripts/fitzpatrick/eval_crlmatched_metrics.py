"""
eval_crlmatched_metrics.py - Concept + diagnosis metrics (CRL-matched scope,
1 seed) trong 1 lan chay, xuat JSON de gop nhieu seed lai (xem
aggregate_seed_results.py). Dung de dien vao Table 1 cua paper: hien tai
Table 1 chi co 1 seed (single held-out test-split) trong khi moi baseline
cua CRL deu la 5-fold CV mean+-std -- khong cong bang. Chay script nay cho
moi seed roi gop lai bang aggregate_seed_results.py de co mean+-std tuong
tu.

Concept metric: cong thuc CRL that (xem eval_concept_metrics.py), tren
concept predictions cua System~1 (concept encoder dung chung cho ca
System~2 lan ensemble).
Diagnosis metric: gated-override ensemble (System~1 + System~2 khong co
label slot), grid-search (s1_thresh, rule_conf_thresh) tren val, bao cao
tren test -- giong het logic eval_ensemble.py.

Usage (Kaggle):
    python -m src.scripts.fitzpatrick.eval_crlmatched_metrics \\
        --data_dir /kaggle/input/datasets/lquangmin/fitzpatrick17k-crl-matched \\
        --img_dir /kaggle/input/datasets/lquangmin/fitzpatrick17k/data/finalfitz17k \\
        --system1_ckpt /kaggle/working/outputs/fitzpatrick_system1_crlmatched_seed{SEED}/best_model.pt \\
        --icrl_dir /kaggle/working/outputs/fitzpatrick_icrl_crlmatched_nolabelslot_seed{SEED} \\
        --seed {SEED} \\
        --output_json /kaggle/working/outputs/crlmatched_seed{SEED}.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.utils.fitzpatrick_concepts import LABEL_NAMES as DEFAULT_LABEL_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Concept + diagnosis metrics, CRL-matched scope, 1 seed.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--system1_ckpt", type=str, required=True)
    p.add_argument("--icrl_dir", type=str, required=True,
                    help="Ban --exclude_label_slot (dung cho ensemble S2).")
    p.add_argument("--label_names", type=str, default="benign,malignant")
    p.add_argument("--seed", type=int, required=True, help="Chi de ghi vao JSON, khong dat lai seed o day.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
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


@torch.no_grad()
def collect_ensemble_inputs(loader, system1, memory, head, centroids, confidences, device):
    all_y, all_s1, all_s2, all_rc = [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        cv = soft_concept_vector(out)
        y = labels["label"].to(device)
        s1p = F.softmax(out["label"], dim=-1)
        rule_ids, _ = memory.match(cv)
        s2p = F.softmax(head(centroids[rule_ids]), dim=-1)
        rc = confidences[rule_ids]
        all_y.append(y); all_s1.append(s1p); all_s2.append(s2p); all_rc.append(rc)
    return torch.cat(all_y), torch.cat(all_s1), torch.cat(all_s2), torch.cat(all_rc)


@torch.no_grad()
def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    label_names = args.label_names.split(",")

    system1, image_size = load_system1(Path(args.system1_ckpt), device)
    train_loader, val_loader, test_loader = make_loaders(
        Path(args.data_dir), args.img_dir, image_size, args.batch_size, args.num_workers,
    )
    concept_names = test_loader.dataset.concept_names
    num_concepts = len(concept_names)

    # --- Concept metrics: System~1's own concept predictions on test ---
    all_pred, all_gt = [], []
    for images, labels in test_loader:
        images = images.to(device)
        out = system1(images)
        pred_hard = (torch.sigmoid(out["concepts"]) > 0.5).float()
        keep = labels["concept_mask"].bool()
        all_pred.append(pred_hard[keep].cpu())
        all_gt.append(labels["concepts"][keep].cpu())
    pred = torch.cat(all_pred).numpy().astype(int)
    gt = torch.cat(all_gt).numpy().astype(int)
    concept_acc, concept_f1 = concept_metrics(pred, gt, concept_names)

    # --- Diagnosis metrics: gated-override ensemble, grid-searched on val ---
    memory = ICRLRuleMemory.load(Path(args.icrl_dir) / "icrl_rule_memory.pt", device=str(device))
    head = nn.Linear(memory.concept_dim, len(label_names)).to(device)
    head.load_state_dict(torch.load(Path(args.icrl_dir) / "prediction_head.pt",
                                     map_location=device, weights_only=False))
    head.eval()
    centroids = memory.get_centroids().to(device)
    confidences = torch.tensor(memory.get_confidences(), device=device)

    y_val, s1_val, s2_val, rc_val = collect_ensemble_inputs(val_loader, system1, memory, head, centroids, confidences, device)
    y_test, s1_test, s2_test, rc_test = collect_ensemble_inputs(test_loader, system1, memory, head, centroids, confidences, device)

    best_combo, best_val_acc = None, -1.0
    for s1_thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for rule_conf_thresh in [0.5, 0.6, 0.7, 0.8]:
            override_val = (s1_val.max(dim=1).values < s1_thresh) & (rc_val > rule_conf_thresh)
            pred_val = s1_val.argmax(1).clone()
            pred_val[override_val] = s2_val.argmax(1)[override_val]
            a = accuracy_score(y_val.cpu().numpy(), pred_val.cpu().numpy())
            if a > best_val_acc:
                best_val_acc, best_combo = a, (s1_thresh, rule_conf_thresh)

    s1_thresh, rule_conf_thresh = best_combo
    override_test = (s1_test.max(dim=1).values < s1_thresh) & (rc_test > rule_conf_thresh)
    pred_test = s1_test.argmax(1).clone()
    pred_test[override_test] = s2_test.argmax(1)[override_test]
    y_test_np, pred_test_np = y_test.cpu().numpy(), pred_test.cpu().numpy()
    diagnosis_acc = accuracy_score(y_test_np, pred_test_np)
    diagnosis_f1 = f1_macro_2class(y_test_np, pred_test_np)

    result = {
        "seed": args.seed,
        "n_concept_eval": int(gt.shape[0]),
        "concept_acc": concept_acc,
        "concept_f1": concept_f1,
        "gated_override_thresholds": {"s1_thresh": s1_thresh, "rule_conf_thresh": rule_conf_thresh},
        "diagnosis_acc": diagnosis_acc,
        "diagnosis_f1": diagnosis_f1,
    }
    print(json.dumps(result, indent=2))
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
