"""
conf_min_ablation.py - Do xem conf_min (nguong Wilson confidence de rule
song sot sau final prune) co phai la nut that thuc su khien so rule qua it
hay khong, tach biet voi n_min (da kiem tra rieng qua --n_min_sweep va cho
thay gan nhu phang quanh 2-4 rule tren toan dai n_min=5-30).

Chi inference lai tren checkpoint da co (System~1 + prediction_head da train
cua n_min duoc recommend), KHONG train lai gi ca:
  - Nap icrl_rule_memory_base.pt (n_min=1, chua co accuracy).
  - Prune theo n_min=recommended (conf_min_override=0.0, giong dung buoc
    --n_min_sweep that trong train_icrl.py).
  - record_rule_accuracy() MOT LAN voi head da train san -- buoc nay
    deterministic (khong co randomness), nen chi can chay 1 lan roi luu
    snapshot, dung lai cho moi gia tri conf_min thay vi train lai head.
  - Voi moi gia tri conf_min trong danh sach: nap lai snapshot, ap dung
    conf_min + dedupe_by_decoded_pattern (dung thu tu that cua
    run_stage3_onward trong train_icrl.py), roi do:
      + so rule cuoi, circular_rate
      + System~2-alone val/test accuracy (evaluate(), giong n_min_sweep)
      + full gated-override ensemble test accuracy (giong eval_crlmatched_
        metrics.py, grid-search threshold tren val)

Usage (local, inference only):
    python -m src.scripts.fitzpatrick.conf_min_ablation \\
        --cv_root outputs/crlmatched_cv_gated \\
        --data_root data/fitzpatrick17k_crl_matched_cv_official \\
        --img_dir data/fitzpatrick17k/data/finalfitz17k \\
        --conf_min_values 0.5,0.45,0.4,0.35,0.3,0.25 \\
        --output_json outputs/conf_min_ablation/conf_min_ablation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scripts.fitzpatrick.train_icrl import (
    load_system1, make_loaders, record_rule_accuracy, evaluate,
    build_full_concept_layout,
)
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY


def parse_args():
    p = argparse.ArgumentParser(description="Ablation: conf_min vs so rule, tren checkpoint da co san.")
    p.add_argument("--cv_root", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--conf_min_values", type=str, default="0.5,0.45,0.4,0.35,0.3,0.25")
    p.add_argument("--label_names", type=str, default="benign,malignant")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_json", type=str, required=True)
    return p.parse_args()


def accuracy_score(y_true, y_pred):
    return float((y_true == y_pred).mean())


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


def gated_override_test_acc(y_val, s1_val, s2_val, rc_val, y_test, s1_test, s2_test, rc_test):
    """Giong het grid-search trong eval_crlmatched_metrics.py."""
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
    return accuracy_score(y_test.cpu().numpy(), pred_test.cpu().numpy()), best_combo


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    cv_root, data_root = Path(args.cv_root), Path(args.data_root)
    label_names = args.label_names.split(",")
    conf_min_values = [float(v) for v in args.conf_min_values.split(",")]

    out_dir = Path(args.output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for k in range(args.num_folds):
        fold_dir = cv_root / f"fold_{k}"
        system1, image_size = load_system1(fold_dir / "system1" / "best_model.pt", device)
        _, val_loader, test_loader = make_loaders(
            data_root / f"fold_{k}", args.img_dir, image_size, args.batch_size, args.num_workers,
        )
        concept_names = val_loader.dataset.concept_names
        full_concept_keys, full_concept_offsets, full_concept_dims, full_cv_dim = \
            build_full_concept_layout(concept_names, label_names)

        sweep_summary = json.load(open(fold_dir / "icrl" / "n_min_sweep_summary.json"))
        rec_n_min = sweep_summary["recommended_n_min"]

        memory = ICRLRuleMemory.load(fold_dir / "icrl" / "icrl_rule_memory_base.pt", device=str(device))
        memory.n_min = rec_n_min
        memory.prune(verbose=False, conf_min_override=0.0)  # filter by n_min only, same as sweep candidate

        head = nn.Linear(memory.concept_dim, len(label_names)).to(device)
        head.load_state_dict(torch.load(fold_dir / "icrl" / "prediction_head.pt",
                                        map_location=device, weights_only=False))
        head.eval()

        record_rule_accuracy(system1, head, memory, val_loader, device)
        snap_path = out_dir / f"_snapshot_fold{k}_postacc.pt"
        memory.save(snap_path)

        fold_rows = []
        for conf_min in conf_min_values:
            mem = ICRLRuleMemory.load(snap_path, device=str(device))
            mem.conf_min = conf_min
            mem.prune(verbose=False)
            if mem.num_rules == 0:
                fold_rows.append({"conf_min": conf_min, "num_rules": 0, "collapsed": True})
                print(f"fold_{k} conf_min={conf_min}: COLLAPSED (0 rules)")
                continue
            dedupe_stats = mem.dedupe_by_decoded_pattern(
                full_concept_keys, full_concept_offsets, full_concept_dims,
                exclude_keys={S1_LABEL_CONCEPT_KEY}, verbose=False,
            )
            if mem.num_rules == 0:
                fold_rows.append({"conf_min": conf_min, "num_rules": 0, "collapsed": True,
                                   "dedupe_stats": dedupe_stats})
                print(f"fold_{k} conf_min={conf_min}: COLLAPSED after dedupe (0 rules)")
                continue

            centroids = mem.get_centroids().to(device)
            confidences = torch.tensor(mem.get_confidences(), device=device)
            s2_val = evaluate(system1, head, val_loader, device, mem, "val")
            s2_test = evaluate(system1, head, test_loader, device, mem, "test")

            y_val, s1_val, s2p_val, rc_val = collect_ensemble_inputs(
                val_loader, system1, mem, head, centroids, confidences, device)
            y_test, s1_test, s2p_test, rc_test = collect_ensemble_inputs(
                test_loader, system1, mem, head, centroids, confidences, device)
            ens_acc, combo = gated_override_test_acc(
                y_val, s1_val, s2p_val, rc_val, y_test, s1_test, s2p_test, rc_test)

            row = {
                "conf_min": conf_min,
                "num_rules": mem.num_rules,
                "circular_rate": dedupe_stats.get("removed_circular", 0) /
                                  max(1, dedupe_stats.get("removed_circular", 0) + mem.num_rules),
                "dedupe_stats": dedupe_stats,
                "s2_alone_val_acc": s2_val["accuracy"],
                "s2_alone_test_acc": s2_test["accuracy"],
                "ensemble_test_acc": ens_acc,
                "ensemble_combo": combo,
            }
            fold_rows.append(row)
            print(f"fold_{k} conf_min={conf_min}: rules={mem.num_rules} "
                  f"S2_test_acc={s2_test['accuracy']:.4f} ensemble_test_acc={ens_acc:.4f}")

        results[f"fold_{k}"] = {"recommended_n_min": rec_n_min, "rows": fold_rows}
        snap_path.unlink(missing_ok=True)

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
