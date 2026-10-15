"""
conf_min_ablation_scope.py - Ban tong quat cua
src/scripts/fitzpatrick/conf_min_ablation.py cho 1-split scope (fulldata,
concept_only, pad_ufes20) thay vi 5-fold CV crlmatched -- dung sau khi build
lai S1+ICRL cho 3 scope nay voi code hien tai (xem
src/scripts/run_scope_rebuild.py, memory project_s1s2_coupling_plan.md).

Muc dich giong nguyen ban: kiem tra conf_min=0.5 (gia tri goc giu lai de so
sanh cong bang khi rebuild) co phai nut that khien so rule qua it hay khong,
tach biet voi n_min (da sweep rieng trong run_scope_rebuild.py).

Chi inference lai tren checkpoint da co (S1 + prediction_head da train cua
n_min duoc recommend), KHONG train lai gi.

Usage (local, inference only, sau khi tai checkpoint tu Kaggle ve):
    python -m src.scripts.conf_min_ablation_scope --scope concept_only \\
        --icrl_root outputs/scope_rebuild/concept_only \\
        --data_dir data/fitzpatrick17k_prepared_concept_only \\
        --img_dir data/fitzpatrick17k/data/finalfitz17k \\
        --conf_min_values 0.5,0.4,0.3,0.25,0.2 \\
        --output_json outputs/conf_min_ablation/concept_only_ablation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scripts.run_scope_rebuild import SCOPE_CONFIGS
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.fitzpatrick.system1 import soft_concept_vector


def parse_args():
    p = argparse.ArgumentParser(description="Ablation conf_min cho 1 scope 1-split (fulldata/concept_only/pad_ufes20).")
    p.add_argument("--scope", type=str, required=True, choices=list(SCOPE_CONFIGS.keys()))
    p.add_argument("--icrl_root", type=str, required=True,
                    help="Thu muc chua system1/best_model.pt va icrl/ (icrl_rule_memory_base.pt, "
                         "n_min_sweep_summary.json, prediction_head.pt da train o n_min recommend).")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--conf_min_values", type=str, default="0.5,0.45,0.4,0.35,0.3,0.25,0.2")
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
    """Giong het grid-search trong eval_crlmatched_metrics.py -- tong quat cho so lop bat ky."""
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


@torch.no_grad()
def evaluate(system1, head, loader, device, memory):
    correct, total = 0, 0
    centroids = memory.get_centroids().to(device)
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        cv = soft_concept_vector(out)
        y = labels["label"].to(device)
        rule_ids, _ = memory.match(cv)
        pred = head(centroids[rule_ids]).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.shape[0]
    return {"accuracy": correct / total}


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    cfg = SCOPE_CONFIGS[args.scope]
    module = cfg["module"]
    label_names = cfg["icrl_args"]["label_names"].split(",")
    conf_min_values = [float(v) for v in args.conf_min_values.split(",")]

    if module == "fitzpatrick":
        from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders, record_rule_accuracy, build_full_concept_layout
        from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY
    else:
        from src.scripts.pad_ufes20.train_icrl import load_system1, make_loaders, record_rule_accuracy, build_full_concept_layout
        from src.utils.pad_ufes20_concepts import S1_LABEL_CONCEPT_KEY

    icrl_root = Path(args.icrl_root)
    out_dir = Path(args.output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    system1, image_size = load_system1(icrl_root / "system1" / "best_model.pt", device)
    _, val_loader, test_loader = make_loaders(Path(args.data_dir), args.img_dir, image_size,
                                               args.batch_size, args.num_workers)
    concept_names = val_loader.dataset.concept_names
    full_concept_keys, full_concept_offsets, full_concept_dims, _ = build_full_concept_layout(
        concept_names, label_names)
    exclude_keys = {S1_LABEL_CONCEPT_KEY}

    sweep_summary = json.load(open(icrl_root / "icrl" / "n_min_sweep_summary.json"))
    rec_n_min = sweep_summary["recommended_n_min"]
    rec_dir = icrl_root / "icrl" / f"n_min_{rec_n_min}"

    memory = ICRLRuleMemory.load(icrl_root / "icrl" / "icrl_rule_memory_base.pt", device=str(device))
    memory.n_min = rec_n_min
    memory.prune(verbose=False, conf_min_override=0.0)

    head = nn.Linear(memory.concept_dim, len(label_names)).to(device)
    head.load_state_dict(torch.load(rec_dir / "prediction_head.pt",
                                     map_location=device, weights_only=False))
    head.eval()

    record_rule_accuracy(system1, head, memory, val_loader, device)
    snap_path = out_dir / f"_snapshot_{args.scope}_postacc.pt"
    memory.save(snap_path)

    rows = []
    for conf_min in conf_min_values:
        mem = ICRLRuleMemory.load(snap_path, device=str(device))
        mem.conf_min = conf_min
        mem.prune(verbose=False)
        if mem.num_rules == 0:
            rows.append({"conf_min": conf_min, "num_rules": 0, "collapsed": True})
            print(f"{args.scope} conf_min={conf_min}: COLLAPSED (0 rules)")
            continue
        dedupe_stats = mem.dedupe_by_decoded_pattern(
            full_concept_keys, full_concept_offsets, full_concept_dims,
            exclude_keys=exclude_keys, verbose=False,
        )
        if mem.num_rules == 0:
            rows.append({"conf_min": conf_min, "num_rules": 0, "collapsed": True,
                          "dedupe_stats": dedupe_stats})
            print(f"{args.scope} conf_min={conf_min}: COLLAPSED after dedupe (0 rules)")
            continue

        centroids = mem.get_centroids().to(device)
        confidences = torch.tensor(mem.get_confidences(), device=device)
        s2_val = evaluate(system1, head, val_loader, device, mem)
        s2_test = evaluate(system1, head, test_loader, device, mem)

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
        rows.append(row)
        print(f"{args.scope} conf_min={conf_min}: rules={mem.num_rules} "
              f"S2_test_acc={s2_test['accuracy']:.4f} ensemble_test_acc={ens_acc:.4f}")

    snap_path.unlink(missing_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"recommended_n_min": rec_n_min, "rows": rows}, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
