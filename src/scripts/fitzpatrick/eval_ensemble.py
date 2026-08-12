"""
eval_ensemble.py - Ket hop du doan nhan cua System 1 va ICRL (thuong la ban
--exclude_label_slot, vi day la truong hop 2 model manh o 2 mat khac nhau:
S1 accuracy nhan cao hon, ICRL concept-reasoning trung thuc hon -- xem
reports/error_breakdown_and_match_contribution.md muc 6-7) de lay accuracy
tot hon ca 2 rieng le.

3 chien luoc, tham so chon tren VAL, bao cao tren TEST (tranh overfit):
  1. weighted average xac suat: alpha*softmax(S1) + (1-alpha)*softmax(S2)
  2. chon theo confidence cao hon (so max-softmax cua S1 vs S2)
  3. gated override: mac dinh S1, chi doi sang S2 khi S1 khong tu tin
     (max_prob < s1_thresh) VA rule S2 du tin cay (confidence > rule_conf_thresh)

CANH BAO: ket qua khong dam bao cai thien o moi scope -- da xac nhan thuc
nghiem tren 3 scope Fitzpatrick, chi 2/3 scope co cai thien that tren test
(xem file report tren). Luon doc ca 3 chien luoc, khong chi tin chien luoc
co val_acc cao nhat (de overfit tren val set nho).

Usage:
    python -m src.scripts.fitzpatrick.eval_ensemble \
        --data_dir data/fitzpatrick17k_prepared \
        --img_dir data/fitzpatrick17k/data/finalfitz17k \
        --system1_ckpt outputs/fitzpatrick_system1/best_model.pt \
        --icrl_dir outputs/fitzpatrick_icrl_nolabelslot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.utils.fitzpatrick_concepts import LABEL_NAMES as DEFAULT_LABEL_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Ensemble System1 + ICRL: weighted-avg / confidence-pick / gated-override.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--system1_ckpt", type=str, required=True)
    p.add_argument("--icrl_dir", type=str, required=True,
                    help="Thuong la ban --exclude_label_slot -- xem docstring.")
    p.add_argument("--label_names", type=str, default=None,
                    help="Comma-separated, override cho scope khac 3-lop mac dinh.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


@torch.no_grad()
def collect(loader, system1, memory, head, centroids, confidences, device, name):
    all_y, all_s1_probs, all_s2_probs, all_rule_conf = [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        s1_out = system1(images)
        cv = soft_concept_vector(s1_out)
        y = labels["label"].to(device)

        s1_probs = F.softmax(s1_out["label"], dim=-1)
        rule_ids, _ = memory.match(cv)
        s2_probs = F.softmax(head(centroids[rule_ids]), dim=-1)
        rule_conf = confidences[rule_ids]

        all_y.append(y)
        all_s1_probs.append(s1_probs)
        all_s2_probs.append(s2_probs)
        all_rule_conf.append(rule_conf)

    y = torch.cat(all_y)
    s1_probs = torch.cat(all_s1_probs)
    s2_probs = torch.cat(all_s2_probs)
    rule_conf = torch.cat(all_rule_conf)
    print(f"[INFO] {name}: N={len(y)}")
    return y, s1_probs, s2_probs, rule_conf


def acc(pred, y):
    return (pred == y).float().mean().item()


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

    label_names = args.label_names.split(",") if args.label_names else list(DEFAULT_LABEL_NAMES)

    system1, image_size = load_system1(Path(args.system1_ckpt), device)
    train_loader, val_loader, test_loader = make_loaders(
        Path(args.data_dir), args.img_dir, image_size, args.batch_size, args.num_workers,
    )

    memory = ICRLRuleMemory.load(Path(args.icrl_dir) / "icrl_rule_memory.pt", device=str(device))
    head = nn.Linear(memory.concept_dim, len(label_names)).to(device)
    head.load_state_dict(torch.load(Path(args.icrl_dir) / "prediction_head.pt",
                                     map_location=device, weights_only=False))
    head.eval()
    centroids = memory.get_centroids().to(device)
    confidences = torch.tensor(memory.get_confidences(), device=device)
    print(f"[INFO] ICRL (S2): {memory.num_rules} rules, cluster_dims={memory.cluster_dims}")

    y_val, s1_val, s2_val, rconf_val = collect(val_loader, system1, memory, head, centroids, confidences, device, "val")
    y_test, s1_test, s2_test, rconf_test = collect(test_loader, system1, memory, head, centroids, confidences, device, "test")

    s1_acc_val, s1_acc_test = acc(s1_val.argmax(1), y_val), acc(s1_test.argmax(1), y_test)
    s2_acc_val, s2_acc_test = acc(s2_val.argmax(1), y_val), acc(s2_test.argmax(1), y_test)
    print(f"\nBaseline S1 alone:  val={s1_acc_val:.4f}  test={s1_acc_test:.4f}")
    print(f"Baseline S2(ICRL) alone: val={s2_acc_val:.4f}  test={s2_acc_test:.4f}")

    print("\n--- Strategy 1: weighted probability average, alpha*S1 + (1-alpha)*S2 ---")
    best_alpha, best_val_acc = None, -1
    for alpha in [i / 10 for i in range(11)]:
        a = acc((alpha * s1_val + (1 - alpha) * s2_val).argmax(1), y_val)
        if a > best_val_acc:
            best_val_acc, best_alpha = a, alpha
        print(f"  alpha={alpha:.1f}: val_acc={a:.4f}")
    combined_test = best_alpha * s1_test + (1 - best_alpha) * s2_test
    print(f"  -> best alpha={best_alpha} (chon tren VAL) => test_acc={acc(combined_test.argmax(1), y_test):.4f}")

    print("\n--- Strategy 2: chon theo confidence cao hon ---")
    def confidence_pick(s1_probs, s2_probs):
        use_s2 = s2_probs.max(dim=1).values > s1_probs.max(dim=1).values
        pred = s1_probs.argmax(1).clone()
        pred[use_s2] = s2_probs.argmax(1)[use_s2]
        return pred, use_s2

    pred_test, use_s2_test = confidence_pick(s1_test, s2_test)
    pred_val, _ = confidence_pick(s1_val, s2_val)
    print(f"  val_acc={acc(pred_val, y_val):.4f}  test_acc={acc(pred_test, y_test):.4f}  "
          f"(dung S2 cho {use_s2_test.float().mean()*100:.1f}% anh test)")

    print("\n--- Strategy 3: gated override (mac dinh S1, doi sang S2 khi S1 khong tu tin + rule S2 tin cay) ---")
    best_combo, best_val_acc3 = None, -1
    for s1_thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for rule_conf_thresh in [0.5, 0.6, 0.7, 0.8]:
            override_val = (s1_val.max(dim=1).values < s1_thresh) & (rconf_val > rule_conf_thresh)
            pred_val3 = s1_val.argmax(1).clone()
            pred_val3[override_val] = s2_val.argmax(1)[override_val]
            a = acc(pred_val3, y_val)
            if a > best_val_acc3:
                best_val_acc3, best_combo = a, (s1_thresh, rule_conf_thresh)

    s1_thresh, rule_conf_thresh = best_combo
    override_test = (s1_test.max(dim=1).values < s1_thresh) & (rconf_test > rule_conf_thresh)
    pred_test3 = s1_test.argmax(1).clone()
    pred_test3[override_test] = s2_test.argmax(1)[override_test]
    print(f"  best (s1_thresh={s1_thresh}, rule_conf_thresh={rule_conf_thresh}) tren VAL: val_acc={best_val_acc3:.4f}")
    print(f"  => test_acc={acc(pred_test3, y_test):.4f}  (override {override_test.float().mean()*100:.1f}% anh test)")


if __name__ == "__main__":
    main()
