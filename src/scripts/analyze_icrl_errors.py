"""
analyze_icrl_errors.py — Phân tích lỗi pipeline S1 → ICRL S2
=============================================================

Với mỗi ảnh trong val/test set, chạy toàn bộ pipeline:
  image → S1 → concept vector → match rule → head → prediction

Ghi lại 4 loại kết quả:
  TP : S1 đúng d3,  S2 đúng  → cả hai đúng
  TN : S1 sai d3,  S2 đúng  → S2 sửa được lỗi S1  ← quan trọng nhất
  FP : S1 đúng d3, S2 sai   → S2 làm hỏng kết quả đúng của S1
  FN : S1 sai d3,  S2 sai   → cả hai sai

Xuất:
  - Bảng thống kê tổng hợp
  - File JSON chi tiết từng sample
  - Grid ảnh trực quan (đúng + sai xen kẽ) dùng matplotlib

Usage:
    python -m src.scripts.analyze_icrl_errors \\
        --data_dir /kaggle/input/datasets/lquangmin/mnist-math \\
        --system1_ckpt /kaggle/working/outputs/system1/best_model.pt \\
        --memory_ckpt  /kaggle/working/outputs/icrl_coherence/icrl_rule_memory.pt \\
        --head_ckpt    /kaggle/working/outputs/icrl_coherence/prediction_head.pt \\
        --output_dir   /kaggle/working/outputs/icrl_analysis \\
        --split val \\
        --n_examples 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm

from src.datasets.mnist_math_dataset import MNISTMathPTDataset
from src.models.multi_head_system1 import MultiHeadSystem1
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.rule_memory import (
    soft_concept_vector,
    CONCEPT_KEYS_ORDERED, CONCEPT_OFFSETS, CONCEPT_DIMS, CONCEPT_TOTAL_DIM,
)
from src.utils.symbols import ID_TO_SYMBOL


# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      required=True)
    p.add_argument("--system1_ckpt",  required=True)
    p.add_argument("--memory_ckpt",   required=True,
                   help="ICRLRuleMemory .pt file")
    p.add_argument("--head_ckpt",     required=True,
                   help="prediction_head.pt")
    p.add_argument("--output_dir",    default="outputs/icrl_analysis")
    p.add_argument("--split",         default="val",
                   choices=["train", "val", "test"])
    p.add_argument("--n_examples",    type=int, default=5,
                   help="Số ví dụ mỗi loại (TP/TN/FP/FN) để visualize")
    p.add_argument("--max_samples",   type=int, default=None,
                   help="Giới hạn số sample phân tích (None = toàn bộ)")
    p.add_argument("--device",        default="auto")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_system1(path, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    model = MultiHeadSystem1(
        feature_dim=saved.get("feature_dim", 256),
        num_slots  =saved.get("num_slots",   4),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval().to(device)


def op_str(op_id: int) -> str:
    return ID_TO_SYMBOL.get(op_id, "?")


def decode_s1(s1_out: dict) -> dict[str, int]:
    """Lấy argmax prediction của S1 cho mỗi slot."""
    return {k: int(s1_out[k][0].argmax().item()) for k in CONCEPT_KEYS_ORDERED}


def rule_id_to_str(memory: ICRLRuleMemory, rule_id: int) -> str:
    """Decode centroid của rule thành string 'd1 op d2 = d3'."""
    decoded = memory.decode_rule(
        rule_id=rule_id,
        concept_keys   =CONCEPT_KEYS_ORDERED,
        concept_offsets=CONCEPT_OFFSETS,
        concept_dims   =CONCEPT_DIMS,
        id_to_symbol   =ID_TO_SYMBOL,
    )
    s = decoded["slots"]
    d1 = s["digit1"]["value"]
    op = s["op1"]["value"]
    d2 = s["digit2"]["value"]
    d3 = decoded["label"]
    return f"{d1} {op} {d2} = {d3}"


# ─────────────────────────────────────────────────────────────
# Run pipeline on one sample
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_sample(image, labels, system1, memory, head, centroids, device):
    """
    Chạy pipeline đầy đủ cho 1 ảnh.
    Returns dict với đầy đủ thông tin để phân tích và visualize.
    """
    image = image.unsqueeze(0).to(device)   # [1, 1, 28, 112]

    # GT
    gt = {k: int(labels[k].item()) for k in CONCEPT_KEYS_ORDERED}

    # S1 predict
    s1_out = system1(image)
    s1_pred = {k: int(F.softmax(s1_out[k][0], dim=0).argmax().item())
               for k in CONCEPT_KEYS_ORDERED}
    s1_conf  = {k: float(F.softmax(s1_out[k][0], dim=0).max().item())
                for k in CONCEPT_KEYS_ORDERED}

    # Concept vector
    cv = soft_concept_vector(s1_out)   # [1, D]

    # S2: match rule
    rule_ids, scores = memory.match(cv, return_scores=True)  # [1], [1, R]
    rule_id  = int(rule_ids[0].item())
    match_sim = float(scores[0, rule_id].item())

    # S2: predict từ centroid của rule matched
    rule_cv  = centroids[rule_ids]   # [1, D]
    s2_logits = head(rule_cv)         # [1, num_classes]
    s2_pred   = int(s2_logits[0].argmax().item())
    s2_conf   = float(F.softmax(s2_logits[0], dim=0).max().item())

    # Decode rule
    rule_str = rule_id_to_str(memory, rule_id)

    # GT expression string
    gt_expr = (f"{gt['digit1']} {op_str(gt['op1'])} {gt['digit2']} = {gt['digit3']}")
    s1_expr = (f"{s1_pred['digit1']} {op_str(s1_pred['op1'])} "
               f"{s1_pred['digit2']} = {s1_pred['digit3']}")

    s1_d3_correct = (s1_pred["digit3"] == gt["digit3"])
    s2_correct    = (s2_pred == gt["digit3"])

    # Classify
    if s1_d3_correct and s2_correct:
        category = "TP"   # cả hai đúng
    elif not s1_d3_correct and s2_correct:
        category = "TN"   # S2 sửa lỗi S1
    elif s1_d3_correct and not s2_correct:
        category = "FP"   # S2 làm hỏng S1
    else:
        category = "FN"   # cả hai sai

    return {
        "gt_expr":        gt_expr,
        "gt":             gt,
        "s1_expr":        s1_expr,
        "s1_pred":        s1_pred,
        "s1_conf":        s1_conf,
        "s1_d3_correct":  s1_d3_correct,
        "rule_id":        rule_id,
        "rule_str":       rule_str,
        "match_sim":      match_sim,
        "s2_pred":        s2_pred,
        "s2_conf":        s2_conf,
        "s2_correct":     s2_correct,
        "category":       category,
        "image":          image[0].cpu(),   # [1, 28, 112]
    }


# ─────────────────────────────────────────────────────────────
# Visualize
# ─────────────────────────────────────────────────────────────

CAT_COLOR = {
    "TP": "#1baf7a",   # teal  — cả hai đúng
    "TN": "#2a78d6",   # blue  — S2 sửa S1
    "FP": "#e8a825",   # amber — S2 làm hỏng
    "FN": "#e34948",   # red   — cả hai sai
}
CAT_LABEL = {
    "TP": "Cả hai đúng",
    "TN": "S2 sửa lỗi S1",
    "FP": "S2 làm hỏng S1",
    "FN": "Cả hai sai",
}


def visualize_examples(examples_by_cat: dict, output_path: Path, n: int = 5):
    """
    Vẽ grid: mỗi hàng = 1 loại (TP/TN/FP/FN), mỗi cột = 1 ví dụ.
    Mỗi ô hiển thị: ảnh + GT + S1 pred + rule matched + S2 pred.
    """
    cats = ["TP", "TN", "FP", "FN"]
    # Chỉ vẽ các loại có đủ ví dụ
    cats_with_data = [c for c in cats if examples_by_cat[c]]
    n_rows = len(cats_with_data)
    n_cols = n

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.2, n_rows * 3.5),
        squeeze=False,
    )
    fig.patch.set_facecolor("#f7f7f5")

    for row, cat in enumerate(cats_with_data):
        examples = examples_by_cat[cat][:n]
        color    = CAT_COLOR[cat]

        for col in range(n_cols):
            ax = axes[row][col]
            ax.set_facecolor("#f7f7f5")
            ax.axis("off")

            # Row label (chỉ cột đầu)
            if col == 0:
                ax.set_ylabel(
                    f"{cat}\n{CAT_LABEL[cat]}",
                    fontsize=9, fontweight="bold",
                    color=color, rotation=0,
                    labelpad=70, va="center",
                )

            if col >= len(examples):
                continue

            ex = examples[col]

            # Ảnh
            img_np = ex["image"][0].numpy()   # [28, 112]
            ax_img = ax.inset_axes([0.0, 0.42, 1.0, 0.55])
            ax_img.imshow(img_np, cmap="gray", vmin=0, vmax=1)
            ax_img.axis("off")
            # Border màu theo category
            for spine in ["top","bottom","left","right"]:
                ax_img.spines[spine].set_visible(True)
                ax_img.spines[spine].set_edgecolor(color)
                ax_img.spines[spine].set_linewidth(2.5)

            # Text info
            gt_str = ex["gt_expr"]
            s1_str = ex["s1_expr"]
            rule_s = ex["rule_str"]
            s2_d3  = ex["s2_pred"]
            s2_ok  = "✓" if ex["s2_correct"] else "✗"
            s1_ok  = "✓" if ex["s1_d3_correct"] else "✗"

            info = (
                f"GT:   {gt_str}\n"
                f"S1:   d3={ex['s1_pred']['digit3']} {s1_ok}  "
                f"({ex['s1_conf']['digit3']:.2f})\n"
                f"Rule: {rule_s}\n"
                f"      sim={ex['match_sim']:.3f}\n"
                f"S2:   d3={s2_d3} {s2_ok}  ({ex['s2_conf']:.2f})"
            )

            ax.text(
                0.5, 0.19, info,
                transform=ax.transAxes,
                fontsize=7.2, fontfamily="monospace",
                va="top", ha="center",
                color="#1a1a18",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor=color, linewidth=1.2,
                    alpha=0.95,
                ),
            )

    # Legend
    patches = [mpatches.Patch(color=CAT_COLOR[c], label=f"{c} — {CAT_LABEL[c]}")
               for c in cats]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    plt.suptitle(
        "ICRL Pipeline Analysis — S1 vs S2 Prediction",
        fontsize=13, fontweight="500", y=1.01,
        color="#1a1a18",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Grid saved → {output_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" \
             else torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    fname    = {"val": "valid.pt", "train": "train.pt", "test": "test.pt"}
    pt_file  = data_dir / fname[args.split]
    if not pt_file.exists():
        pt_file = data_dir / f"{args.split}.pt"
    dataset = MNISTMathPTDataset(pt_file)
    N       = len(dataset)
    print(f"[INFO] {args.split} set: {N} samples | device: {device}")

    # ── Load models ──────────────────────────────────────────
    system1  = load_system1(Path(args.system1_ckpt), device)
    memory   = ICRLRuleMemory.load(args.memory_ckpt, device=str(device))
    centroids = memory.get_centroids().to(device)

    head = nn.Linear(CONCEPT_TOTAL_DIM, 10).to(device)
    head.load_state_dict(torch.load(args.head_ckpt, map_location=device,
                                    weights_only=False))
    head.eval()

    print(f"[INFO] Memory: {memory.num_rules} rules")

    # ── Scan all samples ─────────────────────────────────────
    stats    = defaultdict(int)                      # {category: count}
    examples = defaultdict(list)                     # {category: [sample_dict]}
    records  = []
    n_ex     = args.n_examples

    limit = args.max_samples or N

    for idx in tqdm(range(limit), desc="Analyzing"):
        image, labels = dataset[idx]
        result = run_sample(
            image, labels, system1, memory, head, centroids, device
        )
        cat = result["category"]
        stats[cat] += 1

        # Giữ n_examples cho mỗi loại để visualize
        if len(examples[cat]) < n_ex:
            examples[cat].append(result)

        # Record (không lưu tensor ảnh để JSON nhỏ gọn)
        records.append({
            "idx":           idx,
            "category":      cat,
            "gt_expr":       result["gt_expr"],
            "s1_d3_pred":    result["s1_pred"]["digit3"],
            "s1_d3_correct": result["s1_d3_correct"],
            "rule_id":       result["rule_id"],
            "rule_str":      result["rule_str"],
            "match_sim":     round(result["match_sim"], 4),
            "s2_pred":       result["s2_pred"],
            "s2_correct":    result["s2_correct"],
        })

    # ── Print statistics ─────────────────────────────────────
    total = sum(stats.values())
    print(f"\n{'='*55}")
    print(f"  ICRL Pipeline Analysis — {args.split} set ({total} samples)")
    print(f"{'='*55}")
    print(f"\n  {'Category':6s}  {'Count':>7}  {'%':>6}  Description")
    print(f"  {'-'*50}")

    cat_order = [
        ("TP", "S1 đúng d3,  S2 đúng  — cả hai đúng"),
        ("TN", "S1 sai d3,   S2 đúng  — S2 sửa lỗi S1 ✓"),
        ("FP", "S1 đúng d3,  S2 sai   — S2 làm hỏng S1 ✗"),
        ("FN", "S1 sai d3,   S2 sai   — cả hai sai"),
    ]
    for cat, desc in cat_order:
        n = stats[cat]
        print(f"  {cat:6s}  {n:7d}  {n/total*100:5.1f}%  {desc}")

    s1_acc = (stats["TP"] + stats["FP"]) / total
    s2_acc = (stats["TP"] + stats["TN"]) / total
    fix_rate = stats["TN"] / max(stats["TN"] + stats["FN"], 1)

    print(f"\n  S1 digit3_accuracy  : {s1_acc:.4f}  ({s1_acc*100:.1f}%)")
    print(f"  S2 final_accuracy   : {s2_acc:.4f}  ({s2_acc*100:.1f}%)")
    print(f"  S2 fix rate         : {fix_rate:.4f}  "
          f"(S2 sửa {stats['TN']}/{stats['TN']+stats['FN']} lỗi của S1)")
    print(f"  S2 break rate       : {stats['FP']/max(stats['TP']+stats['FP'],1):.4f}  "
          f"(S2 làm hỏng {stats['FP']}/{stats['TP']+stats['FP']} đúng của S1)")
    print(f"{'='*55}\n")

    # ── Xuất JSON chi tiết ───────────────────────────────────
    json_path = output_dir / f"error_analysis_{args.split}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "total": total,
            "stats": dict(stats),
            "s1_accuracy":  round(s1_acc, 4),
            "s2_accuracy":  round(s2_acc, 4),
            "fix_rate":     round(fix_rate, 4),
            "records":      records,
        }, f, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON saved → {json_path}")

    # ── Visualize grid ───────────────────────────────────────
    grid_path = output_dir / f"error_grid_{args.split}.png"
    visualize_examples(examples, grid_path, n=n_ex)

    # ── In vài ví dụ điển hình ──────────────────────────────
    print(f"\n[INFO] Sample examples:")
    for cat, _ in cat_order:
        exs = examples[cat]
        if not exs:
            print(f"\n  [{cat}] No examples found.")
            continue
        print(f"\n  [{cat}] {CAT_LABEL[cat]}:")
        for e in exs[:3]:
            print(f"    GT={e['gt_expr']:14s}  "
                  f"S1_d3={e['s1_pred']['digit3']}({'✓' if e['s1_d3_correct'] else '✗'})  "
                  f"rule='{e['rule_str']}'  "
                  f"sim={e['match_sim']:.3f}  "
                  f"S2_d3={e['s2_pred']}({'✓' if e['s2_correct'] else '✗'})")


if __name__ == "__main__":
    main()
