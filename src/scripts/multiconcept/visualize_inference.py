"""
visualize_inference.py — Trace suy luận từng ảnh: S1 → concept → rule → S2
================================================================================

Khác với visualize_results.py (tổng hợp số liệu toàn tập), script này vẽ chi
tiết pipeline TRÊN TỪNG ẢNH CỤ THỂ: ảnh gốc, 4 chữ số thật (digits), 10
concept has_digit_X (S1 activate bao nhiêu % + đúng/sai so với ground-truth),
S1 tự đoán nhãn gì (phân phối xác suất 3 lớp), S2 map vào rule nào (concept
signature của rule + nhãn rule gán), và S2 cuối cùng có đổi kết quả so với S1
tự đoán hay không — đúng những gì cần để tự kiểm tra 1 ca cụ thể bằng mắt.

Chọn ảnh bằng 1 trong 2 cách:
  --indices 12 340 5981     chọn đúng các ảnh theo index trong split
  --filter TP/TN/FP/FN/all  lấy ngẫu nhiên --n_examples ảnh theo loại kết quả
      TP: S1 đúng, S2 đúng       (S2 giữ nguyên đúng)
      TN: S1 sai,  S2 đúng       (S2 sửa lỗi S1)   ← quan trọng nhất
      FP: S1 đúng, S2 sai        (S2 làm hỏng)     ← đáng lo nhất
      FN: S1 sai,  S2 sai        (S2 không cứu được)
      all: không lọc theo đúng/sai, chỉ lấy ngẫu nhiên

Usage (Kaggle mặc định, override nếu chạy local):
    python -m src.scripts.multiconcept.visualize_inference \\
        --data_dir /kaggle/input/mnist-multiconcept \\
        --system1_ckpt /kaggle/working/outputs/multiconcept_system1/best_model.pt \\
        --icrl_dir /kaggle/working/outputs/multiconcept_icrl \\
        --output_path /kaggle/working/outputs/multiconcept_icrl/inference_trace.png \\
        --split val --filter TN --n_examples 6
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.multiconcept.system1 import MultiConceptSystem1, soft_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.utils.multiconcept_concepts import CONCEPT_NAMES, NUM_CONCEPTS, LABEL_NAMES, NUM_LABELS, FULL_CV_DIM


BG        = "#f7f7f5"
INK       = "#1a1a18"
INK_SOFT  = "#57564e"
GRID      = "#e2e0d8"
LABEL_COLOR = {"even": "#2a78d6", "equal": "#1baf7a", "odd": "#eb6834"}
STATUS_GOOD = "#0ca30c"
STATUS_CRIT = "#d03b3b"


def parse_args():
    p = argparse.ArgumentParser(description="Vẽ trace suy luận S1->S2 cho từng ảnh cụ thể.")
    p.add_argument("--data_dir",     type=str, default="/kaggle/input/mnist-multiconcept")
    p.add_argument("--system1_ckpt", type=str, default="/kaggle/working/outputs/multiconcept_system1/best_model.pt")
    p.add_argument("--icrl_dir",     type=str, default="/kaggle/working/outputs/multiconcept_icrl")
    p.add_argument("--output_path",  type=str, default=None,
                    help="Mặc định: <icrl_dir>/inference_trace_<filter>.png")
    p.add_argument("--split",        type=str, default="val", choices=["val", "test"])
    p.add_argument("--indices",      type=int, nargs="+", default=None,
                    help="Chọn đúng các ảnh theo index trong split. Nếu set, bỏ qua --filter/--n_examples.")
    p.add_argument("--filter",       type=str, default="all",
                    choices=["all", "TP", "TN", "FP", "FN"],
                    help="TP=S1 đúng+S2 đúng, TN=S1 sai+S2 đúng (S2 sửa), "
                         "FP=S1 đúng+S2 sai (S2 làm hỏng), FN=cả hai sai.")
    p.add_argument("--n_examples",   type=int, default=6)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device",       type=str, default="auto")
    return p.parse_args()


def load_system1(ckpt_path: Path, device: torch.device) -> MultiConceptSystem1:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    model = MultiConceptSystem1(
        feature_dim=saved_args.get("feature_dim", 256),
        num_concepts=saved_args.get("num_concepts", NUM_CONCEPTS),
        num_labels=saved_args.get("num_labels", NUM_LABELS),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    return model


@torch.no_grad()
def run_one(idx, images, concepts_gt, labels_gt, digits, system1, memory, head, centroids,
            rules_by_id, device):
    image = images[idx:idx + 1].to(device)
    out = system1(image)

    concept_probs = torch.sigmoid(out["concepts"])[0].cpu()
    concept_pred = (concept_probs > 0.5).float()
    label_probs = F.softmax(out["label"], dim=-1)[0].cpu()
    s1_pred = int(label_probs.argmax())

    full_cv = soft_concept_vector(out)
    rule_ids, scores = memory.match(full_cv, return_scores=True)
    rule_id = int(rule_ids[0])
    match_sim = float(scores[0, rule_id])
    rule_cv = centroids[rule_ids]
    s2_logits = head(rule_cv)
    s2_probs = F.softmax(s2_logits, dim=-1)[0].cpu()
    s2_pred = int(s2_logits[0].argmax())

    true_label = int(labels_gt[idx])
    dt = digits[idx].tolist()
    n_odd = sum(1 for d in dt if d % 2 == 1)
    n_even = len(dt) - n_odd

    return {
        "idx": idx,
        "image": images[idx, 0].numpy(),
        "digits": dt,
        "n_odd": n_odd, "n_even": n_even,
        "concept_probs": concept_probs.tolist(),
        "concept_pred": concept_pred.tolist(),
        "concept_gt": concepts_gt[idx].tolist(),
        "label_probs": label_probs.tolist(),
        "s1_pred": s1_pred,
        "s1_correct": s1_pred == true_label,
        "rule_id": rule_id,
        "rule_info": rules_by_id.get(rule_id),
        "match_sim": match_sim,
        "s2_probs": s2_probs.tolist(),
        "s2_pred": s2_pred,
        "s2_correct": s2_pred == true_label,
        "true_label": true_label,
        "changed": s1_pred != s2_pred,
    }


def categorize(s1_correct: bool, s2_correct: bool) -> str:
    if s1_correct and s2_correct:
        return "TP"
    if not s1_correct and s2_correct:
        return "TN"
    if s1_correct and not s2_correct:
        return "FP"
    return "FN"


@torch.no_grad()
def scan_categories(images, labels_gt, system1, memory, head, centroids, device, batch_size=512):
    """Quét cả split theo BATCH (nhanh) chỉ để lấy category từng ảnh — không
    tính concept/label chi tiết (việc đó chỉ cần cho vài ảnh cuối cùng)."""
    N = len(images)
    cats = [None] * N
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        img_b = images[start:end].to(device)
        out = system1(img_b)
        full_cv = soft_concept_vector(out)
        rule_ids, _ = memory.match(full_cv)
        rule_cvs = centroids[rule_ids]
        s2_pred = head(rule_cvs).argmax(dim=1)
        s1_pred = out["label"].argmax(dim=1)
        true_b = labels_gt[start:end].to(device)
        for i in range(end - start):
            cats[start + i] = categorize(bool(s1_pred[i] == true_b[i]), bool(s2_pred[i] == true_b[i]))
    return cats


def draw_card(fig, gs_row, rec):
    true_name = LABEL_NAMES[rec["true_label"]]
    s1_name = LABEL_NAMES[rec["s1_pred"]]
    s2_name = LABEL_NAMES[rec["s2_pred"]]
    border_color = STATUS_GOOD if rec["s2_correct"] else STATUS_CRIT

    # ── ảnh + digits + verdict ──
    ax_img = fig.add_subplot(gs_row[0, 0])
    ax_img.set_facecolor(BG)
    ax_img.imshow(rec["image"], cmap="gray", vmin=0, vmax=1)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_edgecolor(border_color); spine.set_linewidth(3)
    ax_img.set_title(f"idx={rec['idx']}   digits={rec['digits']}\n"
                      f"{rec['n_odd']} lẻ, {rec['n_even']} chẵn  →  thật: {true_name}",
                      fontsize=8.5, color=INK)

    # ── concept activation (10 has_digit_X) ──
    ax_c = fig.add_subplot(gs_row[0, 1])
    ax_c.set_facecolor(BG)
    y = np.arange(NUM_CONCEPTS)
    probs = rec["concept_probs"]
    pred = rec["concept_pred"]
    gt = rec["concept_gt"]
    colors = [STATUS_GOOD if pred[i] == gt[i] else STATUS_CRIT for i in range(NUM_CONCEPTS)]
    ax_c.barh(y, probs, color=colors, height=0.65, zorder=3)
    ax_c.axvline(0.5, color=INK_SOFT, linewidth=0.8, linestyle=":", zorder=2)
    for i in range(NUM_CONCEPTS):
        marker = "●" if gt[i] == 1 else "○"
        ax_c.text(1.03, i, marker, fontsize=8, color=INK_SOFT, va="center")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([n.replace("has_digit_", "d") for n in CONCEPT_NAMES], fontsize=7)
    ax_c.set_xlim(0, 1.12)
    ax_c.set_ylim(-0.6, NUM_CONCEPTS - 0.4)
    ax_c.invert_yaxis()
    ax_c.tick_params(colors=INK_SOFT, labelsize=7)
    for spine in ["top", "right"]:
        ax_c.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_c.spines[spine].set_color(GRID)

    # ── S1 label distribution ──
    ax_l = fig.add_subplot(gs_row[0, 2])
    ax_l.set_facecolor(BG)
    yl = np.arange(NUM_LABELS)
    colors_l = [LABEL_COLOR[n] for n in LABEL_NAMES]
    ax_l.barh(yl, rec["label_probs"], color=colors_l, height=0.6, zorder=3)
    ax_l.set_yticks(yl); ax_l.set_yticklabels(LABEL_NAMES, fontsize=8)
    ax_l.set_xlim(0, 1.02)
    true_pos = rec["true_label"]
    ax_l.get_yticklabels()[true_pos].set_fontweight("bold")
    ax_l.axhline(true_pos, color=INK, linewidth=0, zorder=1)
    ax_l.barh([true_pos], [1.0], color="none", edgecolor=INK, linewidth=1.5,
              height=0.6, zorder=4)
    for spine in ["top", "right"]:
        ax_l.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_l.spines[spine].set_color(GRID)
    ax_l.tick_params(colors=INK_SOFT, labelsize=7)

    # ── rule + verdict text ──
    ax_t = fig.add_subplot(gs_row[0, 3])
    ax_t.set_facecolor(BG)
    ax_t.set_xticks([]); ax_t.set_yticks([])
    for spine in ax_t.spines.values():
        spine.set_visible(False)

    ri = rec["rule_info"]
    if ri is not None:
        present = ", ".join(c.replace("has_digit_", "") for c in ri.get("present_concepts", [])) or "(rỗng)"
        rule_txt = (f"Rule #{rec['rule_id']}  (n={ri['n']}, coh={ri['coherence']:.3f})\n"
                    f"digit hiện diện: {{{present}}}\n"
                    f"nhãn rule: {ri['label_name']}\n"
                    f"match sim: {rec['match_sim']:.3f}")
    else:
        rule_txt = f"Rule #{rec['rule_id']}  (không có metadata)\nmatch sim: {rec['match_sim']:.3f}"

    s1_mark = "✓" if rec["s1_correct"] else "✗"
    s2_mark = "✓" if rec["s2_correct"] else "✗"
    s1_color = STATUS_GOOD if rec["s1_correct"] else STATUS_CRIT
    s2_color = STATUS_GOOD if rec["s2_correct"] else STATUS_CRIT
    change_txt = "→ KHÔNG đổi" if not rec["changed"] else "→ CÓ đổi"
    cat = categorize(rec["s1_correct"], rec["s2_correct"])

    ax_t.text(0.0, 0.97, rule_txt, fontsize=8.3, color=INK, va="top", ha="left",
              transform=ax_t.transAxes, family="monospace")
    ax_t.text(0.0, 0.42, f"S1 tự đoán: {s1_name} {s1_mark}", fontsize=9, color=s1_color,
              va="top", ha="left", transform=ax_t.transAxes, fontweight="bold")
    ax_t.text(0.0, 0.30, f"S2 (qua rule): {s2_name} {s2_mark}", fontsize=9, color=s2_color,
              va="top", ha="left", transform=ax_t.transAxes, fontweight="bold")
    ax_t.text(0.0, 0.18, change_txt, fontsize=8.3, color=INK_SOFT, va="top", ha="left",
              transform=ax_t.transAxes)
    ax_t.text(0.0, 0.03, f"[{cat}]", fontsize=9, color=border_color, va="top", ha="left",
              transform=ax_t.transAxes, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=border_color))


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" \
             else torch.device(args.device)
    rng = random.Random(args.seed)

    data_dir = Path(args.data_dir)
    split_file = {"val": "valid.pt", "test": "test.pt"}[args.split]
    data = torch.load(data_dir / split_file, map_location="cpu", weights_only=True)
    images, concepts_gt, labels_gt, digits = data["images"], data["concepts"], data["label"], data["digits"]
    N = len(images)
    print(f"[INFO] {args.split} set: {N} samples")

    system1 = load_system1(Path(args.system1_ckpt), device)
    icrl_dir = Path(args.icrl_dir)
    memory = ICRLRuleMemory.load(icrl_dir / "icrl_rule_memory.pt", device=str(device))
    head = nn.Linear(FULL_CV_DIM, NUM_LABELS).to(device)
    head.load_state_dict(torch.load(icrl_dir / "prediction_head.pt", map_location=device, weights_only=False))
    head.eval()
    centroids = memory.get_centroids().to(device)

    with open(icrl_dir / "icrl_rules.json", encoding="utf-8") as f:
        rules_by_id = {r["rule_id"]: r for r in json.load(f)}

    if args.indices:
        chosen = args.indices
    else:
        print(f"[INFO] Quét theo batch toàn bộ {args.split} để lọc theo filter='{args.filter}'...")
        if args.filter == "all":
            pool = list(range(N))
        else:
            cats = scan_categories(images, labels_gt, system1, memory, head, centroids, device)
            pool = [i for i, c in enumerate(cats) if c == args.filter]
        if not pool:
            raise RuntimeError(f"Không tìm thấy ảnh nào khớp filter='{args.filter}' trong {args.split}.")
        rng.shuffle(pool)
        chosen = pool[:args.n_examples]

    print(f"[INFO] Vẽ {len(chosen)} ảnh: {chosen}")
    records = [run_one(idx, images, concepts_gt, labels_gt, digits, system1, memory,
                       head, centroids, rules_by_id, device) for idx in chosen]

    n = len(records)
    row_h = 2.6
    fig_h = row_h * n + 1.15   # dải cố định trên cùng cho suptitle + header cột, không co lại khi n tăng
    fig = plt.figure(figsize=(15, fig_h), facecolor=BG)
    gs = fig.add_gridspec(n, 1,
                           left=0.03, right=0.98,
                           top=1 - 1.0 / fig_h, bottom=0.15 / fig_h,
                           hspace=0.8)

    dataset_name = data_dir.name
    fig.suptitle(f"MNIST-MultiConcept — Trace suy luận từng ảnh  ·  {dataset_name}  ·  "
                 f"split={args.split}  ·  filter={args.filter if not args.indices else 'indices'}",
                 fontsize=13, fontweight="bold", color=INK, x=0.03, y=1 - 0.3 / fig_h, ha="left")

    # Header cột — chỉ 1 lần cho cả ảnh, thay vì lặp lại mỗi hàng (4 cột dùng
    # đúng width_ratios của draw_card để căn giữa đúng vị trí).
    col_ratios = np.array([1, 1.3, 0.9, 1.3], dtype=float)
    col_edges = np.concatenate([[0], np.cumsum(col_ratios)]) / col_ratios.sum()
    col_centers = (col_edges[:-1] + col_edges[1:]) / 2
    x0, x1 = 0.03, 0.98
    col_headers = [
        "Ảnh + ground-truth",
        "S1: concept has_digit_X\n(bar = P(present), ● = GT present)",
        "S1: tự đoán nhãn\n(viền đen = nhãn thật)",
        "S2: rule đã map + kết luận cuối",
    ]
    header_y = 1 - 0.65 / fig_h
    for cx, htxt in zip(col_centers, col_headers):
        fig.text(x0 + cx * (x1 - x0), header_y, htxt, fontsize=9, color=INK_SOFT,
                  fontweight="bold", ha="center", va="top")

    for row, rec in enumerate(records):
        gs_row = gs[row, 0].subgridspec(1, 4, width_ratios=[1, 1.3, 0.9, 1.3], wspace=0.35)
        draw_card(fig, gs_row, rec)

    out_path = Path(args.output_path) if args.output_path else \
        icrl_dir / f"inference_trace_{args.filter if not args.indices else 'indices'}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"[DONE] Saved -> {out_path}")


if __name__ == "__main__":
    main()
