"""
visualize_results.py — Dashboard tổng hợp kết quả MultiConcept (S1 + ICRL)
================================================================================

Gộp toàn bộ chỉ số quan trọng đã dùng để đánh giá pipeline trong quá trình
phát triển (accuracy, ổn định training S1, chất lượng từng concept, phân bố
rule, độ thuần khiết rule, tỉ lệ MÂU THUẪN giữa các rule cùng pattern, độ phủ
pattern) vào MỘT ảnh duy nhất, để so sánh nhanh giữa các lần chạy (vd. dataset
có/không cho phép trùng lặp digit) mà không cần chạy lại từng đoạn script rời
rạc như lúc debug.

Usage (Kaggle mặc định, override nếu chạy local):
    python -m src.scripts.multiconcept.visualize_results \\
        --data_dir /kaggle/input/mnist-multiconcept \\
        --system1_ckpt /kaggle/working/outputs/multiconcept_system1/best_model.pt \\
        --system1_metrics /kaggle/working/outputs/multiconcept_system1/metrics.json \\
        --icrl_dir /kaggle/working/outputs/multiconcept_icrl \\
        --output_path /kaggle/working/outputs/multiconcept_icrl/results_dashboard.png \\
        --split val
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.multiconcept.mnist_multiconcept_dataset import MNISTMultiConceptPTDataset
from src.models.multiconcept.system1 import MultiConceptSystem1, soft_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.utils.multiconcept_concepts import (
    CONCEPT_NAMES, NUM_CONCEPTS, LABEL_NAMES, NUM_LABELS, FULL_CV_DIM,
)


# ─────────────────────────────────────────────────────────────
# Palette — cố định, không cycle (xem dataviz skill: categorical theme +
# status palette, giữ 2 vai trò tách biệt để không nhầm "loại nhãn" với
# "tốt/xấu").
# ─────────────────────────────────────────────────────────────

BG        = "#f7f7f5"
INK       = "#1a1a18"
INK_SOFT  = "#57564e"
GRID      = "#e2e0d8"

LABEL_COLOR = {"even": "#2a78d6", "equal": "#1baf7a", "odd": "#eb6834"}  # blue/aqua/orange
STATUS_GOOD = "#0ca30c"
STATUS_WARN = "#fab219"
STATUS_CRIT = "#d03b3b"


def parse_args():
    p = argparse.ArgumentParser(description="Dashboard tổng hợp kết quả MultiConcept.")
    p.add_argument("--data_dir",        type=str, default="/kaggle/input/mnist-multiconcept")
    p.add_argument("--system1_ckpt",    type=str, default="/kaggle/working/outputs/multiconcept_system1/best_model.pt")
    p.add_argument("--system1_metrics", type=str, default=None,
                    help="Mặc định: <thư mục chứa system1_ckpt>/metrics.json")
    p.add_argument("--icrl_dir",        type=str, default="/kaggle/working/outputs/multiconcept_icrl")
    p.add_argument("--output_path",     type=str, default=None,
                    help="Mặc định: <icrl_dir>/results_dashboard.png")
    p.add_argument("--split",           type=str, default="val", choices=["val", "test"])
    p.add_argument("--batch_size",      type=int, default=512)
    p.add_argument("--num_workers",     type=int, default=2)
    p.add_argument("--device",          type=str, default="auto")
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


# ─────────────────────────────────────────────────────────────
# Live inference over 1 split: confusion matrix, per-concept acc/F1,
# match-confidence, per-rule purity thực đo trên ảnh thật.
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_live_stats(system1, memory, head, loader, device, num_labels):
    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)  # [true, pred]
    concept_tp = torch.zeros(NUM_CONCEPTS)
    concept_fp = torch.zeros(NUM_CONCEPTS)
    concept_fn = torch.zeros(NUM_CONCEPTS)
    concept_correct = torch.zeros(NUM_CONCEPTS)
    n_total = 0

    match_sims = []
    rule_stats = defaultdict(lambda: {"n": 0, "correct": 0, "labels": Counter()})

    centroids = memory.get_centroids().to(device)

    for images, labels in tqdm(loader, desc=f"  Inference", leave=False):
        images = images.to(device)
        true_label = labels["label"].to(device)
        true_concepts = labels["concepts"].to(device)

        out = system1(images)
        full_cv = soft_concept_vector(out)
        rule_ids, scores = memory.match(full_cv, return_scores=True)
        best_sim = scores.gather(1, rule_ids.unsqueeze(1)).squeeze(1)
        rule_cvs = centroids[rule_ids]
        pred_label = head(rule_cvs).argmax(dim=1)

        for t, p_ in zip(true_label.tolist(), pred_label.tolist()):
            confusion[t, p_] += 1

        concept_pred = (torch.sigmoid(out["concepts"]) > 0.5).float().cpu()
        tc = true_concepts.cpu()
        concept_tp += ((concept_pred == 1) & (tc == 1)).sum(dim=0)
        concept_fp += ((concept_pred == 1) & (tc == 0)).sum(dim=0)
        concept_fn += ((concept_pred == 0) & (tc == 1)).sum(dim=0)
        concept_correct += (concept_pred == tc).sum(dim=0)
        n_total += images.size(0)

        match_sims.extend(best_sim.cpu().tolist())

        for rid, t, p_ in zip(rule_ids.tolist(), true_label.tolist(), pred_label.tolist()):
            st = rule_stats[rid]
            st["n"] += 1
            st["correct"] += int(p_ == t)
            st["labels"][t] += 1

    precision = concept_tp / (concept_tp + concept_fp).clamp(min=1e-8)
    recall = concept_tp / (concept_tp + concept_fn).clamp(min=1e-8)
    f1 = (2 * precision * recall / (precision + recall).clamp(min=1e-8)).tolist()
    acc = (concept_correct / n_total).tolist()

    return {
        "confusion": confusion,
        "n_total": n_total,
        "concept_f1": f1,
        "concept_acc": acc,
        "match_sims": match_sims,
        "rule_stats": rule_stats,
    }


def pattern_contradiction_stats(rules_json: list[dict]) -> dict:
    by_pattern = defaultdict(list)
    for r in rules_json:
        present = frozenset(
            int(k.split("_")[-1]) for k, v in r["slots"].items()
            if k.startswith("has_digit_") and v["value"] == "present"
        )
        by_pattern[present].append(r)

    total_n = sum(r["n"] for r in rules_json)
    contradictory_patterns = 0
    contradictory_n = 0
    for pat, rs in by_pattern.items():
        if len({r["label_name"] for r in rs}) > 1:
            contradictory_patterns += 1
            contradictory_n += sum(r["n"] for r in rs)

    return {
        "n_patterns": len(by_pattern),
        "contradictory_patterns": contradictory_patterns,
        "contradictory_n": contradictory_n,
        "total_n": total_n,
    }


# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────

def style_axis(ax):
    ax.set_facecolor(BG)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def stat_tile(ax, label, value, sub, color=INK):
    ax.set_facecolor("#ffffff")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.text(0.06, 0.72, label.upper(), fontsize=9.5, color=INK_SOFT,
             fontweight="bold", transform=ax.transAxes, ha="left")
    ax.text(0.06, 0.36, value, fontsize=26, color=color, fontweight="bold",
             transform=ax.transAxes, ha="left", va="center")
    ax.text(0.06, 0.12, sub, fontsize=8.5, color=INK_SOFT,
             transform=ax.transAxes, ha="left")


def plot_dashboard(args, s1_metrics, icrl_metrics, rules_json, live, out_path):
    fig = plt.figure(figsize=(19, 15), facecolor=BG)

    dataset_name = Path(args.data_dir).name
    contra = pattern_contradiction_stats(rules_json)
    contra_rate = contra["contradictory_n"] / max(contra["total_n"], 1)

    # Hàng 0: 4 stat tile riêng; phần thân: lưới 3x3 cho các biểu đồ.
    gs_top = fig.add_gridspec(1, 4, left=0.045, right=0.98, top=0.93, bottom=0.80, wspace=0.25)
    gs_main = fig.add_gridspec(3, 3, left=0.045, right=0.98, top=0.75, bottom=0.04,
                                hspace=0.55, wspace=0.32)

    fig.suptitle(f"MNIST-MultiConcept — Kết quả pipeline S1 → ICRL  ·  {dataset_name}  ·  split={args.split}",
                 fontsize=16, fontweight="bold", color=INK, x=0.045, ha="left", y=0.985)

    tiles = [
        ("Accuracy (val)", f"{icrl_metrics['val_accuracy']*100:.2f}%",
         f"test = {icrl_metrics['test_accuracy']*100:.2f}%", INK),
        ("S1 concept F1", f"{s1_metrics['test_metrics']['concept_macro_f1']*100:.1f}%",
         f"mean acc = {s1_metrics['test_metrics']['concept_mean_acc']*100:.1f}%", INK),
        ("Số rule", f"{icrl_metrics['num_rules']}",
         f"{contra['n_patterns']} concept pattern khác nhau", INK),
        ("Rule mâu thuẫn", f"{contra_rate*100:.2f}%",
         f"{contra['contradictory_patterns']}/{contra['n_patterns']} pattern có ≥2 nhãn",
         STATUS_GOOD if contra_rate < 0.02 else STATUS_WARN if contra_rate < 0.15 else STATUS_CRIT),
    ]
    for i, (lbl, val, sub, color) in enumerate(tiles):
        ax = fig.add_subplot(gs_top[0, i])
        stat_tile(ax, lbl, val, sub, color)

    # ── (1,1) S1 training curves ────────────────────────────────
    ax = fig.add_subplot(gs_main[0, 0:2])
    style_axis(ax)
    hist = s1_metrics["history"]
    epochs = [h["epoch"] for h in hist]
    ax.plot(epochs, [h["val_concept_macro_f1"] for h in hist], color="#2a78d6",
             linewidth=2, label="concept_macro_f1", zorder=3)
    ax.plot(epochs, [h["val_label_acc"] for h in hist], color="#eb6834",
             linewidth=2, label="label_acc", zorder=3)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("epoch", fontsize=9, color=INK_SOFT)
    ax.set_title("S1 training — val concept F1 vs label acc theo epoch\n"
                 "(chỗ label_acc tụt sâu = bất ổn training của label_head)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    # ── (1,3) Confusion matrix ──────────────────────────────────
    ax = fig.add_subplot(gs_main[0, 2])
    ax.set_facecolor(BG)
    cm = live["confusion"]
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt_color = "white" if cm_norm[i, j] > 0.5 else INK
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)", ha="center", va="center",
                    fontsize=8, color=txt_color)
    ax.set_xticks(range(NUM_LABELS)); ax.set_xticklabels(LABEL_NAMES, fontsize=9)
    ax.set_yticks(range(NUM_LABELS)); ax.set_yticklabels(LABEL_NAMES, fontsize=9)
    ax.set_xlabel("dự đoán (S2)", fontsize=9, color=INK_SOFT)
    ax.set_ylabel("nhãn thật", fontsize=9, color=INK_SOFT)
    ax.set_title("Confusion matrix (pipeline đầy đủ)", fontsize=10, color=INK, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── (2,1) Per-concept accuracy ──────────────────────────────
    ax = fig.add_subplot(gs_main[1, 0])
    style_axis(ax)
    order = np.argsort(live["concept_acc"])
    names = [CONCEPT_NAMES[i].replace("has_digit_", "") for i in order]
    accs = [live["concept_acc"][i] for i in order]
    f1s = [live["concept_f1"][i] for i in order]
    y = np.arange(len(names))
    # Dot plot (không phải bar) — giá trị đều dồn gần 1.0, bar bắt buộc bắt
    # đầu từ 0 sẽ xoá hết khác biệt; dot plot cho phép zoom trục X an toàn
    # (không có yêu cầu "phải bắt đầu từ 0" như bar).
    x_floor = min(min(accs), min(f1s))
    x_lo = max(0.0, np.floor(x_floor * 20) / 20 - 0.03)
    for yi in y:
        ax.plot([x_lo, 1.01], [yi, yi], color=GRID, linewidth=1, zorder=1)
    ax.plot(accs, y, "o", color="#2a78d6", markersize=7, label="accuracy", zorder=3)
    ax.plot(f1s, y, "o", color="#eb6834", markersize=7, label="F1", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([f"digit {n}" for n in names], fontsize=8)
    ax.set_xlim(x_lo, 1.01)
    ax.set_title("Chất lượng nhận diện từng concept (has_digit_X)\n"
                 "(trục X thu hẹp quanh vùng dữ liệu — mọi concept đều >90%)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=True, fontsize=8, loc="lower left", framealpha=0.95,
              facecolor="white", edgecolor=GRID)

    # ── (2,2) Rule size distribution ────────────────────────────
    ax = fig.add_subplot(gs_main[1, 1])
    style_axis(ax)
    ns = [r["n"] for r in rules_json]
    bins = np.logspace(0, np.log10(max(ns) + 1), 25)
    ax.hist(ns, bins=bins, color="#4a3aa7", edgecolor=BG, linewidth=0.6, zorder=3)
    ax.set_xscale("log")
    ax.axvline(5, color=STATUS_CRIT, linestyle="--", linewidth=1, label="n_min=5", zorder=4)
    ax.set_xlabel("n (số mẫu train mỗi rule, log scale)", fontsize=9, color=INK_SOFT)
    ax.set_title(f"Phân bố kích thước rule ({len(rules_json)} rule)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)

    # ── (2,3) Rule purity distribution (đo thật trên live inference) ──
    ax = fig.add_subplot(gs_main[1, 2])
    style_axis(ax)
    purities = []
    weights = []
    for rid, st in live["rule_stats"].items():
        maj = st["labels"].most_common(1)[0][1]
        purities.append(maj / st["n"])
        weights.append(st["n"])
    buckets = [(0.95, 1.01), (0.85, 0.95), (0.70, 0.85), (0.50, 0.70), (0.0, 0.50)]
    bucket_labels = ["≥95%", "85-95%", "70-85%", "50-70%", "<50%"]
    bucket_mass = []
    for lo, hi in buckets:
        mass = sum(w for p, w in zip(purities, weights) if lo <= p < hi)
        bucket_mass.append(mass / max(sum(weights), 1) * 100)
    colors = [STATUS_GOOD, STATUS_GOOD, STATUS_WARN, STATUS_CRIT, STATUS_CRIT]
    ax.bar(bucket_labels, bucket_mass, color=colors, zorder=3)
    for i, v in enumerate(bucket_mass):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(0, 105)
    ax.set_ylabel(f"% mẫu {args.split} (n-weighted)", fontsize=9, color=INK_SOFT)
    ax.set_title("Độ thuần khiết rule — đo trực tiếp trên ảnh thật\n"
                 "(purity = tỉ lệ ảnh matched vào rule có nhãn == nhãn rule)",
                 fontsize=10, color=INK, loc="left")

    # ── (3,1) Contradiction breakdown ───────────────────────────
    ax = fig.add_subplot(gs_main[2, 0])
    style_axis(ax)
    consistent_n = contra["total_n"] - contra["contradictory_n"]
    ax.bar(["Nhất quán", "Mâu thuẫn"],
           [consistent_n / contra["total_n"] * 100, contra_rate * 100],
           color=[STATUS_GOOD, STATUS_CRIT], zorder=3)
    ax.set_ylabel("% mẫu train", fontsize=9, color=INK_SOFT)
    ax.set_ylim(0, 105)
    for i, v in enumerate([consistent_n / contra["total_n"] * 100, contra_rate * 100]):
        ax.text(i, v + 1, f"{v:.2f}%", ha="center", fontsize=9, color=INK)
    ax.set_title("Cùng 1 pattern concept — rule có đồng thuận nhãn không?\n"
                 f"({contra['contradictory_patterns']}/{contra['n_patterns']} pattern có ≥2 rule khác nhãn)",
                 fontsize=10, color=INK, loc="left")

    # ── (3,2) True vs predicted label distribution ──────────────
    ax = fig.add_subplot(gs_main[2, 1])
    style_axis(ax)
    true_counts = live["confusion"].sum(axis=1)
    pred_counts = live["confusion"].sum(axis=0)
    x = np.arange(NUM_LABELS)
    w = 0.35
    ax.bar(x - w/2, true_counts, width=w, label="thật",
           color=[LABEL_COLOR[n] for n in LABEL_NAMES], alpha=1.0, zorder=3)
    ax.bar(x + w/2, pred_counts, width=w, label="dự đoán",
           color=[LABEL_COLOR[n] for n in LABEL_NAMES], alpha=0.45, zorder=3,
           hatch="//", edgecolor=INK)
    ax.set_xticks(x); ax.set_xticklabels(LABEL_NAMES, fontsize=9)
    ax.set_ylabel(f"số mẫu ({args.split})", fontsize=9, color=INK_SOFT)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Phân phối nhãn: thật vs dự đoán", fontsize=10, color=INK, loc="left")

    # ── (3,3) Match confidence / coverage ───────────────────────
    ax = fig.add_subplot(gs_main[2, 2])
    style_axis(ax)
    sims = np.array(live["match_sims"])
    ax.hist(sims, bins=40, color="#1baf7a", edgecolor=BG, linewidth=0.5, zorder=3)
    ax.set_yscale("log")  # đa số mẫu dồn sát 1.0 — log scale mới thấy được phần đuôi nhỏ
    ax.axvline(0.85, color=STATUS_CRIT, linestyle="--", linewidth=1,
               label=f"<0.85: {(sims<0.85).mean()*100:.2f}% mẫu")
    ax.set_xlabel("cosine similarity với rule khớp nhất", fontsize=9, color=INK_SOFT)
    ax.set_ylabel("số mẫu (log scale)", fontsize=9, color=INK_SOFT)
    ax.set_title(f"Độ tin cậy match ({args.split}, {live['n_total']} ảnh)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" \
             else torch.device(args.device)

    system1_ckpt = Path(args.system1_ckpt)
    system1_metrics_path = Path(args.system1_metrics) if args.system1_metrics \
        else system1_ckpt.parent / "metrics.json"
    icrl_dir = Path(args.icrl_dir)
    out_path = Path(args.output_path) if args.output_path else icrl_dir / "results_dashboard.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading S1: {system1_ckpt}")
    system1 = load_system1(system1_ckpt, device)
    with open(system1_metrics_path, encoding="utf-8") as f:
        s1_metrics = json.load(f)

    print(f"[INFO] Loading ICRL: {icrl_dir}")
    with open(icrl_dir / "metrics.json", encoding="utf-8") as f:
        icrl_metrics = json.load(f)
    with open(icrl_dir / "icrl_rules.json", encoding="utf-8") as f:
        rules_json = json.load(f)
    memory = ICRLRuleMemory.load(icrl_dir / "icrl_rule_memory.pt", device=str(device))
    head = nn.Linear(FULL_CV_DIM, NUM_LABELS).to(device)
    head.load_state_dict(torch.load(icrl_dir / "prediction_head.pt", map_location=device, weights_only=False))
    head.eval()

    data_dir = Path(args.data_dir)
    split_file = {"val": "valid.pt", "test": "test.pt"}[args.split]
    dataset = MNISTMultiConceptPTDataset(data_dir / split_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    print(f"[INFO] {args.split} set: {len(dataset)} samples")

    live = run_live_stats(system1, memory, head, loader, device, NUM_LABELS)

    print("[INFO] Rendering dashboard...")
    plot_dashboard(args, s1_metrics, icrl_metrics, rules_json, live, out_path)
    print(f"[DONE] Saved -> {out_path}")


if __name__ == "__main__":
    main()
