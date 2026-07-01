"""
visualize_pipeline.py
=====================
Visualize một ảnh từ MNIST Math dataset và toàn bộ pipeline output:
  - Ảnh gốc + spatial slot boundaries
  - System 1: predicted probabilities per slot (bar chart)
  - System 2: rule được chọn + per-slot cosine match scores
  - So sánh GT vs S1 prediction vs S2 rule

Usage:
    python -m src.scripts.visualize_pipeline \\
        --data_dir data/mnist_math \\
        --system1_ckpt outputs/system1_v2/best_model.pt \\
        --system2_ckpt outputs/system2_v4/best_system2.pt \\
        --sample_idx 200 \\
        --output_path outputs/pipeline_vis.png

    # Random sample:
    python -m src.scripts.visualize_pipeline \\
        --data_dir data/mnist_math --random

    # Chọn loại: valid / invalid
    python -m src.scripts.visualize_pipeline \\
        --data_dir data/mnist_math --pick invalid
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

from src.models.multi_head_system1 import MultiHeadSystem1
from src.models.system2_model import System2Rules
from src.models.rule_memory import (
    soft_concept_vector,
    CONCEPT_KEYS_ORDERED,
    CONCEPT_DIMS,
    CONCEPT_OFFSETS,
)
from src.utils.symbols import ID_TO_SYMBOL


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

SLOT_COLORS = {
    "digit1": "#378ADD",
    "op1":    "#1D9E75",
    "digit2": "#7F77DD",
    "op2":    "#D85A30",
    "digit3": "#BA7517",
}

SLOT_DISPLAY = {
    "digit1": "digit 1",
    "op1":    "op 1",
    "digit2": "digit 2",
    "op2":    "op 2",
    "digit3": "digit 3",
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def label_to_str(key: str, idx: int) -> str:
    """Convert concept index to readable label."""
    if key in ("op1", "op2"):
        return ID_TO_SYMBOL.get(idx, str(idx))
    return str(idx)


def load_system1(ckpt_path: Path | None, device: torch.device) -> MultiHeadSystem1:
    if ckpt_path is None or not ckpt_path.exists():
        print("[WARN] No System1 checkpoint — using random weights")
        return MultiHeadSystem1(feature_dim=256, num_slots=5).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    saved = ckpt.get("args", {})
    model = MultiHeadSystem1(
        feature_dim=saved.get("feature_dim", 256),
        num_slots=saved.get("num_slots", 5),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"[INFO] System1 loaded: {ckpt_path}")
    return model


def load_system2(ckpt_path: Path | None, device: torch.device) -> System2Rules:
    if ckpt_path is None or not ckpt_path.exists():
        print("[WARN] No System2 checkpoint — using random weights")
        return System2Rules(num_rules=128).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    saved = ckpt.get("args", {})
    model = System2Rules(
        num_rules=saved.get("num_rules", 128),
        score_mode=saved.get("score_mode", "slot_cosine"),
        temperature=saved.get("T_min", 0.07),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"[INFO] System2 loaded: {ckpt_path}")
    return model


def get_gt_labels(data: dict, idx: int) -> dict[str, int]:
    return {k: data[k][idx].item() for k in CONCEPT_KEYS_ORDERED}


# ─────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────

def _bar_probs(ax, probs: np.ndarray, key: str, gt_idx: int, pred_idx: int,
               title: str, max_bars: int = 10):
    """Draw probability bar chart for one slot."""
    n     = len(probs)
    xs    = np.arange(n)
    color = SLOT_COLORS[key]

    # Base bars
    bars = ax.bar(xs, probs, color=color, alpha=0.35, width=0.7, zorder=2)

    # Highlight GT (green border) and pred (solid fill)
    for i, bar in enumerate(bars):
        if i == pred_idx:
            bar.set_alpha(0.9)
            bar.set_linewidth(0)
        if i == gt_idx:
            bar.set_edgecolor("#1a1a1a")
            bar.set_linewidth(2.0)
            bar.set_linestyle("--")

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, min(1.05, probs.max() * 1.25 + 0.05))
    ax.set_xticks(xs)

    # X-tick labels
    if key in ("op1", "op2"):
        xlabels = [ID_TO_SYMBOL.get(i, str(i)) for i in range(n)]
    elif key == "valid":
        xlabels = ["no", "yes"]
    else:
        xlabels = [str(i) for i in range(n)]
    ax.set_xticklabels(xlabels, fontsize=8)

    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", ".5", "1"], fontsize=7)
    ax.set_title(title, fontsize=9, pad=3, color=color, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5, zorder=1)


def _match_bar(ax, slot_scores: np.ndarray, slot_keys: list[str],
               slot_match: np.ndarray, threshold: float = 0.7):
    """Horizontal bar chart of per-slot cosine match scores."""
    n      = len(slot_keys)
    ys     = np.arange(n)
    colors = [SLOT_COLORS[k] for k in slot_keys]
    alphas = [0.9 if m else 0.35 for m in slot_match]

    bars = ax.barh(ys, slot_scores, color=colors, alpha=0.7, height=0.6, zorder=2)
    for bar, alpha, matched in zip(bars, alphas, slot_match):
        bar.set_alpha(alpha)
        if matched:
            bar.set_edgecolor("#1a1a1a")
            bar.set_linewidth(1.5)

    # Threshold line
    ax.axvline(threshold, color="#E24B4A", linewidth=1.0, linestyle="--",
               alpha=0.7, label=f"threshold={threshold}")

    ax.set_xlim(0, 1.05)
    ax.set_yticks(ys)
    ax.set_yticklabels([SLOT_DISPLAY[k] for k in slot_keys], fontsize=8)
    ax.set_xlabel("Cosine similarity", fontsize=8)
    ax.set_title("Per-slot match scores", fontsize=9, fontweight="bold",
                 color="#444441", pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5, zorder=1)
    ax.legend(fontsize=7, loc="lower right")

    # Score labels
    for y, score in zip(ys, slot_scores):
        ax.text(min(score + 0.02, 0.97), y, f"{score:.2f}",
                va="center", fontsize=7, color="#2c2c2a")


# ─────────────────────────────────────────────────────────────
# Main visualization
# ─────────────────────────────────────────────────────────────

def visualize(
    image:      torch.Tensor,
    gt_labels:  dict[str, int],
    s1_outputs: dict[str, torch.Tensor],
    s2_infer:   dict,
    sample_idx: int,
    output_path: Path,
    symbol_width: int = 28,
):
    """
    Layout:
    ┌─────────────────────────────────────────────────────────┐
    │  ROW 0: Ảnh gốc (full width, với slot boundaries)      │
    ├────────────────────────┬────────────────────────────────┤
    │  ROW 1: S1 probs       │  ROW 1: S2 rule info           │
    │  digit1 op1 digit2     │    rule string                 │
    │  op2    digit3  valid  │    per-slot cosine scores      │
    ├────────────────────────┴────────────────────────────────┤
    │  ROW 2: Comparison table (GT vs S1 pred vs S2 rule)    │
    └─────────────────────────────────────────────────────────┘
    """
    img_np = image.squeeze().numpy()   # [28, 140]
    H, W   = img_np.shape
    slots  = CONCEPT_KEYS_ORDERED      # 6 keys

    # ── S1 data ──────────────────────────────────────────────
    s1_probs = {}
    s1_preds = {}
    for key in slots:
        logits       = s1_outputs[key][0]
        probs        = F.softmax(logits, dim=0).numpy()
        s1_probs[key] = probs
        s1_preds[key] = int(probs.argmax())

    # ── S2 data ──────────────────────────────────────────────
    best_r        = s2_infer["best_rule_idx"][0].item()
    rule_str      = s2_infer["rule_strings"][0]
    s2_pred_slot  = {k: s2_infer["pred_slot"][k][0].item() for k in slots}
    slot_scores_r = s2_infer["slot_scores"][0, best_r].numpy()   # [6]
    slot_match_r  = s2_infer["slot_match"][0, best_r].numpy()    # [6]
    all_scores    = s2_infer["rule_scores"][0].numpy()            # [R]
    top3_rules    = all_scores.argsort()[::-1][:3]

    # ── Figure setup ─────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11), facecolor="#fafafa")
    fig.patch.set_facecolor("#fafafa")

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.4, 3.8, 1.5],
        hspace=0.35,
    )

    # ── ROW 0: Image ─────────────────────────────────────────
    ax_img = fig.add_subplot(outer[0])
    ax_img.imshow(img_np, cmap="gray", aspect="auto",
                  interpolation="nearest", vmin=0, vmax=1)

    slot_concept_keys = ["digit1", "op1", "digit2", "op2", "digit3"]
    for i, key in enumerate(slot_concept_keys):
        x0 = i * symbol_width
        x1 = x0 + symbol_width
        color = SLOT_COLORS[key]
        ax_img.axvline(x0, color=color, linewidth=1.5, alpha=0.7)
        ax_img.axvline(x1, color=color, linewidth=1.5, alpha=0.7)
        ax_img.add_patch(mpatches.FancyArrowPatch(
            (x0, -3), (x1, -3),
            arrowstyle="<->", color=color, linewidth=1,
            clip_on=False,
        ))
        ax_img.text(
            (x0 + x1) / 2, -7, SLOT_DISPLAY[key],
            ha="center", va="top", fontsize=7.5, color=color,
            fontweight="bold", clip_on=False,
        )

    ax_img.set_xlim(-0.5, W - 0.5)
    ax_img.set_ylim(H - 0.5, -0.5)
    ax_img.axis("off")

    # Expression string in title
    has_valid_label = "valid" in gt_labels
    _validity = f"  ({'valid' if gt_labels['valid'] == 1 else 'invalid'})" if has_valid_label else ""
    gt_expr = (
        f"{label_to_str('digit1', gt_labels['digit1'])} "
        f"{label_to_str('op1', gt_labels['op1'])} "
        f"{label_to_str('digit2', gt_labels['digit2'])} "
        f"{label_to_str('op2', gt_labels['op2'])} "
        f"{label_to_str('digit3', gt_labels['digit3'])}"
        f"{_validity}"
    )
    ax_img.set_title(
        f"Sample #{sample_idx}   GT: {gt_expr}",
        fontsize=12, pad=8, fontweight="bold", color="#2c2c2a",
        loc="left",
    )

    # ── ROW 1: Left — S1 probabilities ───────────────────────
    inner_mid = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[1],
        wspace=0.35, hspace=0.45,
    )
    left_gs = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=inner_mid[:, 0],
        wspace=0.4, hspace=0.6,
    )
    # 6 slots: 2 rows × 3 cols
    slot_axes_positions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
    ]
    for (r, c), key in zip(slot_axes_positions, slots):
        ax = fig.add_subplot(left_gs[r, c])
        probs   = s1_probs[key]
        gt_idx  = gt_labels[key]
        pred_idx = s1_preds[key]
        correct = (gt_idx == pred_idx)
        title   = f"{SLOT_DISPLAY[key]}  {'✓' if correct else '✗'}"
        _bar_probs(ax, probs, key, gt_idx, pred_idx, title)

    # S1 section label (fig-level để không conflict với axes)
    # (sẽ thêm vào sau khi axes đã được vẽ)

    # ── ROW 1: Right — S2 rule + scores ──────────────────────
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=inner_mid[:, 1],
        hspace=0.5,
    )

    # Top-right: match score bar
    ax_match = fig.add_subplot(right_gs[0])
    _match_bar(ax_match, slot_scores_r, slots, slot_match_r.astype(bool))

    # Bottom-right: rule info text
    ax_rule = fig.add_subplot(right_gs[1])
    ax_rule.axis("off")

    # Build rule display
    rule_parts = rule_str.split(" AND ")
    rule_color_parts = []
    for part in rule_parts:
        key = part.split("=")[0]
        c = SLOT_COLORS.get(key, "#2c2c2a")
        rule_color_parts.append((part, c))

    ax_rule.set_title("System 2 — best matched rule", fontsize=9.5,
                       fontweight="bold", color="#2c2c2a", pad=4, loc="left")

    # Rule string — hai dòng, mỗi dòng 3 điều kiện
    y_pos = 0.86
    parts_line1 = rule_parts[:3]
    parts_line2 = rule_parts[3:]
    for li, line_parts in enumerate([parts_line1, parts_line2]):
        x_cur = 0.0
        for pi, part in enumerate(line_parts):
            key_p = part.split("=")[0]
            col   = SLOT_COLORS.get(key_p, "#2c2c2a")
            ax_rule.text(x_cur, y_pos - li * 0.22, part,
                         fontsize=9.5, color=col, fontweight="bold",
                         transform=ax_rule.transAxes, va="top")
            x_cur += len(part) * 0.062 + 0.01
            if pi < len(line_parts) - 1:
                ax_rule.text(x_cur, y_pos - li * 0.22, "AND",
                             fontsize=8, color="#aaa", style="italic",
                             transform=ax_rule.transAxes, va="top")
                x_cur += 0.115

    # Rule score info
    rule_score = float(all_scores[best_r])
    ax_rule.text(0, y_pos - 0.50,
                 f"Rule #{best_r}   score = {rule_score:.4f}",
                 fontsize=8.5, color="#444441",
                 transform=ax_rule.transAxes, va="top")

    # Top-3 rules
    y_top3 = y_pos - 0.66
    ax_rule.text(0, y_top3, "Top-3 candidate rules:",
                 fontsize=8, color="#5f5e5a",
                 transform=ax_rule.transAxes, va="top",
                 fontweight="bold")
    for rank, ridx in enumerate(top3_rules):
        sc = float(all_scores[ridx])
        slot_probs_r = {
            k: F.softmax(model_s2_ref.rule_slot_logits[k], dim=-1)[ridx]
            for k in CONCEPT_KEYS_ORDERED
        }
        vals = []
        for k in CONCEPT_KEYS_ORDERED:
            vi = slot_probs_r[k].argmax().item()
            vals.append(label_to_str(k, vi))
        # Compact: "3 + 5 = 8"
        d1,o1,d2,o2,d3 = vals
        r_compact = f"  {d1} {o1} {d2} {o2} {d3}"
        marker = "▶" if ridx == best_r else " "
        ax_rule.text(
            0, y_top3 - 0.15 - rank * 0.16,
            f"{marker} #{ridx:3d}  score={sc:.3f}  {r_compact}",
            fontsize=8, color="#2c2c2a" if ridx == best_r else "#888780",
            transform=ax_rule.transAxes, va="top",
            fontfamily="monospace",
        )

    # ── ROW 2: Comparison table ───────────────────────────────
    ax_cmp = fig.add_subplot(outer[2])
    ax_cmp.axis("off")
    ax_cmp.set_title("Slot-by-slot comparison", fontsize=9.5,
                      fontweight="bold", color="#2c2c2a", pad=4, loc="left")

    col_labels = ["Slot", "Ground truth", "System 1 pred", "S1 correct?",
                  "System 2 rule", "S2 correct?", "S1→S2 match?"]
    rows = []
    for key in CONCEPT_KEYS_ORDERED:
        gt_v   = label_to_str(key, gt_labels[key])
        s1_v   = label_to_str(key, s1_preds[key])
        s2_v   = label_to_str(key, s2_pred_slot[key])
        s1_ok  = "✓" if s1_preds[key] == gt_labels[key] else "✗"
        s2_ok  = "✓" if s2_pred_slot[key] == gt_labels[key] else "✗"
        agree  = "✓" if s1_preds[key] == s2_pred_slot[key] else "✗"
        rows.append([SLOT_DISPLAY[key], gt_v, s1_v, s1_ok, s2_v, s2_ok, agree])

    tbl = ax_cmp.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, -0.05, 1, 1.05],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Style header
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#2c2c2a")
        cell.set_text_props(color="white", fontweight="bold")

    # Style rows
    for i, (key, row) in enumerate(zip(CONCEPT_KEYS_ORDERED, rows)):
        base_color = SLOT_COLORS[key]
        r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)
        bg = (r/255, g/255, b/255, 0.10)
        for j in range(len(col_labels)):
            cell = tbl[i + 1, j]
            cell.set_facecolor(bg)
            val = row[j]
            if val == "✓":
                cell.set_text_props(color="#1D9E75", fontweight="bold")
            elif val == "✗":
                cell.set_text_props(color="#E24B4A", fontweight="bold")
            elif j == 0:
                cell.set_text_props(color=base_color, fontweight="bold")

    # ── Save ─────────────────────────────────────────────────
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#fafafa", edgecolor="none")
    plt.close(fig)
    print(f"[INFO] Saved: {output_path}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

model_s2_ref: System2Rules | None = None  # global ref for top-3 decode


def main():
    global model_s2_ref

    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      type=str, required=True)
    p.add_argument("--system1_ckpt",  type=str, default=None)
    p.add_argument("--system2_ckpt",  type=str, default=None)
    p.add_argument("--split",         type=str, default="val",
                   choices=["train", "val", "test"])
    p.add_argument("--sample_idx",    type=int, default=None,
                   help="Chỉ định index cụ thể.")
    p.add_argument("--random",        action="store_true",
                   help="Chọn ngẫu nhiên.")
    p.add_argument("--pick",          type=str, default=None,
                   choices=["valid", "invalid"],
                   help="Lọc theo loại sample.")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--output_path",   type=str, default="outputs/pipeline_vis.png")
    args = p.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data — tự động tìm file .pt khớp với split name
    # Dataset v1: val.pt / train.pt / test.pt
    # Dataset v2: valid.pt / train.pt / test.pt  (val → valid)
    data_dir = Path(args.data_dir)
    from src.datasets.mnist_math_dataset import MNISTMathPTDataset

    _SPLIT_ALIASES = {
        "val":   ["val.pt", "valid.pt"],
        "valid": ["valid.pt", "val.pt"],
        "train": ["train.pt"],
        "test":  ["test.pt"],
    }
    _candidates = _SPLIT_ALIASES.get(args.split, [f"{args.split}.pt"])
    _pt_path = None
    for _fname in _candidates:
        _candidate = data_dir / _fname
        if _candidate.exists():
            _pt_path = _candidate
            break
    if _pt_path is None:
        raise FileNotFoundError(
            f"No dataset file found for split '{args.split}' in {data_dir}. "
            f"Tried: {_candidates}"
        )
    print(f"[INFO] Loading: {_pt_path}")
    _ds   = MNISTMathPTDataset(_pt_path)
    data  = _ds.data   # raw dict
    N     = _ds.length

    # Select sample index
    if args.sample_idx is not None:
        idx = args.sample_idx
    elif args.pick is not None:
        if "valid" in data and args.pick in ("valid", "invalid"):
            target_valid = 1 if args.pick == "valid" else 0
            pool = [i for i in range(N) if data["valid"][i].item() == target_valid]
        else:
            pool = list(range(N))   # v2: tất cả đều valid
        idx = random.choice(pool)
    elif args.random:
        idx = random.randint(0, N - 1)
    else:
        idx = 0

    print(f"[INFO] Sample #{idx} from {args.split} set")

    gt_labels = {k: data[k][idx].item() for k in CONCEPT_KEYS_ORDERED}
    if "valid" in data:
        gt_labels["valid"] = data["valid"][idx].item()
    print(
        f"[INFO] GT: "
        f"digit1={gt_labels['digit1']} "
        f"op1={label_to_str('op1', gt_labels['op1'])} "
        f"digit2={gt_labels['digit2']} "
        f"op2={label_to_str('op2', gt_labels['op2'])} "
        f"digit3={gt_labels['digit3']}"
    )

    # Load models
    s1_ckpt = Path(args.system1_ckpt) if args.system1_ckpt else None
    s2_ckpt = Path(args.system2_ckpt) if args.system2_ckpt else None
    model_s1 = load_system1(s1_ckpt, device)
    model_s2 = load_system2(s2_ckpt, device)
    model_s2_ref = model_s2

    image = data["images"][[idx]].to(device)

    with torch.no_grad():
        s1_out  = model_s1(image)
        cv      = soft_concept_vector(s1_out)
        s2_out  = model_s2.infer(cv)

    # Move to CPU for plotting
    s1_out_cpu = {k: v.cpu() for k, v in s1_out.items()}
    s2_out_cpu = {
        k: (v.cpu() if hasattr(v, "cpu") else v)
        for k, v in s2_out.items()
    }
    if isinstance(s2_out_cpu["pred_slot"], dict):
        s2_out_cpu["pred_slot"] = {
            k: v.cpu() for k, v in s2_out_cpu["pred_slot"].items()
        }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    visualize(
        image      = image.cpu(),
        gt_labels  = gt_labels,
        s1_outputs = s1_out_cpu,
        s2_infer   = s2_out_cpu,
        sample_idx = idx,
        output_path= output_path,
        symbol_width= 28,
    )


if __name__ == "__main__":
    main()