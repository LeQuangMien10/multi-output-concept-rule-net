"""
visualize_gt_contradiction.py - Show 2 REAL images with the IDENTICAL
ground-truth concept pattern but OPPOSITE true labels -- direct visual proof
for the finding from train_icrl_gt_ablation.py (rules 4 & 5: "Plaque AND
Brown(Hyperpigmentation) AND NOT Papule/Scale/Erythema" -> malignant for one
cluster, benign for another, both 100% pure).

Same visual language as visualize_inference.py's draw_card() (colors, image
panel, concept checklist) but no S1/rule inference involved -- this is pure
ground-truth data, the point is to show the CONCEPTS THEMSELVES (not any
model's prediction) are identical while the true label differs.

Usage:
    python -m src.scripts.fitzpatrick.visualize_gt_contradiction \
        --data_dir data/fitzpatrick17k_crl_matched \
        --img_dir data/fitzpatrick17k/data/finalfitz17k \
        --output_dir outputs/fitzpatrick_icrl_gt_ablation/contradiction_examples
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

BG = "#f7f7f5"
INK = "#1a1a18"
INK_SOFT = "#57564e"
GRID = "#e2e0d8"
LABEL_COLOR = {"benign": "#1a52a0", "malignant": "#c0392b"}

# The 5 concepts that define this specific pattern (from the GT ablation rule pair).
PATTERN_PRESENT = ["Plaque", "Brown(Hyperpigmentation)"]
PATTERN_ABSENT = ["Papule", "Scale", "Erythema"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/fitzpatrick17k_crl_matched")
    p.add_argument("--img_dir", type=str, default="data/fitzpatrick17k/data/finalfitz17k")
    p.add_argument("--output_dir", type=str, default="outputs/fitzpatrick_icrl_gt_ablation/contradiction_examples")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_display_image(img_dir: Path, filename: str, image_size: int) -> np.ndarray:
    display_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
    ])
    img = Image.open(img_dir / filename).convert("RGB")
    return np.asarray(display_transform(img))


def draw_card(row, concept_names, img_dir, image_size, save_path):
    label = row["label"]
    border_color = LABEL_COLOR[label]
    display_img = load_display_image(img_dir, row["filename"], image_size)

    fig = plt.figure(figsize=(11, 6.5), facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.4,
                           left=0.05, right=0.97, top=0.86, bottom=0.06)

    fig.suptitle(f"Fitzpatrick17k — cùng ground-truth concept, nhãn thật khác nhau  ·  {row['filename']}",
                 fontsize=12, fontweight="bold", color=INK, x=0.05, y=0.96, ha="left")

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.set_facecolor(BG)
    ax_img.imshow(display_img)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_edgecolor(border_color); spine.set_linewidth(4)
    ax_img.set_title(f"nhãn thật (ground-truth): {label}", fontsize=11, color=border_color, fontweight="bold")

    ax_c = fig.add_subplot(gs[0, 1])
    ax_c.set_facecolor(BG)
    y = np.arange(len(concept_names))
    values = [float(row[c]) for c in concept_names]
    colors = []
    for c in concept_names:
        if c in PATTERN_PRESENT:
            colors.append("#0d7a5f")   # highlight: part of the shared pattern, present
        elif c in PATTERN_ABSENT:
            colors.append("#9a5f00")   # highlight: part of the shared pattern, absent
        else:
            colors.append(GRID if float(row[c]) == 0 else INK_SOFT)
    ax_c.barh(y, values, color=colors, height=0.6, zorder=3)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(concept_names, fontsize=6.2)
    ax_c.set_xlim(0, 1.05)
    ax_c.set_ylim(-0.6, len(concept_names) - 0.4)
    ax_c.invert_yaxis()
    ax_c.set_title("48 concept ground-truth thật (xanh đậm/nâu = 5 concept định nghĩa\npattern chung; xám = concept khác, cũng giống hệt ảnh kia)",
                    fontsize=8, color=INK_SOFT)
    for spine in ["top", "right"]:
        ax_c.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_c.spines[spine].set_color(GRID)

    fig.savefig(save_path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"[INFO] Saved {save_path}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / f"{args.split}.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        concept_names = [c for c in reader.fieldnames if c not in
                         ("md5hash", "filename", "label", "label_idx", "concept_mask")]
        rows = list(reader)

    # Require an EXACT match across all 48 concepts (not just the 5
    # pattern-defining ones) -- the cleanest possible demonstration that two
    # images can be identical on every recorded concept yet have opposite
    # ground-truth labels.
    present_set = set(PATTERN_PRESENT)
    matches = [
        r for r in rows
        if set(c for c in concept_names if r[c] == "1") == present_set
    ]
    by_label = {"benign": [], "malignant": []}
    for r in matches:
        by_label[r["label"]].append(r)

    print(f"[INFO] {len(matches)} images match pattern "
          f"({'+'.join(PATTERN_PRESENT)}, NOT {'/'.join(PATTERN_ABSENT)}): "
          f"benign={len(by_label['benign'])}  malignant={len(by_label['malignant'])}")

    rng = random.Random(args.seed)
    for label in ("benign", "malignant"):
        if not by_label[label]:
            print(f"[WARN] No {label} example found.")
            continue
        row = rng.choice(by_label[label])
        draw_card(row, concept_names, img_dir, args.image_size,
                  out_dir / f"contradiction_{label}_{row['md5hash']}.png")


if __name__ == "__main__":
    main()
