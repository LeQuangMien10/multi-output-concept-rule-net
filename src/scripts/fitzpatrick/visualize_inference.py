"""
visualize_inference.py - Trace suy luan tung anh: S1 -> concept -> rule -> S2
================================================================================

Giong tinh than multiconcept/visualize_inference.py NHUNG moi anh xuat ra 1
FILE RIENG (khong gop nhieu anh vao 1 dashboard) -- de moi file dung y 1 input
cu the, de dua vao bao cao / xem tung ca rieng le.

Ve 4 phan cho 1 anh: anh goc + nhan that, 35 concept S1 du doan (mau xanh/do
= dung/sai so voi ground-truth THAT cho anh co concept_mask=1; mau xam = anh
khong co nhan concept that de doi chieu -- van hien thi xac suat du doan),
phan phoi nhan 3 lop S1 tu du doan (s1_label_pred -- CUNG la 1 "concept" duoc
noi vao vector dung de cluster o Stage 2), va rule ma S2 da match + ket luan
cuoi cung (co doi so voi S1 khong).

Chon anh bang 1 trong 2 cach:
  --indices 12 340          chon dung cac dong theo index trong split
  --filter TP/TN/FP/FN/all  lay ngau nhien --n_examples anh theo loai ket qua
      TP: S1 dung, S2 dung       (S2 giu nguyen dung)
      TN: S1 sai,  S2 dung       (S2 sua loi S1)   <- quan trong nhat
      FP: S1 dung, S2 sai        (S2 lam hong)     <- dang lo nhat
      FN: S1 sai,  S2 sai        (S2 khong cuu duoc)
      all: khong loc, lay ngau nhien

Usage (Kaggle mac dinh, override neu chay local):
    python -m src.scripts.fitzpatrick.visualize_inference \\
        --data_dir /kaggle/input/datasets/lquangmin/fitzpatrick17k-prepared \\
        --img_dir /kaggle/input/datasets/lquangmin/fitzpatrick17k/data/finalfitz17k \\
        --system1_ckpt /kaggle/working/outputs/fitzpatrick_system1/best_model.pt \\
        --icrl_dir /kaggle/working/outputs/fitzpatrick_icrl \\
        --output_dir /kaggle/working/outputs/fitzpatrick_icrl/inference_cards \\
        --split val --filter TN --n_examples 3
"""
from __future__ import annotations

import argparse
import csv
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
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.fitzpatrick.fitzpatrick_dataset import FitzpatrickDataset, build_transforms
from src.models.fitzpatrick.system1 import FitzpatrickSystem1, soft_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.utils.fitzpatrick_concepts import LABEL_NAMES as DEFAULT_LABEL_NAMES


BG = "#f7f7f5"
INK = "#1a1a18"
INK_SOFT = "#57564e"
GRID = "#e2e0d8"
NO_GT = "#9a9a94"
LABEL_COLOR = {"benign": "#1a52a0", "malignant": "#c0392b", "non-neoplastic": "#0d7a5f"}
LABEL_COLOR_FALLBACK = ["#1a52a0", "#c0392b", "#0d7a5f", "#8e5ea8", "#c77c1f"]
STATUS_GOOD = "#0ca30c"
STATUS_CRIT = "#d03b3b"


def label_color(name: str, idx: int) -> str:
    return LABEL_COLOR.get(name, LABEL_COLOR_FALLBACK[idx % len(LABEL_COLOR_FALLBACK)])

# Ten hien thi rut gon cho cac concept ten dai -- tranh y-tick label de len cot anh ben canh.
CONCEPT_DISPLAY_NAME = {
    "Brown(Hyperpigmentation)": "Brown(Hyperpig.)",
    "White(Hypopigmentation)": "White(Hypopig.)",
    "Warty/Papillomatous": "Warty/Papillom.",
    "Exophytic/Fungating": "Exophytic/Fung.",
}


def display_name(concept: str) -> str:
    return CONCEPT_DISPLAY_NAME.get(concept, concept)


def parse_args():
    p = argparse.ArgumentParser(description="Ve trace suy luan S1->S2 cho tung anh cu the (Fitzpatrick17k).")
    p.add_argument("--data_dir", type=str, default="/kaggle/input/datasets/lquangmin/fitzpatrick17k-prepared")
    p.add_argument("--img_dir", type=str, default="/kaggle/input/datasets/lquangmin/fitzpatrick17k/data/finalfitz17k")
    p.add_argument("--system1_ckpt", type=str, default="/kaggle/working/outputs/fitzpatrick_system1/best_model.pt")
    p.add_argument("--icrl_dir", type=str, default="/kaggle/working/outputs/fitzpatrick_icrl")
    p.add_argument("--output_dir", type=str, default=None,
                    help="Mac dinh: <icrl_dir>/inference_cards")
    p.add_argument("--label_names", type=str, default=None,
                    help="Comma-separated, override cho scope khac 3-lop mac dinh "
                         "(vd. --label_names benign,malignant cho CRL-matched).")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--indices", type=int, nargs="+", default=None,
                    help="Chon dung cac dong theo index trong split. Neu set, bo qua --filter/--n_examples.")
    p.add_argument("--filter", type=str, default="all", choices=["all", "TP", "TN", "FP", "FN"],
                    help="TP=S1 dung+S2 dung, TN=S1 sai+S2 dung (S2 sua), "
                         "FP=S1 dung+S2 sai (S2 lam hong), FN=ca hai sai.")
    p.add_argument("--n_examples", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def load_system1(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    model = FitzpatrickSystem1(
        backbone_name=saved_args.get("backbone", "resnet50"),
        pretrained=False,
        num_concepts=saved_args["num_concepts"],
        num_labels=saved_args["num_labels"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    return model, saved_args.get("image_size", 224)


def categorize(s1_correct: bool, s2_correct: bool) -> str:
    if s1_correct and s2_correct:
        return "TP"
    if not s1_correct and s2_correct:
        return "TN"
    if s1_correct and not s2_correct:
        return "FP"
    return "FN"


@torch.no_grad()
def scan_categories(loader, system1, memory, head, centroids, device):
    """Quet ca split theo BATCH (nhanh) chi de lay category tung anh -- khong
    tinh concept/label chi tiet (viec do chi can cho vai anh cuoi cung)."""
    cats = []
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        full_cv = soft_concept_vector(out)
        rule_ids, _ = memory.match(full_cv)
        rule_cvs = centroids[rule_ids]
        s2_pred = head(rule_cvs).argmax(dim=1)
        s1_pred = out["label"].argmax(dim=1)
        true_b = labels["label"].to(device)
        for i in range(len(true_b)):
            cats.append(categorize(bool(s1_pred[i] == true_b[i]), bool(s2_pred[i] == true_b[i])))
    return cats


def load_display_image(img_dir: Path, filename: str, image_size: int) -> np.ndarray:
    """Anh de HIEN THI (RGB, da resize+crop giong luc infer nhung KHONG
    normalize) -- dung ImageNet-normalized tensor de imshow se ra mau sai."""
    display_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
    ])
    img = Image.open(img_dir / filename).convert("RGB")
    return np.asarray(display_transform(img))


@torch.no_grad()
def run_one(row, img_dir, image_size, system1, memory, head, centroids, device,
            concept_names, label_names):
    num_concepts = len(concept_names)
    model_transform = build_transforms("val", image_size)
    pil_img = Image.open(img_dir / row["filename"]).convert("RGB")
    image = model_transform(pil_img).unsqueeze(0).to(device)

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
    s2_pred = int(s2_logits[0].argmax())

    # Ly do match THAT SU cho ANH NAY -- khac voi rule["present_concepts"]
    # (threshold 0.5 tren centroid TRUNG BINH cua ca cum, khong phai tren anh
    # nay). "present_concepts" co the liet ke 1 concept ma chinh anh nay gan
    # nhu khong co (centroid cao vi cac anh KHAC trong cum co, con anh nay
    # thi khong), va bo sot concept anh nay THAT SU co manh nhung centroid
    # (bi pha loang boi cac anh khac) khong vuot 0.5. Phan ra tung chieu
    # dong gop bao nhieu vao cosine similarity moi la ly do THAT SU.
    dim_names = list(concept_names) + [f"label:{n}" for n in label_names]
    s, e = memory.cluster_dims if memory.cluster_dims else (0, full_cv.shape[1])
    cv_n = F.normalize(full_cv[:, s:e], dim=1)[0]
    centroid_n = F.normalize(centroids[rule_id, s:e].unsqueeze(0), dim=1)[0]
    contrib = cv_n * centroid_n
    order = torch.argsort(contrib, descending=True)
    top_contributors = [
        {"dim": dim_names[s + i], "contrib": float(contrib[i]),
         "pct_of_sim": float(contrib[i] / max(match_sim, 1e-9))}
        for i in order[:5].tolist()
    ]

    true_label = int(row["label_idx"])
    has_concept_gt = row["concept_mask"] == "1"
    concept_gt = [float(row[c]) for c in concept_names] if has_concept_gt else [None] * num_concepts

    return {
        "filename": row["filename"],
        "display_image": load_display_image(img_dir, row["filename"], image_size),
        "concept_probs": concept_probs.tolist(),
        "concept_pred": concept_pred.tolist(),
        "concept_gt": concept_gt,
        "has_concept_gt": has_concept_gt,
        "label_probs": label_probs.tolist(),
        "s1_pred": s1_pred,
        "s1_correct": s1_pred == true_label,
        "rule_id": rule_id,
        "match_sim": match_sim,
        "top_contributors": top_contributors,
        "s2_pred": s2_pred,
        "s2_correct": s2_pred == true_label,
        "true_label": true_label,
        "changed": s1_pred != s2_pred,
    }


def draw_card(rec, rules_by_id, save_path, concept_names, label_names):
    num_concepts = len(concept_names)
    num_labels = len(label_names)
    cat = categorize(rec["s1_correct"], rec["s2_correct"])
    border_color = STATUS_GOOD if rec["s2_correct"] else STATUS_CRIT

    fig = plt.figure(figsize=(15, 7.2), facecolor=BG)
    gs = fig.add_gridspec(1, 4, width_ratios=[0.85, 1.65, 0.85, 1.25], wspace=0.55,
                           left=0.035, right=0.98, top=0.86, bottom=0.06)

    true_name = label_names[rec["true_label"]]
    s1_name = label_names[rec["s1_pred"]]
    s2_name = label_names[rec["s2_pred"]]
    fig.suptitle(f"Fitzpatrick17k — Trace suy luận 1 ảnh  ·  {rec['filename']}  ·  [{cat}]",
                 fontsize=13, fontweight="bold", color=INK, x=0.035, y=0.97, ha="left")

    # -- anh + ground-truth --
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.set_facecolor(BG)
    ax_img.imshow(rec["display_image"])
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_edgecolor(border_color); spine.set_linewidth(3)
    gt_note = "" if rec["has_concept_gt"] else "\n(ảnh không có nhãn concept thật)"
    ax_img.set_title(f"nhãn thật: {true_name}{gt_note}", fontsize=9, color=INK)

    # -- 35 concept, sap theo xac suat giam dan --
    ax_c = fig.add_subplot(gs[0, 1])
    ax_c.set_facecolor(BG)
    order = sorted(range(num_concepts), key=lambda i: -rec["concept_probs"][i])
    probs = [rec["concept_probs"][i] for i in order]
    pred = [rec["concept_pred"][i] for i in order]
    gt = [rec["concept_gt"][i] for i in order]
    names = [display_name(concept_names[i]) for i in order]
    y = np.arange(num_concepts)
    colors = []
    for i in range(num_concepts):
        if gt[i] is None:
            colors.append(NO_GT)
        else:
            colors.append(STATUS_GOOD if pred[i] == gt[i] else STATUS_CRIT)
    ax_c.barh(y, probs, color=colors, height=0.65, zorder=3)
    ax_c.axvline(0.5, color=INK_SOFT, linewidth=0.8, linestyle=":", zorder=2)
    for i in range(num_concepts):
        marker = "?" if gt[i] is None else ("●" if gt[i] == 1 else "○")
        ax_c.text(1.03, i, marker, fontsize=7.5, color=INK_SOFT, va="center")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(names, fontsize=6.3)
    ax_c.set_xlim(0, 1.12)
    ax_c.set_ylim(-0.6, num_concepts - 0.4)
    ax_c.invert_yaxis()
    ax_c.tick_params(colors=INK_SOFT, labelsize=6.5)
    ax_c.set_title(f"S1: {num_concepts} concept (xanh/đỏ = đúng/sai so GT thật,\nxám = ảnh không có GT)  ● GT có  ○ GT không",
                    fontsize=7.5, color=INK_SOFT)
    for spine in ["top", "right"]:
        ax_c.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_c.spines[spine].set_color(GRID)

    # -- S1 label distribution (= s1_label_pred, cung la 1 concept dung de cluster) --
    ax_l = fig.add_subplot(gs[0, 2])
    ax_l.set_facecolor(BG)
    yl = np.arange(num_labels)
    colors_l = [label_color(n, i) for i, n in enumerate(label_names)]
    ax_l.barh(yl, rec["label_probs"], color=colors_l, height=0.6, zorder=3)
    ax_l.set_yticks(yl); ax_l.set_yticklabels(label_names, fontsize=8)
    ax_l.set_xlim(0, 1.02)
    true_pos = rec["true_label"]
    ax_l.get_yticklabels()[true_pos].set_fontweight("bold")
    ax_l.barh([true_pos], [1.0], color="none", edgecolor=INK, linewidth=1.5, height=0.6, zorder=4)
    for spine in ["top", "right"]:
        ax_l.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_l.spines[spine].set_color(GRID)
    ax_l.tick_params(colors=INK_SOFT, labelsize=7)
    ax_l.set_title("S1: s1_label_pred\n(viền đen = nhãn thật)", fontsize=8, color=INK_SOFT)

    # -- rule + verdict --
    ax_t = fig.add_subplot(gs[0, 3])
    ax_t.set_facecolor(BG)
    ax_t.set_xticks([]); ax_t.set_yticks([])
    for spine in ax_t.spines.values():
        spine.set_visible(False)

    # "present_concepts" la threshold 0.5 tren centroid TRUNG BINH ca cum --
    # co the KHONG phai ly do anh NAY match (vd. concept present o centroid
    # vi cac anh KHAC trong cum co, con anh nay gan nhu khong; hoac nguoc
    # lai, concept anh nay THAT SU co manh nhung centroid bi pha loang
    # khong vuot 0.5 nen khong duoc liet ke). top_contributors la phan ra
    # thuc te tung chieu dong gop bao nhieu vao cosine similarity CHO CHINH
    # ANH NAY -- moi la ly do that su khien no match vao rule nay.
    contrib_lines = "\n".join(
        f"  {i+1}. {c['dim']:<20s} {c['pct_of_sim']*100:3.0f}%"
        for i, c in enumerate(rec["top_contributors"])
    )
    ri = rules_by_id.get(rec["rule_id"])
    if ri is not None:
        present = ", ".join(ri.get("present_concepts", [])) or "(rỗng)"
        rule_txt = (f"Rule #{rec['rule_id']}  (n={ri['n']}, conf={ri['confidence']:.3f})\n"
                    f"concept rule (TB cả cụm): {{{present}}}\n"
                    f"nhãn rule: {ri['label_name']}\n"
                    f"match sim: {rec['match_sim']:.3f}\n"
                    f"Vì sao match ảnh NÀY (top đóng góp):\n{contrib_lines}")
    else:
        rule_txt = (f"Rule #{rec['rule_id']}  (không có metadata)\nmatch sim: {rec['match_sim']:.3f}\n"
                    f"Vì sao match ảnh NÀY (top đóng góp):\n{contrib_lines}")

    s1_mark = "✓" if rec["s1_correct"] else "✗"
    s2_mark = "✓" if rec["s2_correct"] else "✗"
    s1_color = STATUS_GOOD if rec["s1_correct"] else STATUS_CRIT
    s2_color = STATUS_GOOD if rec["s2_correct"] else STATUS_CRIT
    change_txt = "→ KHÔNG đổi" if not rec["changed"] else "→ CÓ đổi"

    ax_t.text(0.0, 0.97, rule_txt, fontsize=8.0, color=INK, va="top", ha="left",
              transform=ax_t.transAxes, family="monospace", wrap=True)
    ax_t.text(0.0, 0.33, f"S1 tự đoán: {s1_name} {s1_mark}", fontsize=9.5, color=s1_color,
              va="top", ha="left", transform=ax_t.transAxes, fontweight="bold")
    ax_t.text(0.0, 0.28, f"S2 (qua rule): {s2_name} {s2_mark}", fontsize=9.5, color=s2_color,
              va="top", ha="left", transform=ax_t.transAxes, fontweight="bold")
    ax_t.text(0.0, 0.16, change_txt, fontsize=8.6, color=INK_SOFT, va="top", ha="left",
              transform=ax_t.transAxes)
    ax_t.text(0.0, 0.02, f"[{cat}]", fontsize=10, color=border_color, va="top", ha="left",
              transform=ax_t.transAxes, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=border_color))

    fig.savefig(save_path, dpi=150, facecolor=BG)
    plt.close(fig)


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    rng = random.Random(args.seed)

    label_names = args.label_names.split(",") if args.label_names else list(DEFAULT_LABEL_NAMES)

    system1, image_size = load_system1(Path(args.system1_ckpt), device)
    icrl_dir = Path(args.icrl_dir)
    memory = ICRLRuleMemory.load(icrl_dir / "icrl_rule_memory.pt", device=str(device))

    data_dir = Path(args.data_dir)
    csv_path = data_dir / f"{args.split}.csv"
    dataset = FitzpatrickDataset(csv_path, args.img_dir, build_transforms("val", image_size))
    concept_names = dataset.concept_names
    full_cv_dim = len(concept_names) + len(label_names)
    print(f"[INFO] {len(concept_names)} concepts, {len(label_names)} labels ({label_names})", flush=True)

    head = nn.Linear(full_cv_dim, len(label_names)).to(device)
    head.load_state_dict(torch.load(icrl_dir / "prediction_head.pt", map_location=device, weights_only=False))
    head.eval()
    centroids = memory.get_centroids().to(device)

    with open(icrl_dir / "icrl_rules.json", encoding="utf-8") as f:
        rules_by_id = {r["rule_id"]: r for r in json.load(f)}

    rows = dataset.rows
    print(f"[INFO] {args.split} set: {len(rows)} samples")

    if args.indices:
        chosen = args.indices
    else:
        print(f"[INFO] Scanning {args.split} in batches to filter by filter='{args.filter}'...")
        if args.filter == "all":
            pool = list(range(len(rows)))
        else:
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            cats = scan_categories(loader, system1, memory, head, centroids, device)
            pool = [i for i, c in enumerate(cats) if c == args.filter]
        if not pool:
            raise RuntimeError(f"Không tìm thấy ảnh nào khớp filter='{args.filter}' trong {args.split}.")
        # Uu tien anh CO nhan concept that (concept_mask=1) -- de con thay
        # mau xanh/do that su (dung/sai) thay vi toan xam "?" khong doi chieu
        # duoc. Chi lay anh khong co nhan neu khong du so luong yeu cau.
        labeled = [i for i in pool if rows[i]["concept_mask"] == "1"]
        unlabeled = [i for i in pool if rows[i]["concept_mask"] != "1"]
        rng.shuffle(labeled)
        rng.shuffle(unlabeled)
        print(f"[INFO] {len(labeled)}/{len(pool)} images in pool have real concept GT -- prioritized first.")
        chosen = (labeled + unlabeled)[:args.n_examples]

    print(f"[INFO] Drawing {len(chosen)} images: {chosen}")

    output_dir = Path(args.output_dir) if args.output_dir else icrl_dir / "inference_cards"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in chosen:
        rec = run_one(rows[idx], Path(args.img_dir), image_size, system1, memory, head, centroids, device,
                      concept_names, label_names)
        cat = categorize(rec["s1_correct"], rec["s2_correct"])
        out_path = output_dir / f"{args.split}_{idx}_{cat}.png"
        draw_card(rec, rules_by_id, out_path, concept_names, label_names)
        print(f"[DONE] {idx} [{cat}] -> {out_path}")


if __name__ == "__main__":
    main()
