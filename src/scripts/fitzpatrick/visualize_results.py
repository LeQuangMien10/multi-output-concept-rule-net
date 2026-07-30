"""
visualize_results.py - Danh gia chi tiet checkpoint S1 Fitzpatrick (Buoc 5 chuan bi)
========================================================================================

Chay INFERENCE (khong train) tren checkpoint da co (vd. best_model.pt tai ve tu Kaggle)
de tra loi cau hoi khong the thay chi qua concept_macro_f1 tong hop:
  - Trong 35 concept, bao nhieu concept THAT SU hoc duoc, bao nhieu con ~0?
  - Diem F1 thap co tuong quan voi so mau (support) it khong?
  - Training curve co dau hieu overfit khong (train/val loss tach nhau tu epoch nao)?
  - Confusion matrix 3 lop the nao (co bi lech ve non-neoplastic khong)?

Usage:
    python -m src.scripts.fitzpatrick.visualize_results \\
        --checkpoint outputs/fitzpatrick_system1/best_model.pt \\
        --metrics_json outputs/fitzpatrick_system1/metrics.json \\
        --data_dir data/fitzpatrick17k_prepared \\
        --img_dir data/fitzpatrick17k/data/finalfitz17k \\
        --split val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.fitzpatrick.fitzpatrick_dataset import FitzpatrickDataset, build_transforms
from src.models.fitzpatrick.system1 import FitzpatrickSystem1
from src.utils.fitzpatrick_concepts import CONCEPT_NAMES, LABEL_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate + visualize a trained Fitzpatrick S1 checkpoint.")
    p.add_argument("--checkpoint", type=str, default="outputs/fitzpatrick_system1/best_model.pt")
    p.add_argument("--metrics_json", type=str, default="outputs/fitzpatrick_system1/metrics.json")
    p.add_argument("--data_dir", type=str, default="data/fitzpatrick17k_prepared")
    p.add_argument("--img_dir", type=str, default="data/fitzpatrick17k/data/finalfitz17k")
    p.add_argument("--output_path", type=str, default="outputs/fitzpatrick_audit/s1_dashboard.png")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    all_concept_logits, all_concept_targets, all_concept_mask = [], [], []
    all_label_logits, all_label_targets = [], []
    for images, labels in tqdm(loader, desc="Infer"):
        images = images.to(device)
        out = model(images)
        all_concept_logits.append(out["concepts"].cpu())
        all_concept_targets.append(labels["concepts"])
        all_concept_mask.append(labels["concept_mask"])
        all_label_logits.append(out["label"].cpu())
        all_label_targets.append(labels["label"])
    return {
        "concept_logits": torch.cat(all_concept_logits),
        "concept_targets": torch.cat(all_concept_targets),
        "concept_mask": torch.cat(all_concept_mask),
        "label_logits": torch.cat(all_label_logits),
        "label_targets": torch.cat(all_label_targets),
    }


def per_concept_stats(concept_logits, concept_targets, concept_mask):
    keep = concept_mask.bool()
    logits, targets = concept_logits[keep], concept_targets[keep]
    preds = (torch.sigmoid(logits) > 0.5).float()

    stats = []
    for i, name in enumerate(CONCEPT_NAMES):
        t, p = targets[:, i], preds[:, i]
        support = int(t.sum().item())
        tp = int(((p == 1) & (t == 1)).sum().item())
        fp = int(((p == 1) & (t == 0)).sum().item())
        fn = int(((p == 0) & (t == 1)).sum().item())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        stats.append({"concept": name, "support": support, "precision": precision,
                       "recall": recall, "f1": f1, "n_eval": int(keep.sum().item())})
    return stats


def confusion_matrix_3class(label_logits, label_targets):
    preds = label_logits.argmax(dim=1)
    n = len(LABEL_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(label_targets.tolist(), preds.tolist()):
        cm[t, p] += 1
    return cm


def plot_dashboard(history, concept_stats, cm, split_name, save_path):
    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(f"Fitzpatrick S1 — Danh gia checkpoint tren split '{split_name}'", fontsize=14, y=0.995)
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.3, top=0.93, bottom=0.06, left=0.09, right=0.97)

    # (a) training curves: train_loss vs val_loss -- overfit inflection point
    ax = fig.add_subplot(gs[0, 0])
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train_loss"] for h in history], label="train_loss", color="#2980b9")
    ax.plot(epochs, [h["val_loss"] for h in history], label="val_loss", color="#c0392b")
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    ax.axvline(best_epoch, color="#7f8c8d", linestyle="--", alpha=0.7,
               label=f"val_loss thap nhat @ epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Val loss (khoang cach = overfit)")
    ax.legend(fontsize=8)

    # (b) val_concept_macro_f1 + val_label_acc over epoch
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, [h["val_concept_macro_f1"] for h in history], label="val_concept_macro_f1", color="#27ae60")
    ax2 = ax.twinx()
    ax2.plot(epochs, [h["val_label_acc"] for h in history], label="val_label_acc", color="#e67e22")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("concept_macro_f1", color="#27ae60")
    ax2.set_ylabel("label_acc", color="#e67e22")
    ax.set_title("Concept F1 vs Label accuracy qua cac epoch")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")

    # (c) per-concept F1 sorted, mau theo support
    ax = fig.add_subplot(gs[1, :])
    sorted_stats = sorted(concept_stats, key=lambda s: s["f1"], reverse=True)
    names = [s["concept"] for s in sorted_stats]
    f1s = [s["f1"] for s in sorted_stats]
    supports = [s["support"] for s in sorted_stats]
    colors = ["#27ae60" if s >= 20 else "#c0392b" for s in supports]
    ax.bar(range(len(names)), f1s, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("F1")
    n_learned = sum(1 for f in f1s if f > 0.3)
    ax.set_title(f"Per-concept F1 tren {sorted_stats[0]['n_eval']} anh co concept-label "
                 f"(xanh = support>=20 mau, do = <20 mau; {n_learned}/{len(names)} concept co F1>0.3)")

    # (d) F1 vs support scatter
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(supports, f1s, alpha=0.7, color="#2980b9")
    for s in sorted_stats:
        if s["f1"] > 0.5 or s["support"] > 150:
            ax.annotate(s["concept"], (s["support"], s["f1"]), fontsize=6, alpha=0.8)
    ax.set_xlabel("Support (so anh co concept nay trong split)")
    ax.set_ylabel("F1")
    ax.set_title("F1 co tuong quan voi support khong?")

    # (e) confusion matrix
    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (nhan 3 lop)")
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    print(f"[INFO] Loaded checkpoint: epoch={ckpt['epoch']} monitor={ckpt['monitor']} "
          f"best_val_metric={ckpt['best_val_metric']:.4f}")

    model = FitzpatrickSystem1(
        backbone_name=ckpt_args["backbone"],
        pretrained=False,   # trong so se duoc load tu checkpoint ngay ben duoi
        num_concepts=ckpt_args["num_concepts"],
        num_labels=ckpt_args["num_labels"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = FitzpatrickDataset(
        Path(args.data_dir) / f"{args.split}.csv", args.img_dir,
        build_transforms(args.split, ckpt_args.get("image_size", 224)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"[INFO] {len(dataset)} images in '{args.split}' split.")

    out = run_inference(model, loader, device)
    concept_stats = per_concept_stats(out["concept_logits"], out["concept_targets"], out["concept_mask"])
    cm = confusion_matrix_3class(out["label_logits"], out["label_targets"])

    with open(args.metrics_json, encoding="utf-8") as f:
        metrics = json.load(f)
    history = metrics["history"]

    n_zero_f1 = sum(1 for s in concept_stats if s["f1"] == 0.0)
    n_good = sum(1 for s in concept_stats if s["f1"] > 0.5)
    print(f"\n[SUMMARY] {n_zero_f1}/{len(concept_stats)} concept co F1=0 (chua hoc duoc gi)")
    print(f"[SUMMARY] {n_good}/{len(concept_stats)} concept co F1>0.5")
    print("\nTop 10 concept theo F1:")
    for s in sorted(concept_stats, key=lambda s: s["f1"], reverse=True)[:10]:
        print(f"  {s['concept']:28s} F1={s['f1']:.3f}  support={s['support']}")
    print("\nBottom 10 concept theo F1:")
    for s in sorted(concept_stats, key=lambda s: s["f1"])[:10]:
        print(f"  {s['concept']:28s} F1={s['f1']:.3f}  support={s['support']}")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    plot_dashboard(history, concept_stats, cm, args.split, args.output_path)

    result = {
        "split": args.split,
        "checkpoint_epoch": ckpt["epoch"],
        "n_zero_f1_concepts": n_zero_f1,
        "n_good_f1_concepts_gt0.5": n_good,
        "concept_stats": concept_stats,
        "confusion_matrix": cm.tolist(),
        "label_names": LABEL_NAMES,
    }
    result_path = Path(args.output_path).with_suffix(".json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n[DONE] Saved {args.output_path} and {result_path}")


if __name__ == "__main__":
    main()
