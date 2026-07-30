"""
train_system1_baseline.py - Train S1 cho Fitzpatrick17k
==========================================================

Cung 2-dau design nhu MultiConcept (multiconcept/train_system1_baseline.py):
  - concept_head : 35 concept nhi phan (multi-label, sigmoid), chi supervised
                    tren concept_mask=1 (~19-20% train, xem prepare_dataset.py).
  - label_head   : nhan 3 lop (benign/malignant/non-neoplastic), supervised
                    FULL (100% anh luon co nhan).

Khac MultiConcept: backbone la ResNet-50 PRETRAINED ImageNet (khong train tu
dau -- 11.6k anh train la qua nho de hoc dac trung thi giac tu dau, xem phan
tich cong nghe S1 da thong nhat truoc khi trien khai). Dung differential LR:
backbone thap hon head, giong tinh than label_lr_scale da dung o MultiConcept
nhung ap dung nguoc lai (o day backbone can LR THAP de khong pha vo pretrained
weight, con head moi khoi tao can LR binh thuong).

MUC TIEU DOI CHIEU: ~62.4% accuracy 3 lop (baseline paper goc Groh et al. 2021,
VGG-16/ResNet-18 pretrained). Dat gan muc do la dau hieu pipeline dung, KHONG
ky vong cao hon nhieu o lan chay dau.

Usage (Kaggle mac dinh, override --data_dir/--img_dir neu chay local):
    python -m src.scripts.fitzpatrick.train_system1_baseline \\
        --data_dir /kaggle/input/fitzpatrick17k-prepared \\
        --img_dir /kaggle/input/fitzpatrick17k/data/finalfitz17k \\
        --output_dir /kaggle/working/outputs/fitzpatrick_system1 \\
        --epochs 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.fitzpatrick.fitzpatrick_dataset import FitzpatrickDataset, build_transforms
from src.models.fitzpatrick.system1 import FitzpatrickSystem1
from src.utils.fitzpatrick_concepts import NUM_CONCEPTS, NUM_LABELS
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train FitzpatrickSystem1 (S1) baseline.")

    p.add_argument("--data_dir", type=str, default="/kaggle/input/fitzpatrick17k-prepared",
                    help="Thu muc chua train.csv/val.csv/test.csv/meta.json (output cua prepare_dataset.py).")
    p.add_argument("--img_dir", type=str, default="/kaggle/input/fitzpatrick17k/data/finalfitz17k",
                    help="Thu muc chua *.jpg goc (ten file = md5hash.jpg).")
    p.add_argument("--output_dir", type=str, default="/kaggle/working/outputs/fitzpatrick_system1")

    p.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--pretrained", type=lambda s: s.lower() != "false", default=True,
                    help="True (mac dinh) = tai pretrained ImageNet weights (can mang, on tren Kaggle). "
                         "False = random init, chi dung de smoke-test pipeline local khong can mang.")
    p.add_argument("--freeze_backbone_stages", type=int, default=0)
    p.add_argument("--image_size", type=int, default=224)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4,
                    help="LR cho head (concept_head + label_head, khoi tao ngau nhien).")
    p.add_argument("--backbone_lr_scale", type=float, default=0.1,
                    help="LR backbone = lr * backbone_lr_scale. Backbone da pretrained, "
                         "can LR thap hon nhieu de khong pha vo dac trung da hoc; "
                         "head moi khoi tao can LR binh thuong de hoc nhanh.")
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num_concepts", type=int, default=NUM_CONCEPTS)
    p.add_argument("--num_labels", type=int, default=NUM_LABELS)

    p.add_argument("--monitor", type=str, default="concept_macro_f1",
                    choices=["concept_macro_f1", "concept_mean_acc", "label_acc"],
                    help="Metric dung de luu best checkpoint.")

    return p.parse_args()


def make_loaders(data_dir: Path, img_dir: Path, image_size: int, batch_size: int, num_workers: int):
    train_dataset = FitzpatrickDataset(data_dir / "train.csv", img_dir, build_transforms("train", image_size))
    val_dataset = FitzpatrickDataset(data_dir / "val.csv", img_dir, build_transforms("val", image_size))
    test_dataset = FitzpatrickDataset(data_dir / "test.csv", img_dir, build_transforms("test", image_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def masked_bce_loss(concept_logits: torch.Tensor, concepts: torch.Tensor,
                     concept_mask: torch.Tensor) -> torch.Tensor:
    per_sample = F.binary_cross_entropy_with_logits(
        concept_logits, concepts, reduction="none"
    ).mean(dim=1)
    denom = concept_mask.sum().clamp(min=1.0)
    return (per_sample * concept_mask).sum() / denom


@torch.no_grad()
def compute_concept_metrics(all_logits: torch.Tensor, all_targets: torch.Tensor,
                             all_mask: torch.Tensor) -> dict:
    """Chi tinh tren sample co concept_mask=1 (khac MultiConcept: concept GT
    KHONG co full-supervision tren val/test giong het train o day vi val/test
    cua prepare_dataset.py van chi co ~19-20% anh co SkinCon that -- xem note
    trong meta.json. Neu can concept metric tren TOAN BO val/test, phai gan
    nhan SkinCon cho toan bo truoc, hien chua co."""
    keep = all_mask.bool()
    if keep.sum() == 0:
        return {"concept_mean_acc": 0.0, "concept_macro_f1": 0.0, "n_concept_eval": 0}
    logits, targets = all_logits[keep], all_targets[keep]
    preds = (torch.sigmoid(logits) > 0.5).float()

    tp = ((preds == 1) & (targets == 1)).sum(dim=0).float()
    fp = ((preds == 1) & (targets == 0)).sum(dim=0).float()
    fn = ((preds == 0) & (targets == 1)).sum(dim=0).float()

    precision = tp / (tp + fp).clamp(min=1e-8)
    recall = tp / (tp + fn).clamp(min=1e-8)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    has_pos = (targets.sum(dim=0) > 0)
    f1 = torch.where(has_pos, f1, torch.zeros_like(f1))

    acc = (preds == targets).float().mean(dim=0)

    return {
        "concept_mean_acc": acc.mean().item(),
        "concept_macro_f1": f1[has_pos].mean().item() if has_pos.any() else 0.0,
        "n_concept_eval": int(keep.sum().item()),
    }


def train_one_epoch(model, loader, optimizer, device, grad_clip_norm):
    model.train()
    total_loss, total_n = 0.0, 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        concepts = labels["concepts"].to(device)
        concept_mask = labels["concept_mask"].to(device)
        target_label = labels["label"].to(device)

        out = model(images)
        concept_loss = masked_bce_loss(out["concepts"], concepts, concept_mask)
        label_loss = F.cross_entropy(out["label"], target_label)
        loss = concept_loss + label_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_n += images.size(0)

    return {"loss": total_loss / total_n}


@torch.no_grad()
def evaluate(model, loader, device, split_name="Val"):
    model.eval()
    all_concept_logits, all_concept_targets, all_concept_mask = [], [], []
    label_correct, label_total = 0, 0
    total_loss, total_n = 0.0, 0

    for images, labels in tqdm(loader, desc=split_name, leave=False):
        images = images.to(device)
        concepts = labels["concepts"].to(device)
        concept_mask = labels["concept_mask"].to(device)
        target_label = labels["label"].to(device)

        out = model(images)
        concept_loss = masked_bce_loss(out["concepts"], concepts, concept_mask)
        label_loss = F.cross_entropy(out["label"], target_label)
        loss = concept_loss + label_loss

        total_loss += loss.item() * images.size(0)
        total_n += images.size(0)

        all_concept_logits.append(out["concepts"].cpu())
        all_concept_targets.append(concepts.cpu())
        all_concept_mask.append(concept_mask.cpu())

        label_correct += (out["label"].argmax(dim=1) == target_label).sum().item()
        label_total += images.size(0)

    metrics = compute_concept_metrics(
        torch.cat(all_concept_logits), torch.cat(all_concept_targets), torch.cat(all_concept_mask)
    )
    metrics["label_acc"] = label_correct / label_total
    metrics["loss"] = total_loss / total_n
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Data dir: {data_dir}")
    print(f"[INFO] Img dir: {img_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Backbone: {args.backbone}  pretrained={args.pretrained}")

    train_loader, val_loader, test_loader = make_loaders(
        data_dir=data_dir, img_dir=img_dir, image_size=args.image_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    model = FitzpatrickSystem1(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        num_concepts=args.num_concepts,
        num_labels=args.num_labels,
        freeze_backbone_stages=args.freeze_backbone_stages,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": args.lr * args.backbone_lr_scale},
            {"params": model.head_parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    best_val_metric = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.grad_clip_norm)
        val_metrics = evaluate(model, val_loader, device, split_name="Val")

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(f"train_loss={row['train_loss']:.4f} | val_loss={row['val_loss']:.4f} | "
              f"val_concept_macro_f1={row['val_concept_macro_f1']:.4f} | "
              f"val_concept_mean_acc={row['val_concept_mean_acc']:.4f} | "
              f"val_label_acc={row['val_label_acc']:.4f}")

        monitored = val_metrics[args.monitor]
        if monitored > best_val_metric:
            best_val_metric = monitored
            ckpt_path = output_dir / "best_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "best_val_metric": best_val_metric,
                "monitor": args.monitor,
                "epoch": epoch,
            }, ckpt_path)
            print(f"[INFO] Saved best checkpoint ({args.monitor}={best_val_metric:.4f})")

    best_ckpt = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device, split_name="Test")

    results = {
        "best_val_metric": best_val_metric,
        "monitor": args.monitor,
        "test_metrics": test_metrics,
        "history": history,
        "args": vars(args),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n[DONE] Fitzpatrick System1 results:")
    print(f"  best_val_{args.monitor} = {best_val_metric:.4f}")
    print(f"  test_concept_macro_f1  = {test_metrics['concept_macro_f1']:.4f}")
    print(f"  test_concept_mean_acc  = {test_metrics['concept_mean_acc']:.4f}")
    print(f"  test_label_acc         = {test_metrics['label_acc']:.4f}  "
          f"(so sanh voi baseline paper goc ~0.624)")


if __name__ == "__main__":
    main()
