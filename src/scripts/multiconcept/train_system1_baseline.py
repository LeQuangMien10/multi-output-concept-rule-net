"""
train_system1_baseline.py — Train S1 cho MNIST-MultiConcept
==============================================================

S1 học 2 việc song song, cả hai đều là "concept" theo đúng tinh thần MNIST
Math (digit3 vừa là target vừa là 1 slot trong concept vector):
  - concept_head : 16 concept thị giác nhị phân (multi-label, sigmoid),
                    chỉ supervised trên concept_mask=1 (~25% train, mô
                    phỏng SkinCon ~22% coverage).
  - label_head   : nhãn 3 lớp (non_neoplastic/benign/malignant), supervised
                    FULL (100% ảnh luôn có nhãn, giống Fitzpatrick thật —
                    three_partition_label có ở mọi ảnh, chỉ SkinCon concept
                    mới khan hiếm).

Output của label_head sẽ được nối vào concept vector ở Stage 2 (ICRL) để
cluster — xem soft_concept_vector trong models/multiconcept/system1.py.
Nhãn dùng để train prediction head Stage 3 vẫn lấy từ memory.get_labels()
(ground-truth ngoài), KHÔNG đọc trực tiếp label_head.

Usage (Kaggle mặc định, override --data_dir nếu chạy local):
    python -m src.scripts.multiconcept.train_system1_baseline \\
        --data_dir /kaggle/input/mnist-multiconcept \\
        --output_dir /kaggle/working/outputs/multiconcept_system1 \\
        --epochs 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.multiconcept.mnist_multiconcept_dataset import MNISTMultiConceptPTDataset
from src.models.multiconcept.system1 import MultiConceptSystem1
from src.utils.multiconcept_concepts import NUM_CONCEPTS, NUM_LABELS
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train MultiConceptSystem1 (S1) baseline.")

    # Kaggle-first defaults — override bằng đường dẫn local nếu cần.
    p.add_argument("--data_dir", type=str, default="/kaggle/input/mnist-multiconcept",
                    help="Thư mục chứa train.pt/valid.pt/test.pt. "
                         "Mặc định trỏ Kaggle input; đổi sang path local (vd. data/mnist_multiconcept_v1) nếu chạy máy local.")
    p.add_argument("--output_dir", type=str, default="/kaggle/working/outputs/multiconcept_system1")

    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--label_lr_scale", type=float, default=0.2,
                    help="LR của label_head = lr * label_lr_scale. label_head hội tụ "
                         "nhanh hơn concept_head nhiều (task dễ hơn) nên cùng LR dễ "
                         "overshoot gây val_label_acc sập định kỳ (quan sát thực nghiệm: "
                         "val_loss vọt lên 3-5x giữa các epoch dù concept vẫn cải thiện đều).")
    p.add_argument("--grad_clip_norm", type=float, default=1.0,
                    help="Gradient clipping (max L2 norm) — chặn spike gradient hiếm gặp "
                         "làm hỏng vài batch huấn luyện, cùng nguyên nhân bất ổn ở trên.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)

    p.add_argument("--feature_dim",  type=int, default=256)
    p.add_argument("--num_concepts", type=int, default=NUM_CONCEPTS)
    p.add_argument("--num_labels",   type=int, default=NUM_LABELS)

    p.add_argument("--monitor", type=str, default="concept_macro_f1",
                    choices=["concept_macro_f1", "concept_mean_acc", "label_acc"],
                    help="Metric dùng để lưu best checkpoint.")

    return p.parse_args()


def make_loaders(data_dir: Path, batch_size: int, num_workers: int):
    train_dataset = MNISTMultiConceptPTDataset(data_dir / "train.pt")
    val_dataset   = MNISTMultiConceptPTDataset(data_dir / "valid.pt")
    test_dataset  = MNISTMultiConceptPTDataset(data_dir / "test.pt")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def masked_bce_loss(concept_logits: torch.Tensor, concepts: torch.Tensor,
                     concept_mask: torch.Tensor) -> torch.Tensor:
    """
    BCE trung bình theo concept, chỉ tính trên sample có concept_mask=1.
    concept_logits, concepts: [B, C]   concept_mask: [B]
    """
    per_sample = F.binary_cross_entropy_with_logits(
        concept_logits, concepts, reduction="none"
    ).mean(dim=1)                                    # [B]
    denom = concept_mask.sum().clamp(min=1.0)
    return (per_sample * concept_mask).sum() / denom


@torch.no_grad()
def compute_concept_metrics(all_logits: torch.Tensor, all_targets: torch.Tensor) -> dict:
    """Per-concept accuracy + macro-F1 trên toàn bộ split (luôn full supervision)."""
    preds = (torch.sigmoid(all_logits) > 0.5).float()

    tp = ((preds == 1) & (all_targets == 1)).sum(dim=0).float()
    fp = ((preds == 1) & (all_targets == 0)).sum(dim=0).float()
    fn = ((preds == 0) & (all_targets == 1)).sum(dim=0).float()

    precision = tp / (tp + fp).clamp(min=1e-8)
    recall    = tp / (tp + fn).clamp(min=1e-8)
    f1        = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    # Concept không xuất hiện dương lần nào trong split -> F1 không xác định, coi là 0
    has_pos = (all_targets.sum(dim=0) > 0)
    f1 = torch.where(has_pos, f1, torch.zeros_like(f1))

    acc = (preds == all_targets).float().mean(dim=0)

    return {
        "concept_mean_acc": acc.mean().item(),
        "concept_macro_f1": f1[has_pos].mean().item() if has_pos.any() else 0.0,
    }


def train_one_epoch(model, loader, optimizer, device, grad_clip_norm):
    model.train()
    total_loss, total_n = 0.0, 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images       = images.to(device)
        concepts     = labels["concepts"].to(device)
        concept_mask = labels["concept_mask"].to(device)
        target_label = labels["label"].to(device)

        out = model(images)
        concept_loss = masked_bce_loss(out["concepts"], concepts, concept_mask)
        label_loss   = F.cross_entropy(out["label"], target_label)   # full supervision
        loss = concept_loss + label_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_n    += images.size(0)

    return {"loss": total_loss / total_n}


@torch.no_grad()
def evaluate(model, loader, device, split_name="Val"):
    model.eval()
    all_concept_logits, all_concept_targets = [], []
    label_correct, label_total = 0, 0
    total_loss, total_n = 0.0, 0

    for images, labels in tqdm(loader, desc=split_name, leave=False):
        images       = images.to(device)
        concepts     = labels["concepts"].to(device)
        concept_mask = labels["concept_mask"].to(device)
        target_label = labels["label"].to(device)

        out = model(images)
        concept_loss = masked_bce_loss(out["concepts"], concepts, concept_mask)
        label_loss   = F.cross_entropy(out["label"], target_label)
        loss = concept_loss + label_loss

        total_loss += loss.item() * images.size(0)
        total_n    += images.size(0)

        all_concept_logits.append(out["concepts"].cpu())
        all_concept_targets.append(concepts.cpu())

        label_correct += (out["label"].argmax(dim=1) == target_label).sum().item()
        label_total   += images.size(0)

    metrics = compute_concept_metrics(torch.cat(all_concept_logits), torch.cat(all_concept_targets))
    metrics["label_acc"] = label_correct / label_total
    metrics["loss"] = total_loss / total_n
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Data dir: {data_dir}")
    print(f"[INFO] Output dir: {output_dir}")

    train_loader, val_loader, test_loader = make_loaders(
        data_dir=data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    model = MultiConceptSystem1(
        feature_dim=args.feature_dim,
        num_concepts=args.num_concepts,
        num_labels=args.num_labels,
    ).to(device)

    label_head_params = list(model.label_head.parameters())
    other_params = [p for p in model.parameters()
                    if not any(p is q for q in label_head_params)]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params,      "lr": args.lr},
            {"params": label_head_params, "lr": args.lr * args.label_lr_scale},
        ],
        weight_decay=args.weight_decay,
    )

    best_val_metric = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.grad_clip_norm)
        val_metrics   = evaluate(model, val_loader, device, split_name="Val")

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

    print("\n[DONE] MultiConcept System1 results:")
    print(f"  best_val_{args.monitor} = {best_val_metric:.4f}")
    print(f"  test_concept_macro_f1  = {test_metrics['concept_macro_f1']:.4f}")
    print(f"  test_concept_mean_acc  = {test_metrics['concept_mean_acc']:.4f}")
    print(f"  test_label_acc         = {test_metrics['label_acc']:.4f}  (S1-alone baseline, so sánh với ICRL sau)")


if __name__ == "__main__":
    main()
