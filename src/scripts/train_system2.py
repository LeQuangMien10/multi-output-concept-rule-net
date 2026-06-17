"""
train_system2.py
================
Train System 2 rule masks từ dataset.

Luồng:
    1. Load System 1 (đã train sẵn, frozen)
    2. Với mỗi batch: System1 → soft concept vector
    3. System2 tính rule scores → classification loss
    4. Chỉ update tham số System 2

Usage:
    python -m src.training.train_system2 \\
        --data_dir data/mnist_math \\
        --system1_ckpt outputs/system1_baseline/best_model.pt \\
        --output_dir outputs/system2 \\
        --num_rules 64 \\
        --epochs 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.mnist_math_dataset import MNISTMathPTDataset
from src.models.multi_head_system1 import MultiHeadSystem1
from src.models.system2_model import System2Rules, system1_outputs_to_concept
from src.models.rule_memory import labels_to_concept_vector
from src.training.metrics import AverageMeter, AccuracyMeter
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train System 2 rule network.")

    # Data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--system1_ckpt", type=str, required=True,
                   help="Path to System1 best_model.pt checkpoint.")
    p.add_argument("--output_dir", type=str, default="outputs/system2")

    # System 2 architecture
    p.add_argument("--num_rules", type=int, default=64,
                   help="Number of rule prototypes to learn.")
    p.add_argument("--score_mode", type=str, default="weighted",
                   choices=["dot", "weighted", "cosine"])
    p.add_argument("--temperature", type=float, default=1.0)

    # Loss weights
    p.add_argument("--sparsity_weight", type=float, default=0.01)
    p.add_argument("--coverage_weight", type=float, default=0.01)
    p.add_argument("--use_alt_head", action="store_true",
                   help="Use linear score head instead of rule_valid_logits.")

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # Concept input to System2
    p.add_argument("--use_gt_concepts", action="store_true",
                   help="Dùng ground-truth concept (label) thay vì System1 predictions.")

    return p.parse_args()


def make_loaders(data_dir: Path, batch_size: int, num_workers: int):
    train_ds = MNISTMathPTDataset(data_dir / "train.pt")
    val_ds = MNISTMathPTDataset(data_dir / "val.pt")
    test_ds = MNISTMathPTDataset(data_dir / "test.pt")

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
    )


def load_system1(ckpt_path: Path, device: torch.device) -> MultiHeadSystem1:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})
    feature_dim = args.get("feature_dim", 256)
    model = MultiHeadSystem1(feature_dim=feature_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[INFO] System1 loaded from {ckpt_path} (frozen)")
    return model


def get_concept_vec(
    system1: MultiHeadSystem1,
    images: torch.Tensor,
    labels: dict[str, torch.Tensor],
    use_gt: bool,
    soft: bool = True,
) -> torch.Tensor:
    """
    Nếu use_gt=True → dùng ground truth labels làm concept.
    Ngược lại → System1 dự đoán.
    """
    if use_gt:
        return labels_to_concept_vector(labels).float()

    with torch.no_grad():
        s1_out = system1(images)

    return system1_outputs_to_concept(s1_out, soft=soft)


def train_one_epoch(
    system1: MultiHeadSystem1,
    system2: System2Rules,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
) -> dict[str, float]:
    system2.train()
    loss_meter = AverageMeter()
    cls_meter = AverageMeter()
    acc_meter = AccuracyMeter()

    for images, labels in tqdm(loader, desc="Train S2", leave=False):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        concept_vec = get_concept_vec(
            system1, images, labels,
            use_gt=args.use_gt_concepts,
            soft=True,
        )

        outputs = system2(concept_vec)

        total_loss, loss_dict = System2Rules.compute_loss(
            outputs, labels,
            sparsity_weight=args.sparsity_weight,
            coverage_weight=args.coverage_weight,
            use_alt_head=args.use_alt_head,
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        B = images.size(0)
        loss_meter.update(total_loss.item(), B)
        cls_meter.update(loss_dict["loss_cls"].item(), B)

        logits = (
            outputs["valid_logits_alt"] if args.use_alt_head
            else outputs["valid_logits"]
        )
        preds = logits.argmax(dim=1)
        correct = (preds == labels["valid"]).sum().item()
        acc_meter.update(correct, B)

    return {
        "loss": loss_meter.avg,
        "cls_loss": cls_meter.avg,
        "valid_acc": acc_meter.acc,
    }


@torch.no_grad()
def evaluate(
    system1: MultiHeadSystem1,
    system2: System2Rules,
    loader: DataLoader,
    device: torch.device,
    args,
    split: str = "Val",
) -> dict[str, float]:
    system2.eval()
    loss_meter = AverageMeter()
    acc_meter = AccuracyMeter()

    for images, labels in tqdm(loader, desc=f"Eval S2 [{split}]", leave=False):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        concept_vec = get_concept_vec(
            system1, images, labels,
            use_gt=args.use_gt_concepts,
            soft=True,
        )

        outputs = system2(concept_vec)

        total_loss, _ = System2Rules.compute_loss(
            outputs, labels,
            sparsity_weight=args.sparsity_weight,
            coverage_weight=args.coverage_weight,
            use_alt_head=args.use_alt_head,
        )

        B = images.size(0)
        loss_meter.update(total_loss.item(), B)

        logits = (
            outputs["valid_logits_alt"] if args.use_alt_head
            else outputs["valid_logits"]
        )
        preds = logits.argmax(dim=1)
        correct = (preds == labels["valid"]).sum().item()
        acc_meter.update(correct, B)

    return {
        "loss": loss_meter.avg,
        "valid_acc": acc_meter.acc,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    train_loader, val_loader, test_loader = make_loaders(
        data_dir, args.batch_size, args.num_workers
    )

    system1 = load_system1(Path(args.system1_ckpt), device)

    system2 = System2Rules(
        num_rules=args.num_rules,
        score_mode=args.score_mode,
        temperature=args.temperature,
    ).to(device)

    optimizer = torch.optim.AdamW(
        system2.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"[INFO] System2: {args.num_rules} rules, score_mode={args.score_mode}")
    print(f"[INFO] Concept source: {'GT labels' if args.use_gt_concepts else 'System1 predictions'}")

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(system1, system2, train_loader, optimizer, device, args)
        val_m = evaluate(system1, system2, val_loader, device, args, "Val")

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_m['loss']:.4f} | "
            f"train_valid_acc={train_m['valid_acc']:.4f} | "
            f"val_loss={val_m['loss']:.4f} | "
            f"val_valid_acc={val_m['valid_acc']:.4f}"
        )

        if val_m["valid_acc"] > best_val_acc:
            best_val_acc = val_m["valid_acc"]
            torch.save(
                {
                    "model_state_dict": system2.state_dict(),
                    "args": vars(args),
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                output_dir / "best_system2.pt",
            )
            print(f"  → Saved best checkpoint (val_acc={best_val_acc:.4f})")

    # Load best & evaluate on test
    ckpt = torch.load(output_dir / "best_system2.pt", map_location=device)
    system2.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate(system1, system2, test_loader, device, args, "Test")

    # In các rule đã học
    print("\n[INFO] Learned rules (top 20, threshold=0.5):")
    all_rules = system2.memory.decode_all_rules(threshold=0.5)
    for i, rule_str in enumerate(all_rules[:20]):
        valid_pred = system2.rule_valid_logits[i].argmax().item()
        print(f"  Rule {i:3d} [valid={valid_pred}]: {rule_str}")

    results = {
        "best_val_acc": best_val_acc,
        "test_metrics": test_m,
        "history": history,
        "args": vars(args),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Test valid_acc={test_m['valid_acc']:.4f}")
    print(f"[INFO] Results saved to {output_dir}/metrics.json")


if __name__ == "__main__":
    main()