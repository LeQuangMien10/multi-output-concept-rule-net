from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.mnist_math_dataset import MNISTMathPTDataset
from src.models.image_only_baseline import ImageOnlyBaseline
from src.training.metrics import AverageMeter, AccuracyMeter, batch_correct_from_logits
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train image-only CNN baseline.")

    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing train.pt, val.pt, test.pt.")
    parser.add_argument("--output_dir", type=str, default="outputs/image_only_baseline")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--feature_dim", type=int, default=256)

    return parser.parse_args()


def make_loaders(data_dir: Path, batch_size: int, num_workers: int):
    train_dataset = MNISTMathPTDataset(data_dir / "train.pt")
    val_dataset = MNISTMathPTDataset(data_dir / "val.pt")
    test_dataset = MNISTMathPTDataset(data_dir / "test.pt")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AccuracyMeter()

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        valid = labels["valid"].to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, valid)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)

        correct, total = batch_correct_from_logits(logits, valid)
        acc_meter.update(correct, total)

    return {
        "loss": loss_meter.avg,
        "valid_acc": acc_meter.acc,
    }


@torch.no_grad()
def evaluate(model, loader, device, split_name="Val"):
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AccuracyMeter()

    for images, labels in tqdm(loader, desc=split_name, leave=False):
        images = images.to(device)
        valid = labels["valid"].to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, valid)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)

        correct, total = batch_correct_from_logits(logits, valid)
        acc_meter.update(correct, total)

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
    print(f"[INFO] Data dir: {data_dir}")
    print(f"[INFO] Output dir: {output_dir}")

    train_loader, val_loader, test_loader = make_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = ImageOnlyBaseline(feature_dim=args.feature_dim).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, split_name="Val")

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_valid_acc": train_metrics["valid_acc"],
            "val_loss": val_metrics["loss"],
            "val_valid_acc": val_metrics["valid_acc"],
        }
        history.append(row)

        print(
            f"train_loss={row['train_loss']:.4f} | "
            f"train_valid_acc={row['train_valid_acc']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_valid_acc={row['val_valid_acc']:.4f}"
        )

        if val_metrics["valid_acc"] > best_val_acc:
            best_val_acc = val_metrics["valid_acc"]
            ckpt_path = output_dir / "best_model.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                ckpt_path,
            )

            print(f"[INFO] Saved best checkpoint to {ckpt_path}")

    best_ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device, split_name="Test")

    results = {
        "best_val_acc": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_valid_acc": test_metrics["valid_acc"],
        "history": history,
        "args": vars(args),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n[DONE] Image-only baseline results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()