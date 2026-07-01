from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.mnist_math_dataset import MNISTMathPTDataset
from src.models.multi_head_system1 import MultiHeadSystem1
from src.training.metrics import (
    AverageMeter,
    AccuracyMeter,
    batch_correct_from_logits,
    compute_expression_correct,
)
from src.utils.seed import set_seed


# v2: không còn "valid" — tất cả expressions đều valid by construction
CONCEPT_KEYS = ["digit1", "op1", "digit2", "op2", "digit3"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-head System 1 baseline.")

    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing train.pt, val.pt, test.pt.")
    parser.add_argument("--output_dir", type=str, default="outputs/system1_baseline")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--num_slots", type=int, default=4,
                        help="Số slot trong ảnh v2 (4 cho MNIST Math v2: digit1,op1,digit2,op2).")
    # valid_loss_weight đã bỏ (không còn valid label trong v2)
    parser.add_argument(
        "--monitor",
        type=str,
        default="expression_acc",
        choices=["expression_acc", "digit1_acc", "digit2_acc", "digit3_acc"],
        help="Metric dùng để lưu best checkpoint.",
    )

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


def system1_loss(outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]):
    """CE loss trên 5 concept slots (v2: không có valid)."""
    loss_digit1 = F.cross_entropy(outputs["digit1"], labels["digit1"])
    loss_op1    = F.cross_entropy(outputs["op1"],    labels["op1"])
    loss_digit2 = F.cross_entropy(outputs["digit2"], labels["digit2"])
    loss_op2    = F.cross_entropy(outputs["op2"],    labels["op2"])
    loss_digit3 = F.cross_entropy(outputs["digit3"], labels["digit3"])

    total_loss = loss_digit1 + loss_op1 + loss_digit2 + loss_op2 + loss_digit3

    loss_dict = {
        "loss_total":  total_loss,
        "loss_digit1": loss_digit1.detach(),
        "loss_op1":    loss_op1.detach(),
        "loss_digit2": loss_digit2.detach(),
        "loss_op2":    loss_op2.detach(),
        "loss_digit3": loss_digit3.detach(),
    }

    return total_loss, loss_dict


def init_metric_meters():
    return {
        "digit1_acc":     AccuracyMeter(),
        "op1_acc":        AccuracyMeter(),
        "digit2_acc":     AccuracyMeter(),
        "op2_acc":        AccuracyMeter(),
        "digit3_acc":     AccuracyMeter(),
        "expression_acc": AccuracyMeter(),
    }


def update_metric_meters(meters, outputs, labels):
    for key in CONCEPT_KEYS:
        correct, total = batch_correct_from_logits(outputs[key], labels[key])
        meters[f"{key}_acc"].update(correct, total)

    expr_correct, expr_total = compute_expression_correct(outputs, labels)
    meters["expression_acc"].update(expr_correct, expr_total)


def collect_metric_results(loss_meter, meters):
    results = {
        "loss": loss_meter.avg,
    }

    for key, meter in meters.items():
        results[key] = meter.acc

    return results


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    loss_meter = AverageMeter()
    meters = init_metric_meters()

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        outputs = model(images)
        loss, _ = system1_loss(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)

        update_metric_meters(meters, outputs, labels)

    return collect_metric_results(loss_meter, meters)


@torch.no_grad()
def evaluate(model, loader, device, split_name="Val"):
    model.eval()

    loss_meter = AverageMeter()
    meters = init_metric_meters()

    for images, labels in tqdm(loader, desc=split_name, leave=False):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        outputs = model(images)
        loss, _ = system1_loss(outputs, labels)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)

        update_metric_meters(meters, outputs, labels)

    return collect_metric_results(loss_meter, meters)


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

    model = MultiHeadSystem1(
        feature_dim=args.feature_dim,
        num_slots=args.num_slots,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_metric = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            split_name="Val",
        )

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        history.append(row)

        print(
            f"train_loss={row['train_loss']:.4f} | "
            f"train_expr_acc={row['train_expression_acc']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_expr_acc={row['val_expression_acc']:.4f}"
        )

        print(
            f"Val concepts: "
            f"d1={row['val_digit1_acc']:.4f}, "
            f"op1={row['val_op1_acc']:.4f}, "
            f"d2={row['val_digit2_acc']:.4f}, "
            f"op2={row['val_op2_acc']:.4f}, "
            f"d3={row['val_digit3_acc']:.4f}"
        )

        # Lưu checkpoint theo metric được chọn (mặc định expression_acc)
        monitored = val_metrics[args.monitor]
        if monitored > best_val_metric:
            best_val_metric = monitored
            ckpt_path = output_dir / "best_model.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_metric": best_val_metric,
                    "monitor": args.monitor,
                    "epoch": epoch,
                },
                ckpt_path,
            )

            print(f"[INFO] Saved best checkpoint ({args.monitor}={best_val_metric:.4f})")

    best_ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        split_name="Test",
    )

    results = {
        "best_val_metric": best_val_metric,
        "monitor": args.monitor,
        "test_metrics": test_metrics,
        "history": history,
        "args": vars(args),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n[DONE] System 1 results:")
    print(f"  best_val_{args.monitor} = {best_val_metric:.4f}")
    print(f"  test_expression_acc    = {test_metrics['expression_acc']:.4f}")
    print(
        f"  test slot acc: "
        f"d1={test_metrics['digit1_acc']:.4f}  "
        f"op1={test_metrics['op1_acc']:.4f}  "
        f"d2={test_metrics['digit2_acc']:.4f}  "
        f"op2={test_metrics['op2_acc']:.4f}  "
        f"d3={test_metrics['digit3_acc']:.4f}"
    )


if __name__ == "__main__":
    main()