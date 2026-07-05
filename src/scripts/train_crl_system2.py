"""
train_crl_system2.py — Training script cho CRL-based System 2
==============================================================
Usage:
    python -m src.scripts.train_crl_system2 \\
        --data_dir /kaggle/input/datasets/lquangmin/mnist-math \\
        --system1_ckpt /kaggle/working/outputs/system1_v4/best_model.pt \\
        --output_dir /kaggle/working/outputs/crl_system2_v1 \\
        --num_rules 64 \\
        --epochs 50
"""
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
from src.models.crl_system2 import CRLSystem2, _idx_to_concept_name
from src.models.rule_memory import (
    labels_to_input_concept_vector,
    soft_input_concept_vector,
    CONCEPT_KEYS_ORDERED,
)
from src.training.metrics import AverageMeter
from src.utils.seed import set_seed


# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CRL-based System 2.")

    p.add_argument("--data_dir",     type=str, required=True)
    p.add_argument("--system1_ckpt", type=str, required=True)
    p.add_argument("--output_dir",   type=str, default="outputs/crl_system2")

    p.add_argument("--num_rules",    type=int,   default=64,
                   help="Số rules. Không cần khớp số expressions (rules emerge từ data).")
    p.add_argument("--num_classes",  type=int,   default=10,
                   help="Số class output (10 cho digit3).")
    p.add_argument("--init_std",     type=float, default=0.1,
                   help="Std của random init cho rule_weights.")

    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)

    p.add_argument("--sparsity_weight",  type=float, default=0.01,
                   help="L1 regularization trên rule_weights.")
    p.add_argument("--diversity_weight", type=float, default=0.01,
                   help="Penalize similar rules.")

    p.add_argument("--warmup_gt_epochs", type=int, default=5,
                   help="Số epoch đầu dùng GT concept labels thay vì S1 output.")
    p.add_argument("--monitor", type=str, default="expression_acc",
                   choices=["expression_acc", "digit3_acc"])

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Data & Model helpers
# ─────────────────────────────────────────────────────────────

def make_loaders(data_dir: Path, batch_size: int, num_workers: int):
    def _ds(split):
        for name in (split, "valid" if split == "val" else split):
            p = data_dir / f"{name}.pt"
            if p.exists():
                return MNISTMathPTDataset(p)
        raise FileNotFoundError(f"No {split}.pt in {data_dir}")

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(_ds("train"), shuffle=True,  **kw),
        DataLoader(_ds("val"),   shuffle=False, **kw),
        DataLoader(_ds("test"),  shuffle=False, **kw),
    )


def load_system1(ckpt_path: Path, device: torch.device) -> MultiHeadSystem1:
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    args  = ckpt.get("args", {})
    model = MultiHeadSystem1(
        feature_dim=args.get("feature_dim", 256),
        num_slots   =args.get("num_slots",   4),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def get_concept_vec(
    system1: MultiHeadSystem1,
    images : torch.Tensor,
    labels : dict,
    use_gt : bool,
) -> torch.Tensor:
    if use_gt:
        return labels_to_input_concept_vector(labels)
    s1_out = system1(images)
    return soft_input_concept_vector(s1_out)


# ─────────────────────────────────────────────────────────────
# Train / Evaluate
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    system1: MultiHeadSystem1,
    system2: CRLSystem2,
    loader : DataLoader,
    optim  : torch.optim.Optimizer,
    device : torch.device,
    args   : argparse.Namespace,
    epoch  : int,
) -> dict[str, float]:
    system2.train()
    use_gt = epoch <= args.warmup_gt_epochs

    meters = {k: AverageMeter() for k in
              ["loss_total", "loss_task", "loss_sparsity", "loss_diversity",
               "digit3_acc"]}

    for images, labels in tqdm(loader, desc=f"Train ep{epoch}", leave=False):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        cv  = get_concept_vec(system1, images, labels, use_gt=use_gt)
        out = system2(cv)

        loss, loss_dict = CRLSystem2.compute_loss(
            outputs          = out,
            labels           = labels,
            rule_weights     = system2.rule_weights,
            sparsity_weight  = args.sparsity_weight,
            diversity_weight = args.diversity_weight,
        )

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Accuracy
        acc = CRLSystem2.compute_accuracy(out, labels)
        B   = images.size(0)
        for k, v in loss_dict.items():
            if k in meters:
                meters[k].update(v.item(), B)
        meters["digit3_acc"].update(acc["digit3_acc"], B)

    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def evaluate(
    system1    : MultiHeadSystem1,
    system2    : CRLSystem2,
    loader     : DataLoader,
    device     : torch.device,
    args       : argparse.Namespace,
    split_name : str = "Val",
) -> dict[str, float]:
    system2.eval()

    meters = {k: AverageMeter() for k in
              ["loss_total", "loss_task", "digit3_acc"]}

    for images, labels in loader:
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        cv  = get_concept_vec(system1, images, labels, use_gt=False)
        out = system2(cv)

        _, loss_dict = CRLSystem2.compute_loss(
            out, labels, system2.rule_weights,
            sparsity_weight=args.sparsity_weight,
            diversity_weight=args.diversity_weight,
        )
        acc = CRLSystem2.compute_accuracy(out, labels)
        B   = images.size(0)

        for k in ["loss_total", "loss_task"]:
            meters[k].update(loss_dict[k].item(), B)
        meters["digit3_acc"].update(acc["digit3_acc"], B)

    result = {k: m.avg for k, m in meters.items()}
    result["expression_acc"] = result["digit3_acc"]  # same for single-output
    return result


# ─────────────────────────────────────────────────────────────
# Rule printing
# ─────────────────────────────────────────────────────────────

def print_and_save_rules(system2: CRLSystem2, output_dir: Path, n: int = 20):
    decoded = system2.decode_rules(top_k=3)

    print(f"\n[INFO] Top {n} learned rules (từ |weight| lớn nhất):")
    for r in decoded[:n]:
        pred  = r["predicts_digit3"]
        rstr  = r["rule_string"]
        print(f"  Rule {r['rule_id']:3d} → digit3={pred}: {rstr}")

    out_path = output_dir / "rule_generated.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(decoded, f, indent=2, ensure_ascii=False)
    print(f"[INFO] All {len(decoded)} rules saved to {out_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    set_seed(args.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir  = Path(args.data_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Data dir: {data_dir}")
    print(f"[INFO] Output dir: {out_dir}")

    train_loader, val_loader, test_loader = make_loaders(
        data_dir, args.batch_size, args.num_workers
    )

    # ── Load frozen System1 ──────────────────────────────────
    system1 = load_system1(Path(args.system1_ckpt), device)
    print(f"[INFO] System1 loaded (frozen): {args.system1_ckpt}")

    # ── CRL System2 ──────────────────────────────────────────
    system2 = CRLSystem2(
        num_rules   = args.num_rules,
        concept_dim = 30,  # input-only: digit1+op1+digit2+op2, NO digit3
        num_classes = args.num_classes,
        init_std    = args.init_std,
    ).to(device)

    print(f"[INFO] CRL System2: {args.num_rules} rules, random init (std={args.init_std})")
    print(f"[INFO] Input concept dim: 30 (digit1+op1+digit2+op2, NO digit3 — prevents circular reasoning)")
    print(f"[INFO] Loss: task_CE + {args.sparsity_weight}×L1 + {args.diversity_weight}×diversity")
    print(f"[INFO] Warmup GT epochs: {args.warmup_gt_epochs}")

    optim = torch.optim.AdamW(
        system2.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_metric = -1.0
    history         = []

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(
            system1, system2, train_loader, optim, device, args, epoch
        )
        val_m = evaluate(system1, system2, val_loader, device, args, "Val")
        scheduler.step()

        monitor_val = val_m[args.monitor]
        row = {
            "epoch"                 : epoch,
            "train_loss"            : train_m["loss_total"],
            "train_task_loss"       : train_m["loss_task"],
            "train_sparsity_loss"   : train_m["loss_sparsity"],
            "train_digit3_acc"      : train_m["digit3_acc"],
            "val_loss"              : val_m["loss_total"],
            "val_digit3_acc"        : val_m["digit3_acc"],
            "val_expression_acc"    : val_m["expression_acc"],
            # Sparsity indicator: mean |w| (thấp = rules sparse)
            "rule_weight_l1"        : system2.rule_weights.abs().mean().item(),
        }
        history.append(row)

        saved = ""
        if monitor_val > best_val_metric:
            best_val_metric = monitor_val
            torch.save(
                {"model_state_dict": system2.state_dict(), "args": vars(args)},
                out_dir / "best_system2.pt",
            )
            saved = f"  → Saved best ({args.monitor}={monitor_val:.4f})"

        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f" | loss={train_m['loss_total']:.4f}"
            f" (task={train_m['loss_task']:.4f}"
            f", spar={train_m['loss_sparsity']:.4f})"
            f" | val_d3={val_m['digit3_acc']:.4f}"
            f" | |W|={row['rule_weight_l1']:.4f}"
            + saved
        )

    # ── Test ─────────────────────────────────────────────────
    best_ckpt = torch.load(out_dir / "best_system2.pt", map_location=device,
                           weights_only=False)
    system2.load_state_dict(best_ckpt["model_state_dict"])
    test_m = evaluate(system1, system2, test_loader, device, args, "Test")

    print(f"\n[DONE] Test results:")
    print(f"  digit3_acc     = {test_m['digit3_acc']:.4f}")
    print(f"  expression_acc = {test_m['expression_acc']:.4f}")

    # ── Save rules ───────────────────────────────────────────
    print_and_save_rules(system2, out_dir, n=20)

    # ── Save metrics ─────────────────────────────────────────
    metrics = {
        "best_val_metric"    : best_val_metric,
        "monitor"            : args.monitor,
        "test_metrics"       : test_m,
        "history"            : history,
        "args"               : vars(args),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Results saved to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()