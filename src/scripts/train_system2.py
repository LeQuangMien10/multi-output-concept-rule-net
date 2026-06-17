"""
train_system2.py  (v2 — concept-balanced)
==========================================
Train System 2 với multi-objective loss đảm bảo mọi concept
slot (digit1, op1, digit2, op2, digit3, valid) đều ngang nhau.

Usage:
    python -m src.training.train_system2 \
        --data_dir data/mnist_math \
        --system1_ckpt outputs/system1_baseline/best_model.pt \
        --output_dir outputs/system2 \
        --num_rules 64 \
        --epochs 30

Loss weights:
    --concept_weight   (default 1.0)  CE trên 6 slot
    --recon_weight     (default 0.5)  MSE reconstruction
    --sparsity_weight  (default 0.05) entropy regularization
    --coverage_weight  (default 0.05) penalize unused rules
    --diversity_weight (default 0.01) penalize identical rules
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.mnist_math_dataset import MNISTMathPTDataset
from src.models.multi_head_system1 import MultiHeadSystem1
from src.models.system2_model import (
    System2Rules,
    compute_system2_accuracy,
    system1_outputs_to_concept,
)
from src.models.rule_memory import (
    labels_to_concept_vector,
    CONCEPT_KEYS_ORDERED,
)
from src.training.metrics import AverageMeter
from src.utils.seed import set_seed


# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train System 2 (concept-balanced).")

    # Paths
    p.add_argument("--data_dir",      type=str, required=True)
    p.add_argument("--system1_ckpt",  type=str, required=True)
    p.add_argument("--output_dir",    type=str, default="outputs/system2")

    # Architecture
    p.add_argument("--num_rules",    type=int,   default=64)
    p.add_argument("--score_mode",   type=str,   default="weighted",
                   choices=["dot", "weighted", "cosine"])
    p.add_argument("--temperature",  type=float, default=1.0)

    # Loss weights
    p.add_argument("--concept_weight",   type=float, default=1.0,
                   help="Weight cho CE loss trên 6 concept slot.")
    p.add_argument("--recon_weight",     type=float, default=0.5,
                   help="Weight cho MSE reconstruction loss.")
    p.add_argument("--sparsity_weight",  type=float, default=0.05)
    p.add_argument("--coverage_weight",  type=float, default=0.05)
    p.add_argument("--diversity_weight", type=float, default=0.01)

    # Per-slot weight override (JSON string)
    # e.g. '{"digit1":1.0,"op1":1.0,"digit2":1.0,"op2":1.0,"digit3":1.0,"valid":1.0}'
    p.add_argument("--slot_weights", type=str, default=None,
                   help="JSON dict override per-slot weight.")

    # Training
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)

    # Concept source
    p.add_argument("--use_gt_concepts", action="store_true",
                   help="Dùng GT labels làm concept thay vì System1.")

    # Checkpoint monitor
    p.add_argument("--monitor", type=str, default="expression_acc",
                   choices=["expression_acc", "concept_acc", "valid_acc",
                            "digit1_acc", "digit2_acc", "digit3_acc"],
                   help="Metric để chọn best checkpoint.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────

def make_loaders(data_dir: Path, batch_size: int, num_workers: int):
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(MNISTMathPTDataset(data_dir / "train.pt"), shuffle=True,  **kw),
        DataLoader(MNISTMathPTDataset(data_dir / "val.pt"),   shuffle=False, **kw),
        DataLoader(MNISTMathPTDataset(data_dir / "test.pt"),  shuffle=False, **kw),
    )


def load_system1(ckpt_path: Path, device: torch.device) -> MultiHeadSystem1:
    ckpt = torch.load(ckpt_path, map_location=device)
    feature_dim = ckpt.get("args", {}).get("feature_dim", 256)
    model = MultiHeadSystem1(feature_dim=feature_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[INFO] System1 loaded (frozen): {ckpt_path}")
    return model


def get_concept_vec(
    system1   : MultiHeadSystem1,
    images    : torch.Tensor,
    labels    : dict[str, torch.Tensor],
    use_gt    : bool,
    soft      : bool = True,
) -> torch.Tensor:
    if use_gt:
        return labels_to_concept_vector(labels).float()
    with torch.no_grad():
        s1_out = system1(images)
    return system1_outputs_to_concept(s1_out, soft=soft)


# ─────────────────────────────────────────────────────────────
# Metrics accumulator
# ─────────────────────────────────────────────────────────────

class SlotAccumulator:
    """Tích lũy per-slot accuracy + expression accuracy qua batches."""

    def __init__(self):
        self.counts: dict[str, list] = {
            k: [0, 0] for k in CONCEPT_KEYS_ORDERED   # [correct, total]
        }
        self.expr_correct = 0
        self.expr_total   = 0

    def update(
        self,
        outputs: dict[str, torch.Tensor],
        labels : dict[str, torch.Tensor],
        B      : int,
    ):
        acc_dict = compute_system2_accuracy(outputs, labels)

        for key in CONCEPT_KEYS_ORDERED:
            n_correct = int(round(acc_dict[f"{key}_acc"] * B))
            self.counts[key][0] += n_correct
            self.counts[key][1] += B

        n_expr = int(round(acc_dict["expression_acc"] * B))
        self.expr_correct += n_expr
        self.expr_total   += B

    def result(self) -> dict[str, float]:
        out = {}
        for key in CONCEPT_KEYS_ORDERED:
            c, t = self.counts[key]
            out[f"{key}_acc"] = c / t if t > 0 else 0.0
        out["expression_acc"] = (
            self.expr_correct / self.expr_total
            if self.expr_total > 0 else 0.0
        )
        out["concept_acc"] = sum(
            out[f"{k}_acc"] for k in CONCEPT_KEYS_ORDERED
        ) / len(CONCEPT_KEYS_ORDERED)
        return out


# ─────────────────────────────────────────────────────────────
# Train / Eval loops
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    system1  : MultiHeadSystem1,
    system2  : System2Rules,
    loader   : DataLoader,
    optimizer: torch.optim.Optimizer,
    device   : torch.device,
    args,
    slot_weights: dict[str, float] | None,
) -> dict[str, float]:
    system2.train()

    loss_meters = {
        k: AverageMeter()
        for k in ["total", "concept", "recon", "sparsity", "coverage", "diversity"]
    }
    acc_accum = SlotAccumulator()

    for images, labels in tqdm(loader, desc="Train S2", leave=False):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        concept_vec = get_concept_vec(
            system1, images, labels, use_gt=args.use_gt_concepts, soft=True
        )

        outputs = system2(concept_vec)

        total_loss, loss_dict = System2Rules.compute_loss(
            outputs=outputs,
            concept_vec=concept_vec,
            labels=labels,
            concept_weight   = args.concept_weight,
            recon_weight     = args.recon_weight,
            sparsity_weight  = args.sparsity_weight,
            coverage_weight  = args.coverage_weight,
            diversity_weight = args.diversity_weight,
            slot_weights     = slot_weights,
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        B = images.size(0)
        loss_meters["total"    ].update(loss_dict["loss_total"    ].item(), B)
        loss_meters["concept"  ].update(loss_dict["loss_concept"  ].item(), B)
        loss_meters["recon"    ].update(loss_dict["loss_recon"    ].item(), B)
        loss_meters["sparsity" ].update(loss_dict["loss_sparsity" ].item(), B)
        loss_meters["coverage" ].update(loss_dict["loss_coverage" ].item(), B)
        loss_meters["diversity"].update(loss_dict["loss_diversity"].item(), B)

        acc_accum.update(outputs, labels, B)

    result = {f"loss_{k}": v.avg for k, v in loss_meters.items()}
    result.update(acc_accum.result())
    return result


@torch.no_grad()
def evaluate(
    system1 : MultiHeadSystem1,
    system2 : System2Rules,
    loader  : DataLoader,
    device  : torch.device,
    args,
    split   : str = "Val",
    slot_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    system2.eval()

    loss_meter = AverageMeter()
    acc_accum  = SlotAccumulator()

    for images, labels in tqdm(loader, desc=f"Eval [{split}]", leave=False):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        concept_vec = get_concept_vec(
            system1, images, labels, use_gt=args.use_gt_concepts, soft=True
        )

        outputs = system2(concept_vec)

        total_loss, _ = System2Rules.compute_loss(
            outputs=outputs,
            concept_vec=concept_vec,
            labels=labels,
            concept_weight   = args.concept_weight,
            recon_weight     = args.recon_weight,
            sparsity_weight  = args.sparsity_weight,
            coverage_weight  = args.coverage_weight,
            diversity_weight = args.diversity_weight,
            slot_weights     = slot_weights,
        )

        B = images.size(0)
        loss_meter.update(total_loss.item(), B)
        acc_accum.update(outputs, labels, B)

    result = {"loss_total": loss_meter.avg}
    result.update(acc_accum.result())
    return result


# ─────────────────────────────────────────────────────────────
# Pretty print helpers
# ─────────────────────────────────────────────────────────────

def print_epoch(epoch: int, total: int, train_m: dict, val_m: dict):
    print(
        f"Epoch {epoch:3d}/{total} | "
        f"loss={train_m['loss_total']:.4f} "
        f"(concept={train_m['loss_concept']:.4f}, "
        f"recon={train_m['loss_recon']:.4f}) | "
        f"val_loss={val_m['loss_total']:.4f}"
    )
    # Per-slot accuracy
    slot_str = "  ".join(
        f"{k}={val_m[f'{k}_acc']:.4f}" for k in CONCEPT_KEYS_ORDERED
    )
    print(f"  Val slot acc: {slot_str}")
    print(
        f"  Val concept_acc={val_m['concept_acc']:.4f}  "
        f"expression_acc={val_m['expression_acc']:.4f}"
    )


def print_learned_rules(system2: System2Rules, n: int = 20, threshold: float = 0.5):
    print(f"\n[INFO] Top {n} learned rules (mask threshold={threshold}):")
    slot_probs = system2.get_rule_slot_probs()

    for i in range(min(n, system2.num_rules)):
        # Decode prototype prediction per slot
        parts = []
        for key in CONCEPT_KEYS_ORDERED:
            pred_idx = slot_probs[key][i].argmax().item()
            if key in ("op1", "op2"):
                from src.utils.symbols import ID_TO_SYMBOL
                label = ID_TO_SYMBOL.get(pred_idx, str(pred_idx))
            else:
                label = str(pred_idx)
            parts.append(f"{key}={label}")

        rule_str = " AND ".join(parts)
        # Mask info
        mask = system2.memory.get_hard_masks(threshold)[i]
        n_active = int(mask.sum().item())
        print(f"  Rule {i:3d} [{n_active:2d} active slots]: {rule_str}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Slot weights
    slot_weights: dict[str, float] | None = None
    if args.slot_weights:
        slot_weights = json.loads(args.slot_weights)
        print(f"[INFO] Slot weights: {slot_weights}")

    train_loader, val_loader, test_loader = make_loaders(
        data_dir, args.batch_size, args.num_workers
    )

    system1 = load_system1(Path(args.system1_ckpt), device)

    system2 = System2Rules(
        num_rules   = args.num_rules,
        score_mode  = args.score_mode,
        temperature = args.temperature,
    ).to(device)

    optimizer = torch.optim.AdamW(
        system2.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"[INFO] System2: {args.num_rules} rules, score_mode={args.score_mode}")
    print(f"[INFO] Loss weights: concept={args.concept_weight}, "
          f"recon={args.recon_weight}, sparsity={args.sparsity_weight}, "
          f"coverage={args.coverage_weight}, diversity={args.diversity_weight}")
    print(f"[INFO] Monitor: {args.monitor}")
    print(f"[INFO] Concept source: "
          f"{'GT labels' if args.use_gt_concepts else 'System1 predictions'}")

    best_val_metric = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(
            system1, system2, train_loader, optimizer, device, args, slot_weights
        )
        val_m = evaluate(
            system1, system2, val_loader, device, args, "Val", slot_weights
        )

        print_epoch(epoch, args.epochs, train_m, val_m)

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}":   v for k, v in val_m.items()},
        }
        history.append(row)

        monitored = val_m[args.monitor]
        if monitored > best_val_metric:
            best_val_metric = monitored
            torch.save(
                {
                    "model_state_dict"  : system2.state_dict(),
                    "args"              : vars(args),
                    "best_val_metric"   : best_val_metric,
                    "monitor"           : args.monitor,
                    "epoch"             : epoch,
                },
                output_dir / "best_system2.pt",
            )
            print(f"  → Saved best ({args.monitor}={best_val_metric:.4f})")

    # ── Final evaluation ─────────────────────────────────────
    ckpt = torch.load(output_dir / "best_system2.pt", map_location=device)
    system2.load_state_dict(ckpt["model_state_dict"])

    test_m = evaluate(
        system1, system2, test_loader, device, args, "Test", slot_weights
    )

    print("\n[DONE] Test results:")
    slot_str = "  ".join(
        f"{k}={test_m[f'{k}_acc']:.4f}" for k in CONCEPT_KEYS_ORDERED
    )
    print(f"  Slot acc: {slot_str}")
    print(f"  concept_acc={test_m['concept_acc']:.4f}  "
          f"expression_acc={test_m['expression_acc']:.4f}")

    # In rules đã học
    print_learned_rules(system2, n=20, threshold=0.5)

    # Save results
    results = {
        "best_val_metric" : best_val_metric,
        "monitor"         : args.monitor,
        "test_metrics"    : test_m,
        "history"         : history,
        "args"            : vars(args),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[INFO] Results saved to {output_dir}/metrics.json")


if __name__ == "__main__":
    main()