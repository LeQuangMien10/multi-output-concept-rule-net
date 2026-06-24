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

import math
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
    p.add_argument("--num_rules",    type=int,   default=128,
                   help="Số rule prototype. 128 cho MNIST Math (100+ valid expressions).")
    p.add_argument("--score_mode",   type=str,   default="slot_cosine",
                   choices=["slot_cosine", "flat_cosine"],
                   help="slot_cosine: mỗi slot đóng góp đều nhau (recommended).")
    p.add_argument("--T_max",        type=float, default=2.0,
                   help="Temperature ban đầu (cosine annealing). Cao → exploration.")
    p.add_argument("--T_min",        type=float, default=0.07,
                   help="Temperature cuối (cosine annealing). Thấp → peaked assignment.")
    p.add_argument("--init_sharp",   type=float, default=8.0,
                   help="Logit sharpness khi khởi tạo prototype từ expressions.")
    p.add_argument("--hard_threshold", type=float, default=0.7,
                   help="Cosine threshold để coi slot khớp khi inference.")

    # Loss weights
    p.add_argument("--concept_weight",   type=float, default=1.0,
                   help="Weight cho CE loss trên 6 concept slot.")
    p.add_argument("--recon_weight",     type=float, default=0.3,
                   help="Weight cho MSE reconstruction loss.")
    p.add_argument("--sparsity_weight",  type=float, default=0.05)
    p.add_argument("--coverage_weight",  type=float, default=0.05)
    p.add_argument("--diversity_weight", type=float, default=0.02)

    # Per-slot weight override (JSON string)
    p.add_argument("--slot_weights", type=str, default=None,
                   help="JSON dict, e.g. '{\"digit3\":2.0}'")

    # Training
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)

    # Concept source
    p.add_argument("--use_gt_concepts", action="store_true",
                   help="Dùng GT labels làm concept thay vì System1.")
    p.add_argument("--warmup_gt_epochs", type=int, default=5,
                   help="Số epoch đầu dùng GT concept (warm-up). Đặt 0 để tắt.")

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
    saved_args = ckpt.get("args", {})
    feature_dim = saved_args.get("feature_dim", 256)
    num_slots   = saved_args.get("num_slots", 5)
    model = MultiHeadSystem1(feature_dim=feature_dim, num_slots=num_slots)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[INFO] System1 loaded (frozen): {ckpt_path}")
    print(f"       feature_dim={feature_dim}, num_slots={num_slots}, slot_dim={model.slot_dim}")
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
    epoch    : int = 0,
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

        # Warm-up: dùng GT concept trong warmup_gt_epochs đầu
        use_gt_now = args.use_gt_concepts or (epoch <= args.warmup_gt_epochs)
        concept_vec = get_concept_vec(
            system1, images, labels, use_gt=use_gt_now, soft=True
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
    temp_str = f" T={train_m.get('temperature', 0):.3f}" if 'temperature' in train_m else ""
    print(
        f"Epoch {epoch:3d}/{total}{temp_str} | "
        f"loss={train_m['loss_total']:.4f} "
        f"(concept={train_m['loss_concept']:.4f}, "
        f"recon={train_m['loss_recon']:.4f}, "
        f"spar={train_m['loss_sparsity']:.3f}) | "
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


def print_learned_rules(system2: System2Rules, output_dir: Path, n: int = 20):
    """
    Decode tất cả rules từ prototype probs (không dùng mask nữa).
    Export ra JSON và in n rules đầu ra terminal.
    """
    from src.utils.symbols import ID_TO_SYMBOL
    print(f"\n[INFO] Top {n} learned rules (decoded from prototype argmax):")

    slot_probs = system2.get_rule_slot_probs()  # key → [R, dim_k]
    rules_json_data = []

    for i in range(system2.num_rules):
        parts = []
        slots_dict = {}
        slot_confidence = {}

        for key in CONCEPT_KEYS_ORDERED:
            probs    = slot_probs[key][i]              # [dim_k]
            pred_idx = int(probs.argmax().item())
            conf     = float(probs.max().item())       # confidence = max prob

            label = ID_TO_SYMBOL.get(pred_idx, str(pred_idx)) if key in ("op1","op2") else str(pred_idx)
            parts.append(f"{key}={label}")
            slots_dict[key] = label
            slot_confidence[key] = round(conf, 4)

        rule_str = " AND ".join(parts)
        min_conf = min(slot_confidence.values())

        if i < n:
            conf_str = " | ".join(f"{k}:{v:.2f}" for k, v in slot_confidence.items())
            print(f"  Rule {i:3d} [min_conf={min_conf:.2f}]: {rule_str}")
            print(f"          conf: {conf_str}")

        rules_json_data.append({
            "rule_id"        : i,
            "rule_string"    : rule_str,
            "slots"          : slots_dict,
            "slot_confidence": slot_confidence,
            "min_confidence" : round(min_conf, 4),
        })

    json_path = output_dir / "rule_generated.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rules_json_data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] All {system2.num_rules} rules exported to {json_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def get_temperature(epoch: int, total_epochs: int, T_max: float, T_min: float) -> float:
    """Cosine annealing schedule: T_max → T_min qua total_epochs."""
    return T_min + 0.5 * (T_max - T_min) * (1 + math.cos(math.pi * epoch / total_epochs))


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
        num_rules      = args.num_rules,
        score_mode     = args.score_mode,
        temperature    = args.T_max,   # sẽ được override mỗi epoch bởi annealing
        hard_threshold = args.hard_threshold,
        init_sharp     = args.init_sharp,
    ).to(device)
    # lưu T_max/T_min vào system2 để checkpoint có thể restore
    system2._T_max = args.T_max
    system2._T_min = args.T_min

    optimizer = torch.optim.AdamW(
        system2.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"[INFO] System2: {args.num_rules} rules, score_mode={args.score_mode}, "
          f"T_max={args.T_max} → T_min={args.T_min} (cosine annealing), "
          f"hard_threshold={args.hard_threshold}")
    print(f"[INFO] Prototype init: sharp={args.init_sharp}")
    print(f"[INFO] Warm-up GT epochs: {args.warmup_gt_epochs}")
    print(f"[INFO] Loss weights: concept={args.concept_weight}, "
          f"recon={args.recon_weight}, sparsity={args.sparsity_weight}, "
          f"coverage={args.coverage_weight}, diversity={args.diversity_weight}")
    print(f"[INFO] Monitor: {args.monitor}")
    print(f"[INFO] Concept source: "
          f"{'GT labels' if args.use_gt_concepts else 'System1 predictions'}")

    best_val_metric = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Temperature annealing: T_max → T_min theo cosine schedule
        current_temp = get_temperature(epoch, args.epochs, args.T_max, args.T_min)
        system2.temperature = current_temp

        train_m = train_one_epoch(
            system1, system2, train_loader, optimizer, device, args, slot_weights,
            epoch=epoch,
        )
        val_m = evaluate(
            system1, system2, val_loader, device, args, "Val", slot_weights
        )

        train_m["temperature"] = current_temp
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

    # In rules đã học và export ra JSON
    print_learned_rules(system2, output_dir=output_dir, n=20)

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