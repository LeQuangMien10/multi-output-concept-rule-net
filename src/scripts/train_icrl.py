"""
train_icrl.py — Build Incremental Concept-driven Rule Memory
=============================================================

3 giai đoạn:
  Stage 1: Load S1 đã train (frozen)
  Stage 2: Pass training set qua S1 → collect concept vectors
           Incremental clustering: CREATE / MATCH / UPDATE
           Prune sau mỗi epoch
  Stage 3: Train linear prediction head trên rule assignments

Usage (MNIST Math +/−):
    python -m src.scripts.train_icrl \\
        --data_dir /kaggle/input/datasets/lquangmin/mnist-math \\
        --system1_ckpt /kaggle/working/outputs/system1_v4/best_model.pt \\
        --output_dir /kaggle/working/outputs/icrl_v1 \\
        --theta 0.85 \\
        --epochs 3

Hyperparameters quan trọng:
    --theta       : similarity threshold để CREATE rule (default 0.85)
    --theta_merge : similarity threshold để MERGE rules (default 0.95)
    --n_min       : minimum samples để rule survive (default 5)
    --conf_min    : minimum confidence để rule survive (default 0.1)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.mnist_math_dataset import MNISTMathPTDataset
from src.models.multi_head_system1 import MultiHeadSystem1
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.rule_memory import (
    soft_concept_vector,
    logits_to_concept_vector,
    CONCEPT_KEYS_ORDERED,
    CONCEPT_OFFSETS,
    CONCEPT_DIMS,
    CONCEPT_TOTAL_DIM,
)
from src.utils.seed import set_seed
from src.utils.symbols import ID_TO_SYMBOL


# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build ICRL rule memory.")

    p.add_argument("--data_dir",     type=str, required=True)
    p.add_argument("--system1_ckpt", type=str, required=True)
    p.add_argument("--output_dir",   type=str, default="outputs/icrl")

    # ICRL hyperparameters
    p.add_argument("--theta",        type=float, default=0.85,
                   help="Similarity threshold để CREATE rule mới.")
    p.add_argument("--theta_merge",  type=float, default=0.95,
                   help="Similarity threshold để MERGE 2 rules.")
    p.add_argument("--n_min",        type=int,   default=5,
                   help="Số samples tối thiểu để rule survive prune.")
    p.add_argument("--conf_min",     type=float, default=0.1,
                   help="Confidence tối thiểu để rule survive prune.")

    # Stage 2 options
    p.add_argument("--epochs",       type=int,   default=3,
                   help="Số lần pass qua training set để build rule memory.")
    p.add_argument("--use_hard_cv",  action="store_true",
                   help="Dùng hard (argmax) concept vector thay vì soft (softmax).")

    # Stage 3 (prediction head)
    p.add_argument("--head_epochs",  type=int,   default=20,
                   help="Số epochs train prediction head.")
    p.add_argument("--head_lr",      type=float, default=1e-3)
    p.add_argument("--num_classes",  type=int,   default=10,
                   help="Số classes của target (digit3=10, benign/malignant=2).")

    # General
    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       type=str,   default="auto")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────

def load_system1(ckpt_path: Path, device: torch.device) -> MultiHeadSystem1:
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args  = ckpt.get("args", {})
    feature_dim = saved_args.get("feature_dim", 256)
    num_slots   = saved_args.get("num_slots", 4)
    model       = MultiHeadSystem1(feature_dim=feature_dim, num_slots=num_slots)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_loaders(data_dir: Path, batch_size: int, num_workers: int):
    def _loader(split, shuffle):
        candidates = [f"{split}.pt", "valid.pt" if split == "val" else f"{split}.pt"]
        for fname in (["val.pt","valid.pt"] if split == "val" else [f"{split}.pt"]):
            pt = data_dir / fname
            if pt.exists():
                ds = MNISTMathPTDataset(pt)
                return DataLoader(ds, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=True)
        raise FileNotFoundError(f"No {split} split found in {data_dir}")

    return _loader("train", True), _loader("val", False), _loader("test", False)


# ─────────────────────────────────────────────────────────────
# Stage 2: Build rule memory
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def build_rule_memory(
    system1:    MultiHeadSystem1,
    loader:     DataLoader,
    memory:     ICRLRuleMemory,
    device:     torch.device,
    use_hard:   bool = False,
    epoch_label: str = "Epoch",
) -> dict[str, int]:
    """Pass qua loader một lần, update rule memory."""
    total_stats = {"created": 0, "matched": 0, "total": 0}

    for images, labels in tqdm(loader, desc=f"  Build [{epoch_label}]", leave=False):
        images = images.to(device)

        # Concept vector từ S1 (frozen)
        with torch.no_grad():
            s1_out = system1(images)

        if use_hard:
            cv = logits_to_concept_vector(s1_out)
        else:
            cv = soft_concept_vector(s1_out)   # [B, D]

        # S1 confidence = mean max-prob across slots (excluding op2 = always =)
        slot_confs = []
        for key in ["digit1", "op1", "digit2", "digit3"]:
            probs = F.softmax(s1_out[key], dim=1)
            slot_confs.append(probs.max(dim=1).values)
        s1_conf = torch.stack(slot_confs, dim=1).mean(dim=1)   # [B]

        # Target label = digit3 (hoặc thay bằng label khác nếu dataset khác)
        y = labels["digit3"].to(device)

        stats = memory.process_batch(cv, y, s1_conf)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    return total_stats


# ─────────────────────────────────────────────────────────────
# Stage 2b: Update accuracy (sau mỗi epoch, dùng current head)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def update_rule_accuracy(
    system1: MultiHeadSystem1,
    head:    nn.Linear,
    loader:  DataLoader,
    memory:  ICRLRuleMemory,
    device:  torch.device,
    use_hard: bool = False,
) -> None:
    """Compute predictions với head hiện tại và update accuracy trong memory."""
    if memory.is_empty or head is None:
        return

    # Reset accuracy counters
    for i in range(memory.num_rules):
        memory._correct[i]    = 0
        memory._total_pred[i] = 0

    centroids = memory.get_centroids().to(device)   # [R, D]

    for images, labels in tqdm(loader, desc="  Update accuracy", leave=False):
        images = images.to(device)
        s1_out = system1(images)
        cv     = logits_to_concept_vector(s1_out) if use_hard else soft_concept_vector(s1_out)
        y      = labels["digit3"].to(device)

        # Match → rule_ids
        rule_ids, _ = memory.match(cv)            # [B]
        # Predict via head
        rule_cvs    = centroids[rule_ids]          # [B, D]
        preds       = head(rule_cvs).argmax(dim=1) # [B]

        memory.update_accuracy(cv, y, preds)


# ─────────────────────────────────────────────────────────────
# Stage 3: Train prediction head
# ─────────────────────────────────────────────────────────────

def train_head(
    system1:     MultiHeadSystem1,
    memory:      ICRLRuleMemory,
    train_loader: DataLoader,
    val_loader:  DataLoader,
    num_classes: int,
    epochs:      int,
    lr:          float,
    device:      torch.device,
    use_hard:    bool = False,
) -> nn.Linear:
    """
    Train linear head: concept_vec → class.
    Head input = concept vector (D-dim), output = num_classes.
    """
    head = nn.Linear(CONCEPT_TOTAL_DIM, num_classes).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    best_val  = 0.0
    best_state = None

    centroids = memory.get_centroids().to(device)   # [R, D]
    rule_labels = torch.tensor(memory.get_labels(), dtype=torch.long, device=device)  # [R]

    print(f"\n[Stage 3] Train prediction head ({epochs} epochs)")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        head.train()
        train_correct = 0; train_total = 0

        for images, labels in tqdm(train_loader, desc=f"  Head ep{epoch:2d}", leave=False):
            images = images.to(device)
            with torch.no_grad():
                s1_out = system1(images)
                cv     = logits_to_concept_vector(s1_out) if use_hard else soft_concept_vector(s1_out)
            y = labels["digit3"].to(device)

            logits = head(cv)
            loss   = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total   += len(y)

        # ── Val ──
        head.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                s1_out = system1(images)
                cv     = logits_to_concept_vector(s1_out) if use_hard else soft_concept_vector(s1_out)
                y      = labels["digit3"].to(device)
                preds  = head(cv).argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total   += len(y)

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total

        print(f"  Ep {epoch:2d}/{epochs}: train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    if best_state:
        head.load_state_dict(best_state)
    print(f"  Best val_acc = {best_val:.4f}")
    return head


# ─────────────────────────────────────────────────────────────
# Evaluate & export
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    system1:  MultiHeadSystem1,
    head:     nn.Linear,
    loader:   DataLoader,
    device:   torch.device,
    split:    str = "test",
    use_hard: bool = False,
) -> dict[str, float]:
    head.eval()
    correct = 0; total = 0

    for images, labels in tqdm(loader, desc=f"  Eval {split}", leave=False):
        images = images.to(device)
        s1_out = system1(images)
        cv     = logits_to_concept_vector(s1_out) if use_hard else soft_concept_vector(s1_out)
        y      = labels["digit3"].to(device)
        preds  = head(cv).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += len(y)

    return {"accuracy": correct / total, "correct": correct, "total": total}


def export_rules(
    memory:     ICRLRuleMemory,
    output_dir: Path,
    n_show:     int = 20,
) -> None:
    """Export rules to JSON và print top rules."""
    rules_data = []

    for r in range(memory.num_rules):
        decoded = memory.decode_rule(
            rule_id        = r,
            concept_keys   = CONCEPT_KEYS_ORDERED,
            concept_offsets= CONCEPT_OFFSETS,
            concept_dims   = CONCEPT_DIMS,
            id_to_symbol   = ID_TO_SYMBOL,
        )
        # Build rule string (MNIST Math format)
        slots = decoded["slots"]
        d1 = slots.get("digit1", {}).get("value", "?")
        op = slots.get("op1", {}).get("value", "?")
        d2 = slots.get("digit2", {}).get("value", "?")
        d3 = decoded["label"]
        rule_str = f"{d1} {op} {d2} = {d3}"

        decoded["rule_string"] = rule_str
        rules_data.append(decoded)

    # Sort by confidence desc
    rules_data.sort(key=lambda x: -x["confidence"])

    json_path = output_dir / "icrl_rules.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rules_data, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] {len(rules_data)} rules exported to {json_path}")

    # Print top rules
    print(f"\n[INFO] Top {min(n_show, len(rules_data))} rules (sorted by confidence):")
    for r in rules_data[:n_show]:
        bar = "█" * int(r["confidence"] * 20)
        print(f"  [{r['confidence']:.3f}] {r['rule_string']:20s}  "
              f"n={r['n']:4d}  coh={r['coherence']:.3f}  acc={r['accuracy']:.3f}  {bar}")

    # Print confidence distribution
    confs = [r["confidence"] for r in rules_data]
    bins  = [(0.0,.3),(.3,.5),(.5,.7),(.7,.9),(.9,1.01)]
    print(f"\n  Confidence distribution ({len(rules_data)} rules):")
    for lo, hi in bins:
        n = sum(1 for c in confs if lo <= c < hi)
        print(f"    [{lo:.1f},{hi:.1f}): {n:3d}  {'█'*n}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" \
             else torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] ICRL params: θ={args.theta}  θ_merge={args.theta_merge}  "
          f"n_min={args.n_min}  conf_min={args.conf_min}")

    # ── Data ────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    train_loader, val_loader, test_loader = make_loaders(
        data_dir, args.batch_size, args.num_workers
    )
    print(f"[INFO] Data: {data_dir}")

    # ── Stage 1: Load S1 ────────────────────────────────────
    system1 = load_system1(Path(args.system1_ckpt), device)
    print(f"[INFO] System1 loaded (frozen): {args.system1_ckpt}")

    # ── Stage 2: Build rule memory ──────────────────────────
    memory = ICRLRuleMemory(
        concept_dim  = CONCEPT_TOTAL_DIM,
        theta        = args.theta,
        theta_merge  = args.theta_merge,
        n_min        = args.n_min,
        conf_min     = args.conf_min,
        device       = str(device),
    )

    print(f"\n[Stage 2] Building rule memory ({args.epochs} epochs)")
    head = None   # head không có ở epoch đầu

    for epoch in range(1, args.epochs + 1):
        print(f"\n  Epoch {epoch}/{args.epochs}")

        # Pass training set
        stats = build_rule_memory(
            system1, train_loader, memory, device,
            use_hard=args.use_hard_cv,
            epoch_label=f"{epoch}/{args.epochs}",
        )
        print(f"  Created={stats['created']}  Matched={stats['matched']}  "
              f"Rules so far={memory.num_rules}")

        # Update accuracy nếu đã có head
        if head is not None:
            update_rule_accuracy(system1, head, train_loader, memory, device, args.use_hard_cv)

        # Prune: Stage 2 chỉ dùng coherence (accuracy signal chưa ổn định)
        memory.prune(verbose=True, coherence_only=True)
        print(f"  After prune: {memory.num_rules} rules")
        print(f"  Confidence: "
              f"mean={sum(memory.get_confidences())/max(1,memory.num_rules):.3f}  "
              f"min={min(memory.get_confidences()):.3f}  "
              f"max={max(memory.get_confidences()):.3f}")

        # Quick head để cung cấp accuracy signal cho epoch tiếp theo.
        # Dùng head(cv) — KHÔNG phải head(centroid).
        # Lý do: head(centroid) cho acc=100% mọi rule (circular, centroid
        # đã encode digit3 đúng) → signal không phân biệt rules tốt/xấu.
        # head(cv) đo khả năng predict từ noisy concept vector thực →
        # rules có concept coherent thì acc cao → survive prune đúng.
        if epoch < args.epochs:
            head = nn.Linear(CONCEPT_TOTAL_DIM, args.num_classes).to(device)
            head_opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

            with torch.no_grad():
                all_cvs, all_ys = [], []
                for images, labels in train_loader:
                    images = images.to(device)
                    s1_out = system1(images)
                    cv = (logits_to_concept_vector(s1_out) if args.use_hard_cv
                          else soft_concept_vector(s1_out))
                    all_cvs.append(cv)
                    all_ys.append(labels["digit3"])

            all_cvs = torch.cat(all_cvs).to(device)
            all_ys  = torch.cat(all_ys).to(device)

            head.train()
            for _ in range(5):
                perm = torch.randperm(len(all_cvs), device=device)
                for i in range(0, len(all_cvs), args.batch_size):
                    idx  = perm[i:i + args.batch_size]
                    loss = F.cross_entropy(head(all_cvs[idx]), all_ys[idx])
                    head_opt.zero_grad(); loss.backward(); head_opt.step()

    # Save rule memory after building
    memory_path = output_dir / "icrl_rule_memory.pt"
    memory.save(memory_path)
    print(f"\n[INFO] Rule memory saved: {memory_path}  ({memory.num_rules} rules)")

    # ── Stage 3: Train prediction head ──────────────────────
    head = train_head(
        system1, memory, train_loader, val_loader,
        num_classes=args.num_classes,
        epochs=args.head_epochs,
        lr=args.head_lr,
        device=device,
        use_hard=args.use_hard_cv,
    )

    # Save head
    torch.save(head.state_dict(), output_dir / "prediction_head.pt")

    # ── Evaluate ────────────────────────────────────────────
    print("\n[INFO] Evaluating...")
    test_metrics = evaluate(system1, head, test_loader, device, "test", args.use_hard_cv)
    val_metrics  = evaluate(system1, head, val_loader,  device, "val",  args.use_hard_cv)

    print(f"\n[DONE] Results:")
    print(f"  val_accuracy  = {val_metrics['accuracy']:.4f}")
    print(f"  test_accuracy = {test_metrics['accuracy']:.4f}")

    # ── Export rules ─────────────────────────────────────────
    export_rules(memory, output_dir)

    # ── Save metrics ─────────────────────────────────────────
    metrics = {
        "val_accuracy":  val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "num_rules":     memory.num_rules,
        "args": vars(args),
        "rule_confidence_stats": {
            "mean": sum(memory.get_confidences()) / max(1, memory.num_rules),
            "min":  min(memory.get_confidences()) if memory.num_rules else 0,
            "max":  max(memory.get_confidences()) if memory.num_rules else 0,
        }
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[INFO] Results saved to {output_dir}/metrics.json")


if __name__ == "__main__":
    main()