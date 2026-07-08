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
    labels_to_concept_vector,
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
    p.add_argument("--theta_merge",  type=float, default=0.98,
                   help="Similarity threshold để MERGE 2 rules. Cao hơn theta để chỉ merge rules rất gần nhau.")
    p.add_argument("--n_min",        type=int,   default=2,
                   help="Số samples tối thiểu để rule survive prune. Thấp hơn → ít mất coverage.")
    p.add_argument("--conf_min",     type=float, default=0.05,
                   help="Confidence tối thiểu để rule survive prune.")
    p.add_argument("--match_input_only", action="store_true",
                   help="Dùng chỉ input slots (bỏ target slot và op2 trivial) cho MATCH/CREATE/MERGE.\n"
                        "MNIST: slot-wise cosine trên digit1+op1+digit2 (25 dims).\n"
                        "Fitzpatrick: đặt match_offsets thủ công qua --match_offsets_json.\n"
                        "Giúp clustering theo input pattern thay vì output label.")
    p.add_argument("--match_offsets_json", type=str, default=None,
                   help="JSON list of [start,end] pairs. Override --match_input_only.\n"
                        "Ví dụ: '[[0,10],[10,15],[15,25]]' cho digit1+op1+digit2.")

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
    p.add_argument("--target_key",   type=str,   default="digit3",
                   help="Key trong labels dict để lấy target. MNIST='digit3', Fitzpatrick='label'.")
    p.add_argument("--use_gt_concepts", action="store_true",
                   help="Dùng Ground Truth labels làm concept vector thay vì S1 predictions.\n"
                        "Loại bỏ hoàn toàn S1 noise — cho phép đo ceiling accuracy của ICRL framework\n"
                        "khi concept vector hoàn hảo. Hữu ích để debug: nếu GT vẫn thấp thì vấn đề\n"
                        "nằm ở framework, không phải S1 noise.")

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

def get_concept_vec(
    system1:       "MultiHeadSystem1",
    images:        torch.Tensor,
    labels:        dict,
    use_gt:        bool  = False,
    use_hard:      bool  = False,
    device:        torch.device | None = None,
) -> torch.Tensor:
    """
    Lấy concept vector theo mode được chọn:
      use_gt=True  : GT one-hot từ labels (loại bỏ S1 noise hoàn toàn)
      use_gt=False : S1 predictions (soft hoặc hard argmax)

    Dùng GT cho phép đo ceiling của ICRL framework độc lập với S1 quality.
    """
    if use_gt:
        return labels_to_concept_vector(
            {k: v.to(device or images.device) for k, v in labels.items()
             if k in CONCEPT_KEYS_ORDERED}
        ).float()
    # S1 prediction
    with torch.no_grad():
        s1_out = system1(images.to(device or images.device))
    return logits_to_concept_vector(s1_out) if use_hard else soft_concept_vector(s1_out)


@torch.no_grad()
def build_rule_memory(
    system1:    MultiHeadSystem1,
    loader:     DataLoader,
    memory:     ICRLRuleMemory,
    device:     torch.device,
    use_hard:   bool = False,
    epoch_label: str = "Epoch",
    target_key:  str = "digit3",
    s1_conf_keys: list[str] | None = None,
) -> dict[str, int]:
    """
    Pass qua loader một lần, update rule memory.

    target_key    : key trong labels dict để lấy y (MNIST='digit3', Fitzpatrick='label')
    s1_conf_keys  : list slots dùng để tính S1 confidence.
                    None → tự detect từ s1_out keys (bỏ qua op2 = trivial slot)
    """
    total_stats = {"created": 0, "matched": 0, "total": 0}

    use_gt = getattr(build_rule_memory, '_use_gt', False)

    for images, labels in tqdm(loader, desc=f"  Build [{epoch_label}]", leave=False):
        images = images.to(device)

        cv = get_concept_vec(system1, images, labels,
                             use_gt=use_gt, use_hard=use_hard, device=device)

        # S1 confidence:
        # GT mode: confidence=1.0 (perfect concept → full weight)
        # S1 mode: mean max-prob across non-trivial slots
        if use_gt:
            s1_conf = torch.ones(cv.shape[0], device=device)
        else:
            with torch.no_grad():
                s1_out_for_conf = system1(images)
            conf_keys = s1_conf_keys or [k for k in s1_out_for_conf if k != "op2"]
            slot_confs = [F.softmax(s1_out_for_conf[k], dim=1).max(dim=1).values
                          for k in conf_keys if k in s1_out_for_conf]
            s1_conf = (torch.stack(slot_confs, dim=1).mean(dim=1)
                       if slot_confs else torch.ones(cv.shape[0], device=device))

        # Target label
        if target_key in labels:
            y = labels[target_key].to(device)
        else:
            candidate_keys = [k for k in labels if k != "images"]
            y = labels[candidate_keys[0]].to(device)

        stats = memory.process_batch(cv, y, s1_conf)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    return total_stats


# ─────────────────────────────────────────────────────────────
# Stage 2b: Update accuracy (sau mỗi epoch, dùng current head)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def update_rule_accuracy(
    system1:    MultiHeadSystem1,
    head:       nn.Linear,
    loader:     DataLoader,
    memory:     ICRLRuleMemory,
    device:     torch.device,
    use_hard:   bool = False,
    target_key: str  = "digit3",
) -> None:
    """
    Compute predictions via centroid-based head và update accuracy trong memory.
    Dùng head(centroid[match(cv)]) — không phải head(cv) trực tiếp.
    """
    if memory.is_empty or head is None:
        return

    for i in range(memory.num_rules):
        memory._correct[i]    = 0
        memory._total_pred[i] = 0

    centroids = memory.get_centroids().to(device)   # [R, D]

    use_gt = getattr(update_rule_accuracy, '_use_gt', False)

    for images, labels in tqdm(loader, desc="  Update accuracy", leave=False):
        images = images.to(device)
        cv     = get_concept_vec(system1, images, labels,
                                 use_gt=use_gt, use_hard=use_hard, device=device)
        y      = labels[target_key].to(device) if target_key in labels else labels[list(labels.keys())[0]].to(device)

        rule_ids, _ = memory.match(cv)            # [B]
        rule_cvs    = centroids[rule_ids]          # [B, D]  ← centroid, not noisy cv
        preds       = head(rule_cvs).argmax(dim=1) # [B]

        memory.update_accuracy(cv, y, preds)


# ─────────────────────────────────────────────────────────────
# Stage 3: Train prediction head
# ─────────────────────────────────────────────────────────────

def train_head(
    system1:      MultiHeadSystem1,
    memory:       ICRLRuleMemory,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    num_classes:  int,
    epochs:       int,
    lr:           float,
    device:       torch.device,
    use_hard:     bool = False,
    target_key:   str  = "digit3",
    use_gt:       bool = False,
) -> nn.Linear:
    """
    Train linear head: centroid[match(cv)] → class.

    Key design: head nhận CENTROID của best-matched rule, không phải cv trực tiếp.
    Centroid = mean của ~1000+ ảnh → ít noise hơn single concept vector nhiều.
    Điều này giúp ICRL tạo ra sự khác biệt so với head(cv) thuần túy.

    General: target_key có thể là 'digit3' (MNIST) hoặc 'label' (Fitzpatrick).
    """
    head = nn.Linear(CONCEPT_TOTAL_DIM, num_classes).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    best_val   = 0.0
    best_state = None

    centroids = memory.get_centroids().to(device)   # [R, D]  — frozen trong suốt stage 3

    print(f"\n[Stage 3] Train prediction head ({epochs} epochs)")
    print(f"  Inference mode: head(centroid[match(cv)])  ← centroid reduces noise")

    def _get_target(labels):
        if target_key in labels:
            return labels[target_key].to(device)
        return labels[list(labels.keys())[0]].to(device)

    for epoch in range(1, epochs + 1):
        # ── Train ──
        head.train()
        train_correct = 0; train_total = 0

        for images, labels in tqdm(train_loader, desc=f"  Head ep{epoch:2d}", leave=False):
            images = images.to(device)
            with torch.no_grad():
                cv       = get_concept_vec(system1, images, labels,
                                           use_gt=use_gt, use_hard=use_hard, device=device)
                rule_ids, _ = memory.match(cv)
                rule_cvs = centroids[rule_ids]
            y = _get_target(labels)

            logits = head(rule_cvs)
            loss   = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total   += len(y)

        # ── Val ──
        head.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images   = images.to(device)
                cv       = get_concept_vec(system1, images, labels,
                                           use_gt=use_gt, use_hard=use_hard, device=device)
                y        = _get_target(labels)
                rule_ids, _ = memory.match(cv)
                rule_cvs = centroids[rule_ids]
                preds    = head(rule_cvs).argmax(dim=1)
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
    system1:    MultiHeadSystem1,
    head:       nn.Linear,
    loader:     DataLoader,
    device:     torch.device,
    memory:     ICRLRuleMemory,
    split:      str  = "test",
    use_hard:   bool = False,
    target_key: str  = "digit3",
) -> dict[str, float]:
    """
    Inference: head(centroid[match(cv)]) → prediction.
    Centroid-based inference — nhất quán với train_head.
    """
    head.eval()
    correct = 0; total = 0
    centroids = memory.get_centroids().to(device)   # [R, D]

    use_gt = getattr(evaluate, '_use_gt', False)

    for images, labels in tqdm(loader, desc=f"  Eval {split}", leave=False):
        images   = images.to(device)
        cv       = get_concept_vec(system1, images, labels,
                                   use_gt=use_gt, use_hard=use_hard, device=device)
        y        = labels[target_key].to(device) if target_key in labels else labels[list(labels.keys())[0]].to(device)
        rule_ids, _ = memory.match(cv)
        rule_cvs = centroids[rule_ids]
        preds    = head(rule_cvs).argmax(dim=1)
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
    use_gt = args.use_gt_concepts
    print(f"[INFO] ICRL params: θ={args.theta}  θ_merge={args.theta_merge}  "
          f"n_min={args.n_min}  conf_min={args.conf_min}  "
          f"match_input_only={args.match_input_only}")
    print(f"[INFO] Concept source: {'Ground Truth labels' if use_gt else 'System1 predictions'}")
    if use_gt:
        print(f"[INFO] GT mode: measuring ceiling accuracy of ICRL framework (no S1 noise)")

    # Propagate use_gt flag via function attribute (avoids threading issues)
    build_rule_memory._use_gt    = use_gt
    update_rule_accuracy._use_gt = use_gt
    evaluate._use_gt             = use_gt

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
    # ── Resolve match_offsets ───────────────────────────────
    match_offsets = None
    if args.match_offsets_json:
        import json as _json
        match_offsets = [tuple(p) for p in _json.loads(args.match_offsets_json)]
        print(f"[INFO] match_offsets (custom): {match_offsets}")
    elif args.match_input_only:
        # MNIST Math: digit1[0:10] + op1[10:15] + digit2[15:25]
        # Bỏ op2[25:30] (trivial, luôn =) và digit3[30:40] (target)
        # General: user chỉ định slot layout qua --match_offsets_json
        from src.models.rule_memory import CONCEPT_OFFSETS, CONCEPT_DIMS
        input_keys  = [k for k in CONCEPT_KEYS_ORDERED
                       if k not in ("op2", args.target_key)]
        match_offsets = [(CONCEPT_OFFSETS[k], CONCEPT_OFFSETS[k] + CONCEPT_DIMS[k])
                         for k in input_keys]
        print(f"[INFO] match_input_only: slots={input_keys}  offsets={match_offsets}")

    memory = ICRLRuleMemory(
        concept_dim   = CONCEPT_TOTAL_DIM,
        theta         = args.theta,
        theta_merge   = args.theta_merge,
        n_min         = args.n_min,
        conf_min      = args.conf_min,
        match_offsets = match_offsets,
        device        = str(device),
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
            target_key=args.target_key,
        )
        print(f"  Created={stats['created']}  Matched={stats['matched']}  "
              f"Rules so far={memory.num_rules}")

        # Update accuracy nếu đã có head
        if head is not None:
            update_rule_accuracy(system1, head, train_loader, memory, device,
                                 use_hard=args.use_hard_cv, target_key=args.target_key)

        # Prune
        memory.prune(verbose=True)
        print(f"  After prune: {memory.num_rules} rules")
        print(f"  Confidence: "
              f"mean={sum(memory.get_confidences())/max(1,memory.num_rules):.3f}  "
              f"min={min(memory.get_confidences()):.3f}  "
              f"max={max(memory.get_confidences()):.3f}")

        # Quick head để cung cấp accuracy signal cho epoch tiếp theo
        if epoch < args.epochs:
            # Quick head để cung cấp accuracy signal cho epoch tiếp theo
            # Dùng centroid-based inference — nhất quán với stage 3
            head = nn.Linear(CONCEPT_TOTAL_DIM, args.num_classes).to(device)
            head_opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

            # Collect centroids và labels (one-pass, không cần lưu toàn bộ cv)
            centroids_now = memory.get_centroids().to(device)  # [R, D]

            with torch.no_grad():
                all_rule_cvs, all_ys = [], []
                for images, batch_labels in train_loader:
                    images = images.to(device)
                    s1_out = system1(images)
                    cv     = get_concept_vec(system1, images, batch_labels,
                                               use_gt=use_gt, use_hard=args.use_hard_cv, device=device)
                    rule_ids, _ = memory.match(cv)            # [B]
                    rule_cvs    = centroids_now[rule_ids]     # [B, D]
                    all_rule_cvs.append(rule_cvs.cpu())

                    # Fix: _y nằm TRONG for loop
                    _y = (batch_labels[args.target_key]
                          if args.target_key in batch_labels
                          else batch_labels[list(batch_labels.keys())[0]])
                    all_ys.append(_y)

            all_rule_cvs = torch.cat(all_rule_cvs).to(device)
            all_ys       = torch.cat(all_ys).to(device)

            # Validate label range
            assert all_ys.min() >= 0 and all_ys.max() < args.num_classes, (
                f"Label out of range: [{all_ys.min()}, {all_ys.max()}] "
                f"but num_classes={args.num_classes}. Set --num_classes correctly."
            )

            head.train()
            for _ in range(5):
                perm = torch.randperm(len(all_rule_cvs), device=device)
                for i in range(0, len(all_rule_cvs), args.batch_size):
                    idx  = perm[i:i + args.batch_size]
                    loss = F.cross_entropy(head(all_rule_cvs[idx]), all_ys[idx])
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
        target_key=args.target_key,
        use_gt=use_gt,
    )

    # Save head
    torch.save(head.state_dict(), output_dir / "prediction_head.pt")

    # ── Evaluate ────────────────────────────────────────────
    print("\n[INFO] Evaluating...")
    test_metrics = evaluate(system1, head, test_loader, device, memory,
                            split="test", use_hard=args.use_hard_cv, target_key=args.target_key)
    val_metrics  = evaluate(system1, head, val_loader,  device, memory,
                            split="val",  use_hard=args.use_hard_cv, target_key=args.target_key)

    print(f"\n[DONE] Results:")
    print(f"  val_accuracy  = {val_metrics['accuracy']:.4f}")
    print(f"  test_accuracy = {test_metrics['accuracy']:.4f}")

    # ── Export rules ─────────────────────────────────────────
    export_rules(memory, output_dir)

    # ── Save metrics ─────────────────────────────────────────
    metrics = {
        "val_accuracy":    val_metrics["accuracy"],
        "test_accuracy":   test_metrics["accuracy"],
        "num_rules":       memory.num_rules,
        "concept_source":  "gt_labels" if use_gt else "system1_predictions",
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