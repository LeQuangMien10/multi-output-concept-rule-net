"""
train_icrl_multiconcept.py — ICRL Stage 2/3 cho MNIST-MultiConcept
======================================================================

Giống train_icrl.py (MNIST Math) nhưng:
  - concept vector là 16 concept nhị phân ĐỘC LẬP (sigmoid), không phải 5 slot
    loại trừ nhau (softmax theo nhóm) -> cluster_dims=None dùng toàn bộ vector,
    không có khái niệm "input-only slots" vs "target slot" (đúng thiết lập
    Fitzpatrick: nhãn đích KHÔNG nằm trong concept vector).
  - nhãn đích (--target_key mặc định "label", 3 lớp non_neoplastic/benign/
    malignant) được sample từ softmax lúc sinh dataset -> cluster có thể
    KHÔNG thuần khiết 100% (khác MNIST Math nơi digit3 tất định từ d1/op/d2).
  - Stage 3 dùng đúng fix đã verify trên MNIST Math: train head trực tiếp
    trên R centroids với rule_labels = memory.get_labels() (majority-vote
    ground-truth), weight_decay=0, đủ step để đạt separable fit.

Usage (Kaggle mặc định, override --data_dir/--system1_ckpt nếu chạy local):
    python -m src.scripts.multiconcept.train_icrl \\
        --data_dir /kaggle/input/mnist-multiconcept \\
        --system1_ckpt /kaggle/working/outputs/multiconcept_system1/best_model.pt \\
        --output_dir /kaggle/working/outputs/multiconcept_icrl \\
        --theta 0.85 --theta_merge 0.95 --n_min 5 --conf_min 0.1 \\
        --epochs 3 --head_epochs 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.multiconcept.mnist_multiconcept_dataset import MNISTMultiConceptPTDataset
from src.models.multiconcept.system1 import (
    MultiConceptSystem1, soft_concept_vector, hard_concept_vector,
)
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.utils.seed import set_seed
from src.utils.multiconcept_concepts import CONCEPT_NAMES, NUM_CONCEPTS, LABEL_NAMES, NUM_LABELS


# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build ICRL rule memory cho MNIST-MultiConcept.")

    # Kaggle-first defaults — override bằng đường dẫn local nếu cần.
    p.add_argument("--data_dir",     type=str, default="/kaggle/input/mnist-multiconcept")
    p.add_argument("--system1_ckpt", type=str, default="/kaggle/working/outputs/multiconcept_system1/best_model.pt")
    p.add_argument("--output_dir",   type=str, default="/kaggle/working/outputs/multiconcept_icrl")

    p.add_argument("--theta",        type=float, default=0.85)
    p.add_argument("--theta_merge",  type=float, default=0.95)
    p.add_argument("--n_min",        type=int,   default=5)
    p.add_argument("--conf_min",     type=float, default=0.1)

    p.add_argument("--epochs",       type=int,   default=3,
                    help="Số lần pass qua training set để build rule memory (Stage 2).")
    p.add_argument("--use_hard_cv",  action="store_true")

    p.add_argument("--head_epochs",  type=int,   default=20)
    p.add_argument("--head_lr",      type=float, default=1e-3)
    p.add_argument("--head_steps_per_epoch", type=int, default=250,
                    help="Full-batch gradient step mỗi epoch khi train head trên R centroids.")
    p.add_argument("--num_classes",  type=int,   default=NUM_LABELS)

    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       type=str,   default="auto")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────

def load_system1(ckpt_path: Path, device: torch.device) -> MultiConceptSystem1:
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args  = ckpt.get("args", {})
    feature_dim = saved_args.get("feature_dim", 256)
    num_concepts = saved_args.get("num_concepts", NUM_CONCEPTS)
    model = MultiConceptSystem1(feature_dim=feature_dim, num_concepts=num_concepts)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_loaders(data_dir: Path, batch_size: int, num_workers: int):
    def _loader(split, shuffle):
        for fname in (["val.pt", "valid.pt"] if split == "val" else [f"{split}.pt"]):
            pt = data_dir / fname
            if pt.exists():
                ds = MNISTMultiConceptPTDataset(pt)
                return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=True)
        raise FileNotFoundError(f"No {split} split found in {data_dir}")

    return _loader("train", True), _loader("val", False), _loader("test", False)


# ─────────────────────────────────────────────────────────────
# Stage 2: Build rule memory
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def build_rule_memory(
    system1: MultiConceptSystem1,
    loader: DataLoader,
    memory: ICRLRuleMemory,
    device: torch.device,
    use_hard: bool = False,
    epoch_label: str = "Epoch",
) -> dict[str, int]:
    total_stats = {"created": 0, "matched": 0, "total": 0}

    for images, labels in tqdm(loader, desc=f"  Build [{epoch_label}]", leave=False):
        images = images.to(device)

        with torch.no_grad():
            concept_logits = system1(images)

        cv = hard_concept_vector(concept_logits) if use_hard else soft_concept_vector(concept_logits)

        # S1 confidence per sample: trung bình |sigmoid - 0.5| * 2 across concepts
        # (0 = hoàn toàn không chắc, 1 = tuyệt đối chắc) — thay cho max-softmax-prob
        # vốn chỉ áp dụng cho phân loại categorical.
        probs = torch.sigmoid(concept_logits)
        s1_conf = (2.0 * (probs - 0.5).abs()).mean(dim=1)   # [B]

        y = labels["label"].to(device)

        stats = memory.process_batch(cv, y, s1_conf)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    return total_stats


# ─────────────────────────────────────────────────────────────
# Stage 3: Train prediction head trực tiếp trên R rule centroids
# ─────────────────────────────────────────────────────────────

def train_head(
    system1:     MultiConceptSystem1,
    memory:      ICRLRuleMemory,
    val_loader:  DataLoader,
    num_classes: int,
    epochs:      int,
    lr:          float,
    device:      torch.device,
    use_hard:    bool = False,
    steps_per_epoch: int = 250,
) -> nn.Linear:
    """
    Xem giải thích chi tiết trong train_icrl.py::train_head (MNIST Math) —
    cùng nguyên tắc: nhãn train head = memory.get_labels() (majority-vote
    ground-truth per rule), weight_decay=0 để đạt separable fit trong ngân
    sách step hợp lý. Ở MultiConcept, nguyên tắc này càng bắt buộc vì nhãn
    đích không hề nằm trong concept vector — memory.get_labels() là nguồn
    nhãn DUY NHẤT khả dụng để train head.
    """
    head = nn.Linear(NUM_CONCEPTS, num_classes).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)

    best_val   = 0.0
    best_state = None

    centroids   = memory.get_centroids().to(device)
    rule_labels = torch.tensor(memory.get_labels(), dtype=torch.long, device=device)

    print(f"\n[Stage 3] Train prediction head trực tiếp trên {memory.num_rules} rule centroids "
          f"({epochs} epochs x {steps_per_epoch} steps)")
    print(f"  Nhãn: memory.get_labels() (majority-vote ground-truth mỗi rule)")

    for epoch in range(1, epochs + 1):
        head.train()
        for _ in range(steps_per_epoch):
            logits = head(centroids)
            loss   = F.cross_entropy(logits, rule_labels)
            opt.zero_grad(); loss.backward(); opt.step()
        train_acc = (logits.argmax(dim=1) == rule_labels).float().mean().item()

        head.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                concept_logits = system1(images)
                cv = hard_concept_vector(concept_logits) if use_hard else soft_concept_vector(concept_logits)
                y = labels["label"].to(device)
                rule_ids, _ = memory.match(cv)
                rule_cvs = centroids[rule_ids]
                preds = head(rule_cvs).argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total   += len(y)

        val_acc = val_correct / val_total
        print(f"  Ep {epoch:2d}/{epochs}: rule_train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

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
def evaluate(system1, head, loader, device, memory, split="test", use_hard=False):
    head.eval()
    correct = 0; total = 0
    centroids = memory.get_centroids().to(device)

    for images, labels in tqdm(loader, desc=f"  Eval {split}", leave=False):
        images = images.to(device)
        concept_logits = system1(images)
        cv = hard_concept_vector(concept_logits) if use_hard else soft_concept_vector(concept_logits)
        y = labels["label"].to(device)
        rule_ids, _ = memory.match(cv)
        rule_cvs = centroids[rule_ids]
        preds = head(rule_cvs).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += len(y)

    return {"accuracy": correct / total, "correct": correct, "total": total}


def export_rules(memory: ICRLRuleMemory, output_dir: Path, n_show: int = 20) -> None:
    concept_offsets = {name: i for i, name in enumerate(CONCEPT_NAMES)}
    concept_dims    = {name: 1 for name in CONCEPT_NAMES}

    rules_data = []
    for r in range(memory.num_rules):
        decoded = memory.decode_rule(
            rule_id=r,
            concept_keys=CONCEPT_NAMES,
            concept_offsets=concept_offsets,
            concept_dims=concept_dims,
            id_to_symbol=None,
        )
        present = [k for k, v in decoded["slots"].items() if v["value"] == "present"]
        decoded["label_name"] = LABEL_NAMES[decoded["label"]] if 0 <= decoded["label"] < NUM_LABELS else "?"
        decoded["present_concepts"] = present
        rules_data.append(decoded)

    rules_data.sort(key=lambda x: -x["confidence"])

    json_path = output_dir / "icrl_rules.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rules_data, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] {len(rules_data)} rules exported to {json_path}")

    print(f"\n[INFO] Top {min(n_show, len(rules_data))} rules (sorted by confidence):")
    for r in rules_data[:n_show]:
        bar = "█" * int(r["confidence"] * 20)
        concepts_str = "+".join(r["present_concepts"]) or "(none)"
        print(f"  [{r['confidence']:.3f}] {r['label_name']:15s}  n={r['n']:4d}  "
              f"coh={r['coherence']:.3f}  {concepts_str[:60]:60s}  {bar}")


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

    data_dir = Path(args.data_dir)
    train_loader, val_loader, test_loader = make_loaders(
        data_dir, args.batch_size, args.num_workers
    )
    print(f"[INFO] Data: {data_dir}")

    system1 = load_system1(Path(args.system1_ckpt), device)
    print(f"[INFO] System1 loaded (frozen): {args.system1_ckpt}")

    memory = ICRLRuleMemory(
        concept_dim  = NUM_CONCEPTS,
        theta        = args.theta,
        theta_merge  = args.theta_merge,
        n_min        = args.n_min,
        conf_min     = args.conf_min,
        cluster_dims = None,   # toàn bộ 16 concept — không có "target slot" để loại trừ
        device       = str(device),
    )

    print(f"\n[Stage 2] Building rule memory ({args.epochs} epochs)")
    for epoch in range(1, args.epochs + 1):
        print(f"\n  Epoch {epoch}/{args.epochs}")
        stats = build_rule_memory(
            system1, train_loader, memory, device,
            use_hard=args.use_hard_cv, epoch_label=f"{epoch}/{args.epochs}",
        )
        print(f"  Created={stats['created']}  Matched={stats['matched']}  "
              f"Rules so far={memory.num_rules}")

        memory.prune(verbose=True, conf_min_override=0.0)
        print(f"  After prune: {memory.num_rules} rules")
        cohs = [memory._compute_coherence(i) for i in range(memory.num_rules)]
        print(f"  Coherence: mean={sum(cohs)/max(1,len(cohs)):.3f}  "
              f"min={min(cohs) if cohs else 0:.3f}  max={max(cohs) if cohs else 0:.3f}")

    memory_path = output_dir / "icrl_rule_memory.pt"
    memory.save(memory_path)
    print(f"\n[INFO] Rule memory saved: {memory_path}  ({memory.num_rules} rules)")

    head = train_head(
        system1, memory, val_loader,
        num_classes=args.num_classes,
        epochs=args.head_epochs,
        lr=args.head_lr,
        device=device,
        use_hard=args.use_hard_cv,
        steps_per_epoch=args.head_steps_per_epoch,
    )
    torch.save(head.state_dict(), output_dir / "prediction_head.pt")

    print("\n[INFO] Evaluating...")
    test_metrics = evaluate(system1, head, test_loader, device, memory, "test", args.use_hard_cv)
    val_metrics  = evaluate(system1, head, val_loader,  device, memory, "val",  args.use_hard_cv)

    print(f"\n[DONE] Results:")
    print(f"  val_accuracy  = {val_metrics['accuracy']:.4f}")
    print(f"  test_accuracy = {test_metrics['accuracy']:.4f}")

    export_rules(memory, output_dir)

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
