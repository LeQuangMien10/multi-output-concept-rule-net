"""
train_system2.py
----------------
Training System 2 với GROUND TRUTH concept vectors (teacher forcing).

Pipeline:
  1. Load dataset → lấy (image, concept_vector, label)
  2. Nếu System 1 đã train → dùng System 1 extract concept_vec
     Nếu chưa → dùng ground truth concept one-hot làm concept_vec
  3. Khởi tạo RuleMemory từ concept vectors (kmeans hoặc random)
  4. Train System2Rules bằng soft forward + cls_loss + diversity_loss
  5. Save checkpoint

Giả định:
  - Dataset trả về (image, concept_labels_dict, class_label)
  - concept_labels_dict: {"d1": int, "o1": int, "d2": int, ...}
  - Có sẵn hàm build_concept_vector() để chuyển dict → one-hot vector
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional, Callable
import numpy as np

from src.models.system2_model import System2Rules
from src.models.rule_memory import RuleMemory


# ======================================================================
# Helper: build concept vector từ ground truth labels
# ======================================================================

def build_concept_vector(
    labels_dict: Dict[str, torch.Tensor],
    concept_groups: Dict[str, Tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """
    Chuyển dict nhãn ground truth thành one-hot concept vector.

    Args:
        labels_dict   : {"d1": [B] int, "o1": [B] int, ...}
        concept_groups: {"d1": (start, size), ...}
        device        : torch device

    Returns:
        [B, concept_dim] float tensor (one-hot concatenated)
    """
    B = next(iter(labels_dict.values())).shape[0]
    concept_dim = sum(size for _, size in concept_groups.values())
    vec = torch.zeros(B, concept_dim, device=device)

    for group_name, (start, size) in concept_groups.items():
        if group_name in labels_dict:
            idx = labels_dict[group_name].long().to(device)  # [B]
            # One-hot scatter
            vec[:, start : start + size].scatter_(1, idx.unsqueeze(1), 1.0)

    return vec


# ======================================================================
# Collect all concept vectors từ dataset (để init memory)
# ======================================================================

@torch.no_grad()
def collect_concept_vectors(
    dataloader: DataLoader,
    concept_groups: Dict[str, Tuple[int, int]],
    device: torch.device,
    system1: Optional[nn.Module] = None,
    max_samples: int = 10000,
) -> torch.Tensor:
    """
    Thu thập concept vectors từ toàn bộ dataset.
    Nếu system1 không None → dùng System 1 predictions.
    Ngược lại → dùng ground truth one-hot.

    Returns: [N, concept_dim]
    """
    all_vecs = []
    collected = 0

    for batch in tqdm(dataloader, desc="Collecting concept vectors"):
        images = batch["image"].to(device)
        B = images.shape[0]

        if system1 is not None:
            # Dùng System 1 predictions
            system1.eval()
            preds = system1(images)  # dict {group_name: [B, size] logits}
            vec_parts = []
            for group_name, (start, size) in concept_groups.items():
                if group_name in preds:
                    logits = preds[group_name]            # [B, size]
                    vec_parts.append(logits.cpu())
            vec = torch.cat(vec_parts, dim=-1)           # [B, concept_dim]
        else:
            # Ground truth one-hot
            labels_dict = {k: batch[k] for k in concept_groups if k in batch}
            vec = build_concept_vector(labels_dict, concept_groups, device).cpu()

        all_vecs.append(vec)
        collected += B
        if collected >= max_samples:
            break

    return torch.cat(all_vecs, dim=0)[:max_samples]


# ======================================================================
# Training loop
# ======================================================================

def train_system2(
    system2: System2Rules,
    dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    concept_groups: Dict[str, Tuple[int, int]],
    output_group: str,
    device: torch.device,
    num_epochs: int = 50,
    lr: float = 1e-3,
    diversity_weight: float = 0.01,
    system1: Optional[nn.Module] = None,
    save_dir: str = "checkpoints/system2",
    log_interval: int = 10,
) -> Dict[str, List[float]]:
    """
    Main training loop cho System 2.

    Args:
        system2         : System2Rules model
        dataloader      : training dataloader
        val_dataloader  : validation dataloader (optional)
        concept_groups  : {name: (start, size)}
        output_group    : tên group label
        device          : torch device
        num_epochs      : số epochs
        lr              : learning rate
        diversity_weight: weight cho diversity loss
        system1         : nếu có → dùng system1 predictions thay GT concepts
        save_dir        : nơi lưu checkpoint
        log_interval    : log mỗi bao nhiêu batch

    Returns:
        history: {"train_loss": [...], "train_acc": [...], "val_acc": [...]}
    """
    system2 = system2.to(device)
    optimizer = optim.Adam(system2.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    os.makedirs(save_dir, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        system2.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            labels = batch[output_group].long().to(device)   # [B]

            # --- Tạo concept vector ---
            if system1 is not None:
                # System 1 predictions
                system1.eval()
                with torch.no_grad():
                    preds = system1(images)
                concept_vec = torch.cat(
                    [preds[g] for g in concept_groups], dim=-1
                ).to(device)
            else:
                # Ground truth teacher forcing
                labels_dict = {k: batch[k] for k in concept_groups if k in batch}
                concept_vec = build_concept_vector(labels_dict, concept_groups, device)

            # --- Forward + loss ---
            optimizer.zero_grad()
            loss_dict = system2.compute_loss(
                concept_vec=concept_vec,
                labels=labels,
                use_soft=True,
                diversity_weight=diversity_weight,
            )
            loss = loss_dict["loss"]
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(system2.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += loss_dict["acc"].item()
            n_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                avg_loss = epoch_loss / n_batches
                avg_acc = epoch_acc / n_batches
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{avg_acc:.3f}",
                    "cls": f"{loss_dict['cls_loss'].item():.4f}",
                    "div": f"{loss_dict['div_loss'].item():.4f}",
                })

        scheduler.step()

        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_acc = epoch_acc / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)

        # --- Validation ---
        val_acc = 0.0
        if val_dataloader is not None:
            val_acc = evaluate_system2_accuracy(
                system2, val_dataloader, concept_groups, output_group,
                device, system1=system1
            )
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                system2.save(os.path.join(save_dir, "best"))
                print(f"  ✓ New best val_acc = {val_acc:.4f} → saved")

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"loss={avg_train_loss:.4f} | train_acc={avg_train_acc:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    # Save final checkpoint
    system2.save(os.path.join(save_dir, "final"))
    print(f"\n[train_system2] Training done. Best val_acc = {best_val_acc:.4f}")

    return history


# ======================================================================
# Evaluate accuracy (helper dùng chung cho train và evaluate scripts)
# ======================================================================

@torch.no_grad()
def evaluate_system2_accuracy(
    system2: System2Rules,
    dataloader: DataLoader,
    concept_groups: Dict[str, Tuple[int, int]],
    output_group: str,
    device: torch.device,
    system1: Optional[nn.Module] = None,
    use_gt_concepts: bool = False,
) -> float:
    """
    Tính accuracy của System 2.

    Args:
        use_gt_concepts: True → dùng GT concepts (upper bound);
                         False → dùng System 1 predictions (thực tế)
    """
    system2.eval()
    correct = 0
    total = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch[output_group].long().to(device)
        B = images.shape[0]

        if use_gt_concepts or system1 is None:
            labels_dict = {k: batch[k] for k in concept_groups if k in batch}
            concept_vec = build_concept_vector(labels_dict, concept_groups, device)
        else:
            system1.eval()
            preds = system1(images)
            concept_vec = torch.cat(
                [preds[g] for g in concept_groups], dim=-1
            ).to(device)

        out = system2.forward(concept_vec)
        pred_labels = out["logits"].argmax(dim=-1)
        correct += (pred_labels == labels).sum().item()
        total += B

    return correct / max(total, 1)


# ======================================================================
# Main entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Train System 2 (Rule-based)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--system1_ckpt", type=str, default=None,
                        help="Path to System 1 checkpoint (optional)")
    parser.add_argument("--save_dir", type=str, default="checkpoints/system2")
    parser.add_argument("--num_rules", type=int, default=None,
                        help="Override num_rules from config")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--init_strategy", type=str, default="kmeans",
                        choices=["kmeans", "random_sample"])
    parser.add_argument("--use_gt_concepts", action="store_true",
                        help="Use GT concepts instead of System 1 predictions")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_system2] Device: {device}")

    # Lấy concept_groups từ config
    # Expected format trong config:
    # concept_groups:
    #   d1: [0, 10]
    #   o1: [10, 4]
    #   d2: [14, 10]
    #   o2: [24, 4]
    #   d3: [28, 10]
    #   output: [38, 2]
    concept_groups_raw = cfg["concept_groups"]
    concept_groups = {
        k: tuple(v) for k, v in concept_groups_raw.items()
    }

    output_group = cfg.get("output_group", "output")
    num_rules = args.num_rules or cfg.get("system2", {}).get("num_rules", 100)
    num_epochs = args.epochs or cfg.get("system2", {}).get("num_epochs", 50)
    lr = args.lr or cfg.get("system2", {}).get("lr", 1e-3)
    diversity_weight = cfg.get("system2", {}).get("diversity_weight", 0.01)

    # ---- Load dataset ----
    # Người dùng cần implement hàm get_dataloaders() phù hợp với project
    # Đây là placeholder interface
    try:
        from src.data.dataset import get_dataloaders
        train_loader, val_loader = get_dataloaders(cfg, split="train"), \
                                   get_dataloaders(cfg, split="val")
    except ImportError:
        raise RuntimeError(
            "Cần implement src/data/dataset.py với hàm get_dataloaders(cfg, split)"
        )

    # ---- Load System 1 (optional) ----
    system1 = None
    if args.system1_ckpt and not args.use_gt_concepts:
        try:
            from src.models.multi_output_net import MultiOutputNet
            system1 = MultiOutputNet.load(args.system1_ckpt)
            system1 = system1.to(device)
            system1.eval()
            print(f"[train_system2] Loaded System 1 from {args.system1_ckpt}")
        except Exception as e:
            print(f"[train_system2] Warning: Could not load System 1: {e}")
            print("[train_system2] Falling back to GT concepts")

    # ---- Collect concept vectors để init memory ----
    print(f"[train_system2] Collecting concept vectors for memory init...")
    concept_vectors = collect_concept_vectors(
        train_loader, concept_groups, device,
        system1=system1 if not args.use_gt_concepts else None,
        max_samples=5000,
    )
    print(f"[train_system2] Collected {concept_vectors.shape[0]} concept vectors")

    # ---- Khởi tạo System 2 ----
    system2 = System2Rules(
        num_rules=num_rules,
        concept_groups=concept_groups,
        output_group=output_group,
        match_mode=cfg.get("system2", {}).get("match_mode", "cosine"),
        temperature=cfg.get("system2", {}).get("temperature", 1.0),
        diversity_weight=diversity_weight if False else None,  # handled in compute_loss
    )

    # Init memory từ data
    print(f"[train_system2] Initializing {num_rules} rule prototypes via {args.init_strategy}...")
    system2.init_memory_from_data(concept_vectors, strategy=args.init_strategy)

    # Set concept labels nếu có trong config
    if "concept_labels" in cfg:
        system2.set_concept_labels(cfg["concept_labels"])

    print(f"[train_system2] System 2 initialized: {num_rules} rules, "
          f"concept_dim={system2.concept_dim}")

    # ---- Training ----
    history = train_system2(
        system2=system2,
        dataloader=train_loader,
        val_dataloader=val_loader,
        concept_groups=concept_groups,
        output_group=output_group,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
        diversity_weight=diversity_weight,
        system1=system1 if not args.use_gt_concepts else None,
        save_dir=args.save_dir,
    )

    print("\n[train_system2] Done!")
    print(f"  Best val_acc: {max(history['val_acc']) if history['val_acc'] else 'N/A':.4f}")


if __name__ == "__main__":
    main()