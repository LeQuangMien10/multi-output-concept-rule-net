"""
crl_system2.py — CRL-inspired System 2
========================================
Thay thế prototype cosine matching bằng differentiable logic layers,
lấy cảm hứng từ CRL (MICCAI 2025) nhưng thiết kế độc lập.

Kiến trúc:
    concept_vec [B, C]                   (C=40, softmax probs từ S1)
        ↓  rule_weights [R, C]           (random init, học từ data)
    rule_logit  [B, R] = cv @ W.T
        ↓  sigmoid
    rule_act    [B, R]                   (soft rule firing, 0-1)
        ↓  pred_head Linear(R, 10)
    digit3_pred [B, 10]                  (CE loss)

Điểm khác biệt so với prototype cosine (hiện tại):
    - Init: RANDOM N(0,0.1) — rule content emerge từ data
    - Không có temperature annealing
    - Không dùng cosine similarity → dùng dot product + sigmoid
    - rule_weights có thể âm (concept "không được có" trong rule)
    - Prediction = linear combination of rule activations, không phải
      weighted sum of prototypes

Loss:
    CE(digit3_pred, y)              task loss
    + λ₁ * L1(rule_weights)        sparsity: mỗi rule chọn ít concepts
    + λ₂ * diversity(rule_weights) rules học pattern khác nhau
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rule_memory import (
    CONCEPT_TOTAL_DIM,
    CONCEPT_KEYS_ORDERED,
    CONCEPT_DIMS,
    CONCEPT_OFFSETS,
)
from src.utils.symbols import ID_TO_SYMBOL


# ─────────────────────────────────────────────────────────────
# CRL System 2
# ─────────────────────────────────────────────────────────────

class CRLSystem2(nn.Module):
    """
    CRL-inspired differentiable rule learning.

    Parameters
    ----------
    num_rules     : int   — số rules (không biết trước, hyperparameter)
    concept_dim   : int   — chiều concept vector (40 cho MNIST Math v3)
    num_classes   : int   — số class output (10 cho digit3 ∈ {0..9})
    init_std      : float — std của random init cho rule_weights
    """

    def __init__(
        self,
        num_rules   : int   = 64,
        concept_dim : int   = CONCEPT_TOTAL_DIM,  # 40
        num_classes : int   = 10,
        init_std    : float = 0.1,
    ):
        super().__init__()

        self.num_rules   = num_rules
        self.concept_dim = concept_dim
        self.num_classes = num_classes

        # ── Rule weights: random init, fully learnable ────────
        # Shape [R, C] — mỗi hàng = 1 rule, mỗi cột = 1 concept
        # Dương: concept cần có mặt để rule fire
        # Âm:   concept không được có để rule fire
        self.rule_weights = nn.Parameter(
            torch.randn(num_rules, concept_dim) * init_std
        )

        # ── Prediction head: rule activations → digit3 ────────
        # Tách biệt khỏi rule_weights để 2 gradients độc lập
        self.pred_head = nn.Linear(num_rules, num_classes)

    # ── Forward ──────────────────────────────────────────────

    def forward(self, concept_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        concept_vec : FloatTensor[B, 40]

        Returns
        -------
        dict với keys:
            rule_logit    [B, R]   — raw score trước sigmoid
            rule_act      [B, R]   — soft rule firing (0-1)
            digit3_logit  [B, 10]  — dùng cho CE loss
        """
        # Dot product: how well concept_vec aligns with each rule
        rule_logit = concept_vec @ self.rule_weights.T   # [B, R]

        # Soft AND: sigmoid maps (-∞,+∞) → (0,1)
        # rule_act[b,r] ≈ 1: rule r fires strongly for sample b
        # rule_act[b,r] ≈ 0: rule r does not fire
        rule_act = torch.sigmoid(rule_logit)             # [B, R]

        # Linear prediction from rule activations
        digit3_logit = self.pred_head(rule_act)          # [B, 10]

        return {
            "rule_logit"   : rule_logit,
            "rule_act"     : rule_act,
            "digit3_logit" : digit3_logit,
        }

    # ── Loss ─────────────────────────────────────────────────

    @staticmethod
    def compute_loss(
        outputs          : dict[str, torch.Tensor],
        labels           : dict[str, torch.Tensor],
        rule_weights     : torch.Tensor,
        sparsity_weight  : float = 0.01,
        diversity_weight : float = 0.01,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Loss = task_CE + sparsity_L1 + diversity

        task_CE:
            CE(digit3_logit, labels['digit3'])

        sparsity_L1:
            mean(|rule_weights|)
            Ép rule_weights về 0 → mỗi rule chỉ "quan tâm"
            đến ít concepts (sparse rule condition).

        diversity:
            mean pairwise cosine similarity giữa các rules
            Ép rules học patterns khác nhau, tránh collapse.
        """
        digit3_logit = outputs["digit3_logit"]

        # 1. Task loss
        task_loss = F.cross_entropy(digit3_logit, labels["digit3"].long())

        # 2. Sparsity: L1 trên rule_weights
        sparsity_loss = rule_weights.abs().mean()

        # 3. Diversity: penalize similar rules
        rw_norm       = F.normalize(rule_weights, dim=1)      # [R, C]
        sim_mat       = rw_norm @ rw_norm.T                   # [R, R]
        R             = sim_mat.shape[0]
        upper         = torch.triu(
            torch.ones(R, R, device=sim_mat.device), diagonal=1
        ).bool()
        diversity_loss = sim_mat[upper].mean()

        total = (
            task_loss
            + sparsity_weight  * sparsity_loss
            + diversity_weight * diversity_loss
        )

        loss_dict = {
            "loss_total"    : total,
            "loss_task"     : task_loss.detach(),
            "loss_sparsity" : sparsity_loss.detach(),
            "loss_diversity": diversity_loss.detach(),
        }
        return total, loss_dict

    # ── Accuracy ─────────────────────────────────────────────

    @staticmethod
    def compute_accuracy(
        outputs : dict[str, torch.Tensor],
        labels  : dict[str, torch.Tensor],
    ) -> dict[str, float]:
        preds   = outputs["digit3_logit"].argmax(dim=1)
        targets = labels["digit3"].long()
        acc     = (preds == targets).float().mean().item()
        return {"digit3_acc": acc, "expression_acc": acc}

    # ── Interpretability ─────────────────────────────────────

    @torch.no_grad()
    def decode_rules(self, top_k: int = 3) -> list[dict]:
        """
        Decode learned rules thành dạng có thể đọc được.

        Với mỗi rule r:
            Tìm top_k concepts có |weight| lớn nhất
            Phân loại thành positive (w > 0) và negative (w < 0)

        Returns list[dict] với keys:
            rule_id, positive_concepts, negative_concepts, rule_string
        """
        W = self.rule_weights.detach()   # [R, C]
        decoded = []

        for r in range(self.num_rules):
            w_r    = W[r]                                    # [C]
            abs_w  = w_r.abs()
            top_idx = abs_w.topk(top_k).indices.tolist()

            pos_concepts = []
            neg_concepts = []

            for idx in top_idx:
                concept_name = _idx_to_concept_name(idx)
                weight_val   = w_r[idx].item()
                if weight_val > 0:
                    pos_concepts.append((concept_name, round(weight_val, 3)))
                else:
                    neg_concepts.append((concept_name, round(weight_val, 3)))

            # Build readable rule string
            parts = []
            for name, w in pos_concepts:
                parts.append(f"{name}")
            for name, w in neg_concepts:
                parts.append(f"NOT {name}")
            rule_string = " AND ".join(parts) if parts else "(no dominant concept)"

            # Prediction tendency
            pred_head_w = self.pred_head.weight[:, r].detach()  # [10]
            pred_class  = pred_head_w.argmax().item()

            decoded.append({
                "rule_id"          : r,
                "positive_concepts": pos_concepts,
                "negative_concepts": neg_concepts,
                "rule_string"      : rule_string,
                "predicts_digit3"  : pred_class,
                "pred_head_weight" : pred_head_w.tolist(),
            })

        return decoded

    @torch.no_grad()
    def infer(
        self,
        concept_vec: torch.Tensor,
        top_k_rules: int = 3,
    ) -> dict:
        """
        Inference với explanation.

        Returns
        -------
        dict:
            digit3_pred    LongTensor[B]
            digit3_prob    FloatTensor[B, 10]
            rule_act       FloatTensor[B, R]
            top_rules      list[list[int]]  — top firing rules per sample
        """
        out = self.forward(concept_vec)
        B   = concept_vec.shape[0]

        digit3_pred = out["digit3_logit"].argmax(dim=1)
        digit3_prob = F.softmax(out["digit3_logit"], dim=1)
        rule_act    = out["rule_act"]

        # Top-k firing rules per sample
        top_rules = [
            rule_act[b].topk(top_k_rules).indices.tolist()
            for b in range(B)
        ]

        return {
            "digit3_pred": digit3_pred,
            "digit3_prob": digit3_prob,
            "rule_act"   : rule_act,
            "top_rules"  : top_rules,
        }

    def extra_repr(self) -> str:
        return (
            f"num_rules={self.num_rules}, "
            f"concept_dim={self.concept_dim}, "
            f"num_classes={self.num_classes}"
        )


# ─────────────────────────────────────────────────────────────
# Concept index → human-readable name
# ─────────────────────────────────────────────────────────────

def _idx_to_concept_name(idx: int) -> str:
    """
    Chuyển index trong concept vector [0..39] thành tên dễ đọc.
    Layout: digit1[0:10], op1[10:15], digit2[15:25], op2[25:30], digit3[30:40]
    """
    for key in CONCEPT_KEYS_ORDERED:
        offset = CONCEPT_OFFSETS[key]
        dim    = CONCEPT_DIMS[key]
        if offset <= idx < offset + dim:
            local = idx - offset
            if key in ("op1", "op2"):
                label = ID_TO_SYMBOL.get(local, str(local))
                return f"{key}={label}"
            return f"{key}={local}"
    return f"concept[{idx}]"