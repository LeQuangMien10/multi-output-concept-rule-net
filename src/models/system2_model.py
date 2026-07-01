"""
system2_model.py  (dataset v2 — "a + b = ?")
=============================================
Cập nhật cho dataset v2:
    - 5 concept slots (bỏ "valid"): digit1, op1, digit2, op2, digit3
    - concept_dim: 40 thay vì 42
    - _enumerate_mnist_math_expressions: chỉ sinh valid expressions (55)
      không còn invalid peers (không cần thiết vì dataset chỉ có valid)
    - Rule được decode thành "a + b = c" thay vì có valid=0/1

Giữ nguyên toàn bộ logic:
    - Prototype cosine scoring (Bottleneck 2 fix)
    - Expression init với sharp=8.0 (Bottleneck 3 fix)
    - Cosine annealing temperature
    - Multi-objective loss (concept + recon + sparsity + coverage + diversity)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rule_memory import (
    RuleMemory,
    CONCEPT_TOTAL_DIM,
    CONCEPT_KEYS_ORDERED,
    CONCEPT_DIMS,
    soft_concept_vector,
    logits_to_concept_vector,
)
from src.models.rule_matching import RuleMatcher


# ─────────────────────────────────────────────────────────────
# Prototype initialization (dataset v2: chỉ valid expressions)
# ─────────────────────────────────────────────────────────────

def _enumerate_valid_expressions(op1_id: int = 0, op2_id: int = 4) -> list[tuple]:
    """
    Liệt kê 55 valid expressions: a + b = c  với a+b ≤ 9.
    Tuple: (digit1, op1, digit2, op2, digit3)
    — không có "valid" slot vì dataset v2 chỉ có valid expressions.
    """
    exprs = []
    for a in range(10):
        for b in range(10):
            c = a + b
            if c <= 9:
                exprs.append((a, op1_id, b, op2_id, c))
    return exprs  # 55 expressions


def _build_prototype_logits(
    num_rules:    int,
    concept_dims: dict,
    concept_keys: list,
    sharp:        float = 8.0,
    seed:         int   = 42,
) -> dict:
    """
    Khởi tạo rule_slot_logits từ 55 valid expressions.

    Với 128 rules và 55 expressions: pad bằng cách lặp lại.
    Kết quả: mỗi rule bắt đầu từ một expression cụ thể,
    score spread cao ngay từ epoch 1.
    """
    import random as _rng
    _rng.seed(seed)

    exprs = _enumerate_valid_expressions()
    # Pad đến num_rules nếu cần
    while len(exprs) < num_rules:
        exprs.append(_rng.choice(exprs))
    _rng.shuffle(exprs)
    exprs = exprs[:num_rules]

    logits = {k: torch.zeros(num_rules, concept_dims[k]) for k in concept_keys}
    for r, expr in enumerate(exprs):
        vals = dict(zip(concept_keys, expr))
        for key in concept_keys:
            logits[key][r, vals[key]] = sharp

    return logits


# ─────────────────────────────────────────────────────────────
# System2Rules
# ─────────────────────────────────────────────────────────────

class System2Rules(nn.Module):
    """
    System 2: học rule prototype cho 5 concept slots (dataset v2).

    concept_vec [B, 40]
        ↓  slot_wise_cosine vs rule_proto_cv [R, 40]
    rule_scores [B, R]
        ↓  softmax / temperature (cosine annealing)
    rule_assign [B, R]
        ↓  weighted sum
    pred_slot   dict key→[B, dim_k]
        ↓  NLL loss vs GT labels[digit1..digit3]
    """

    def __init__(
        self,
        num_rules:      int   = 55,
        concept_dim:    int   = CONCEPT_TOTAL_DIM,   # 40
        score_mode:     str   = "slot_cosine",
        temperature:    float = 2.0,
        hard_threshold: float = 0.7,
        init_sharp:     float = 8.0,
    ):
        super().__init__()

        self.num_rules   = num_rules
        self.concept_dim = concept_dim
        self.temperature = temperature
        self.init_sharp  = init_sharp

        # ── Rule prototypes ──────────────────────────────────
        init_logits = _build_prototype_logits(
            num_rules=num_rules,
            concept_dims=CONCEPT_DIMS,
            concept_keys=CONCEPT_KEYS_ORDERED,
            sharp=init_sharp,
        )
        self.rule_slot_logits = nn.ParameterDict({
            key: nn.Parameter(init_logits[key])
            for key in CONCEPT_DIMS
        })

        # ── Matcher ──────────────────────────────────────────
        self.matcher = RuleMatcher(
            score_mode=score_mode,
            hard_threshold=hard_threshold,
        )

        # ── RuleMemory (decode only) ─────────────────────────
        self.memory = RuleMemory(num_rules=num_rules, concept_dim=concept_dim)

    # ── Prototype helpers ────────────────────────────────────

    def get_rule_slot_probs(self) -> dict[str, torch.Tensor]:
        """key → FloatTensor[R, dim_k]"""
        return {
            key: F.softmax(self.rule_slot_logits[key], dim=-1)
            for key in CONCEPT_KEYS_ORDERED
        }

    def get_rule_concept_vec(self) -> torch.Tensor:
        """Concat prototype probs → [R, 40]"""
        return torch.cat(
            [F.softmax(self.rule_slot_logits[key], dim=-1)
             for key in CONCEPT_KEYS_ORDERED],
            dim=1,
        )

    # ── Forward ──────────────────────────────────────────────

    def forward(self, concept_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        concept_vec : FloatTensor[B, 40]

        Returns dict:
          rule_scores       [B, R]
          rule_assignment   [B, R]
          pred_slot_logits  dict key→[B, dim_k]
          pred_concept      [B, 40]
          rule_concept_vec  [R, 40]
        """
        rule_cv     = self.get_rule_concept_vec()
        rule_scores = self.matcher(concept_vec, rule_cv)
        rule_assign = F.softmax(rule_scores / self.temperature, dim=1)

        slot_probs = self.get_rule_slot_probs()
        pred_slot_logits = {
            key: rule_assign @ slot_probs[key]
            for key in CONCEPT_KEYS_ORDERED
        }

        pred_concept = torch.cat(
            [pred_slot_logits[k] for k in CONCEPT_KEYS_ORDERED], dim=1
        )

        return {
            "rule_scores"     : rule_scores,
            "rule_assignment" : rule_assign,
            "pred_slot_logits": pred_slot_logits,
            "pred_concept"    : pred_concept,
            "rule_concept_vec": rule_cv,
        }

    # ── Inference ────────────────────────────────────────────

    @torch.no_grad()
    def infer(self, concept_vec: torch.Tensor) -> dict:
        """
        Returns dict:
          best_rule_idx  LongTensor[B]
          pred_slot      dict key→LongTensor[B]
          rule_strings   list[str]  e.g. "3 + 5 = 8"
          slot_scores    FloatTensor[B, R, 5]
          slot_match     BoolTensor[B, R, 5]
          rule_scores    FloatTensor[B, R]
        """
        rule_cv = self.get_rule_concept_vec()

        best_rule_idx, slot_scores, slot_match = self.matcher.predict(
            concept_vec, rule_cv
        )
        rule_scores = self.matcher(concept_vec, rule_cv)

        slot_probs = self.get_rule_slot_probs()
        pred_slot = {
            key: slot_probs[key][best_rule_idx].argmax(dim=1)
            for key in CONCEPT_KEYS_ORDERED
        }

        rule_strings = [
            self.memory.decode_rule_from_probs(slot_probs, idx.item())
            for idx in best_rule_idx
        ]

        return {
            "best_rule_idx": best_rule_idx,
            "pred_slot"    : pred_slot,
            "rule_strings" : rule_strings,
            "slot_scores"  : slot_scores,
            "slot_match"   : slot_match,
            "rule_scores"  : rule_scores,
        }

    # ── Loss ─────────────────────────────────────────────────

    @staticmethod
    def compute_loss(
        outputs          : dict[str, torch.Tensor],
        concept_vec      : torch.Tensor,
        labels           : dict[str, torch.Tensor],
        concept_weight   : float = 1.0,
        recon_weight     : float = 0.3,
        sparsity_weight  : float = 0.05,
        coverage_weight  : float = 0.05,
        diversity_weight : float = 0.02,
        slot_weights     : dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Loss = concept + recon + sparsity + coverage + diversity
        Tất cả 5 concept slots ngang nhau (không có "valid").
        """
        pred_slot   = outputs["pred_slot_logits"]
        rule_assign = outputs["rule_assignment"]
        pred_cv     = outputs["pred_concept"]
        rule_cv     = outputs["rule_concept_vec"]

        if slot_weights is None:
            slot_weights = {k: 1.0 for k in CONCEPT_KEYS_ORDERED}

        # 1. Concept NLL loss (per slot)
        slot_losses = {}
        for key in CONCEPT_KEYS_ORDERED:
            log_pred       = (pred_slot[key] + 1e-8).log()
            slot_losses[key] = F.nll_loss(log_pred, labels[key].long())

        concept_loss = sum(
            slot_weights[k] * slot_losses[k] for k in CONCEPT_KEYS_ORDERED
        ) / sum(slot_weights.values())

        # 2. Reconstruction MSE
        recon_loss = F.mse_loss(pred_cv, concept_vec.detach())

        # 3. Sparsity: penalize uniform assignment
        entropy = -(rule_assign * (rule_assign + 1e-8).log()).sum(dim=1)
        sparsity_loss = entropy.mean()

        # 4. Coverage: penalize unused rules
        avg_assign    = rule_assign.mean(dim=0)
        uniform_thr   = 0.5 / rule_assign.shape[1]
        coverage_loss = F.relu(uniform_thr - avg_assign).mean()

        # 5. Diversity: penalize identical prototypes
        rv_norm        = F.normalize(rule_cv, dim=1)
        sim_mat        = rv_norm @ rv_norm.T
        R              = sim_mat.shape[0]
        upper          = torch.triu(torch.ones(R, R, device=sim_mat.device), diagonal=1).bool()
        diversity_loss = sim_mat[upper].mean()

        total = (
              concept_weight   * concept_loss
            + recon_weight     * recon_loss
            + sparsity_weight  * sparsity_loss
            + coverage_weight  * coverage_loss
            + diversity_weight * diversity_loss
        )

        loss_dict = {
            "loss_total"    : total,
            "loss_concept"  : concept_loss.detach(),
            "loss_recon"    : recon_loss.detach(),
            "loss_sparsity" : sparsity_loss.detach(),
            "loss_coverage" : coverage_loss.detach(),
            "loss_diversity": diversity_loss.detach(),
            **{f"loss_slot_{k}": slot_losses[k].detach() for k in CONCEPT_KEYS_ORDERED},
        }
        return total, loss_dict

    def extra_repr(self) -> str:
        return (
            f"num_rules={self.num_rules}, "
            f"concept_dim={self.concept_dim}, "
            f"temperature={self.temperature}"
        )


# ─────────────────────────────────────────────────────────────
# Accuracy helpers
# ─────────────────────────────────────────────────────────────

def compute_system2_accuracy(
    outputs: dict[str, torch.Tensor],
    labels : dict[str, torch.Tensor],
) -> dict[str, float]:
    """Per-slot accuracy + expression accuracy (tất cả 5 slots đúng)."""
    pred_slot = outputs["pred_slot_logits"]
    B = labels["digit1"].shape[0]
    device = labels["digit1"].device

    per_slot_correct: dict[str, torch.Tensor] = {}
    all_correct = torch.ones(B, dtype=torch.bool, device=device)

    for key in CONCEPT_KEYS_ORDERED:
        preds   = pred_slot[key].argmax(dim=1)
        targets = labels[key].long()
        correct = (preds == targets)
        per_slot_correct[key] = correct
        all_correct = all_correct & correct

    result = {
        f"{key}_acc": per_slot_correct[key].float().mean().item()
        for key in CONCEPT_KEYS_ORDERED
    }
    result["expression_acc"] = all_correct.float().mean().item()
    result["concept_acc"]    = sum(
        result[f"{k}_acc"] for k in CONCEPT_KEYS_ORDERED
    ) / len(CONCEPT_KEYS_ORDERED)
    return result


def system1_outputs_to_concept(
    s1_outputs: dict[str, torch.Tensor],
    soft: bool = True,
) -> torch.Tensor:
    if soft:
        return soft_concept_vector(s1_outputs)
    return logits_to_concept_vector(s1_outputs)