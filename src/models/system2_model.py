"""
system2_rules.py
================
Full System 2 model: nhận concept vector → học/activate rules
→ đưa ra dự đoán "valid/invalid" dựa trên rule.

Architecture:
    concept_vec [B, 42]
        ↓  RuleMatcher  (soft scores khi train, hard khi infer)
    scores      [B, R]
        ↓  softmax / linear head
    logits_valid [B, 2]   (valid/invalid classification)

Training objective:
    - Classification loss: softmax(scores) → valid label
    - Sparsity loss: khuyến khích mỗi ảnh khớp ít rule
    - Coverage loss: khuyến khích mỗi rule được dùng ít nhất 1 lần
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rule_memory import (
    RuleMemory,
    CONCEPT_TOTAL_DIM,
    labels_to_concept_vector,
    soft_concept_vector,
    logits_to_concept_vector,
    CONCEPT_OFFSETS,
    CONCEPT_DIMS,
    CONCEPT_KEYS_ORDERED,
)
from src.models.rule_matching import RuleMatcher


class System2Rules(nn.Module):
    """
    System 2: symbolic rule learner.

    Parameters
    ----------
    num_rules : int
        Số rule prototype cần học.
    concept_dim : int
        Kích thước concept vector (default 42).
    score_mode : str
        Cách tính match score: "dot" | "weighted" | "cosine".
    hard_threshold : float
        Ngưỡng activation khi inference.
    temperature : float
        Temperature của softmax trên rule scores → logits.
    """

    def __init__(
        self,
        num_rules: int = 64,
        concept_dim: int = CONCEPT_TOTAL_DIM,
        score_mode: str = "weighted",
        hard_threshold: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.num_rules = num_rules
        self.concept_dim = concept_dim
        self.temperature = temperature

        # RuleMemory: lưu các rule prototype
        self.memory = RuleMemory(
            num_rules=num_rules,
            concept_dim=concept_dim,
        )

        # RuleMatcher: tính score match
        self.matcher = RuleMatcher(
            memory=self.memory,
            score_mode=score_mode,
            hard_threshold=hard_threshold,
        )

        # Mỗi rule có label "valid" riêng (0 hoặc 1)
        # → học xem rule nào tương ứng valid/invalid expression
        self.rule_valid_logits = nn.Parameter(
            torch.zeros(num_rules, 2)
        )  # [R, 2]

        # Optional: projection head để map scores → valid logits
        self.score_to_valid = nn.Linear(num_rules, 2)

    # ── Forward (training) ───────────────────────────────────

    def forward(
        self,
        concept_vec: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        concept_vec : FloatTensor[B, 42]  — soft concept vector

        Returns dict:
            "valid_logits"   : [B, 2]   — dự đoán valid/invalid
            "rule_scores"    : [B, R]   — điểm khớp với từng rule
            "rule_assignment": [B, R]   — softmax weights
            "rule_valid_logits": [R, 2] — mỗi rule dự đoán valid/invalid
        """
        # [B, R] — soft match scores
        rule_scores = self.matcher(concept_vec)

        # Softmax → assignment distribution
        rule_assign = F.softmax(rule_scores / self.temperature, dim=1)  # [B, R]

        # --- Approach 1: weighted sum của rule_valid_logits
        # [B, R] x [R, 2] → [B, 2]
        valid_logits = rule_assign @ self.rule_valid_logits

        # --- Approach 2 (phụ): linear head từ scores
        valid_logits_alt = self.score_to_valid(rule_scores)

        return {
            "valid_logits": valid_logits,          # [B, 2]  — dùng để tính loss
            "valid_logits_alt": valid_logits_alt,  # [B, 2]  — alternative head
            "rule_scores": rule_scores,             # [B, R]
            "rule_assignment": rule_assign,         # [B, R]
            "rule_valid_logits": self.rule_valid_logits,  # [R, 2]
        }

    # ── Inference ────────────────────────────────────────────

    @torch.no_grad()
    def infer(
        self,
        concept_vec_hard: torch.Tensor,
        memory: "RuleMemory | None" = None,
    ) -> dict:
        """
        Hard inference: activate rules, chọn rule match 100%.

        Parameters
        ----------
        concept_vec_hard : FloatTensor[B, 42]  — hard one-hot concept

        Returns
        -------
        dict:
            "activated_rules" : list[list[int]]  — rule indices per sample
            "match_ratios"    : [B, R]
            "predicted_valid" : LongTensor[B]    — 0/1
            "rule_strings"    : list[list[str]]  — human-readable rules
        """
        mem = memory or self.memory
        rule_list, ratios = self.matcher.predict(concept_vec_hard)

        # Dự đoán valid từ rule được chọn
        B = concept_vec_hard.shape[0]
        predicted_valid = torch.zeros(B, dtype=torch.long,
                                      device=concept_vec_hard.device)

        for b, rules in enumerate(rule_list):
            if not rules:
                # Không có rule nào activate → fallback: argmax score
                scores = self.matcher(concept_vec_hard[b: b + 1])  # [1, R]
                best = scores.argmax(dim=1).item()
                rules = [best]

            # Lấy rule dự đoán valid từ rule đầu tiên active
            rule_idx = rules[0]
            pred_v = self.rule_valid_logits[rule_idx].argmax().item()
            predicted_valid[b] = pred_v

        # Decode rules sang string
        rule_strings: list[list[str]] = [
            [mem.decode_rule(r) for r in rules] if rules else ["(no rule)"]
            for rules in rule_list
        ]

        return {
            "activated_rules": rule_list,
            "match_ratios": ratios,
            "predicted_valid": predicted_valid,
            "rule_strings": rule_strings,
        }

    # ── Loss ────────────────────────────────────────────────

    @staticmethod
    def compute_loss(
        outputs: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        sparsity_weight: float = 0.01,
        coverage_weight: float = 0.01,
        use_alt_head: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Tổng loss = classification_loss + sparsity_loss + coverage_loss.

        Parameters
        ----------
        sparsity_weight  : hệ số phạt khi 1 ảnh khớp nhiều rule
        coverage_weight  : hệ số phạt khi rule không được dùng

        Returns
        -------
        (total_loss, loss_dict)
        """
        valid_target = labels["valid"]                # [B]

        # 1. Classification loss
        logits = (
            outputs["valid_logits_alt"] if use_alt_head
            else outputs["valid_logits"]
        )
        cls_loss = F.cross_entropy(logits, valid_target)

        # 2. Sparsity loss: entropy của rule assignment → khuyến khích peaked
        rule_assign = outputs["rule_assignment"]      # [B, R]
        entropy = -(rule_assign * (rule_assign + 1e-8).log()).sum(dim=1)
        sparsity_loss = entropy.mean()

        # 3. Coverage loss: mỗi rule nên được assign ít nhất 1 lần
        avg_assign = rule_assign.mean(dim=0)          # [R]
        # Phạt nếu avg_assign quá thấp (rule không được dùng)
        coverage_loss = F.relu(0.5 / rule_assign.shape[1] - avg_assign).mean()

        total = cls_loss + sparsity_weight * sparsity_loss + coverage_weight * coverage_loss

        loss_dict = {
            "loss_total": total,
            "loss_cls": cls_loss.detach(),
            "loss_sparsity": sparsity_loss.detach(),
            "loss_coverage": coverage_loss.detach(),
        }

        return total, loss_dict

    def extra_repr(self) -> str:
        return (
            f"num_rules={self.num_rules}, "
            f"concept_dim={self.concept_dim}, "
            f"temperature={self.temperature}"
        )


# ─────────────────────────────────────────────────────────────
# Convenience: build concept vector from System1 outputs
# ─────────────────────────────────────────────────────────────

def system1_outputs_to_concept(
    s1_outputs: dict[str, torch.Tensor],
    soft: bool = True,
) -> torch.Tensor:
    """
    Chuyển outputs của System1 sang concept vector cho System2.

    soft=True  → dùng softmax probs (khi training System2)
    soft=False → dùng argmax one-hot (khi inference)
    """
    if soft:
        return soft_concept_vector(s1_outputs)
    else:
        return logits_to_concept_vector(s1_outputs)