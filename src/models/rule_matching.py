"""
rule_matching.py
================
Tính điểm match giữa concept vector (đầu ra System 1)
và các rule mask (từ RuleMemory).

Matching score cao → rule được "activate" bởi ảnh đó.

Hai mode chính:
    - soft  : dùng khi training (differentiable)
    - hard  : dùng khi inference (argmax / threshold)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rule_memory import (
    RuleMemory,
    CONCEPT_KEYS_ORDERED,
    CONCEPT_OFFSETS,
    CONCEPT_DIMS,
)


# ─────────────────────────────────────────────────────────────
# Core scoring functions
# ─────────────────────────────────────────────────────────────

def compute_match_scores(
    concept_vec: torch.Tensor,
    rule_masks: torch.Tensor,
    mode: str = "dot",
) -> torch.Tensor:
    """
    Tính điểm match giữa mỗi ảnh và mỗi rule.

    Parameters
    ----------
    concept_vec : FloatTensor[B, D]
        Concept vector của batch (soft one-hot hoặc hard one-hot).
    rule_masks : FloatTensor[R, D]
        Rule masks từ RuleMemory.rule_masks.
    mode : str
        "dot"      — tổng c_i * m_i  (unnormalized)
        "weighted" — dot / (sum of mask), tránh thiên về rule dài
        "cosine"   — cosine similarity

    Returns
    -------
    scores : FloatTensor[B, R]
        scores[b, r] = mức độ rule r match với ảnh b.
    """
    if mode == "dot":
        # [B, D] x [D, R] → [B, R]
        return concept_vec @ rule_masks.T

    elif mode == "weighted":
        dot = concept_vec @ rule_masks.T                              # [B, R]
        norm = rule_masks.sum(dim=1, keepdim=True).T.clamp(min=1e-6) # [1, R]
        return dot / norm

    elif mode == "cosine":
        cv_norm = F.normalize(concept_vec, dim=1)                     # [B, D]
        rm_norm = F.normalize(rule_masks, dim=1)                      # [R, D]
        return cv_norm @ rm_norm.T                                    # [B, R]

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'dot'/'weighted'/'cosine'.")


def slot_wise_match(
    concept_vec: torch.Tensor,
    rule_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Match từng slot riêng lẻ (digit1, op1, ..., valid).

    Returns
    -------
    slot_scores : FloatTensor[B, R, num_slots]
        slot_scores[b, r, s] = điểm match của slot s cho cặp (b, r).
    """
    B = concept_vec.shape[0]
    R = rule_masks.shape[0]
    S = len(CONCEPT_KEYS_ORDERED)

    slot_scores = torch.zeros(B, R, S, device=concept_vec.device)

    for s, key in enumerate(CONCEPT_KEYS_ORDERED):
        offset = CONCEPT_OFFSETS[key]
        dim = CONCEPT_DIMS[key]
        cv_slot = concept_vec[:, offset: offset + dim]      # [B, dim]
        rm_slot = rule_masks[:, offset: offset + dim]       # [R, dim]
        # dot product per slot: [B, dim] x [dim, R] → [B, R]
        slot_scores[:, :, s] = cv_slot @ rm_slot.T

    return slot_scores


# ─────────────────────────────────────────────────────────────
# Hard activation (inference)
# ─────────────────────────────────────────────────────────────

def activate_rules_hard(
    concept_vec_hard: torch.Tensor,
    rule_masks_hard: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dùng khi inference: một rule được "fully activated" khi
    mọi slot mà rule yêu cầu (mask=1) đều khớp với concept.

    Parameters
    ----------
    concept_vec_hard : FloatTensor[B, D]  — hard one-hot concept
    rule_masks_hard  : FloatTensor[R, D]  — binary mask (0/1)
    threshold        : float              — min activation ratio

    Returns
    -------
    activated   : BoolTensor[B, R]   — True nếu rule fully activated
    match_ratio : FloatTensor[B, R]  — tỷ lệ slot khớp trong rule
    """
    # Số slot rule yêu cầu: [R]
    rule_length = rule_masks_hard.sum(dim=1).clamp(min=1)

    # Số slot đúng: concept_vec AND rule_mask
    # concept_vec_hard[B, D] * rule_masks_hard[R, D] → phải broadcast
    hits = concept_vec_hard.unsqueeze(1) * rule_masks_hard.unsqueeze(0)  # [B, R, D]
    hits_per_rule = hits.sum(dim=2)                                       # [B, R]

    match_ratio = hits_per_rule / rule_length.unsqueeze(0)               # [B, R]
    activated = match_ratio >= threshold                                   # [B, R]

    return activated, match_ratio


def predict_rules(
    concept_vec_hard: torch.Tensor,
    rule_masks_hard: torch.Tensor,
    threshold: float = 1.0,
) -> list[list[int]]:
    """
    Với mỗi ảnh trong batch, trả về danh sách các rule index
    được fully activated.

    threshold=1.0 → yêu cầu 100% slot match (strict rule).
    """
    activated, _ = activate_rules_hard(
        concept_vec_hard, rule_masks_hard, threshold
    )
    # activated: [B, R]
    results: list[list[int]] = []
    for b in range(activated.shape[0]):
        active_rules = activated[b].nonzero(as_tuple=False).squeeze(-1).tolist()
        results.append(active_rules)
    return results


# ─────────────────────────────────────────────────────────────
# RuleMatcher module (dùng trong System2)
# ─────────────────────────────────────────────────────────────

class RuleMatcher(nn.Module):
    """
    Module ghép RuleMemory với logic matching.

    Parameters
    ----------
    memory : RuleMemory
    score_mode : str  — "dot" | "weighted" | "cosine"
    hard_threshold : float  — threshold cho hard activation
    """

    def __init__(
        self,
        memory: RuleMemory,
        score_mode: str = "weighted",
        hard_threshold: float = 1.0,
    ):
        super().__init__()
        self.memory = memory
        self.score_mode = score_mode
        self.hard_threshold = hard_threshold

    def forward(self, concept_vec: torch.Tensor) -> torch.Tensor:
        """
        Soft matching → dùng khi training.

        Returns
        -------
        scores : FloatTensor[B, R]
        """
        return compute_match_scores(
            concept_vec,
            self.memory.rule_masks,
            mode=self.score_mode,
        )

    @torch.no_grad()
    def predict(
        self,
        concept_vec_hard: torch.Tensor,
    ) -> tuple[list[list[int]], torch.Tensor]:
        """
        Hard matching → dùng khi inference.

        Returns
        -------
        activated_rules : list[list[int]]  — per sample
        match_ratios    : FloatTensor[B, R]
        """
        hard_masks = self.memory.get_hard_masks(self.hard_threshold)
        rule_list = predict_rules(
            concept_vec_hard, hard_masks, self.hard_threshold
        )
        _, ratios = activate_rules_hard(
            concept_vec_hard, hard_masks, self.hard_threshold
        )
        return rule_list, ratios