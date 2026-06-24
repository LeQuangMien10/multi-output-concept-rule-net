"""
rule_matching.py  (v3 — prototype-based scoring)
=================================================
Bottleneck 2 fix: thay thế mask-based scoring bằng prototype cosine scoring.

Vấn đề cũ:
    score = concept_vec @ rule_masks.T
    rule_masks = sigmoid(rule_logits)   ← chỉ là binary gate, không encode giá trị
    → gradient từ score KHÔNG chảy vào rule_slot_logits (prototype)
    → rule_slot_logits và rule_masks diverge, không đồng bộ

Giải pháp mới:
    score = slot_wise_cosine(concept_vec, rule_proto_cv)
    rule_proto_cv = concat(softmax(rule_slot_logits[k]) for k in slots)
    → gradient từ score chảy trực tiếp vào rule_slot_logits
    → một tham số (rule_slot_logits) vừa làm scoring vừa làm prediction

Slot-wise cosine (không phải flat cosine):
    Mỗi slot k đóng góp đều nhau 1/6 vào tổng score,
    bất kể dim_k (digit dim=10 không lấn át op dim=5).
    Phù hợp với thiết lý "mọi concept slot ngang nhau".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rule_memory import (
    CONCEPT_KEYS_ORDERED,
    CONCEPT_OFFSETS,
    CONCEPT_DIMS,
)


# ─────────────────────────────────────────────────────────────
# Prototype-based scoring (THAY THẾ mask-based scoring)
# ─────────────────────────────────────────────────────────────

def slot_wise_cosine_scores(
    concept_vec:   torch.Tensor,
    rule_proto_cv: torch.Tensor,
) -> torch.Tensor:
    """
    Tính slot-wise cosine similarity giữa concept vector và rule prototypes.
    Mỗi slot đóng góp đều nhau vào tổng score (1/num_slots).

    Parameters
    ----------
    concept_vec   : FloatTensor[B, 42]  — softmax probs từ System1
    rule_proto_cv : FloatTensor[R, 42]  — concat softmax probs từ rule_slot_logits

    Returns
    -------
    scores : FloatTensor[B, R]
        scores[b, r] = mean cosine similarity qua 6 slots
    """
    num_slots = len(CONCEPT_KEYS_ORDERED)
    B = concept_vec.shape[0]
    R = rule_proto_cv.shape[0]
    total = torch.zeros(B, R, device=concept_vec.device)

    for key in CONCEPT_KEYS_ORDERED:
        offset = CONCEPT_OFFSETS[key]
        dim    = CONCEPT_DIMS[key]
        cv_s   = F.normalize(concept_vec[:, offset: offset + dim],   dim=1)  # [B, dim]
        rv_s   = F.normalize(rule_proto_cv[:, offset: offset + dim], dim=1)  # [R, dim]
        total += cv_s @ rv_s.T                                                # [B, R]

    return total / num_slots   # [B, R], range ≈ [-1, 1]


def flat_cosine_scores(
    concept_vec:   torch.Tensor,
    rule_proto_cv: torch.Tensor,
) -> torch.Tensor:
    """
    Flat cosine trên toàn bộ 42-dim (alternative, ít preferred hơn slot_wise).
    """
    cv_n = F.normalize(concept_vec,   dim=1)  # [B, 42]
    rv_n = F.normalize(rule_proto_cv, dim=1)  # [R, 42]
    return cv_n @ rv_n.T                       # [B, R]


# ─────────────────────────────────────────────────────────────
# Hard activation (inference)
# ─────────────────────────────────────────────────────────────

def slot_wise_match_detail(
    concept_vec:   torch.Tensor,
    rule_proto_cv: torch.Tensor,
    threshold:     float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-slot cosine score và boolean match cho mỗi cặp (sample, rule).
    Dùng khi inference để giải thích tại sao rule được chọn.

    Parameters
    ----------
    concept_vec   : FloatTensor[B, 42]
    rule_proto_cv : FloatTensor[R, 42]
    threshold     : float — ngưỡng cosine để coi là slot "khớp"

    Returns
    -------
    slot_scores : FloatTensor[B, R, num_slots]  — cosine per slot
    slot_match  : BoolTensor[B, R, num_slots]   — score >= threshold
    """
    B = concept_vec.shape[0]
    R = rule_proto_cv.shape[0]
    S = len(CONCEPT_KEYS_ORDERED)

    slot_scores = torch.zeros(B, R, S, device=concept_vec.device)

    for s, key in enumerate(CONCEPT_KEYS_ORDERED):
        offset = CONCEPT_OFFSETS[key]
        dim    = CONCEPT_DIMS[key]
        cv_s   = F.normalize(concept_vec[:, offset: offset + dim],   dim=1)
        rv_s   = F.normalize(rule_proto_cv[:, offset: offset + dim], dim=1)
        slot_scores[:, :, s] = cv_s @ rv_s.T

    slot_match = slot_scores >= threshold
    return slot_scores, slot_match


def predict_best_rule(
    concept_vec:   torch.Tensor,
    rule_proto_cv: torch.Tensor,
    temperature:   float = 0.5,
) -> torch.Tensor:
    """
    Chọn rule có score cao nhất cho mỗi sample.

    Returns
    -------
    best_rule_idx : LongTensor[B]
    """
    scores = slot_wise_cosine_scores(concept_vec, rule_proto_cv)  # [B, R]
    return scores.argmax(dim=1)                                    # [B]


# ─────────────────────────────────────────────────────────────
# RuleMatcher module (dùng trong System2)
# ─────────────────────────────────────────────────────────────

class RuleMatcher(nn.Module):
    """
    Module tính score match giữa concept vector và rule prototypes.

    Bottleneck 2 fix:
        - Nhận rule_proto_cv (từ System2Rules.get_rule_concept_vec())
          thay vì rule_masks từ RuleMemory.
        - Score = slot_wise_cosine(concept_vec, rule_proto_cv)
        - Gradient chảy thẳng vào rule_slot_logits.

    Parameters
    ----------
    score_mode     : "slot_cosine" (default) | "flat_cosine"
    hard_threshold : float — cosine threshold để coi slot là "khớp" khi inference
    """

    def __init__(
        self,
        score_mode:     str   = "slot_cosine",
        hard_threshold: float = 0.7,
    ):
        super().__init__()
        self.score_mode     = score_mode
        self.hard_threshold = hard_threshold

    def forward(
        self,
        concept_vec:   torch.Tensor,
        rule_proto_cv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Soft matching — dùng khi training.

        Parameters
        ----------
        concept_vec   : FloatTensor[B, 42]
        rule_proto_cv : FloatTensor[R, 42]  — từ System2Rules.get_rule_concept_vec()

        Returns
        -------
        scores : FloatTensor[B, R]
        """
        if self.score_mode == "slot_cosine":
            return slot_wise_cosine_scores(concept_vec, rule_proto_cv)
        elif self.score_mode == "flat_cosine":
            return flat_cosine_scores(concept_vec, rule_proto_cv)
        else:
            raise ValueError(f"Unknown score_mode: {self.score_mode!r}")

    @torch.no_grad()
    def predict(
        self,
        concept_vec:   torch.Tensor,
        rule_proto_cv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hard inference — chọn best rule và tính per-slot match detail.

        Returns
        -------
        best_rule_idx : LongTensor[B]
        slot_scores   : FloatTensor[B, R, num_slots]
        slot_match    : BoolTensor[B, R, num_slots]
        """
        best_rule_idx = predict_best_rule(concept_vec, rule_proto_cv)
        slot_scores, slot_match = slot_wise_match_detail(
            concept_vec, rule_proto_cv, self.hard_threshold
        )
        return best_rule_idx, slot_scores, slot_match