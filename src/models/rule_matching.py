"""
rule_matching.py  (v3 — prototype cosine, dataset v2 compatible)
=================================================================
Slot-wise cosine scoring giữa concept vector [B, 40] và rule prototypes [R, 40].
Mỗi trong 5 slots đóng góp đều nhau 1/5.
(Dataset v2: 5 slots thay vì 6 — không có "valid")
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


def slot_wise_cosine_scores(
    concept_vec:   torch.Tensor,
    rule_proto_cv: torch.Tensor,
) -> torch.Tensor:
    """
    Slot-wise cosine similarity, equal weight per slot (1/num_slots).

    concept_vec   : FloatTensor[B, 40]
    rule_proto_cv : FloatTensor[R, 40]
    returns       : FloatTensor[B, R]
    """
    num_slots = len(CONCEPT_KEYS_ORDERED)   # 5
    B = concept_vec.shape[0]
    R = rule_proto_cv.shape[0]
    total = torch.zeros(B, R, device=concept_vec.device)

    for key in CONCEPT_KEYS_ORDERED:
        offset = CONCEPT_OFFSETS[key]
        dim    = CONCEPT_DIMS[key]
        cv_s   = F.normalize(concept_vec[:, offset: offset + dim],   dim=1)
        rv_s   = F.normalize(rule_proto_cv[:, offset: offset + dim], dim=1)
        total += cv_s @ rv_s.T

    return total / num_slots


def flat_cosine_scores(
    concept_vec:   torch.Tensor,
    rule_proto_cv: torch.Tensor,
) -> torch.Tensor:
    cv_n = F.normalize(concept_vec,   dim=1)
    rv_n = F.normalize(rule_proto_cv, dim=1)
    return cv_n @ rv_n.T


def slot_wise_match_detail(
    concept_vec:   torch.Tensor,
    rule_proto_cv: torch.Tensor,
    threshold:     float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-slot cosine score và boolean match cho mỗi cặp (sample, rule).

    Returns
    -------
    slot_scores : FloatTensor[B, R, num_slots]
    slot_match  : BoolTensor[B, R, num_slots]
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
) -> torch.Tensor:
    scores = slot_wise_cosine_scores(concept_vec, rule_proto_cv)
    return scores.argmax(dim=1)


class RuleMatcher(nn.Module):
    def __init__(self, score_mode: str = "slot_cosine", hard_threshold: float = 0.7):
        super().__init__()
        self.score_mode     = score_mode
        self.hard_threshold = hard_threshold

    def forward(self, concept_vec: torch.Tensor, rule_proto_cv: torch.Tensor) -> torch.Tensor:
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
        best_rule_idx = predict_best_rule(concept_vec, rule_proto_cv)
        slot_scores, slot_match = slot_wise_match_detail(
            concept_vec, rule_proto_cv, self.hard_threshold
        )
        return best_rule_idx, slot_scores, slot_match