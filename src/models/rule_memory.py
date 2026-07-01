"""
rule_memory.py  (v2 — dataset v2 compatible)
=============================================
Cập nhật cho dataset v2 "a + b = ?":
    - Bỏ "valid" slot khỏi CONCEPT_KEYS_ORDERED
    - CONCEPT_TOTAL_DIM: 42 → 40
    - concept_vec: [B, 40] thay vì [B, 42]

Layout concept vector (40-dim):
    [0:10]   digit1  (softmax prob, dim=10)
    [10:15]  op1     (softmax prob, dim=5)
    [15:25]  digit2  (softmax prob, dim=10)
    [25:30]  op2     (softmax prob, dim=5)
    [30:40]  digit3  (softmax prob, dim=10)  ← target, nhưng vẫn là concept
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Concept layout constants  (dataset v2: không có "valid")
# ─────────────────────────────────────────────────────────────

CONCEPT_DIMS: dict[str, int] = {
    "digit1": 10,
    "op1":    5,
    "digit2": 10,
    "op2":    5,
    "digit3": 10,
}

CONCEPT_KEYS_ORDERED: list[str] = [
    "digit1", "op1", "digit2", "op2", "digit3"
]

CONCEPT_TOTAL_DIM: int = sum(CONCEPT_DIMS.values())  # 40

# Start index of each slot in the flat vector
CONCEPT_OFFSETS: dict[str, int] = {}
_off = 0
for _k in CONCEPT_KEYS_ORDERED:
    CONCEPT_OFFSETS[_k] = _off
    _off += CONCEPT_DIMS[_k]


# ─────────────────────────────────────────────────────────────
# Concept vector builders
# ─────────────────────────────────────────────────────────────

def labels_to_concept_vector(labels: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    GT labels → hard one-hot concept vector [B, 40].
    labels: dict key → LongTensor[B]
    Chỉ dùng các key trong CONCEPT_KEYS_ORDERED (bỏ qua "valid" nếu có).
    """
    B = labels["digit1"].shape[0]
    vec = torch.zeros(B, CONCEPT_TOTAL_DIM, device=labels["digit1"].device)
    for key in CONCEPT_KEYS_ORDERED:
        offset = CONCEPT_OFFSETS[key]
        dim    = CONCEPT_DIMS[key]
        idx    = labels[key].long()
        one_hot = torch.zeros(B, dim, device=idx.device)
        one_hot.scatter_(1, idx.unsqueeze(1), 1.0)
        vec[:, offset: offset + dim] = one_hot
    return vec


def logits_to_concept_vector(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    System1 logits → hard one-hot concept vector [B, 40] (argmax per slot).
    """
    fake_labels = {k: outputs[k].argmax(dim=1) for k in CONCEPT_KEYS_ORDERED}
    return labels_to_concept_vector(fake_labels)


def soft_concept_vector(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    System1 logits → soft concept vector [B, 40] (softmax probs, concat).
    Dùng khi training System2 để gradient flow ngược qua System1.
    """
    parts = [F.softmax(outputs[k], dim=-1) for k in CONCEPT_KEYS_ORDERED]
    return torch.cat(parts, dim=1)  # [B, 40]


# ─────────────────────────────────────────────────────────────
# RuleMemory — decode/interpretability only
# ─────────────────────────────────────────────────────────────

class RuleMemory(nn.Module):
    """
    Decode helper cho rule prototypes.
    Không tham gia scoring hay training — chỉ decode string.
    """

    def __init__(self, num_rules: int, concept_dim: int = CONCEPT_TOTAL_DIM):
        super().__init__()
        self.num_rules   = num_rules
        self.concept_dim = concept_dim
        self.rule_logits = nn.Parameter(
            torch.ones(num_rules, concept_dim),
            requires_grad=False,
        )

    @property
    def rule_masks(self) -> torch.Tensor:
        return torch.sigmoid(self.rule_logits)

    def get_hard_masks(self, threshold: float = 0.5) -> torch.Tensor:
        return (self.rule_masks > threshold).float()

    def forward(self) -> torch.Tensor:
        return self.rule_masks

    def decode_rule_from_probs(
        self,
        slot_probs: dict[str, torch.Tensor],
        rule_idx: int,
    ) -> str:
        """
        Decode rule → string dạng:
            "digit1=3 AND op1=+ AND digit2=5 AND op2== AND digit3=8"
        """
        from src.utils.symbols import ID_TO_SYMBOL, rule_to_string
        pred = {}
        for key in CONCEPT_KEYS_ORDERED:
            pred[key] = int(slot_probs[key][rule_idx].argmax().item())
        # Format: "a + b = c"
        return rule_to_string(
            pred["digit1"], pred["op1"], pred["digit2"], pred["digit3"]
        )

    def decode_all_rules_from_probs(
        self,
        slot_probs: dict[str, torch.Tensor],
    ) -> list[str]:
        return [self.decode_rule_from_probs(slot_probs, i) for i in range(self.num_rules)]

    def extra_repr(self) -> str:
        return f"num_rules={self.num_rules}, concept_dim={self.concept_dim}"