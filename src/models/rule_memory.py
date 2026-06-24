"""
rule_memory.py
==============
Lưu trữ layout constants và các helper để build concept vector.

Bottleneck 2 fix: RuleMemory.rule_logits (mask) đã được loại khỏi
scoring path. Vai trò duy nhất còn lại là decode rule ra string
để hiển thị interpretability. Prototype scoring chuyển sang
System2Rules.rule_slot_logits (xem system2_model.py).

Layout concept vector (42-dim):
    [0:10]   digit1  (softmax prob, dim=10)
    [10:15]  op1     (softmax prob, dim=5)
    [15:25]  digit2  (softmax prob, dim=10)
    [25:30]  op2     (softmax prob, dim=5)
    [30:40]  digit3  (softmax prob, dim=10)
    [40:42]  valid   (softmax prob, dim=2)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Concept layout constants
# ─────────────────────────────────────────────────────────────

CONCEPT_DIMS: dict[str, int] = {
    "digit1": 10,
    "op1":    5,
    "digit2": 10,
    "op2":    5,
    "digit3": 10,
    "valid":  2,
}

CONCEPT_KEYS_ORDERED: list[str] = [
    "digit1", "op1", "digit2", "op2", "digit3", "valid"
]

CONCEPT_TOTAL_DIM: int = sum(CONCEPT_DIMS.values())  # 42

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
    GT labels → hard one-hot concept vector [B, 42].
    labels: dict key → LongTensor[B]
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
    System1 logits → hard one-hot concept vector [B, 42] (argmax per slot).
    """
    fake_labels = {k: outputs[k].argmax(dim=1) for k in CONCEPT_KEYS_ORDERED}
    return labels_to_concept_vector(fake_labels)


def soft_concept_vector(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    System1 logits → soft concept vector [B, 42] (softmax probs, concat).
    Dùng khi training System2 để gradient flow ngược qua System1.
    """
    parts = [F.softmax(outputs[k], dim=-1) for k in CONCEPT_KEYS_ORDERED]
    return torch.cat(parts, dim=1)  # [B, 42]


# ─────────────────────────────────────────────────────────────
# RuleMemory — chỉ dùng cho decode/interpretability
# ─────────────────────────────────────────────────────────────

class RuleMemory(nn.Module):
    """
    Lưu tên và decode helper cho các rule prototype.

    Sau Bottleneck 2 fix, RuleMemory.rule_logits KHÔNG còn được
    dùng trong scoring. Scoring được thực hiện bởi
    System2Rules.rule_slot_logits (prototype có giá trị cụ thể).

    RuleMemory chỉ còn hai nhiệm vụ:
        1. decode_rule_from_probs(): decode prototype probs → string
        2. Tương thích ngược với các checkpoint cũ nếu cần.

    Parameters
    ----------
    num_rules   : số rule
    concept_dim : chiều concept vector (42)
    """

    def __init__(
        self,
        num_rules: int,
        concept_dim: int = CONCEPT_TOTAL_DIM,
    ):
        super().__init__()
        self.num_rules   = num_rules
        self.concept_dim = concept_dim
        # Không còn dùng trong forward/scoring, chỉ giữ để backward compat
        # và decode hard mask nếu cần.
        self.rule_logits = nn.Parameter(
            torch.ones(num_rules, concept_dim),  # sigmoid(1) ≈ 0.73, mask đặc
            requires_grad=False,  # không train, không ảnh hưởng gradient
        )

    @property
    def rule_masks(self) -> torch.Tensor:
        """Soft masks ∈ (0,1) — chỉ dùng cho decode, không dùng trong scoring."""
        return torch.sigmoid(self.rule_logits)

    def get_hard_masks(self, threshold: float = 0.5) -> torch.Tensor:
        return (self.rule_masks > threshold).float()

    def forward(self) -> torch.Tensor:
        return self.rule_masks

    # ── Decode helpers ───────────────────────────────────────

    def decode_rule_from_probs(
        self,
        slot_probs: dict[str, torch.Tensor],
        rule_idx: int,
    ) -> str:
        """
        Decode rule từ prototype probs (argmax per slot) → string.

        Parameters
        ----------
        slot_probs : dict key → [R, dim_k]  — softmax probs từ rule_slot_logits
        rule_idx   : int

        Returns
        -------
        str, e.g. "digit1=3 AND op1=+ AND digit2=5 AND op2== AND digit3=8 AND valid=1"
        """
        from src.utils.symbols import ID_TO_SYMBOL
        parts: list[str] = []
        for key in CONCEPT_KEYS_ORDERED:
            pred_idx = int(slot_probs[key][rule_idx].argmax().item())
            if key in ("op1", "op2"):
                label = ID_TO_SYMBOL.get(pred_idx, str(pred_idx))
            else:
                label = str(pred_idx)
            parts.append(f"{key}={label}")
        return " AND ".join(parts)

    def decode_all_rules_from_probs(
        self,
        slot_probs: dict[str, torch.Tensor],
    ) -> list[str]:
        return [
            self.decode_rule_from_probs(slot_probs, i)
            for i in range(self.num_rules)
        ]

    def extra_repr(self) -> str:
        return f"num_rules={self.num_rules}, concept_dim={self.concept_dim}"