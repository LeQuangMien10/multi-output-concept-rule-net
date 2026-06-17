"""
rule_memory.py
==============
Lưu trữ các rule prototype (learned rule embeddings).

Mỗi rule là một vector nhị phân/soft mask có cùng chiều với
concept vector (40-dim cho MNIST Math: 10+5+10+5+10),
cộng thêm phần "valid" (2-dim) để thành 42-dim.

Layout concept vector (42-dim):
    [0:10]   digit1 (one-hot)
    [10:15]  op1    (one-hot)
    [15:25]  digit2 (one-hot)
    [25:30]  op2    (one-hot)
    [30:40]  digit3 (one-hot)
    [40:42]  valid  (one-hot: 0=invalid, 1=valid)
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Concept layout constants
# ─────────────────────────────────────────────────────────────

CONCEPT_DIMS = {
    "digit1": 10,
    "op1": 5,
    "digit2": 10,
    "op2": 5,
    "digit3": 10,
    "valid": 2,
}

CONCEPT_KEYS_ORDERED = ["digit1", "op1", "digit2", "op2", "digit3", "valid"]

CONCEPT_TOTAL_DIM = sum(CONCEPT_DIMS.values())   # 42

# Start index of each concept slot in the flat vector
CONCEPT_OFFSETS: dict[str, int] = {}
_offset = 0
for _k in CONCEPT_KEYS_ORDERED:
    CONCEPT_OFFSETS[_k] = _offset
    _offset += CONCEPT_DIMS[_k]


# ─────────────────────────────────────────────────────────────
# Helper: build concept vector from label dict
# ─────────────────────────────────────────────────────────────

def labels_to_concept_vector(labels: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    labels: dict với key -> LongTensor[B]
    Return: FloatTensor[B, CONCEPT_TOTAL_DIM]  (one-hot per slot)
    """
    B = labels["digit1"].shape[0]
    vec = torch.zeros(B, CONCEPT_TOTAL_DIM, device=labels["digit1"].device)

    for key in CONCEPT_KEYS_ORDERED:
        offset = CONCEPT_OFFSETS[key]
        dim = CONCEPT_DIMS[key]
        idx = labels[key].long()                        # [B]
        one_hot = torch.zeros(B, dim, device=idx.device)
        one_hot.scatter_(1, idx.unsqueeze(1), 1.0)
        vec[:, offset: offset + dim] = one_hot

    return vec


def logits_to_concept_vector(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    outputs: dict với key -> logits FloatTensor[B, C]
    Lấy argmax rồi one-hot, giống labels_to_concept_vector.
    """
    B = outputs["digit1"].shape[0]
    fake_labels = {k: outputs[k].argmax(dim=1) for k in CONCEPT_KEYS_ORDERED}
    return labels_to_concept_vector(fake_labels)


def soft_concept_vector(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Nối softmax probabilities thành concept vector liên tục [B, 42].
    Dùng cho rule matching khi muốn gradient flow.
    """
    import torch.nn.functional as F
    parts = [F.softmax(outputs[k], dim=-1) for k in CONCEPT_KEYS_ORDERED]
    return torch.cat(parts, dim=1)   # [B, 42]


# ─────────────────────────────────────────────────────────────
# RuleMemory
# ─────────────────────────────────────────────────────────────

class RuleMemory(nn.Module):
    """
    Lưu N rule prototype vectors, mỗi prototype là một soft mask
    kích thước [CONCEPT_TOTAL_DIM] ∈ (0, 1).

    Ý nghĩa: rule_mask[i, j] ≈ 1 nghĩa là slot j phải active
             trong rule i; ≈ 0 nghĩa là slot j không quan trọng.

    Parameters
    ----------
    num_rules : int
        Số lượng rules cần học.
    concept_dim : int
        Chiều concept vector (mặc định 42).
    init : str
        "random" | "ones"  — cách khởi tạo trọng số.
    """

    def __init__(
        self,
        num_rules: int,
        concept_dim: int = CONCEPT_TOTAL_DIM,
        init: str = "random",
    ):
        super().__init__()
        self.num_rules = num_rules
        self.concept_dim = concept_dim

        # Rule mask: tham số học được (logit trước sigmoid)
        raw = torch.empty(num_rules, concept_dim)
        if init == "ones":
            nn.init.constant_(raw, 2.0)   # sigmoid(2) ≈ 0.88
        else:
            nn.init.normal_(raw, mean=0.0, std=0.5)

        self.rule_logits = nn.Parameter(raw)   # [num_rules, concept_dim]

    @property
    def rule_masks(self) -> torch.Tensor:
        """Soft rule masks ∈ (0,1), shape [num_rules, concept_dim]."""
        return torch.sigmoid(self.rule_logits)

    def get_hard_masks(self, threshold: float = 0.5) -> torch.Tensor:
        """Binary rule masks {0,1}, shape [num_rules, concept_dim]."""
        return (self.rule_masks > threshold).float()

    def forward(self) -> torch.Tensor:
        return self.rule_masks

    # ── Interpretability helpers ─────────────────────────────

    def decode_rule(self, rule_idx: int, threshold: float = 0.5) -> str:
        """
        Chuyển rule thứ rule_idx thành chuỗi đọc được:
        e.g. "digit1=2 AND op1=+ AND digit2=3 AND op2== AND digit3=5 AND valid=1"
        """
        from src.utils.symbols import ID_TO_SYMBOL

        mask = self.get_hard_masks(threshold)[rule_idx]   # [42]
        parts: list[str] = []

        for key in CONCEPT_KEYS_ORDERED:
            offset = CONCEPT_OFFSETS[key]
            dim = CONCEPT_DIMS[key]
            slot = mask[offset: offset + dim]             # [dim]
            active = slot.nonzero(as_tuple=False).squeeze(-1).tolist()

            for idx in active:
                if key.startswith("digit") or key == "valid":
                    label = str(idx)
                else:  # op1, op2
                    label = ID_TO_SYMBOL.get(idx, str(idx))
                parts.append(f"{key}={label}")

        return " AND ".join(parts) if parts else "(empty rule)"

    def decode_all_rules(self, threshold: float = 0.5) -> list[str]:
        return [self.decode_rule(i, threshold) for i in range(self.num_rules)]

    def extra_repr(self) -> str:
        return f"num_rules={self.num_rules}, concept_dim={self.concept_dim}"