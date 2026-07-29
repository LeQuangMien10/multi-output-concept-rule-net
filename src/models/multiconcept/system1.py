import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn_backbone import SimpleCNNBackbone
from src.utils.multiconcept_concepts import NUM_CONCEPTS


class MultiConceptSystem1(nn.Module):
    """
    System 1 cho MNIST-MultiConcept: image -> feature GLOBAL -> concept logits.

    Khác MultiHeadSystem1 (MNIST Math): concept ở đây không gắn với 1 vùng ảnh
    cụ thể (vd. "has_repeated_digit" cần nhìn toàn bộ K chữ số), nên dùng
    num_slots=1 (global average pooling) thay vì per-slot — đúng như docstring
    của SimpleCNNBackbone/MultiHeadSystem1 đã dự trù cho "ảnh không có cấu trúc
    positional" (ảnh y tế, da liễu).

    Output: concept_logits [B, NUM_CONCEPTS] — sigmoid độc lập từng concept
    (multi-label, KHÔNG phải softmax theo nhóm như MNIST Math).
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_concepts: int = NUM_CONCEPTS,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.slot_dim     = max(feature_dim // 2, 64)
        self.num_concepts = num_concepts

        self.backbone = SimpleCNNBackbone(
            in_channels=1,
            slot_dim=self.slot_dim,
            num_slots=1,     # global pooling — concept không gắn vị trí cố định
            dropout=dropout,
        )

        self.concept_head = nn.Linear(self.slot_dim, num_concepts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor[B, 1, H, W]

        Returns
        -------
        concept_logits : FloatTensor[B, num_concepts]  (chưa qua sigmoid)
        """
        feats = self.backbone(x)          # [B, 1, slot_dim]
        feats = feats.squeeze(1)          # [B, slot_dim]
        return self.concept_head(feats)   # [B, num_concepts]


def soft_concept_vector(concept_logits: torch.Tensor) -> torch.Tensor:
    """Logits -> soft concept vector [B, num_concepts] (sigmoid probs)."""
    return torch.sigmoid(concept_logits)


def hard_concept_vector(concept_logits: torch.Tensor) -> torch.Tensor:
    """Logits -> hard concept vector [B, num_concepts] (threshold 0.5)."""
    return (torch.sigmoid(concept_logits) > 0.5).float()


if __name__ == "__main__":
    model = MultiConceptSystem1(feature_dim=256, num_concepts=NUM_CONCEPTS)
    dummy = torch.randn(4, 1, 28, 112)
    logits = model(dummy)
    print(f"slot_dim = {model.slot_dim}")
    print(f"concept_logits: {logits.shape}")   # Expected: [4, NUM_CONCEPTS]
    print(f"soft_concept_vector: {soft_concept_vector(logits).shape}")
