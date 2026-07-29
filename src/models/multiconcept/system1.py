import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn_backbone import SimpleCNNBackbone
from src.utils.multiconcept_concepts import NUM_CONCEPTS, NUM_LABELS


class MultiConceptSystem1(nn.Module):
    """
    System 1 cho MNIST-MultiConcept: image -> feature GLOBAL -> {concepts, label}.

    Khác MultiHeadSystem1 (MNIST Math): concept ở đây không gắn với 1 vùng ảnh
    cụ thể (vd. "has_repeated_digit" cần nhìn toàn bộ K chữ số), nên dùng
    num_slots=1 (global average pooling) thay vì per-slot — đúng như docstring
    của SimpleCNNBackbone/MultiHeadSystem1 đã dự trù cho "ảnh không có cấu trúc
    positional" (ảnh y tế, da liễu).

    NHÃN LÀ MỘT CONCEPT: giống hệt cách digit3 (target) được S1 dự đoán và
    nằm trong concept vector của MNIST Math (không tách biệt "target" khỏi
    "concept"), model này có thêm label_head dự đoán nhãn 3 lớp — output
    của head này được nối vào concept vector ở Stage 2 (xem
    soft_concept_vector/hard_concept_vector bên dưới) để ICRL cluster.
    Nhãn luôn có ground-truth đầy đủ (không bị giới hạn 25% như concept)
    nên label_head được train full-supervision mỗi batch.

    QUAN TRỌNG: label_head chỉ cung cấp tín hiệu cho CLUSTERING (Stage 2).
    Nhãn dùng để train prediction head ở Stage 3 vẫn phải lấy từ
    memory.get_labels() (ground-truth ngoài) — không đọc trực tiếp output
    của label_head làm nhãn train, để tránh lặp lại lỗi đã fix ở MNIST Math
    (dùng slot mang nhiễu của S1 làm nhãn thay vì sự thật).

    Output (forward): dict
        "concepts": FloatTensor[B, num_concepts]  (logits, chưa qua sigmoid)
        "label":    FloatTensor[B, num_labels]     (logits, chưa qua softmax)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_concepts: int = NUM_CONCEPTS,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.slot_dim     = max(feature_dim // 2, 64)
        self.num_concepts = num_concepts
        self.num_labels   = num_labels

        self.backbone = SimpleCNNBackbone(
            in_channels=1,
            slot_dim=self.slot_dim,
            num_slots=1,     # global pooling — concept không gắn vị trí cố định
            dropout=dropout,
        )

        self.concept_head = nn.Linear(self.slot_dim, num_concepts)
        self.label_head   = nn.Linear(self.slot_dim, num_labels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : FloatTensor[B, 1, H, W]

        Returns
        -------
        dict với "concepts" [B, num_concepts] và "label" [B, num_labels] — logits.
        """
        feats = self.backbone(x)          # [B, 1, slot_dim]
        feats = feats.squeeze(1)          # [B, slot_dim]

        return {
            "concepts": self.concept_head(feats),
            "label":    self.label_head(feats),
        }


def soft_concept_vector(s1_out: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    S1 output dict -> soft FULL concept vector [B, num_concepts + num_labels].
    concepts: sigmoid độc lập từng concept (multi-label).
    label:    softmax 3-way (nối vào sau, đúng vị trí FULL_CONCEPT_OFFSETS
              trong multiconcept_concepts.py).
    """
    concept_probs = torch.sigmoid(s1_out["concepts"])
    label_probs   = F.softmax(s1_out["label"], dim=-1)
    return torch.cat([concept_probs, label_probs], dim=1)


def hard_concept_vector(s1_out: dict[str, torch.Tensor]) -> torch.Tensor:
    """S1 output dict -> hard FULL concept vector (threshold 0.5 / argmax one-hot)."""
    concept_hard = (torch.sigmoid(s1_out["concepts"]) > 0.5).float()
    label_idx    = s1_out["label"].argmax(dim=1)
    label_hard   = F.one_hot(label_idx, num_classes=s1_out["label"].shape[1]).float()
    return torch.cat([concept_hard, label_hard], dim=1)


if __name__ == "__main__":
    model = MultiConceptSystem1(feature_dim=256, num_concepts=NUM_CONCEPTS, num_labels=NUM_LABELS)
    dummy = torch.randn(4, 1, 28, 112)
    out = model(dummy)
    print(f"slot_dim = {model.slot_dim}")
    print(f"concepts: {out['concepts'].shape}")   # Expected: [4, NUM_CONCEPTS]
    print(f"label:    {out['label'].shape}")       # Expected: [4, NUM_LABELS]
    print(f"soft_concept_vector: {soft_concept_vector(out).shape}")   # [4, NUM_CONCEPTS+NUM_LABELS]
    print(f"hard_concept_vector: {hard_concept_vector(out).shape}")
