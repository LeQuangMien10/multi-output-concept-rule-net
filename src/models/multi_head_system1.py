import torch
import torch.nn as nn

from src.models.cnn_backbone import SimpleCNNBackbone


# Ánh xạ cố định: slot index → concept key
# Khớp chính xác với thứ tự paste trong generate_mnist_math.py:
#   canvas.paste(digit1, i=0), op1 (i=1), digit2 (i=2), op2 (i=3), digit3 (i=4)
_SLOT_TO_KEY = {0: "digit1", 1: "op1", 2: "digit2", 3: "op2"}  # 4 slots trong ảnh; digit3 là label, không có slot riêng


class MultiHeadSystem1(nn.Module):
    """
    System 1: image → spatial slots → per-concept prediction heads.

    Thay đổi so với v1:
    ───────────────────
    v1: backbone → [B, feature_dim] → tất cả heads chia sẻ một vector.
        Digit2/digit3 khó học vì không có gì phân biệt chúng với digit1.

    v2: backbone → [B, 5, slot_dim] → mỗi head nhận đúng slot của mình.
        digit1_head  ← slots[:, 0, :]    (vùng ảnh chứa digit1)
        op1_head     ← slots[:, 1, :]    (vùng ảnh chứa op1)
        digit2_head  ← slots[:, 2, :]    (vùng ảnh chứa digit2)
        op2_head     ← slots[:, 3, :]    (vùng ảnh chứa op2)
        digit3_head  ← slots.mean(dim=1) (predict answer, không có slot riêng)

    Tương thích dài hạn (ảnh y tế, da liễu):
    ─────────────────────────────────────────
    - Multihead architecture được GIỮ NGUYÊN: mỗi concept có head riêng.
    - Chỉ thay đổi cách backbone trích xuất feature.
    - Với ảnh không có cấu trúc positional (ảnh da), dùng num_slots=1
      hoặc truyền pre-computed features từ backbone khác (ViT, ResNet...).
    - Interface __init__ và forward() không thay đổi.

    Parameters
    ----------
    feature_dim : int
        slot_dim = feature_dim // 2 (mặc định 256 → slot_dim=128).
        Giữ tên feature_dim để tương thích với checkpoint cũ.
    num_slots : int
        Số slot trong ảnh (mặc định 4 cho v2: digit1,op1,digit2,op2).
    dropout : float
        Dropout trong slot projection.

    Outputs (forward)
    -----------------
    dict[str, FloatTensor[B, C]]:
        digit1: [B, 10]
        op1:    [B, 5]
        digit2: [B, 10]
        op2:    [B, 5]
        digit3: [B, 10]   ← predict từ global mean (target)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_slots: int = 4,   # v2: [digit1][op1][digit2][op2]
        dropout: float = 0.2,
    ):
        super().__init__()

        # slot_dim = feature_dim // 2 để giữ tổng capacity tương đương v1.
        # feature_dim=256 → slot_dim=128; 5 slots × 128 = 640 > 256 (v1).
        self.slot_dim  = max(feature_dim // 2, 64)
        self.num_slots = num_slots

        self.backbone = SimpleCNNBackbone(
            in_channels=1,
            slot_dim=self.slot_dim,
            num_slots=num_slots,
            dropout=dropout,
        )

        # ── Concept heads ────────────────────────────────────
        # digit heads: nhận slot riêng → [B, slot_dim] → [B, 10]
        self.digit1_head = nn.Linear(self.slot_dim, 10)
        self.op1_head    = nn.Linear(self.slot_dim, 5)
        self.digit2_head = nn.Linear(self.slot_dim, 10)
        self.op2_head    = nn.Linear(self.slot_dim, 5)
        self.digit3_head = nn.Linear(self.slot_dim, 10)



    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : FloatTensor[B, 1, H, W]

        Returns
        -------
        dict[str, FloatTensor]  — logits chưa qua softmax
        """
        # slots: [B, num_slots, slot_dim]
        slots = self.backbone(x)

        outputs = {
            # Các slot trong ảnh: digit1, op1, digit2, op2
            "digit1": self.digit1_head(slots[:, 0, :]),
            "op1":    self.op1_head(slots[:, 1, :]),
            "digit2": self.digit2_head(slots[:, 2, :]),
            "op2":    self.op2_head(slots[:, 3, :]),
            # digit3 = target — predict từ global context (mean của 4 slots)
            "digit3": self.digit3_head(slots.mean(dim=1)),
        }

        return outputs

    def get_slot_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trả về raw slot features [B, num_slots, slot_dim] để debug
        hoặc visualize attention giữa slots và rule matching.
        """
        return self.backbone(x)


if __name__ == "__main__":
    model = MultiHeadSystem1(feature_dim=256, num_slots=4)
    dummy = torch.randn(4, 1, 28, 112)
    outputs = model(dummy)

    print(f"slot_dim = {model.slot_dim}")
    print()
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    print("Slot mapping (4 slots in image, digit3 from mean):")
    for i, key in _SLOT_TO_KEY.items():
        print(f"  slots[:, {i}, :] → {key}_head")
    print("  slots.mean(dim=1)  → digit3_head")