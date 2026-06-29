import torch
import torch.nn as nn


class SimpleCNNBackbone(nn.Module):
    """
    CNN backbone for multi-symbol expression images.

    Thay đổi so với v1: AdaptiveAvgPool2d((1,1)) → (1, num_slots)
    để giữ thông tin vị trí không gian của từng symbol.

    Ảnh đầu vào gồm num_slots ký tự xếp ngang:
[digit1][op1][digit2][op2]  (mặc định 4 slot × 28px = 112px)

    AdaptiveAvgPool2d((1, num_slots)) pool height xuống 1 nhưng giữ
    width dưới dạng num_slots bins, mỗi bin tương ứng đúng một ký tự.

    Input:
        x: [B, in_channels, H, W]   (ví dụ: [B, 1, 28, 112])

    Output:
        slots: [B, num_slots, slot_dim]
            slots[:, i, :] = đặc trưng của ký tự thứ i

    Tính tương thích với mục tiêu dài hạn (ảnh y tế):
        - Đổi num_slots=1 (hoặc dùng global_pool=True) để quay về
          hành vi flatten thông thường.
        - Interface MultiHeadSystem1 không thay đổi; chỉ backbone
          cần điều chỉnh num_slots.
    """

    def __init__(
        self,
        in_channels: int = 1,
        slot_dim: int = 128,
        num_slots: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.slot_dim  = slot_dim
        self.num_slots = num_slots

        # ── Shared convolutional feature extractor ───────────
        # Shape flow (input 28×112, num_slots=4):
        #   [B,  1, 28, 112]
        #   [B, 32, 14,  56]  after MaxPool2d(2)
        #   [B, 64,  7,  28]  after MaxPool2d(2)
        #   [B,128,  7,  28]  after conv3 + conv4
        #   [B,128,  1,   5]  after AdaptiveAvgPool2d((1, num_slots))
        #   Width 28 / num_slots 4 = 7px per slot → exact alignment
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Giữ num_slots bins theo chiều rộng (≠ v1 dùng (1,1))
            nn.AdaptiveAvgPool2d((1, num_slots)),  # [B, 128, 1, num_slots]
        )

        # ── Per-slot projection ──────────────────────────────
        # Áp dụng cùng một Linear lên từng slot (weight sharing).
        # Input: [B, num_slots, 128] → Output: [B, num_slots, slot_dim]
        self.slot_proj = nn.Sequential(
            nn.Linear(128, slot_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        slots : FloatTensor[B, num_slots, slot_dim]
        """
        x = self.conv(x)             # [B, 128, 1, num_slots]
        x = x.squeeze(2)             # [B, 128, num_slots]
        x = x.permute(0, 2, 1)       # [B, num_slots, 128]
        x = self.slot_proj(x)        # [B, num_slots, slot_dim]
        return x


if __name__ == "__main__":
    model = SimpleCNNBackbone(slot_dim=128, num_slots=4)
    dummy = torch.randn(4, 1, 28, 112)  # 4 slots × 28px
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Expected: [4, 5, 128]
    print(f"  slots[b, 0, :] = digit1 features")
    print(f"  slots[b, 1, :] = op1 features")
    print(f"  slots[b, 2, :] = digit2 features")
    print(f"  slots[b, 3, :] = op2 features")
    print(f"  slots[b, 3, :] = op2 features (digit3 is the label, not in image)")