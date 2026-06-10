import torch
import torch.nn as nn


class SimpleCNNBackbone(nn.Module):
    """
    CNN backbone for MNIST Math expression images.

    Input:
        x: [B, 1, 28, 140]

    Output:
        features: [B, feature_dim]
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 256):
        super().__init__()

        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # [B, 32, 14, 70]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # [B, 64, 7, 35]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 128, 1, 1]
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.proj(x)
        return x


if __name__ == "__main__":
    model = SimpleCNNBackbone()
    dummy = torch.randn(4, 1, 28, 140)
    out = model(dummy)
    print(out.shape)  # Expected: [4, 256]