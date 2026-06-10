import torch
import torch.nn as nn

from src.models.cnn_backbone import SimpleCNNBackbone


class MultiHeadSystem1(nn.Module):
    """
    Baseline 2 / System 1:

        image -> CNN backbone -> multiple prediction heads

    Outputs:
        digit1: [B, 10]
        op1:    [B, 5]
        digit2: [B, 10]
        op2:    [B, 5]
        digit3: [B, 10]
        valid:  [B, 2]
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()

        self.backbone = SimpleCNNBackbone(
            in_channels=1,
            feature_dim=feature_dim,
        )

        self.digit1_head = nn.Linear(feature_dim, 10)
        self.op1_head = nn.Linear(feature_dim, 5)
        self.digit2_head = nn.Linear(feature_dim, 10)
        self.op2_head = nn.Linear(feature_dim, 5)
        self.digit3_head = nn.Linear(feature_dim, 10)
        self.valid_head = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)

        outputs = {
            "digit1": self.digit1_head(features),
            "op1": self.op1_head(features),
            "digit2": self.digit2_head(features),
            "op2": self.op2_head(features),
            "digit3": self.digit3_head(features),
            "valid": self.valid_head(features),
        }

        return outputs


if __name__ == "__main__":
    model = MultiHeadSystem1()
    dummy = torch.randn(4, 1, 28, 140)
    outputs = model(dummy)

    for key, value in outputs.items():
        print(key, value.shape)