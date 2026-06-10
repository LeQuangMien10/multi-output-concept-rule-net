import torch
import torch.nn as nn

from src.models.cnn_backbone import SimpleCNNBackbone


class ImageOnlyBaseline(nn.Module):
    """
    Baseline 1:

        image -> CNN backbone -> valid / invalid

    Output:
        logits: [B, 2]
    """

    def __init__(self, feature_dim: int = 256, num_classes: int = 2):
        super().__init__()

        self.backbone = SimpleCNNBackbone(
            in_channels=1,
            feature_dim=feature_dim,
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    model = ImageOnlyBaseline()
    dummy = torch.randn(4, 1, 28, 140)
    logits = model(dummy)
    print(logits.shape)  # Expected: [4, 2]