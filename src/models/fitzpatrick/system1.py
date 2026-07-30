import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from src.utils.fitzpatrick_concepts import NUM_CONCEPTS, NUM_LABELS


class FitzpatrickSystem1(nn.Module):
    """
    System 1 cho Fitzpatrick17k: ảnh RGB thật -> backbone pretrained ImageNet ->
    {concepts, label}. Cùng interface dict {"concepts", "label"} như
    MultiConceptSystem1 (models/multiconcept/system1.py) để phần soft/hard
    concept vector và Stage 2 (ICRL) dùng lại logic tương tự.

    Khác MultiConceptSystem1 (SimpleCNNBackbone train từ đầu trên ảnh xám nhỏ):
    ảnh da liễu thật đa dạng, chỉ ~16.5k ảnh (concept-label chỉ ~3.2k) -- quá
    nhỏ để CNN học đặc trưng thị giác từ đầu (baseline paper gốc Groh et al.
    2021 dùng VGG-16/ResNet-18 pretrained ImageNet, không train from scratch).
    Mặc định dùng ResNet-50 pretrained ImageNet1k; đổi backbone_name nếu muốn
    thử EfficientNet/ConvNeXt sau.

    QUAN TRỌNG (giống hệt MultiConceptSystem1): label_head chỉ cung cấp tín
    hiệu CLUSTERING cho Stage 2. Nhãn dùng để train prediction head Stage 3
    vẫn phải lấy từ memory.get_labels() (ground-truth ngoài).

    Output (forward): dict
        "concepts": FloatTensor[B, num_concepts]  (logits, chưa qua sigmoid)
        "label":    FloatTensor[B, num_labels]     (logits, chưa qua softmax)
    """

    BACKBONE_FEATURE_DIMS = {"resnet50": 2048, "resnet18": 512, "efficientnet_b0": 1280}

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        num_concepts: int = NUM_CONCEPTS,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.3,
        freeze_backbone_stages: int = 0,
    ):
        """
        freeze_backbone_stages: đóng băng N stage đầu của backbone (0 = fine-tune
        toàn bộ). Hữu ích nếu overfit ngay cả với pretrained weight, do dataset
        nhỏ (~11.6k ảnh train sau split) -- thử 0 trước, tăng dần nếu cần.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.num_labels = num_labels

        if backbone_name == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = tv_models.resnet50(weights=weights)
            feature_dim = self.BACKBONE_FEATURE_DIMS["resnet50"]
            backbone.fc = nn.Identity()
            self._stages = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
        elif backbone_name == "resnet18":
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.resnet18(weights=weights)
            feature_dim = self.BACKBONE_FEATURE_DIMS["resnet18"]
            backbone.fc = nn.Identity()
            self._stages = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        for stage in self._stages[:freeze_backbone_stages]:
            for p in stage.parameters():
                p.requires_grad = False

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(dropout)
        self.concept_head = nn.Linear(feature_dim, num_concepts)
        self.label_head = nn.Linear(feature_dim, num_labels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : FloatTensor[B, 3, H, W]  (đã ImageNet-normalize, xem fitzpatrick_dataset.py)

        Returns
        -------
        dict với "concepts" [B, num_concepts] và "label" [B, num_labels] — logits.
        """
        feats = self.backbone(x)          # [B, feature_dim]
        feats = self.dropout(feats)
        return {
            "concepts": self.concept_head(feats),
            "label": self.label_head(feats),
        }

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return list(self.concept_head.parameters()) + list(self.label_head.parameters())


def soft_concept_vector(s1_out: dict[str, torch.Tensor]) -> torch.Tensor:
    """S1 output dict -> soft FULL concept vector [B, num_concepts + num_labels]."""
    concept_probs = torch.sigmoid(s1_out["concepts"])
    label_probs = F.softmax(s1_out["label"], dim=-1)
    return torch.cat([concept_probs, label_probs], dim=1)


def hard_concept_vector(s1_out: dict[str, torch.Tensor]) -> torch.Tensor:
    """S1 output dict -> hard FULL concept vector (threshold 0.5 / argmax one-hot)."""
    concept_hard = (torch.sigmoid(s1_out["concepts"]) > 0.5).float()
    label_idx = s1_out["label"].argmax(dim=1)
    label_hard = F.one_hot(label_idx, num_classes=s1_out["label"].shape[1]).float()
    return torch.cat([concept_hard, label_hard], dim=1)


if __name__ == "__main__":
    model = FitzpatrickSystem1(backbone_name="resnet50", pretrained=False)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"feature_dim = {model.feature_dim}")
    print(f"concepts: {out['concepts'].shape}")   # Expected: [4, NUM_CONCEPTS]
    print(f"label:    {out['label'].shape}")       # Expected: [4, NUM_LABELS]
    print(f"soft_concept_vector: {soft_concept_vector(out).shape}")
    print(f"hard_concept_vector: {hard_concept_vector(out).shape}")
