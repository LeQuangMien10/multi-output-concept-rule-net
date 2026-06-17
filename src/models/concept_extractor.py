import torch
from src.models.multi_head_system1 import MultiHeadSystem1


class ConceptExtractor:
    """
    Wrap System1 model to extract predicted concepts
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = device

        self.model = MultiHeadSystem1().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_concepts(self, images):
        """
        return:
            concept vector (argmax)
        """

        outputs = self.model(images)

        z = torch.cat([
            outputs["digit1"].argmax(dim=-1, keepdim=True),
            outputs["op1"].argmax(dim=-1, keepdim=True),
            outputs["digit2"].argmax(dim=-1, keepdim=True),
            outputs["op2"].argmax(dim=-1, keepdim=True),
            outputs["digit3"].argmax(dim=-1, keepdim=True),
            outputs["valid"].argmax(dim=-1, keepdim=True),
        ], dim=-1)

        return z  # [B, 6]