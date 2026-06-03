from pathlib import Path

import torch
from torch.utils.data import Dataset


class MNISTMathPTDataset(Dataset):
    """
    Load generated MNIST Math dataset from .pt file.

    Expected .pt format:
        {
            "images": Tensor[N, 1, 28, 140],
            "digit1": Tensor[N],
            "op1": Tensor[N],
            "digit2": Tensor[N],
            "op2": Tensor[N],
            "digit3": Tensor[N],
            "valid": Tensor[N],
        }
    """

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        self.data = torch.load(self.data_path, map_location="cpu")

        required_keys = [
            "images",
            "digit1",
            "op1",
            "digit2",
            "op2",
            "digit3",
            "valid",
        ]

        for key in required_keys:
            if key not in self.data:
                raise KeyError(f"Missing key '{key}' in {self.data_path}")

        self.length = self.data["images"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        image = self.data["images"][idx]

        labels = {
            "digit1": self.data["digit1"][idx],
            "op1": self.data["op1"][idx],
            "digit2": self.data["digit2"][idx],
            "op2": self.data["op2"][idx],
            "digit3": self.data["digit3"][idx],
            "valid": self.data["valid"][idx],
        }

        return image, labels


def collate_mnist_math(batch):
    """
    Optional collate function.
    Default PyTorch collate also works, but this keeps format explicit.
    """
    images = torch.stack([item[0] for item in batch], dim=0)

    labels = {
        "digit1": torch.stack([item[1]["digit1"] for item in batch], dim=0),
        "op1": torch.stack([item[1]["op1"] for item in batch], dim=0),
        "digit2": torch.stack([item[1]["digit2"] for item in batch], dim=0),
        "op2": torch.stack([item[1]["op2"] for item in batch], dim=0),
        "digit3": torch.stack([item[1]["digit3"] for item in batch], dim=0),
        "valid": torch.stack([item[1]["valid"] for item in batch], dim=0),
    }

    return images, labels