from pathlib import Path

import torch
from torch.utils.data import Dataset

# Label keys bắt buộc trong dataset v2 (không có "valid")
REQUIRED_KEYS = ["images", "digit1", "op1", "digit2", "op2", "digit3"]


class MNISTMathPTDataset(Dataset):
    """
    Load MNIST Math dataset từ .pt file.

    Hỗ trợ cả hai format:
        v1 (cũ):  có key "valid"  — biểu thức a+b=c, nhãn valid/invalid
        v2 (mới): không có "valid" — biểu thức "a+b=?", predict digit3
    """

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        self.data = torch.load(self.data_path, map_location="cpu", weights_only=True)

        for key in REQUIRED_KEYS:
            if key not in self.data:
                raise KeyError(f"Missing key '{key}' in {self.data_path}")

        self.has_valid = "valid" in self.data
        self.length    = self.data["images"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        image = self.data["images"][idx]

        labels = {
            "digit1": self.data["digit1"][idx],
            "op1":    self.data["op1"][idx],
            "digit2": self.data["digit2"][idx],
            "op2":    self.data["op2"][idx],
            "digit3": self.data["digit3"][idx],
        }
        if self.has_valid:
            labels["valid"] = self.data["valid"][idx]

        return image, labels


def collate_mnist_math(batch):
    images = torch.stack([item[0] for item in batch], dim=0)

    keys = list(batch[0][1].keys())
    labels = {
        k: torch.stack([item[1][k] for item in batch], dim=0)
        for k in keys
    }

    return images, labels