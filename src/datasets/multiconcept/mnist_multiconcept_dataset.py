from pathlib import Path

import torch
from torch.utils.data import Dataset

REQUIRED_KEYS = ["images", "concepts", "concept_mask", "label"]


class MNISTMultiConceptPTDataset(Dataset):
    """
    Load MNIST-MultiConcept dataset từ .pt file (xem generate_mnist_multiconcept.py).

    Mỗi item trả về (image, labels_dict) với:
        labels_dict["concepts"]     FloatTensor[num_concepts]  multi-hot ground-truth
        labels_dict["concept_mask"] scalar float (0/1)         concept có được "công bố"
        labels_dict["label"]        scalar long                nhãn 3 lớp
        labels_dict["label_probs"]  FloatTensor[3] (nếu có)     phân phối xác suất thật
    """

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        self.data = torch.load(self.data_path, map_location="cpu", weights_only=True)
        for key in REQUIRED_KEYS:
            if key not in self.data:
                raise KeyError(f"Missing key '{key}' in {self.data_path}")
        self.has_label_probs = "label_probs" in self.data
        self.length = self.data["images"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        image = self.data["images"][idx]
        labels = {
            "concepts":     self.data["concepts"][idx],
            "concept_mask": self.data["concept_mask"][idx],
            "label":        self.data["label"][idx],
        }
        if self.has_label_probs:
            labels["label_probs"] = self.data["label_probs"][idx]
        return image, labels
