from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.utils.symbols import expression_to_string


def save_dataset_preview(
    batch: dict,
    save_path: str | Path,
    num_samples: int = 16,
) -> None:
    """
    Save a grid preview of generated MNIST Math samples.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    images = batch["images"]
    digit1 = batch["digit1"]
    op1 = batch["op1"]
    digit2 = batch["digit2"]
    op2 = batch["op2"]
    digit3 = batch["digit3"]
    valid = batch["valid"]

    num_samples = min(num_samples, images.shape[0])

    cols = 4
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(cols * 4, rows * 2))

    for i in range(num_samples):
        img = images[i]

        if isinstance(img, torch.Tensor):
            img = img.squeeze(0).cpu().numpy()

        expr = expression_to_string(
            digit1=int(digit1[i]),
            op1_id=int(op1[i]),
            digit2=int(digit2[i]),
            op2_id=int(op2[i]),
            digit3=int(digit3[i]),
        )

        label = "valid" if int(valid[i]) == 1 else "invalid"

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"{expr}\n{label}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()