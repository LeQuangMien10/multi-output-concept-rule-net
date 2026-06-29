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
    Save a grid preview of MNIST Math samples.
    Hỗ trợ cả v1 (có valid) và v2 (không có valid, predict digit3).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    images = batch["images"]
    digit1 = batch["digit1"]
    op1    = batch["op1"]
    digit2 = batch["digit2"]
    op2    = batch["op2"]
    digit3 = batch["digit3"]
    has_valid = "valid" in batch

    num_samples = min(num_samples, images.shape[0])
    cols = 4
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(cols * 4, rows * 2))

    for i in range(num_samples):
        img = images[i]
        if isinstance(img, torch.Tensor):
            img = img.squeeze(0).cpu().numpy()

        # v1: show full expression; v2: show "a + b = → answer=c"
        if has_valid:
            expr  = expression_to_string(int(digit1[i]), int(op1[i]),
                                         int(digit2[i]), int(op2[i]),
                                         int(digit3[i]))
            label = "valid" if int(batch["valid"][i]) == 1 else "invalid"
            title = f"{expr}\n{label}"
        else:
            expr  = expression_to_string(int(digit1[i]), int(op1[i]),
                                         int(digit2[i]))
            title = f"{expr}  →  ans={int(digit3[i])}"

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()