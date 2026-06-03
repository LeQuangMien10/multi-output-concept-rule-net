import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.mnist_math_dataset import MNISTMathPTDataset
from src.utils.visualization import save_dataset_preview
from src.utils.symbols import expression_to_string


def parse_args():
    parser = argparse.ArgumentParser(description="Check generated MNIST Math dataset.")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mnist_math",
        help="Path to generated dataset directory.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to check.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to visualize.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dataset_preview.png",
        help="Path to save preview image.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    split_path = data_dir / f"{args.split}.pt"
    meta_path = data_dir / "meta.json"

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    print(f"[INFO] Loading split: {split_path}")
    dataset = MNISTMathPTDataset(split_path)
    loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True)

    images, labels = next(iter(loader))

    print("[INFO] Dataset loaded successfully.")
    print(f"images shape: {images.shape}")

    for key, value in labels.items():
        print(f"{key} shape: {value.shape}")

    valid_ratio = labels["valid"].float().mean().item()
    print(f"valid ratio in preview batch: {valid_ratio:.3f}")

    print("\n[INFO] Sample labels:")
    for i in range(min(args.num_samples, images.shape[0], 10)):
        expr = expression_to_string(
            digit1=int(labels["digit1"][i]),
            op1_id=int(labels["op1"][i]),
            digit2=int(labels["digit2"][i]),
            op2_id=int(labels["op2"][i]),
            digit3=int(labels["digit3"][i]),
        )

        valid = int(labels["valid"][i])
        print(f"sample {i}: {expr}, valid={valid}")

    batch = {
        "images": images,
        "digit1": labels["digit1"],
        "op1": labels["op1"],
        "digit2": labels["digit2"],
        "op2": labels["op2"],
        "digit3": labels["digit3"],
        "valid": labels["valid"],
    }

    save_dataset_preview(
        batch=batch,
        save_path=args.output,
        num_samples=args.num_samples,
    )

    print(f"\n[DONE] Preview saved to: {args.output}")

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        print("\n[INFO] meta.json:")
        print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()