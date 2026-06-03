import argparse
from pathlib import Path

import yaml

from src.datasets.generate_mnist_math import generate_mnist_math_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MNIST Math dataset.")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_mnist_math.yaml",
        help="Path to dataset config YAML file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    generate_mnist_math_dataset(config)


if __name__ == "__main__":
    main()