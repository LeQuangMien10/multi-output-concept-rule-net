"""
generate_mnist_multiconcept.py — Sinh dataset MNIST-MultiConcept
====================================================================

Dataset trung gian giữa MNIST Math và Fitzpatrick17k — xem chi tiết thiết kế
trong src/utils/multiconcept_concepts.py.

Ảnh: K chữ số MNIST ghép ngang [d1][d2]...[dK] (mặc định K=4 → 28×112,
cùng kích thước với MNIST Math để tái dùng backbone tương thích).

Mỗi sample có:
  - concepts      [NUM_CONCEPTS]  multi-hot nhị phân (độc lập, không loại trừ nhau)
  - concept_mask  scalar          1 nếu concept ground-truth được "công bố" cho
                                   sample này (chỉ áp dụng cho train, mô phỏng
                                   SkinCon chỉ phủ 22% ảnh Fitzpatrick — val/test
                                   luôn được công bố đầy đủ để đánh giá công bằng)
  - label         scalar          nhãn 3 lớp, SAMPLE từ softmax (không tất định)
  - label_probs   [3]             phân phối xác suất thật (ground-truth, chỉ để
                                   phân tích/validate sau này — model không thấy)

Usage:
    python -m src.scripts.multiconcept.make_dataset \\
        --config configs/dataset_mnist_multiconcept.yaml
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm

from src.datasets.generate_mnist_math import build_digit_to_indices, sample_digit_image
from src.utils.seed import set_seed
from src.utils.multiconcept_concepts import (
    CONCEPT_NAMES,
    NUM_CONCEPTS,
    LABEL_NAMES,
    NUM_LABELS,
    LABEL_BIAS,
    LABEL_WEIGHTS,
    INFORMATIVE_CONCEPTS,
    DECOY_CONCEPTS,
    compute_concept_vector,
    sample_label,
)


def render_multiconcept_image(
    digits: tuple[int, ...],
    mnist_dataset: MNIST,
    digit_to_indices: dict[int, list[int]],
    symbol_width: int = 28,
    image_height: int = 28,
) -> "Image.Image":
    """Ghép K ảnh chữ số MNIST ngang hàng: [d1][d2]...[dK]."""
    canvas = None
    for i, digit in enumerate(digits):
        part = sample_digit_image(digit, mnist_dataset, digit_to_indices)
        part = part.resize((symbol_width, image_height))
        if canvas is None:
            from PIL import Image
            canvas = Image.new("L", (symbol_width * len(digits), image_height), color=0)
        canvas.paste(part, (i * symbol_width, 0))
    return canvas


def generate_split(
    split_name: str,
    split_size: int,
    mnist_dataset: MNIST,
    digit_to_indices: dict[int, list[int]],
    num_digits: int,
    symbol_width: int,
    image_height: int,
    concept_supervision_ratio: float,
    rng: random.Random,
) -> dict[str, torch.Tensor]:
    """
    concept_supervision_ratio: tỉ lệ sample được "công bố" concept ground-truth.
    Luôn = 1.0 cho val/test (đánh giá cần concept đầy đủ); chỉ giới hạn ở train
    để mô phỏng SkinCon chỉ phủ ~22% ảnh Fitzpatrick.
    """
    to_tensor = transforms.ToTensor()

    images       = []
    concepts_l   = []
    concept_mask = []
    labels_l     = []
    label_probs  = []
    digits_l     = []

    for _ in tqdm(range(split_size), desc=f"Generating {split_name}"):
        digits = tuple(rng.randint(0, 9) for _ in range(num_digits))

        img = render_multiconcept_image(
            digits=digits,
            mnist_dataset=mnist_dataset,
            digit_to_indices=digit_to_indices,
            symbol_width=symbol_width,
            image_height=image_height,
        )
        images.append(to_tensor(img))

        cvec = compute_concept_vector(digits)
        label, probs = sample_label(cvec, rng)

        concepts_l.append(cvec)
        concept_mask.append(1 if rng.random() < concept_supervision_ratio else 0)
        labels_l.append(label)
        label_probs.append(probs)
        digits_l.append(list(digits))

    return {
        "images":       torch.stack(images, dim=0),
        "concepts":     torch.tensor(concepts_l,   dtype=torch.float32),
        "concept_mask": torch.tensor(concept_mask, dtype=torch.float32),
        "label":        torch.tensor(labels_l,      dtype=torch.long),
        "label_probs":  torch.tensor(label_probs,   dtype=torch.float32),
        "digits":       torch.tensor(digits_l,       dtype=torch.long),
    }


def generate_mnist_multiconcept_dataset(config: dict[str, Any]) -> None:
    dataset_cfg = config["dataset"]

    root_dir   = Path(dataset_cfg["root_dir"])
    output_dir = Path(dataset_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(dataset_cfg.get("seed", 2026))
    set_seed(seed)
    rng = random.Random(seed)

    train_size = int(dataset_cfg["train_size"])
    val_size   = int(dataset_cfg["val_size"])
    test_size  = int(dataset_cfg["test_size"])

    num_digits   = int(dataset_cfg.get("num_digits", 4))
    symbol_width = int(dataset_cfg.get("symbol_width", 28))
    image_height = int(dataset_cfg.get("image_height", 28))

    concept_supervision_ratio = float(dataset_cfg.get("concept_supervision_ratio", 0.25))

    print("[INFO] Loading MNIST...")
    mnist_train = MNIST(root=root_dir, train=True, download=True, transform=None)
    mnist_test  = MNIST(root=root_dir, train=False, download=True, transform=None)

    train_digit_to_indices = build_digit_to_indices(mnist_train)
    test_digit_to_indices  = build_digit_to_indices(mnist_test)

    train_data = generate_split(
        split_name="train", split_size=train_size,
        mnist_dataset=mnist_train, digit_to_indices=train_digit_to_indices,
        num_digits=num_digits, symbol_width=symbol_width, image_height=image_height,
        concept_supervision_ratio=concept_supervision_ratio, rng=rng,
    )
    val_data = generate_split(
        split_name="val", split_size=val_size,
        mnist_dataset=mnist_test, digit_to_indices=test_digit_to_indices,
        num_digits=num_digits, symbol_width=symbol_width, image_height=image_height,
        concept_supervision_ratio=1.0, rng=rng,   # val luôn full concept supervision
    )
    test_data = generate_split(
        split_name="test", split_size=test_size,
        mnist_dataset=mnist_test, digit_to_indices=test_digit_to_indices,
        num_digits=num_digits, symbol_width=symbol_width, image_height=image_height,
        concept_supervision_ratio=1.0, rng=rng,   # test luôn full concept supervision
    )

    torch.save(train_data, output_dir / "train.pt")
    torch.save(val_data,   output_dir / "valid.pt")
    torch.save(test_data,  output_dir / "test.pt")

    label_counts = {
        name: int((train_data["label"] == i).sum().item())
        for i, name in enumerate(LABEL_NAMES)
    }
    concept_pos_rate = {
        name: round(float(train_data["concepts"][:, i].mean().item()), 4)
        for i, name in enumerate(CONCEPT_NAMES)
    }

    meta = {
        "name": dataset_cfg.get("name", "mnist_multiconcept_v1"),
        "task": "K MNIST digits -> multi-label concept vector -> probabilistic 3-way label",
        "purpose": "bridge dataset between mnist_math and fitzpatrick17k "
                   "(non-deterministic concept->label mapping, imbalanced multi-label "
                   "concepts, partial concept supervision on train)",
        "image_shape": [1, image_height, symbol_width * num_digits],
        "symbol_width": symbol_width,
        "image_height": image_height,
        "num_digits": num_digits,
        "num_concepts": NUM_CONCEPTS,
        "concept_names": CONCEPT_NAMES,
        "informative_concepts": INFORMATIVE_CONCEPTS,
        "decoy_concepts": DECOY_CONCEPTS,
        "label_names": LABEL_NAMES,
        "num_labels": NUM_LABELS,
        "label_bias": LABEL_BIAS,
        "label_weights": LABEL_WEIGHTS,
        "concept_supervision_ratio_train": concept_supervision_ratio,
        "concept_supervision_ratio_val_test": 1.0,
        "seed": seed,
        "splits": {"train": train_size, "val": val_size, "test": test_size},
        "train_label_distribution": label_counts,
        "train_concept_positive_rate": concept_pos_rate,
    }

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Dataset saved to: {output_dir}")
    print(f"[DONE] train.pt: {train_data['images'].shape}  label_dist={label_counts}")
    print(f"[DONE] valid.pt: {val_data['images'].shape}")
    print(f"[DONE] test.pt:  {test_data['images'].shape}")
