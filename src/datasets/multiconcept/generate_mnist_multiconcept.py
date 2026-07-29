"""
generate_mnist_multiconcept.py — Sinh dataset MNIST-MultiConcept (v2 — Parity)
====================================================================

Dataset trung gian giữa MNIST Math và Fitzpatrick17k — xem chi tiết thiết kế
trong src/utils/multiconcept_concepts.py. Nhãn = so n_odd vs n_even trong K
chữ số (hàm ĐẾM tất định, không phải bảng trọng số) — tự-kiểm-chứng được
bằng mắt giống hệt digit3 = digit1 − digit2 của MNIST Math.

Ảnh: K chữ số MNIST ghép ngang [d1][d2]...[dK] (mặc định K=4 → 28×112,
cùng kích thước với MNIST Math để tái dùng backbone tương thích).

Mỗi sample có:
  - concepts      [NUM_CONCEPTS]  multi-hot nhị phân: has_digit_0..9 (quyết
                                   định nhãn). Decoy (all_distinct/
                                   has_repeated_digit/contains_closed_loop)
                                   tạm bỏ — xem multiconcept_concepts.py.
  - concept_mask  scalar          1 nếu concept ground-truth được "công bố" cho
                                   sample này (chỉ áp dụng cho train, mô phỏng
                                   SkinCon chỉ phủ 22% ảnh Fitzpatrick — val/test
                                   luôn được công bố đầy đủ để đánh giá công bằng)
  - label         scalar          nhãn 3 lớp (even/equal/odd), TẤT ĐỊNH từ
                                   digits — không sample xác suất nữa. Impurity
                                   của rule-cluster xuất hiện tự nhiên từ việc
                                   concept has_digit_X không ghi nhận số lần lặp
                                   (xem docstring multiconcept_concepts.py)

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
    DIGIT_CONCEPTS,
    DECOY_CONCEPTS,
    compute_concept_vector,
    label_of,
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

        concepts_l.append(compute_concept_vector(digits))
        concept_mask.append(1 if rng.random() < concept_supervision_ratio else 0)
        labels_l.append(label_of(digits))
        digits_l.append(list(digits))

    return {
        "images":       torch.stack(images, dim=0),
        "concepts":     torch.tensor(concepts_l,   dtype=torch.float32),
        "concept_mask": torch.tensor(concept_mask, dtype=torch.float32),
        "label":        torch.tensor(labels_l,      dtype=torch.long),
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
        "task": "K MNIST digits -> has_digit_0..9 concept vector -> label = "
                "majority parity (n_odd vs n_even) among the K digits",
        "purpose": "bridge dataset between mnist_math and fitzpatrick17k. "
                   "label is a DETERMINISTIC, human-checkable function of the "
                   "digits (count odd vs even) -- no weight table to look up, "
                   "unlike v1. Cluster impurity emerges naturally because "
                   "has_digit_X concepts record PRESENCE only, losing "
                   "multiplicity info (e.g. (3,3,4,4) and (3,3,3,4) share the "
                   "same concept pattern {has_digit_3,has_digit_4} but differ "
                   "in true label). Decoy concepts (all_distinct, "
                   "has_repeated_digit, contains_closed_loop) temporarily "
                   "removed -- see multiconcept_concepts.py -- to keep every "
                   "concept shown in a rule directly relevant to the label.",
        "image_shape": [1, image_height, symbol_width * num_digits],
        "symbol_width": symbol_width,
        "image_height": image_height,
        "num_digits": num_digits,
        "num_digits_note": "K is fixed for this experiment; a natural future "
                           "extension is variable K per image (not all "
                           "dermatology images activate the same number of "
                           "concepts either).",
        "num_concepts": NUM_CONCEPTS,
        "concept_names": CONCEPT_NAMES,
        "digit_concepts": DIGIT_CONCEPTS,
        "decoy_concepts": DECOY_CONCEPTS,
        "label_names": LABEL_NAMES,
        "num_labels": NUM_LABELS,
        "label_rule": "n_odd = count of odd digits among the K digits; "
                      "n_odd > K/2 -> odd; n_odd < K/2 -> even; else -> equal",
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
