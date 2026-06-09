import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFilter
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm

from src.utils.seed import set_seed
from src.utils.symbols import SYMBOL_TO_ID, CONCEPT_ORDER, CONCEPT_SPECS


def make_symbol_image(symbol: str, size: int = 28) -> Image.Image:
    """
    Create a grayscale image for math symbols with slight randomness
    so that symbols are not perfectly straight / identical.

    Supported: +, =, -, *, /
    """
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)

    def clamp(x, low=0, high=None):
        if high is None:
            high = size - 1
        return max(low, min(int(x), high))

    # random center shift
    cx = size // 2 + random.randint(-2, 2)
    cy = size // 2 + random.randint(-2, 2)

    # random thickness
    thickness = random.randint(2, 4)

    # useful margins / lengths
    margin = random.randint(6, 8)
    half_len_h = random.randint(8, 10)
    half_len_v = random.randint(8, 10)

    if symbol == "+":
        # horizontal stroke with slight slope
        x1 = clamp(cx - half_len_h)
        y1 = clamp(cy + random.randint(-1, 1))
        x2 = clamp(cx + half_len_h)
        y2 = clamp(cy + random.randint(-1, 1))
        draw.line((x1, y1, x2, y2), fill=255, width=thickness)

        # vertical stroke with slight slope
        x3 = clamp(cx + random.randint(-1, 1))
        y3 = clamp(cy - half_len_v)
        x4 = clamp(cx + random.randint(-1, 1))
        y4 = clamp(cy + half_len_v)
        draw.line((x3, y3, x4, y4), fill=255, width=thickness)

    elif symbol == "=":
        gap = random.randint(5, 8)

        # upper line
        x1 = clamp(margin)
        y1 = clamp(cy - gap // 2 + random.randint(-1, 1))
        x2 = clamp(size - margin)
        y2 = clamp(cy - gap // 2 + random.randint(-1, 1))
        draw.line((x1, y1, x2, y2), fill=255, width=thickness)

        # lower line
        x3 = clamp(margin + random.randint(-1, 1))
        y3 = clamp(cy + gap // 2 + random.randint(-1, 1))
        x4 = clamp(size - margin + random.randint(-1, 1))
        y4 = clamp(cy + gap // 2 + random.randint(-1, 1))
        draw.line((x3, y3, x4, y4), fill=255, width=thickness)

    elif symbol == "-":
        x1 = clamp(margin)
        y1 = clamp(cy + random.randint(-1, 1))
        x2 = clamp(size - margin)
        y2 = clamp(cy + random.randint(-1, 1))
        draw.line((x1, y1, x2, y2), fill=255, width=thickness)

    elif symbol == "*":
        # diagonal 1
        draw.line(
            (
                clamp(margin + random.randint(-1, 1)),
                clamp(margin + random.randint(-1, 1)),
                clamp(size - margin + random.randint(-1, 1)),
                clamp(size - margin + random.randint(-1, 1)),
            ),
            fill=255,
            width=thickness,
        )

        # diagonal 2
        draw.line(
            (
                clamp(size - margin + random.randint(-1, 1)),
                clamp(margin + random.randint(-1, 1)),
                clamp(margin + random.randint(-1, 1)),
                clamp(size - margin + random.randint(-1, 1)),
            ),
            fill=255,
            width=thickness,
        )

        # optional center stroke to make it more star-like
        if random.random() < 0.7:
            draw.line(
                (
                    clamp(cx + random.randint(-1, 1)),
                    clamp(margin + random.randint(-1, 1)),
                    clamp(cx + random.randint(-1, 1)),
                    clamp(size - margin + random.randint(-1, 1)),
                ),
                fill=255,
                width=max(1, thickness - 1),
            )

    elif symbol == "/":
        draw.line(
            (
                clamp(size - margin + random.randint(-1, 1)),
                clamp(margin + random.randint(-1, 1)),
                clamp(margin + random.randint(-1, 1)),
                clamp(size - margin + random.randint(-1, 1)),
            ),
            fill=255,
            width=thickness,
        )

    else:
        raise ValueError(f"Unsupported symbol: {symbol}")

    # random tiny speckle noise
    if random.random() < 0.8:
        for _ in range(random.randint(2, 8)):
            px = random.randint(0, size - 1)
            py = random.randint(0, size - 1)
            img.putpixel((px, py), random.choice([0, 255]))

    # slight rotation so symbols are less perfect
    angle = random.uniform(-8, 8)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)

    # very light blur sometimes
    if random.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))

    return img


def build_digit_to_indices(mnist_dataset: MNIST) -> dict[int, list[int]]:
    """
    Build mapping from digit label to sample indices in MNIST.
    """
    digit_to_indices = {i: [] for i in range(10)}

    for idx, (_, label) in enumerate(mnist_dataset):
        digit_to_indices[int(label)].append(idx)

    return digit_to_indices


def sample_digit_image(
    digit: int,
    mnist_dataset: MNIST,
    digit_to_indices: dict[int, list[int]],
) -> Image.Image:
    """
    Sample one MNIST image corresponding to a specific digit.
    """
    idx = random.choice(digit_to_indices[int(digit)])
    img, _ = mnist_dataset[idx]
    return img


def sample_expression(
    valid_ratio: float = 0.5,
    allow_carry: bool = False,
) -> dict[str, int | str]:
    """
    Generate one expression of the form:

        a + b = c

    If allow_carry=False, only valid samples with a+b <= 9 are generated.
    Invalid samples are made by replacing c with a wrong digit.
    """
    make_valid = random.random() < valid_ratio

    if allow_carry:
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        true_c = a + b
        if true_c > 9:
            # For now this project only supports one-symbol digit3.
            # So if carry happens, resample to keep c as a single digit.
            return sample_expression(valid_ratio=valid_ratio, allow_carry=False)
    else:
        a = random.randint(0, 9)
        b = random.randint(0, 9 - a)
        true_c = a + b

    if make_valid:
        c = true_c
        valid = 1
    else:
        wrong_choices = [x for x in range(10) if x != true_c]
        c = random.choice(wrong_choices)
        valid = 0

    return {
        "digit1": a,
        "op1_symbol": "+",
        "digit2": b,
        "op2_symbol": "=",
        "digit3": c,
        "valid": valid,
    }


def render_expression_image(
    expr: dict[str, int | str],
    mnist_dataset: MNIST,
    digit_to_indices: dict[int, list[int]],
    symbol_width: int = 28,
    image_height: int = 28,
) -> Image.Image:
    """
    Render expression image as:

        [digit1][op1][digit2][op2][digit3]

    Final image size:
        width = 5 * symbol_width
        height = image_height
    """
    digit1 = int(expr["digit1"])
    digit2 = int(expr["digit2"])
    digit3 = int(expr["digit3"])
    op1_symbol = str(expr["op1_symbol"])
    op2_symbol = str(expr["op2_symbol"])

    parts = [
        sample_digit_image(digit1, mnist_dataset, digit_to_indices),
        make_symbol_image(op1_symbol, size=symbol_width),
        sample_digit_image(digit2, mnist_dataset, digit_to_indices),
        make_symbol_image(op2_symbol, size=symbol_width),
        sample_digit_image(digit3, mnist_dataset, digit_to_indices),
    ]

    canvas = Image.new("L", (symbol_width * 5, image_height), color=0)

    for i, part in enumerate(parts):
        part = part.resize((symbol_width, image_height))
        canvas.paste(part, (i * symbol_width, 0))

    return canvas


def generate_split(
    split_name: str,
    split_size: int,
    mnist_dataset: MNIST,
    digit_to_indices: dict[int, list[int]],
    valid_ratio: float,
    allow_carry: bool,
    symbol_width: int,
    image_height: int,
) -> dict[str, torch.Tensor]:
    """
    Generate one split and return tensors.
    """
    to_tensor = transforms.ToTensor()

    images = []
    digit1_list = []
    op1_list = []
    digit2_list = []
    op2_list = []
    digit3_list = []
    valid_list = []

    for _ in tqdm(range(split_size), desc=f"Generating {split_name}"):
        expr = sample_expression(
            valid_ratio=valid_ratio,
            allow_carry=allow_carry,
        )

        img = render_expression_image(
            expr=expr,
            mnist_dataset=mnist_dataset,
            digit_to_indices=digit_to_indices,
            symbol_width=symbol_width,
            image_height=image_height,
        )

        images.append(to_tensor(img))

        digit1_list.append(int(expr["digit1"]))
        op1_list.append(SYMBOL_TO_ID[str(expr["op1_symbol"])])
        digit2_list.append(int(expr["digit2"]))
        op2_list.append(SYMBOL_TO_ID[str(expr["op2_symbol"])])
        digit3_list.append(int(expr["digit3"]))
        valid_list.append(int(expr["valid"]))

    split_data = {
        "images": torch.stack(images, dim=0),
        "digit1": torch.tensor(digit1_list, dtype=torch.long),
        "op1": torch.tensor(op1_list, dtype=torch.long),
        "digit2": torch.tensor(digit2_list, dtype=torch.long),
        "op2": torch.tensor(op2_list, dtype=torch.long),
        "digit3": torch.tensor(digit3_list, dtype=torch.long),
        "valid": torch.tensor(valid_list, dtype=torch.long),
    }

    return split_data


def generate_mnist_math_dataset(config: dict[str, Any]) -> None:
    """
    Main dataset generation function.
    """
    dataset_cfg = config["dataset"]

    root_dir = Path(dataset_cfg["root_dir"])
    output_dir = Path(dataset_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(dataset_cfg.get("seed", 42))
    set_seed(seed)

    train_size = int(dataset_cfg["train_size"])
    val_size = int(dataset_cfg["val_size"])
    test_size = int(dataset_cfg["test_size"])

    valid_ratio = float(dataset_cfg.get("valid_ratio", 0.5))
    allow_carry = bool(dataset_cfg.get("allow_carry", False))

    symbol_width = int(dataset_cfg.get("symbol_width", 28))
    image_height = int(dataset_cfg.get("image_height", 28))

    print("[INFO] Loading MNIST...")
    mnist_train = MNIST(root=root_dir, train=True, download=True, transform=None)
    mnist_test = MNIST(root=root_dir, train=False, download=True, transform=None)

    train_digit_to_indices = build_digit_to_indices(mnist_train)
    test_digit_to_indices = build_digit_to_indices(mnist_test)

    train_data = generate_split(
        split_name="train",
        split_size=train_size,
        mnist_dataset=mnist_train,
        digit_to_indices=train_digit_to_indices,
        valid_ratio=valid_ratio,
        allow_carry=allow_carry,
        symbol_width=symbol_width,
        image_height=image_height,
    )

    val_data = generate_split(
        split_name="val",
        split_size=val_size,
        mnist_dataset=mnist_test,
        digit_to_indices=test_digit_to_indices,
        valid_ratio=valid_ratio,
        allow_carry=allow_carry,
        symbol_width=symbol_width,
        image_height=image_height,
    )

    test_data = generate_split(
        split_name="test",
        split_size=test_size,
        mnist_dataset=mnist_test,
        digit_to_indices=test_digit_to_indices,
        valid_ratio=valid_ratio,
        allow_carry=allow_carry,
        symbol_width=symbol_width,
        image_height=image_height,
    )

    torch.save(train_data, output_dir / "train.pt")
    torch.save(val_data, output_dir / "val.pt")
    torch.save(test_data, output_dir / "test.pt")

    meta = {
        "name": dataset_cfg.get("name", "mnist_math"),
        "task": "a + b = c",
        "image_shape": [1, image_height, symbol_width * 5],
        "symbol_width": symbol_width,
        "image_height": image_height,
        "num_symbols": 5,
        "concept_order": CONCEPT_ORDER,
        "concept_specs": CONCEPT_SPECS,
        "valid_ratio": valid_ratio,
        "allow_carry": allow_carry,
        "seed": seed,
        "splits": {
            "train": train_size,
            "val": val_size,
            "test": test_size,
        },
    }

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Dataset saved to: {output_dir}")
    print(f"[DONE] train.pt: {train_data['images'].shape}")
    print(f"[DONE] val.pt:   {val_data['images'].shape}")
    print(f"[DONE] test.pt:  {test_data['images'].shape}")