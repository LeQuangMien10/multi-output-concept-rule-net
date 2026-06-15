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


import random
import math
from PIL import Image, ImageDraw, ImageFilter


def make_symbol_image(symbol: str, size: int = 28) -> Image.Image:
    """
    Tạo ảnh ký hiệu toán học giả lập nét tay với nhiều biến thể:
    - Độ dài nét (stroke length)
    - Độ lệch tâm (center offset)
    - Độ dày nét (stroke width)
    - Độ nghiêng nét (tilt/skew)
    - Độ cong nét (curvature via multi-segment)
    - Áp lực bút (pressure: fade ở đầu/cuối nét)
    - Chấn động tay (jitter)
    - Blur + noise

    Supported: +, =, -, *, /
    """
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)

    # ─── helpers ───────────────────────────────────────────────────────────────

    def clamp(x, lo=0, hi=None):
        if hi is None:
            hi = size - 1
        return max(lo, min(int(round(x)), hi))

    def jitter(scale=1.0):
        """Rung tay nhỏ."""
        return random.gauss(0, scale)

    def draw_wobbly_line(draw, x1, y1, x2, y2, fill, width, segments=4, wobble=0.8):
        """
        Vẽ đoạn thẳng qua nhiều đoạn nhỏ với nhiễu nhỏ ở điểm giữa
        → nét bút trông tự nhiên hơn nét thẳng hoàn hảo.
        """
        pts = [(x1, y1)]
        for i in range(1, segments):
            t = i / segments
            mx = x1 + (x2 - x1) * t + jitter(wobble)
            my = y1 + (y2 - y1) * t + jitter(wobble)
            pts.append((mx, my))
        pts.append((x2, y2))

        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            # Giả lập áp lực bút: hai đầu mảnh hơn giữa
            t = (i + 0.5) / (len(pts) - 1)
            pressure = math.sin(t * math.pi)          # 0→1→0
            w = max(1, int(round(width * (0.65 + 0.35 * pressure))))
            draw.line(
                (clamp(ax), clamp(ay), clamp(bx), clamp(by)),
                fill=fill, width=w
            )

    def rotate_point(x, y, cx, cy, angle_deg):
        a = math.radians(angle_deg)
        dx, dy = x - cx, y - cy
        rx = dx * math.cos(a) - dy * math.sin(a) + cx
        ry = dx * math.sin(a) + dy * math.cos(a) + cy
        return rx, ry

    # ─── tham số biến thể toàn cục ─────────────────────────────────────────────

    # Lệch tâm: phân phối Gauss rộng hơn, thỉnh thoảng lệch hẳn
    cx = size / 2 + random.gauss(0, 1.8)
    cy = size / 2 + random.gauss(0, 1.8)

    # Độ dày nét: dải rộng hơn, phân phối log-normal giả lập bút bi/bút chì/marker
    base_thick = random.choice([
        random.uniform(1.2, 2.0),   # mảnh (bút chì)
        random.uniform(2.0, 3.2),   # trung bình (bút bi)
        random.uniform(3.2, 4.5),   # dày (marker)
    ])
    thickness = int(round(base_thick))

    # Độ nghiêng của toàn ký hiệu (áp dụng cuối)
    global_tilt = random.gauss(0, 8)        # ±8° phân phối chuẩn, thỉnh thoảng lệch nhiều

    # Độ dài nét: margin & half_len thay đổi theo chiều bút
    margin      = random.uniform(4.5, 9.0)
    half_len_h  = random.uniform(6.5, 10.5)
    half_len_v  = random.uniform(6.5, 10.5)

    # Số đoạn nhỏ & biên độ rung
    segs   = random.choice([3, 4, 5, 6])
    wobble = random.uniform(0.3, 1.2)

    # ─── vẽ theo ký hiệu ───────────────────────────────────────────────────────

    if symbol == "+":
        # Hai nét không nhất thiết đồng đều về chiều dài
        hl = half_len_h * random.uniform(0.85, 1.15)
        vl = half_len_v * random.uniform(0.85, 1.15)

        # Nét ngang (có thể hơi nghiêng độc lập)
        h_tilt = random.gauss(0, 3)
        hx1, hy1 = rotate_point(cx - hl, cy, cx, cy, h_tilt)
        hx2, hy2 = rotate_point(cx + hl, cy, cx, cy, h_tilt)
        draw_wobbly_line(draw, hx1, hy1, hx2, hy2, 255, thickness, segs, wobble)

        # Nét dọc (có thể hơi nghiêng độc lập)
        v_tilt = random.gauss(0, 3)
        vx1, vy1 = rotate_point(cx, cy - vl, cx, cy, v_tilt)
        vx2, vy2 = rotate_point(cx, cy + vl, cx, cy, v_tilt)
        draw_wobbly_line(draw, vx1, vy1, vx2, vy2, 255, thickness, segs, wobble)

    elif symbol == "=":
        gap = random.uniform(4.0, 8.0)

        # Hai nét không song song hoàn toàn
        left_x  = margin + random.gauss(0, 1.0)
        right_x = size - margin + random.gauss(0, 1.0)

        tilt1 = random.gauss(0, 3)
        tilt2 = random.gauss(0, 3)

        # Nét trên
        ux1, uy1 = rotate_point(left_x,  cy - gap / 2, cx, cy, tilt1)
        ux2, uy2 = rotate_point(right_x, cy - gap / 2, cx, cy, tilt1)
        draw_wobbly_line(draw, ux1, uy1, ux2, uy2, 255, thickness, segs, wobble)

        # Nét dưới (bắt đầu & kết thúc không đều với nét trên)
        shift_l = random.gauss(0, 1.5)
        shift_r = random.gauss(0, 1.5)
        lx1, ly1 = rotate_point(left_x  + shift_l, cy + gap / 2, cx, cy, tilt2)
        lx2, ly2 = rotate_point(right_x + shift_r, cy + gap / 2, cx, cy, tilt2)
        draw_wobbly_line(draw, lx1, ly1, lx2, ly2, 255, thickness, segs, wobble)

    elif symbol == "-":
        left_x  = margin + random.gauss(0, 1.2)
        right_x = size - margin + random.gauss(0, 1.2)
        tilt = random.gauss(0, 4)
        x1, y1 = rotate_point(left_x,  cy, cx, cy, tilt)
        x2, y2 = rotate_point(right_x, cy, cx, cy, tilt)
        draw_wobbly_line(draw, x1, y1, x2, y2, 255, thickness, segs, wobble)

    elif symbol == "*":
        # × kiểu: hai đường chéo với góc không nhất thiết là 45°
        angle_bias = random.gauss(0, 6)   # lệch góc so với chuẩn
        reach = random.uniform(7.0, 11.0)

        for base_angle in [45 + angle_bias, 135 + angle_bias]:
            a = math.radians(base_angle)
            ax = cx + reach * math.cos(a) + jitter(0.5)
            ay = cy + reach * math.sin(a) + jitter(0.5)
            bx = cx - reach * math.cos(a) + jitter(0.5)
            by = cy - reach * math.sin(a) + jitter(0.5)
            draw_wobbly_line(draw, ax, ay, bx, by, 255, thickness, segs, wobble)

    elif symbol == "/":
        # Độ dài hai đầu không đều
        top_shift    = random.gauss(0, 1.5)
        bottom_shift = random.gauss(0, 1.5)
        x1 = size - margin + bottom_shift
        y1 = size - margin
        x2 = margin + top_shift
        y2 = margin
        draw_wobbly_line(draw, x1, y1, x2, y2, 255, thickness, segs, wobble)

    else:
        raise ValueError(f"Unsupported symbol: {symbol}")

    # ─── xoay toàn ký hiệu ─────────────────────────────────────────────────────
    img = img.rotate(
        global_tilt,
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
        expand=False
    )

    # ─── blur (mô phỏng mực loang / bút mềm) ───────────────────────────────────
    # Xác suất cao hơn, bán kính rộng hơn cho nét dày
    blur_prob   = 0.80
    blur_radius = random.uniform(0.3, 0.6) + (base_thick - 2.0) * 0.06
    if random.random() < blur_prob:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # ─── noise nhẹ (hạt bút chì / bụi scanner) ────────────────────────────────
    # if random.random() < 0.4:
    #     import numpy as np
    #     arr  = np.array(img, dtype=np.float32)
    #     arr += np.random.normal(0, random.uniform(3, 8), arr.shape)
    #     arr  = np.clip(arr, 0, 255).astype(np.uint8)
    #     img  = Image.fromarray(arr)

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