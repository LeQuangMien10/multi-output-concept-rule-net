"""
cub_concepts.py — Định nghĩa concept + nhãn cho CUB-200-2011
=============================================================

Khác Fitzpatrick/PAD-UFES-20: concept và nhãn của CUB không phải hằng số viết
tay ở đây mà được suy ra từ chính bộ dữ liệu gốc lúc chạy prepare_dataset.py
(312 attribute, 200 loài), rồi lưu vào <data_dir>/meta.json. Mọi script khác
đọc schema qua load_schema(data_dir).

Concept: 112 attribute nhị phân do Koh et al. (ICML 2020) chọn, dạng class-level
(majority vote theo loài trên tập train, mọi ảnh cùng loài dùng chung một vector
concept). Danh sách chỉ số bên dưới là chỉ số 0-based trên 312 attribute gốc
(attributes.txt), chép từ repo Concept Embedding Models
(github.com/mateoespinosa/cem, cem/data/CUB200/cub_loader.py, comment "Set of
CUB attributes selected by original CBM paper"), vốn chuyển thể từ repo gốc
github.com/yewsiang/ConceptBottleneck. prepare_dataset.py tự tính lại phép chọn
theo quy trình của Koh để đối chiếu với danh sách này.
"""
from __future__ import annotations

import json
from pathlib import Path

NUM_RAW_ATTRIBUTES: int = 312
NUM_CLASSES: int = 200
MIN_CLASS_COUNT: int = 10   # Koh 2020: giữ attribute có mặt (sau majority vote) ở >= 10 loài

KOH_SELECTED_CONCEPTS: list[int] = [
    1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56,
    57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101, 106, 110, 111, 116, 117, 119,
    125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179,
    181, 183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221,
    225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277,
    283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311,
]
assert len(KOH_SELECTED_CONCEPTS) == 112

# Cùng key với Fitzpatrick/PAD-UFES-20: slot nhãn S1 dự đoán nối sau các concept.
S1_LABEL_CONCEPT_KEY: str = "s1_label_pred"


def load_schema(data_dir: str | Path) -> dict:
    """Đọc <data_dir>/meta.json do prepare_dataset.py ghi ra. Trả về dict có ít
    nhất concept_names (list[str]) và label_names (list[str])."""
    with open(Path(data_dir) / "meta.json", encoding="utf-8") as f:
        return json.load(f)
