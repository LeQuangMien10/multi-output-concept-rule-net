"""
multiconcept_concepts.py — Định nghĩa concept + luật sinh nhãn cho MNIST-MultiConcept
========================================================================================

Dataset trung gian giữa MNIST Math và Fitzpatrick17k. Khác với MNIST Math
(5 slot loại trừ nhau, nhãn tất định từ digit1/op1/digit2), dataset này có:

  - Concept NHỊ PHÂN ĐỘC LẬP, không loại trừ nhau (giống 48 concept SkinCon)
    — một ảnh có thể vừa has_repeated_digit vừa sum_high vừa contains_closed_loop.
  - Tần suất concept LỆCH mạnh: có concept phổ biến (~60%) lẫn concept hiếm (~1-2%),
    mô phỏng phổ tần suất Erythema (58%) vs Burrow (0.1%) của SkinCon.
  - Nhãn đích (3 lớp, đặt tên song song với three_partition_label của Fitzpatrick:
    non_neoplastic/benign/malignant) là hàm XÁC SUẤT của MỘT TẬP CON concept
    (không phải toàn bộ) — các concept còn lại là "nhiễu"/không liên quan tới nhãn,
    giống việc không phải concept da liễu nào cũng mang tính chẩn đoán.
  - Nhãn được SAMPLE từ softmax (không lấy argmax) → cùng một concept vector có thể
    ra nhãn khác nhau giữa các lần sample → rule-cluster KHÔNG thuần khiết 100% như
    MNIST Math (nơi label = f(d1,op,d2) tất định tuyệt đối).

Vì ta biết trước z-score/weight thật, có thể so sánh định lượng rule mà ICRL học
được với luật sinh thật — điều không thể làm trên Fitzpatrick thật.
"""
from __future__ import annotations

import math
import random


# ─────────────────────────────────────────────────────────────
# Concept definitions — mỗi concept là 1 hàm digits(tuple[int,...]) -> 0/1
# ─────────────────────────────────────────────────────────────

def _has_digit(value: int):
    return lambda d: int(value in d)


CONCEPT_FUNCS: dict[str, "callable"] = {
    "has_digit_0":          _has_digit(0),
    "has_digit_7":          _has_digit(7),
    "has_repeated_digit":   lambda d: int(len(set(d)) < len(d)),
    "all_distinct":         lambda d: int(len(set(d)) == len(d)),
    "sum_high":             lambda d: int(sum(d) > 5 * len(d)),
    "sum_low":              lambda d: int(sum(d) < 2 * len(d)),
    "contains_closed_loop": lambda d: int(any(x in (0, 6, 8, 9) for x in d)),
    "majority_even":        lambda d: int(sum(1 for x in d if x % 2 == 0) > len(d) / 2),
    "majority_odd":         lambda d: int(sum(1 for x in d if x % 2 == 1) > len(d) / 2),
    "leftmost_is_max":      lambda d: int(d[0] == max(d)),
    "rightmost_is_min":     lambda d: int(d[-1] == min(d)),
    "strictly_increasing":  lambda d: int(all(d[i] < d[i + 1] for i in range(len(d) - 1))),
    "strictly_decreasing":  lambda d: int(all(d[i] > d[i + 1] for i in range(len(d) - 1))),
    "is_palindrome":        lambda d: int(tuple(d) == tuple(reversed(d))),
    "has_pair_sum10":       lambda d: int(any(d[i] + d[j] == 10
                                               for i in range(len(d))
                                               for j in range(i + 1, len(d)))),
    "max_digit_ge8":        lambda d: int(max(d) >= 8),
}

CONCEPT_NAMES: list[str] = list(CONCEPT_FUNCS.keys())
NUM_CONCEPTS: int = len(CONCEPT_NAMES)


# ─────────────────────────────────────────────────────────────
# Luật sinh nhãn — chỉ MỘT TẬP CON concept mang tính "chẩn đoán",
# các concept còn lại có weight=0 (nhiễu/không liên quan tới nhãn).
# ─────────────────────────────────────────────────────────────

LABEL_NAMES: list[str] = ["non_neoplastic", "benign", "malignant"]
NUM_LABELS: int = len(LABEL_NAMES)

# Bias mặc định (khi không concept nào kích hoạt) — thiên về non_neoplastic,
# giống phân bố lệch thật của Fitzpatrick (73% non-neoplastic).
LABEL_BIAS: list[float] = [1.0, -0.5, -1.0]

# weight[concept] = [Δnon_neoplastic, Δbenign, Δmalignant]
# Chỉ 8/16 concept có mặt ở đây — 8 concept còn lại weight=0 (decoy).
LABEL_WEIGHTS: dict[str, list[float]] = {
    "has_repeated_digit":   [0.0, 0.5, 1.2],
    "sum_high":              [0.0, 0.3, 1.0],
    "contains_closed_loop":  [0.0, 0.6, 0.3],
    "strictly_increasing":   [0.0, 1.5, -1.5],   # concept hiếm, tín hiệu benign mạnh
    "is_palindrome":         [0.0, -1.0, 1.8],   # concept hiếm, tín hiệu malignant mạnh
    "has_pair_sum10":        [0.0, 0.2, 0.6],
    "max_digit_ge8":         [0.0, 0.2, 0.7],
    "has_digit_0":           [0.0, -0.4, -0.6],  # bảo vệ (giảm nguy cơ)
}

INFORMATIVE_CONCEPTS: list[str] = list(LABEL_WEIGHTS.keys())
DECOY_CONCEPTS: list[str] = [c for c in CONCEPT_NAMES if c not in LABEL_WEIGHTS]


def compute_concept_vector(digits: tuple[int, ...]) -> list[int]:
    """digits (K chữ số) -> concept vector nhị phân [NUM_CONCEPTS]."""
    return [CONCEPT_FUNCS[name](digits) for name in CONCEPT_NAMES]


def compute_label_logits(concept_vec: list[int]) -> list[float]:
    z = list(LABEL_BIAS)
    for name, value in zip(CONCEPT_NAMES, concept_vec):
        if value and name in LABEL_WEIGHTS:
            w = LABEL_WEIGHTS[name]
            z = [z[i] + w[i] for i in range(NUM_LABELS)]
    return z


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    exps = [math.exp(x - m) for x in z]
    s = sum(exps)
    return [x / s for x in exps]


def compute_label_probs(concept_vec: list[int]) -> list[float]:
    return softmax(compute_label_logits(concept_vec))


def sample_label(concept_vec: list[int], rng: random.Random) -> tuple[int, list[float]]:
    """Sample nhãn (KHÔNG lấy argmax) từ phân phối xác suất thật.

    Đây là nguồn gốc của cluster impurity: hai ảnh cùng concept vector
    vẫn có thể ra 2 nhãn khác nhau, vì nhãn được rút mẫu ngẫu nhiên từ
    p, không phải hàm tất định của concept.
    """
    probs = compute_label_probs(concept_vec)
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i, probs
    return len(probs) - 1, probs
