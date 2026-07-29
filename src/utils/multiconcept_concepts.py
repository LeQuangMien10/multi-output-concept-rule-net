"""
multiconcept_concepts.py — Định nghĩa concept + luật sinh nhãn cho MNIST-MultiConcept
========================================================================================

v2 — "Parity": concept và label liên hệ TRỰC TIẾP bằng 1 phép toán phổ quát
(đếm), không phải bảng trọng số tùy ý phải tra cứu.

Ảnh: K=4 chữ số MNIST ghép ngang. Concept:
  - has_digit_0 .. has_digit_9 (10 concept): chữ số GIÁ TRỊ v có xuất hiện
    trong ảnh hay không (không phân biệt xuất hiện mấy lần).
  - 3 decoy đơn giản, dễ nhận biết, KHÔNG ảnh hưởng nhãn: all_distinct,
    has_repeated_digit, contains_closed_loop.

Nhãn = so sánh số lượng chữ số lẻ vs chẵn trong 4 chữ số:
    n_odd > n_even  → "odd"
    n_even > n_odd  → "even"
    n_odd == n_even → "equal"

Đây là hàm TẤT ĐỊNH, giống hệt digit3 = digit1 − digit2 của MNIST Math:
chỉ cần biết "digit nào lẻ/chẵn" (kiến thức phổ thông), không cần bảng
trọng số nào — is_palindrome trước đây "nặng 1.8 điểm về malignant" phải
tra cứu mới biết; has_digit_3 present bây giờ "là lẻ" thì ai cũng thấy
ngay không cần tra gì.

Impurity của rule-cluster giờ xuất hiện TỰ NHIÊN, không cần sample từ
softmax nữa: concept has_digit_X chỉ ghi nhận "có xuất hiện", MẤT thông
tin về số lần lặp. Ví dụ (3,3,4,4) và (3,3,3,4) cho CÙNG concept pattern
{has_digit_3, has_digit_4} nhưng nhãn thật khác nhau (equal vs odd) — mất
thông tin khi nén ảnh → concept, đúng bản chất khiến concept y khoa
(SkinCon) cũng không quyết định nhãn 100%, không phải nhiễu giả lập.

Vì digit chẵn/lẻ đối xứng tuyệt đối, rule "odd" sẽ có centroid vừa
has_digit_{lẻ} present cao vừa has_digit_{chẵn} ABSENT cao — "NOT" xuất
hiện tự nhiên, không phải trọng số âm tùy ý.

Ghi chú: K=4 cố định (thử nghiệm). Ý tưởng mở rộng: K thay đổi theo ảnh
(giống ảnh da liễu không phải ảnh nào cũng activate cùng số concept) —
để dành cho vòng lặp sau.
"""
from __future__ import annotations


# ─────────────────────────────────────────────────────────────
# Concept definitions — mỗi concept là 1 hàm digits(tuple[int,...]) -> 0/1
# ─────────────────────────────────────────────────────────────

def _has_digit(value: int):
    return lambda d: int(value in d)


DIGIT_CONCEPTS: list[str] = [f"has_digit_{v}" for v in range(10)]

# Decoy: đơn giản, dễ nhận biết bằng mắt, KHÔNG đóng góp vào nhãn (weight=0).
DECOY_CONCEPTS: list[str] = ["all_distinct", "has_repeated_digit", "contains_closed_loop"]

CONCEPT_FUNCS: dict[str, "callable"] = {
    **{f"has_digit_{v}": _has_digit(v) for v in range(10)},
    "all_distinct":         lambda d: int(len(set(d)) == len(d)),
    "has_repeated_digit":   lambda d: int(len(set(d)) < len(d)),
    "contains_closed_loop": lambda d: int(any(x in (0, 6, 8, 9) for x in d)),
}

CONCEPT_NAMES: list[str] = DIGIT_CONCEPTS + DECOY_CONCEPTS
NUM_CONCEPTS: int = len(CONCEPT_NAMES)

# Alias để tương thích code cũ gọi INFORMATIVE_CONCEPTS (= concept quyết định nhãn).
INFORMATIVE_CONCEPTS: list[str] = DIGIT_CONCEPTS


# ─────────────────────────────────────────────────────────────
# Luật sinh nhãn — ĐẾM, không có trọng số nào để tra cứu.
# ─────────────────────────────────────────────────────────────

LABEL_NAMES: list[str] = ["even", "equal", "odd"]
NUM_LABELS: int = len(LABEL_NAMES)


def label_of(digits: tuple[int, ...]) -> int:
    """
    So n_odd (số chữ số lẻ) với n_even (số chữ số chẵn) trong digits.
    Tất định 100% — giống digit3 = digit1 − digit2 của MNIST Math.
    """
    n_odd = sum(1 for d in digits if d % 2 == 1)
    n_even = len(digits) - n_odd
    if n_odd > n_even:
        return LABEL_NAMES.index("odd")
    if n_even > n_odd:
        return LABEL_NAMES.index("even")
    return LABEL_NAMES.index("equal")


def compute_concept_vector(digits: tuple[int, ...]) -> list[int]:
    """digits (K chữ số) -> concept vector nhị phân [NUM_CONCEPTS]."""
    return [CONCEPT_FUNCS[name](digits) for name in CONCEPT_NAMES]


def label_distribution_for_pattern(present_digit_values: set[int], k: int = 4) -> dict[int, int]:
    """
    Diagnostic/debug: với đúng 1 concept pattern (tập giá trị digit xuất
    hiện, bỏ qua decoy), liệt kê phân phối nhãn thật trên MỌI tổ hợp K
    chữ số khớp CHÍNH XÁC tập giá trị đó (dùng để kiểm tra impurity tự
    nhiên — xem phân tích trong hội thoại thiết kế).

    Trả về {label_index: count}. Brute-force 10**k — chỉ dùng offline/debug,
    không gọi trong training loop.
    """
    from itertools import product
    from collections import Counter
    counts = Counter()
    target = frozenset(present_digit_values)
    for combo in product(range(10), repeat=k):
        if frozenset(combo) == target:
            counts[label_of(combo)] += 1
    return dict(counts)


# ─────────────────────────────────────────────────────────────
# Layout của FULL concept vector dùng cho ICRL clustering — nhãn được
# S1 dự đoán như MỘT CONCEPT nữa (softmax 3-way), nối vào sau 13 concept
# nhị phân, giống hệt cách digit3 nằm trong concept vector 40-dim của
# MNIST Math (xem rule_memory.py::CONCEPT_OFFSETS/CONCEPT_DIMS).
#
# QUAN TRỌNG: slot này chỉ dùng để MATCH/CLUSTER (Stage 2). Nhãn dùng để
# train prediction head (Stage 3) vẫn PHẢI lấy từ memory.get_labels()
# (ground-truth ngoài, tracked riêng qua y trong process_batch) — không
# bao giờ đọc trực tiếp giá trị slot này làm nhãn train, vì đó chính là
# lỗi đã phải fix ở MNIST Math (slot mang nhiễu của S1, không phải sự thật).
# ─────────────────────────────────────────────────────────────

S1_LABEL_CONCEPT_KEY: str = "s1_label_pred"

FULL_CONCEPT_KEYS: list[str] = CONCEPT_NAMES + [S1_LABEL_CONCEPT_KEY]

FULL_CONCEPT_DIMS: dict[str, int] = {name: 1 for name in CONCEPT_NAMES}
FULL_CONCEPT_DIMS[S1_LABEL_CONCEPT_KEY] = NUM_LABELS

FULL_CONCEPT_OFFSETS: dict[str, int] = {}
_off = 0
for _name in FULL_CONCEPT_KEYS:
    FULL_CONCEPT_OFFSETS[_name] = _off
    _off += FULL_CONCEPT_DIMS[_name]

FULL_CV_DIM: int = _off   # NUM_CONCEPTS + NUM_LABELS = 13 + 3 = 16
