"""
fitzpatrick_concepts.py — Định nghĩa concept + nhãn cho Fitzpatrick17k + SkinCon
===================================================================================

Tương tự multiconcept_concepts.py (MNIST-MultiConcept) nhưng cho dữ liệu thật.
Khác biệt căn bản: concept ở đây KHÔNG tất định từ nhãn — đây chính là giới hạn
thật của bài toán, không phải thứ có thể "fix" như has_digit_X (xem phân tích
rủi ro đã trao đổi trước khi triển khai).

CONCEPT_NAMES: 35/48 concept SkinCon gốc — đã bỏ 13 concept có <20 ảnh dương
tính trong toàn dataset (Flat topped, Translucent, Macule, Purpura/Petechiae,
Salmon, Acuminate, Cyst, Abscess, Blue, Poikiloderma, Burrow, Gray, Pigmented —
xem outputs/fitzpatrick_audit/metadata_audit.json). Quyết định này do người
dùng chốt (giống việc từng bỏ tạm decoy concept ở MultiConcept): bỏ tạm để vòng
đầu dễ đọc/debug hơn, không phải kết luận các concept đó không quan trọng về
mặt lâm sàng. Giữ nguyên thứ tự cột gốc trong skincon.csv (không sort theo tần
suất) để nếu cần bật lại 1 concept nào đó chỉ cần thêm vào đúng vị trí.

LABEL_NAMES: three_partition_label — 3 lớp (benign/malignant/non-neoplastic,
thứ tự alphabet để cố định canonical, KHÔNG mang ý nghĩa thứ bậc). Đã chốt
dùng 3 lớp (không dùng nine_partition_label) để có baseline đối chiếu được
với paper gốc Groh et al. 2021 (~62.4% accuracy 3 lớp).
"""
from __future__ import annotations

# 48 cột concept gốc trong skincon.csv, trừ 13 concept hiếm (<20 ảnh dương tính
# trong toàn bộ 16,577 ảnh — đo tại outputs/fitzpatrick_audit/metadata_audit.json,
# key "concepts_with_lt20_images"). Giữ nguyên thứ tự cột gốc.
CONCEPT_NAMES: list[str] = [
    "Vesicle", "Papule", "Plaque", "Pustule", "Bulla", "Patch", "Nodule",
    "Ulcer", "Crust", "Erosion", "Excoriation", "Atrophy", "Exudate",
    "Fissure", "Induration", "Xerosis", "Telangiectasia", "Scale", "Scar",
    "Friable", "Sclerosis", "Pedunculated", "Exophytic/Fungating",
    "Warty/Papillomatous", "Dome-shaped", "Brown(Hyperpigmentation)",
    "White(Hypopigmentation)", "Purple", "Yellow", "Black", "Erythema",
    "Comedo", "Lichenification", "Umbilicated", "Wheal",
]
NUM_CONCEPTS: int = len(CONCEPT_NAMES)   # 35

# Concept bị loại — giữ lại tên để dễ bật lại / để script khác tham chiếu khi
# cần giải thích vì sao thiếu (không dùng trực tiếp trong training).
DROPPED_RARE_CONCEPTS: list[str] = [
    "Flat topped", "Translucent", "Macule", "Purpura/Petechiae", "Salmon",
    "Acuminate", "Cyst", "Abscess", "Blue", "Poikiloderma", "Burrow",
    "Gray", "Pigmented",
]

# three_partition_label — thứ tự alphabet, cố định canonical index.
LABEL_NAMES: list[str] = ["benign", "malignant", "non-neoplastic"]
NUM_LABELS: int = len(LABEL_NAMES)
LABEL_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}

# qc flag của tác giả gốc đánh dấu nhãn sai — loại thẳng khi chuẩn bị dữ liệu.
QC_WRONGLY_LABELLED_PREFIX = "3 Wrongly"


# ─────────────────────────────────────────────────────────────
# Layout FULL concept vector cho ICRL clustering (Stage 2) — nhãn dự đoán bởi
# S1 nối vào sau các concept nhị phân, giống hệt pattern FULL_CONCEPT_OFFSETS
# trong multiconcept_concepts.py / rule_memory.py.
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

FULL_CV_DIM: int = _off   # NUM_CONCEPTS + NUM_LABELS = 35 + 3 = 38
