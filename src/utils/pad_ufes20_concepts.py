"""
pad_ufes20_concepts.py — Định nghĩa concept + nhãn cho PAD-UFES-20
=====================================================================

Tương tự fitzpatrick_concepts.py nhưng cho PAD-UFES-20 (Mendeley,
data.mendeley.com/datasets/zr7vgbcyr2/1). Khác biệt căn bản: dataset này
không có concept hình thái do bác sĩ gán như SkinCon — concept vocabulary
ở đây là 6 cột triệu chứng bệnh nhân tự khai đã có sẵn trong metadata.csv
(quyết định đã chốt trước khi triển khai, xem ghi chú dự án).

CONCEPT_NAMES: 6 cột boolean trong metadata.csv (itch, grew, hurt, changed,
bleed, elevation). Khác Fitzpatrick+SkinCon: đây là triệu chứng bệnh nhân
TỰ KHAI, không phải concept hình thái nhìn thấy trên ảnh — System~1 có thể
khó dự đoán tốt các concept này chỉ từ pixel, đây là rủi ro đã biết trước,
không phải lỗi pipeline nếu concept acc/F1 thấp hơn Fitzpatrick.

Missing data: mỗi cột có một số dòng "UNK" (bệnh nhân không trả lời) --
KHÁC Fitzpatrick (nơi 1 ảnh hoặc có đủ 35 concept hoặc không có concept nào).
Ở đây UNK xảy ra theo TỪNG concept riêng lẻ, nên mask phải là 1 vector
[NUM_CONCEPTS] mỗi sample, không phải 1 scalar/ảnh như concept_mask của
Fitzpatrick (xem PadUfes20Dataset).

LABEL_NAMES: cột "diagnostic" gốc, giữ nguyên 6 lớp (NEV, BCC, ACK, SEK,
SCC, MEL) -- không gộp về benign/malignant, để còn so sánh được với các
baseline khác trong literature PAD-UFES-20 (Pacheco & Krohling và các bài
dùng chung dataset này đều báo cáo trên đúng 6 lớp này). Thứ tự alphabet để
cố định canonical index, không mang ý nghĩa thứ bậc.
"""
from __future__ import annotations

CONCEPT_NAMES: list[str] = ["itch", "grew", "hurt", "changed", "bleed", "elevation"]
NUM_CONCEPTS: int = len(CONCEPT_NAMES)

# Giá trị UNK gốc trong metadata.csv -- xử lý riêng ở prepare_dataset.py
# (encode value=0 tạm + mask=0 cho các ô này, không suy diễn giá trị thật).
UNK_VALUE = "UNK"

# diagnostic -- 6 lớp gốc, thứ tự alphabet.
LABEL_NAMES: list[str] = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
NUM_LABELS: int = len(LABEL_NAMES)
LABEL_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}


# ─────────────────────────────────────────────────────────────
# Layout FULL concept vector cho ICRL clustering (Stage 2) -- nhãn dự đoán
# bởi S1 nối vào sau các concept nhị phân, giống hệt pattern
# FULL_CONCEPT_OFFSETS trong fitzpatrick_concepts.py.
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

FULL_CV_DIM: int = _off   # NUM_CONCEPTS + NUM_LABELS = 6 + 6 = 12
