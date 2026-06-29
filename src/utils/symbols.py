DIGIT_CLASSES = list(range(10))

OP_SYMBOLS = ["+", "-", "*", "/", "="]

SYMBOL_TO_ID = {
    "+": 0,
    "-": 1,
    "*": 2,
    "/": 3,
    "=": 4,
}

ID_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_ID.items()}

# ── Format mới: "a op b = ?" — predict digit3 ────────────────────────────────
# Concept slots dùng làm INPUT cho System2 (không bao gồm digit3 = target)
CONCEPT_ORDER = ["digit1", "op1", "digit2", "op2", "digit3"]

CONCEPT_SPECS = {
    "digit1": 10,
    "op1":    5,
    "digit2": 10,
    "op2":    5,
    "digit3": 10,   # target — System2 predicts này từ rule
}

# Label keys trong dataset .pt file
LABEL_KEYS = ["digit1", "op1", "digit2", "op2", "digit3"]

# Concept keys dùng làm INPUT của System2 (không tính digit3)
INPUT_CONCEPT_KEYS  = ["digit1", "op1", "digit2", "op2"]

# Target key để System2 predict
TARGET_KEY = "digit3"


def expression_to_string(digit1: int, op1_id: int, digit2: int) -> str:
    """Render expression dạng 'a + b = ?'"""
    op1 = ID_TO_SYMBOL[int(op1_id)]
    return f"{int(digit1)} {op1} {int(digit2)} = ?"


def rule_to_string(digit1: int, op1_id: int, digit2: int, digit3: int) -> str:
    """Render rule đầy đủ dạng 'a + b = c'"""
    op1 = ID_TO_SYMBOL[int(op1_id)]
    return f"{int(digit1)} {op1} {int(digit2)} = {int(digit3)}"


def build_concept_index_names() -> list[str]:
    """
    Build names cho concept vector 40-dim (giữ nguyên để backward compat):
        digit1: 10  digit2: 10  digit3: 10
        op1: 5      op2: 5
    """
    names: list[str] = []
    for i in range(10):    names.append(f"digit1={i}")
    for s in OP_SYMBOLS:   names.append(f"op1={s}")
    for i in range(10):    names.append(f"digit2={i}")
    for s in OP_SYMBOLS:   names.append(f"op2={s}")
    for i in range(10):    names.append(f"digit3={i}")
    return names