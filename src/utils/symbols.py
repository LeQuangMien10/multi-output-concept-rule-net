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

CONCEPT_ORDER = ["digit1", "op1", "digit2", "op2", "digit3"]

CONCEPT_SPECS = {
    "digit1": 10,
    "op1":    5,
    "digit2": 10,
    "op2":    5,
    "digit3": 10,
}

# v2 helpers
LABEL_KEYS          = ["digit1", "op1", "digit2", "op2", "digit3"]
INPUT_CONCEPT_KEYS  = ["digit1", "op1", "digit2", "op2"]
TARGET_KEY          = "digit3"


def expression_to_string(
    digit1: int,
    op1_id: int,
    digit2: int,
    op2_id: int | None = None,
    digit3: int | None = None,
) -> str:
    """
    Render expression. Tự động chọn format:
        - Có digit3: "a + b = c"  (v1 full, hoặc v2 với answer)
        - Không digit3: "a + b ="  (v2 display, không show answer)
    """
    op1 = ID_TO_SYMBOL[int(op1_id)]
    if digit3 is not None:
        op2 = ID_TO_SYMBOL[int(op2_id)] if op2_id is not None else "="
        return f"{int(digit1)} {op1} {int(digit2)} {op2} {int(digit3)}"
    return f"{int(digit1)} {op1} {int(digit2)} ="


def rule_to_string(digit1: int, op1_id: int, digit2: int, digit3: int) -> str:
    op1 = ID_TO_SYMBOL[int(op1_id)]
    return f"{int(digit1)} {op1} {int(digit2)} = {int(digit3)}"


def build_concept_index_names() -> list[str]:
    names: list[str] = []
    for i in range(10):  names.append(f"digit1={i}")
    for s in OP_SYMBOLS: names.append(f"op1={s}")
    for i in range(10):  names.append(f"digit2={i}")
    for s in OP_SYMBOLS: names.append(f"op2={s}")
    for i in range(10):  names.append(f"digit3={i}")
    return names