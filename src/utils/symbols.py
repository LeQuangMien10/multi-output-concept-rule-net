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
    "op1": 5,
    "digit2": 10,
    "op2": 5,
    "digit3": 10,
}


def expression_to_string(digit1: int, op1_id: int, digit2: int, op2_id: int, digit3: int) -> str:
    op1 = ID_TO_SYMBOL[int(op1_id)]
    op2 = ID_TO_SYMBOL[int(op2_id)]
    return f"{int(digit1)} {op1} {int(digit2)} {op2} {int(digit3)}"


def build_concept_index_names() -> list[str]:
    """
    Build names for the 40-dim concept vector:
        digit1: 10
        op1: 5
        digit2: 10
        op2: 5
        digit3: 10
    """
    names: list[str] = []

    for i in range(10):
        names.append(f"digit1={i}")

    for symbol in OP_SYMBOLS:
        names.append(f"op1={symbol}")

    for i in range(10):
        names.append(f"digit2={i}")

    for symbol in OP_SYMBOLS:
        names.append(f"op2={symbol}")

    for i in range(10):
        names.append(f"digit3={i}")

    return names