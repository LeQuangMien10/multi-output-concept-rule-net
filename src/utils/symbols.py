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

CONCEPT_SPECS = {
    "digit1": 10,
    "op1": 5,
    "digit2": 10,
    "op2": 5,
    "digit3": 10,
}

CONCEPT_ORDER = ["digit1", "op1", "digit2", "op2", "digit3"]