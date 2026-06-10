from __future__ import annotations

import torch


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute accuracy from classification logits.
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / max(total, 1)


@torch.no_grad()
def batch_correct_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    """
    Return number of correct predictions and total samples.
    Useful for accumulating metrics over an epoch.
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct, total


@torch.no_grad()
def compute_expression_correct(outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> tuple[int, int]:
    """
    Expression is correct only if all five concept predictions are correct:

        digit1, op1, digit2, op2, digit3

    Returns:
        correct_count, total_count
    """
    concept_keys = ["digit1", "op1", "digit2", "op2", "digit3"]

    all_correct = None

    for key in concept_keys:
        preds = outputs[key].argmax(dim=-1)
        current_correct = preds == labels[key]

        if all_correct is None:
            all_correct = current_correct
        else:
            all_correct = all_correct & current_correct

    correct = all_correct.sum().item()
    total = labels["digit1"].numel()
    return correct, total


class AverageMeter:
    """
    Track average values such as loss.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


class AccuracyMeter:
    """
    Track accuracy by accumulating correct / total.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, correct: int, total: int) -> None:
        self.correct += int(correct)
        self.total += int(total)

    @property
    def acc(self) -> float:
        return self.correct / max(self.total, 1)