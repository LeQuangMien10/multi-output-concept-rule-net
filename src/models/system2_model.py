import torch
import torch.nn.functional as F
import torch.nn as nn


class System2(nn.Module):
    """
    Rule-based reasoning system
    """

    def __init__(self, rule_memory):
        super().__init__()
        self.rule_memory = rule_memory

    def match(self, z):
        """
        z: [B, 6] concept vector
        """

        rules, rule_logits = self.rule_memory()

        # normalize
        z = z.float()
        rules = torch.sigmoid(rules)

        # similarity: [B, R]
        score = torch.einsum("bd,rd->br", z, rules)

        # soft selection
        weight = F.softmax(score, dim=-1)

        # aggregate rule outputs
        logits = torch.einsum("br,rc->bc", weight, rule_logits)

        # explanation: best rule index
        best_rule = weight.argmax(dim=-1)

        return logits, best_rule, weight