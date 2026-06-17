import torch
import torch.nn as nn


class RuleMemory(nn.Module):
    """
    Each rule is a soft mask over concept dimensions (6 dims groups)
    """

    def __init__(self, num_rules=64, concept_dim=6):
        super().__init__()

        self.num_rules = num_rules
        self.concept_dim = concept_dim

        # learnable rule masks
        self.rules = nn.Parameter(
            torch.randn(num_rules, concept_dim)
        )

        # rule validity logits
        self.rule_logits = nn.Parameter(
            torch.zeros(num_rules, 2)
        )

    def forward(self):
        return self.rules, self.rule_logits