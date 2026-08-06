"""
crl_rrl_net.py — Dataset-agnostic orchestration for the faithful CRL/RRL
logical-layer baseline (see crl_rrl_components.py for the ported layers and
the exact provenance: https://github.com/obiyoag/crl, Apache-2.0).

Equivalent to the official repo's models/crl.py (the CRL LightningModule)
minus the image backbone + concept_predictor: we plug in the concept vector
already produced by our own (frozen) System 1, exactly like ICRLRuleMemory
and the old crl_system2.py do, so the comparison isolates the System-2
mechanism (differentiable logical rule layers vs. ICRL's incremental
clustering) rather than re-deriving concepts from pixels a second time.

Deliberate simplifications vs. the official repo (documented, not silent):
  - No skip connections (their `use_skip`) — adds cross-layer rule-decoding
    complexity for a gain their own ablations don't isolate.
  - No joint concept-prediction loss (`loss_concept`) — concepts come from
    our already-trained, frozen S1, so there is nothing left to supervise
    here; only their `rrl_loss` (CE) + `l2_penalty` survive.
  - Concept vector convention: our S1 stores post-sigmoid probabilities in
    [0,1], while official CRL binarizes raw logits (threshold 0). We shift
    by -0.5 before BinarizeLayer, which is the exact equivalent
    (probability>0.5 <=> logit>0) — see `to_binarize_input`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.baselines.crl_rrl_components import (
    BinarizeLayer, UnionLayer, LRLayer,
)


def to_binarize_input(concept_vec: torch.Tensor) -> torch.Tensor:
    """Shift [0,1] concept probabilities so BinarizeLayer's threshold-at-0 == threshold-at-0.5."""
    return concept_vec - 0.5


class CRLLogicNet(nn.Module):
    """
    dim_list = [concept_dim, *hidden_dims, num_classes]
        idx 0            -> BinarizeLayer
        idx 1..-2         -> UnionLayer (NOT disabled at the first logical
                             layer, exactly as official CRL does, to avoid a
                             redundant double negation right after Binarize's
                             own optional NOT)
        idx -1            -> LRLayer
    """

    def __init__(
        self,
        concept_dim: int,
        num_classes: int,
        hidden_dims: list[int] | tuple[int, ...] = (64,),
        use_not: bool = True,
    ):
        super().__init__()
        self.concept_dim = concept_dim
        self.num_classes = num_classes
        self.use_not = use_not

        dim_list = [concept_dim, *hidden_dims, num_classes]
        self.layer_list = nn.ModuleList()

        prev_dim = None
        for idx, dim in enumerate(dim_list):
            if idx == 0:
                layer = BinarizeLayer(dim, use_not)
            elif idx == len(dim_list) - 1:
                layer = LRLayer(prev_dim, dim)
            else:
                layer_use_not = use_not if idx != 1 else False
                layer = UnionLayer(prev_dim, dim, use_not=layer_use_not)
            prev_dim = layer.output_dim
            self.layer_list.append(layer)

        self.t = nn.Parameter(torch.zeros(1))  # log-temperature, matches official init (temperature=1)

    # ── Forward (continuous, for training) ──────────────────────────

    def forward(self, concept_vec: torch.Tensor) -> torch.Tensor:
        x = to_binarize_input(concept_vec)
        for layer in self.layer_list:
            x = layer(x)
        return x / torch.exp(self.t)

    # ── Binarized forward (for eval / rule extraction) ───────────────

    @torch.no_grad()
    def bi_forward(self, concept_vec: torch.Tensor, count: bool = False) -> torch.Tensor:
        x = to_binarize_input(concept_vec)
        for layer in self.layer_list:
            x = layer.binarized_forward(x)
            if count and layer.layer_type != "linear":
                layer.node_activation_cnt += torch.sum(x, dim=0)
                layer.forward_tot += x.shape[0]
        return x

    # ── Regularization / weight clipping ─────────────────────────────

    def l2_penalty(self) -> torch.Tensor:
        total = torch.zeros((), device=self.t.device)
        for layer in self.layer_list[1:]:
            total = total + layer.l2_norm()
        return total

    def clip_weights(self) -> None:
        """Matches official ClipWeights callback: clip every layer EXCEPT the final LRLayer."""
        for layer in self.layer_list[:-1]:
            layer.clip()

    # ── Rule extraction ────────────────────────────────────────────

    def reset_activation_stats(self, device: torch.device) -> None:
        for layer in self.layer_list[:-1]:
            layer.node_activation_cnt = torch.zeros(layer.output_dim, dtype=torch.double, device=device)
            layer.forward_tot = 0

    def decode_rules(self, concept_names: list[str]) -> list[dict]:
        """
        Must be called AFTER a full pass over data with bi_forward(count=True)
        (see collect_activation_stats in train_crl_rrl.py) so node_activation_cnt
        / forward_tot are populated — dead/always-firing nodes are pruned
        exactly like the official implementation.
        """
        self.layer_list[0].get_rule_name(concept_names)

        prev_layer = self.layer_list[0]
        for i in range(1, len(self.layer_list) - 1):
            layer = self.layer_list[i]
            layer.get_rules(prev_layer)
            layer.get_rule_description(prev_layer.rule_name)
            prev_layer = layer

        lr_layer = self.layer_list[-1]
        lr_layer.get_rule2weights(prev_layer)

        rules = []
        for rid, weights in lr_layer.rule2weights:
            support = (prev_layer.node_activation_cnt[lr_layer.rid2dim[rid]] / prev_layer.forward_tot).item()
            rules.append({
                "rule_id": rid,
                "description": prev_layer.rule_name[rid],
                "support": support,
                "class_weights": {int(k): float(v) for k, v in weights.items()},
                "predicts_class": max(weights.items(), key=lambda kv: kv[1])[0],
            })
        return rules
