"""
crl_rrl_components.py — Logical-layer building blocks, ported from the
OFFICIAL CRL implementation (MICCAI 2025, Yibo Gao / Fudan University):
    https://github.com/obiyoag/crl  (models/components.py, Apache-2.0)

This is a near-verbatim port of BinarizeLayer / ConjunctionLayer /
DisjunctionLayer / UnionLayer / LRLayer, kept mathematically identical to
the original so that the CRL baseline we compare ICRL against is the REAL
architecture (differentiable Boolean logic layers, RRL-style — Wang et al.
NeurIPS 2021 "Scalable Rule-Based Representation Learning"), not the
independently-designed single-layer approximation that previously lived in
src/models/crl_system2.py.

Only change vs. the original: dropped the pytorch_lightning wrapper and the
skip-connection bookkeeping (Connection.is_skip_to_layer / skip_from_layer)
— CRL's own config for the skin task (configs/skin.yaml) turns those on,
but they add real complexity to rule decoding for a marginal accuracy gain
their own ablations don't isolate cleanly. This port documents the
simplification instead of silently matching it. See crl_rrl_net.py for the
orchestration (equivalent to their models/crl.py + callbacks.py).
"""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn

THRESHOLD = 0.5
INIT_RANGE = 0.5
EPSILON = 1e-10


class GradGraft(torch.autograd.Function):
    """Forward = hard (binarized) value; backward = gradient of the continuous relaxation."""

    @staticmethod
    def forward(ctx, X, Y):
        return X

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()


class Binarizer(torch.autograd.Function):
    """Straight-through estimator: hard threshold forward, identity gradient backward."""

    @staticmethod
    def forward(_, x):
        return (x.detach() > 0.0).float()

    @staticmethod
    def backward(_, grad_output):
        return grad_output.clone()


class BinarizeLayer(nn.Module):
    """
    Binarizes the (continuous, pre-threshold) concept vector.

    NOTE on input convention: the original CRL feeds raw concept LOGITS here
    (threshold 0 <=> probability 0.5). Our own System1 stores concept
    PROBABILITIES (post-sigmoid, in [0,1]) in the concept vector, so callers
    must shift by -0.5 before passing in (see crl_rrl_net.py), which is
    mathematically equivalent (probability>0.5 <=> logit>0).
    """

    def __init__(self, n_concepts: int, use_not: bool):
        super().__init__()
        self.n_concepts = n_concepts
        self.use_not = use_not
        self.input_dim = n_concepts
        self.output_dim = 2 * n_concepts if use_not else n_concepts
        self.layer_type = "binarization"
        self.dim2id = {i: i for i in range(self.output_dim)}

    def forward(self, x):
        x = Binarizer.apply(x)
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return x

    @torch.no_grad()
    def binarized_forward(self, x):
        return self.forward(x)

    def clip(self):
        pass

    def get_rule_name(self, concept_names):
        self.rule_name = list(concept_names[: self.n_concepts])
        if self.use_not:
            self.rule_name += [f"~{n}" for n in concept_names[: self.n_concepts]]


class Product(torch.autograd.Function):
    """Product t-norm (log-sum trick) used for the continuous relaxation of AND."""

    @staticmethod
    def forward(ctx, X):
        y = -1.0 / (-1.0 + torch.sum(torch.log(X), dim=1))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + EPSILON))
        return grad_input


class ConjunctionLayer(nn.Module):
    """Learns AND(literals) — soft in training, hard (STE) at inference/rule-extraction."""

    def __init__(self, input_dim: int, output_dim: int, use_not: bool = False):
        super().__init__()
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = output_dim
        self.use_not = use_not
        self.layer_type = "conjunction"
        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.input_dim, self.output_dim))
        self.node_activation_cnt = None

    def forward(self, x):
        res_tilde = self.continuous_forward(x)
        res_bar = self.binarized_forward(x)
        return GradGraft.apply(res_bar, res_tilde)

    def continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return Product.apply(1 - (1 - x).unsqueeze(-1) * self.W)

    @torch.no_grad()
    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarizer.apply(self.W - THRESHOLD)
        return torch.prod(1 - (1 - x).unsqueeze(-1) * Wb, dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class DisjunctionLayer(nn.Module):
    """Learns OR(literals) — De Morgan dual of ConjunctionLayer."""

    def __init__(self, input_dim: int, output_dim: int, use_not: bool = False):
        super().__init__()
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = output_dim
        self.use_not = use_not
        self.layer_type = "disjunction"
        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.input_dim, self.output_dim))
        self.node_activation_cnt = None

    def forward(self, x):
        res_tilde = self.continuous_forward(x)
        res_bar = self.binarized_forward(x)
        return GradGraft.apply(res_bar, res_tilde)

    def continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return 1 - Product.apply(1 - x.unsqueeze(-1) * self.W)

    @torch.no_grad()
    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarizer.apply(self.W - THRESHOLD)
        return 1 - torch.prod(1 - x.unsqueeze(-1) * Wb, dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class UnionLayer(nn.Module):
    """concat(ConjunctionLayer, DisjunctionLayer) — one logical hidden layer."""

    def __init__(self, input_dim: int, output_dim: int, use_not: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim * 2
        self.use_not = use_not
        self.layer_type = "union"
        self.forward_tot = None
        self.node_activation_cnt = None
        self.dim2id = None
        self.rule_list = None
        self.rule_name = None

        self.con_layer = ConjunctionLayer(input_dim, output_dim, use_not=use_not)
        self.dis_layer = DisjunctionLayer(input_dim, output_dim, use_not=use_not)

    def forward(self, x):
        return torch.cat([self.con_layer(x), self.dis_layer(x)], dim=1)

    def binarized_forward(self, x):
        return torch.cat(
            [self.con_layer.binarized_forward(x), self.dis_layer.binarized_forward(x)],
            dim=1,
        )

    def l2_norm(self):
        return torch.sum(self.con_layer.W**2) + torch.sum(self.dis_layer.W**2)

    def clip(self):
        self.con_layer.clip()
        self.dis_layer.clip()

    def get_rules(self, prev_layer):
        self.con_layer.forward_tot = self.dis_layer.forward_tot = self.forward_tot
        self.con_layer.node_activation_cnt = self.dis_layer.node_activation_cnt = (
            self.node_activation_cnt
        )

        con_dim2id, con_rule_list = extract_rules(prev_layer, self.con_layer)
        dis_dim2id, dis_rule_list = extract_rules(
            prev_layer, self.dis_layer, pos_shift=self.con_layer.W.shape[1]
        )

        shift = max(con_dim2id.values()) + 1 if con_dim2id else 0
        dis_dim2id = {k: (-1 if v == -1 else v + shift) for k, v in dis_dim2id.items()}
        dim2id = defaultdict(lambda: -1, {**con_dim2id, **dis_dim2id})

        self.dim2id = dim2id
        self.rule_list = (con_rule_list, dis_rule_list)

    def get_rule_description(self, prev_rule_name):
        self.rule_name = []
        for rl, op in zip(self.rule_list, ("&", "|")):
            for rule in rl:
                parts = []
                for i, (is_not, ri) in enumerate(rule):
                    op_str = f" {op} " if i != 0 else ""
                    not_str = "~" if is_not else ""
                    parts.append(f"{op_str}{not_str}{prev_rule_name[ri]}")
                self.rule_name.append("".join(parts))


class LRLayer(nn.Module):
    """Final linear (logistic-regression-style) layer over the last logical layer's output."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_type = "linear"
        self.rid2dim = None
        self.rule2weights = None
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc1(x)

    @torch.no_grad()
    def binarized_forward(self, x):
        return self.forward(x)

    def l1_norm(self):
        return torch.norm(self.fc1.weight, p=1)

    def l2_norm(self):
        return torch.sum(self.fc1.weight**2)

    def get_rule2weights(self, prev_layer):
        always_act_pos = prev_layer.node_activation_cnt == prev_layer.forward_tot
        dim2id = prev_layer.dim2id

        Wl, bl = list(self.fc1.parameters())
        bl = torch.sum(Wl.T[always_act_pos], dim=0) + bl
        Wl = Wl.detach().cpu().numpy()
        self.bl = bl.detach().cpu().numpy()

        marked = defaultdict(lambda: defaultdict(float))
        rid2dim = {}
        for label_id, wl in enumerate(Wl):
            for i, w in enumerate(wl):
                rid = dim2id[i]
                if rid == -1:
                    continue
                marked[rid][label_id] += w
                rid2dim[rid] = i
        self.rid2dim = rid2dim
        self.rule2weights = sorted(
            marked.items(), key=lambda x: max(map(abs, x[1].values())), reverse=True
        )


def extract_rules(prev_layer, layer, pos_shift: int = 0):
    """
    Decode a Conjunction/DisjunctionLayer's hard weight matrix into symbolic
    rules over the previous layer's (already-deduplicated) rule ids. Ported
    verbatim (minus skip-connections) from the official repo's
    models/components.py::extract_rules.
    """
    dim2id = defaultdict(lambda: -1)
    rules = {}
    rule_list = []
    tmp = 0

    Wb = (layer.W.t() > THRESHOLD).int().detach().cpu().numpy()  # [output_dim, input_dim]
    prev_dim2id = prev_layer.dim2id

    for ri, row in enumerate(Wb):
        no_activated = layer.node_activation_cnt[ri + pos_shift] == 0
        all_activated = layer.node_activation_cnt[ri + pos_shift] == layer.forward_tot
        if no_activated or all_activated:
            dim2id[ri + pos_shift] = -1
            continue

        rule = {}
        for i, w in enumerate(row):
            is_not = False
            if layer.use_not:
                if i >= layer.input_dim // 2:
                    is_not = True
                i = i % (layer.input_dim // 2)
            if w > 0 and prev_dim2id.get(i, -1) != -1:
                rule[(is_not, prev_dim2id[i])] = 1

        rule = tuple(sorted(rule.keys()))
        if rule not in rules:
            rules[rule] = tmp
            rule_list.append(rule)
            dim2id[ri + pos_shift] = tmp
            tmp += 1
        else:
            dim2id[ri + pos_shift] = rules[rule]

    return dim2id, rule_list
