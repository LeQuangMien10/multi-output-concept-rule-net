"""
system2_model.py  (v3 — prototype scoring)
==========================================
Bottleneck 2 fix: scoring và prediction dùng chung một tham số.

Vấn đề v2:
    - rule_slot_logits encode "giá trị" của rule (digit1=3, op1=+, ...)
    - rule_logits (mask) encode "độ quan trọng" của slot
    - Scoring dùng rule_masks → gradient KHÔNG chảy vào rule_slot_logits
    - rule_slot_logits chỉ nhận gradient từ CE loss, không từ scoring
    - Kết quả: rule không học được cách match ảnh với rule đúng

Giải pháp v3 — một tham số, hai nhiệm vụ:
    rule_slot_logits[R, dim_k] per slot k
        ↓ softmax per slot
    rule_slot_probs[R, dim_k]
        ↓ concat → rule_proto_cv [R, 42]

    Scoring:    slot_wise_cosine(concept_vec, rule_proto_cv)  → gradient vào rule_slot_logits
    Prediction: rule_assign @ rule_slot_probs[k]              → gradient vào rule_slot_logits

    → rule_slot_logits nhận gradient từ CẢ HAI đường,
      buộc prototype phải vừa "đúng" (CE loss) vừa "phân biệt được" (score loss)

Architecture:
─────────────
concept_vec [B, 42]  (softmax probs từ System1)
    ↓  slot_wise_cosine vs rule_proto_cv [R, 42]
rule_scores  [B, R]
    ↓  softmax / temperature
rule_assign  [B, R]
    ↓  weighted sum vs rule_slot_probs [R, dim_k]
pred_slot    dict key→[B, dim_k]
    ↓  NLL loss vs GT labels
concept_loss (6 slots, equal weight)

Loss:
    1. concept_loss   — NLL per slot, balanced
    2. recon_loss     — MSE(pred_concept, concept_vec)
    3. sparsity_loss  — penalize uniform assignment (entropy)
    4. coverage_loss  — penalize unused rules
    5. diversity_loss — penalize identical prototypes
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rule_memory import (
    RuleMemory,
    CONCEPT_TOTAL_DIM,
    CONCEPT_KEYS_ORDERED,
    CONCEPT_DIMS,
    soft_concept_vector,
    logits_to_concept_vector,
)
from src.models.rule_matching import RuleMatcher


# ─────────────────────────────────────────────────────────────
# Prototype initialization helper
# ─────────────────────────────────────────────────────────────

def _enumerate_mnist_math_expressions(op1_id: int = 0, op2_id: int = 4) -> list[tuple]:
    """
    Liệt kê tất cả biểu thức a op1 b op2 c với:
        op1=+  (id=0), op2==  (id=4)
        allow_carry=False  → a+b ≤ 9
    Trả về list (digit1, op1, digit2, op2, digit3, valid).
    """
    valid, invalid = [], []
    for a in range(10):
        for b in range(10):
            c_true = a + b
            if c_true > 9:
                continue
            valid.append((a, op1_id, b, op2_id, c_true, 1))
            # 1 invalid peer: same a,b, digit3 sai
            c_wrong = (c_true + 1) % 10
            invalid.append((a, op1_id, b, op2_id, c_wrong, 0))
    return valid, invalid


def _build_prototype_logits(
    num_rules:    int,
    concept_dims: dict,
    concept_keys: list,
    sharp:        float = 8.0,
    seed:         int   = 42,
) -> dict:
    """
    Xây dựng logit tensors để khởi tạo rule_slot_logits.

    Thay vì random N(0, 0.1) dẫn đến softmax đều và score spread ≈ 0,
    mỗi rule được gán một biểu thức toán học cụ thể:
        rule r ← expression (digit1=a, op1=+, digit2=b, op2==, digit3=c, valid=v)
    với logit[r, target_class] = sharp, còn lại = 0.

    Kết quả: softmax(logit) ≈ peaked → cosine giữa 2 rules khác nhau ≈ 0
    → score spread cao ngay từ epoch 1 → softmax/T không bị uniform.

    Returns
    -------
    dict[key → FloatTensor[num_rules, dim_k]]
    """
    import random as _random
    _random.seed(seed)

    valid_exprs, invalid_exprs = _enumerate_mnist_math_expressions()
    all_exprs = valid_exprs + invalid_exprs
    _random.shuffle(all_exprs)

    # Pad nếu num_rules > len(all_exprs)
    while len(all_exprs) < num_rules:
        all_exprs.append(_random.choice(all_exprs))
    exprs = all_exprs[:num_rules]

    logits = {key: torch.zeros(num_rules, concept_dims[key]) for key in concept_keys}
    key_to_pos = {k: i for i, k in enumerate(concept_keys)}

    for r, expr in enumerate(exprs):
        vals = dict(zip(concept_keys, expr))
        for key in concept_keys:
            logits[key][r, vals[key]] = sharp

    return logits


class System2Rules(nn.Module):
    """
    System 2 v3: prototype-based scoring.

    Parameters
    ----------
    num_rules       : số rule prototype cần học
    concept_dim     : chiều concept vector (42)
    score_mode      : "slot_cosine" (default) | "flat_cosine"
    temperature     : softmax temperature cho rule assignment
                      (thấp → peaked, cao → uniform)
    hard_threshold  : cosine threshold để coi slot "khớp" khi inference
    """

    def __init__(
        self,
        num_rules:      int   = 128,
        concept_dim:    int   = CONCEPT_TOTAL_DIM,
        score_mode:     str   = "slot_cosine",
        temperature:    float = 2.0,   # overridden per-epoch by annealing
        hard_threshold: float = 0.7,
        init_sharp:     float = 8.0,   # logit sharpness for prototype init
    ):
        super().__init__()

        self.num_rules   = num_rules
        self.concept_dim = concept_dim
        self.temperature = temperature
        self.init_sharp  = init_sharp

        # ── Rule prototype: một tham số, hai nhiệm vụ ────────
        # rule_slot_logits[k]: [R, dim_k]
        # Khởi tạo từ danh sách expression cụ thể (không phải random):
        #   → các rules bắt đầu từ các điểm khác nhau trong concept space
        #   → score spread cao ngay từ epoch 1 → softmax không bị uniform
        init_logits = _build_prototype_logits(
            num_rules=num_rules,
            concept_dims=CONCEPT_DIMS,
            concept_keys=CONCEPT_KEYS_ORDERED,
            sharp=init_sharp,
        )
        self.rule_slot_logits = nn.ParameterDict({
            key: nn.Parameter(init_logits[key])
            for key in CONCEPT_DIMS
        })

        # ── Matcher ──────────────────────────────────────────
        self.matcher = RuleMatcher(
            score_mode=score_mode,
            hard_threshold=hard_threshold,
        )

        # ── RuleMemory: chỉ dùng cho decode/interpretability ─
        self.memory = RuleMemory(num_rules=num_rules, concept_dim=concept_dim)

    # ── Prototype helpers ────────────────────────────────────

    def get_rule_slot_probs(self) -> dict[str, torch.Tensor]:
        """
        Softmax per slot → prototype probs.
        key → FloatTensor[R, dim_k]
        """
        return {
            key: F.softmax(self.rule_slot_logits[key], dim=-1)
            for key in CONCEPT_KEYS_ORDERED
        }

    def get_rule_concept_vec(self) -> torch.Tensor:
        """
        Nối prototype probs thành concept vector [R, 42].
        Đây là input cho matcher.forward() — gradient chảy vào rule_slot_logits.
        """
        return torch.cat(
            [F.softmax(self.rule_slot_logits[key], dim=-1)
             for key in CONCEPT_KEYS_ORDERED],
            dim=1,
        )  # [R, 42]

    # ── Forward (training) ───────────────────────────────────

    def forward(self, concept_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        concept_vec : FloatTensor[B, 42]  — softmax probs từ System1

        Returns
        -------
        dict:
          rule_scores      : [B, R]           — slot-wise cosine similarity
          rule_assignment  : [B, R]           — softmax weights
          pred_slot_logits : dict key→[B,dim] — weighted sum probs (dùng cho NLL)
          pred_concept     : [B, 42]          — concat pred_slot_logits
          rule_concept_vec : [R, 42]          — rule prototypes
        """
        # 1. Prototype concept vector [R, 42]
        rule_cv = self.get_rule_concept_vec()

        # 2. Slot-wise cosine scores [B, R]
        #    Gradient chảy vào rule_slot_logits qua rule_cv
        rule_scores = self.matcher(concept_vec, rule_cv)

        # 3. Assignment [B, R]
        rule_assign = F.softmax(rule_scores / self.temperature, dim=1)

        # 4. Per-slot prediction: weighted sum của slot probs
        slot_probs = self.get_rule_slot_probs()          # key → [R, dim_k]
        pred_slot_logits: dict[str, torch.Tensor] = {}
        for key in CONCEPT_KEYS_ORDERED:
            # [B, R] @ [R, dim_k] → [B, dim_k]
            pred_slot_logits[key] = rule_assign @ slot_probs[key]

        # 5. Predicted concept vector (concat)
        pred_concept = torch.cat(
            [pred_slot_logits[k] for k in CONCEPT_KEYS_ORDERED], dim=1
        )  # [B, 42]

        return {
            "rule_scores"     : rule_scores,
            "rule_assignment" : rule_assign,
            "pred_slot_logits": pred_slot_logits,
            "pred_concept"    : pred_concept,
            "rule_concept_vec": rule_cv,
        }

    # ── Inference ────────────────────────────────────────────

    @torch.no_grad()
    def infer(self, concept_vec: torch.Tensor) -> dict:
        """
        Hard inference: chọn best rule per sample, decode ra string,
        tính per-slot match detail.

        Parameters
        ----------
        concept_vec : FloatTensor[B, 42]  — soft hoặc hard concept

        Returns
        -------
        dict:
          best_rule_idx  : LongTensor[B]
          pred_slot      : dict key→LongTensor[B]   — argmax per slot từ best rule
          rule_strings   : list[str]                 — decoded rule per sample
          slot_scores    : FloatTensor[B, R, 6]     — cosine per slot per rule
          slot_match     : BoolTensor[B, R, 6]      — score >= threshold
          rule_scores    : FloatTensor[B, R]         — mean slot cosine
        """
        rule_cv = self.get_rule_concept_vec()  # [R, 42]

        # Best rule per sample
        best_rule_idx, slot_scores, slot_match = self.matcher.predict(
            concept_vec, rule_cv
        )  # [B], [B,R,6], [B,R,6]

        # Overall scores [B, R]
        rule_scores = self.matcher(concept_vec, rule_cv)

        # Decode slot values từ best rule prototype
        slot_probs = self.get_rule_slot_probs()  # key → [R, dim_k]
        pred_slot: dict[str, torch.Tensor] = {}
        for key in CONCEPT_KEYS_ORDERED:
            # probs của best rule → argmax → predicted class
            pred_slot[key] = slot_probs[key][best_rule_idx].argmax(dim=1)  # [B]

        # Decode rule string per sample
        rule_strings: list[str] = [
            self.memory.decode_rule_from_probs(slot_probs, idx.item())
            for idx in best_rule_idx
        ]

        return {
            "best_rule_idx": best_rule_idx,
            "pred_slot"    : pred_slot,
            "rule_strings" : rule_strings,
            "slot_scores"  : slot_scores,
            "slot_match"   : slot_match,
            "rule_scores"  : rule_scores,
        }

    # ── Loss ────────────────────────────────────────────────

    @staticmethod
    def compute_loss(
        outputs          : dict[str, torch.Tensor],
        concept_vec      : torch.Tensor,
        labels           : dict[str, torch.Tensor],
        concept_weight   : float = 1.0,
        recon_weight     : float = 0.3,
        sparsity_weight  : float = 0.05,
        coverage_weight  : float = 0.05,
        diversity_weight : float = 0.02,
        slot_weights     : dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Loss = concept + recon + sparsity + coverage + diversity

        1. concept_loss  : NLL per slot, 6 slots ngang nhau
        2. recon_loss    : MSE(pred_concept, concept_vec) — buộc prototype encode đủ info
        3. sparsity_loss : entropy(rule_assign) — mỗi ảnh map peaked vào ít rule
        4. coverage_loss : penalize rule không được dùng
        5. diversity_loss: cosine sim giữa các rule prototype → penalize duplicates
        """
        pred_slot  = outputs["pred_slot_logits"]  # key → [B, dim_k]
        rule_assign= outputs["rule_assignment"]    # [B, R]
        pred_cv    = outputs["pred_concept"]       # [B, 42]
        rule_cv    = outputs["rule_concept_vec"]   # [R, 42]

        if slot_weights is None:
            slot_weights = {k: 1.0 for k in CONCEPT_KEYS_ORDERED}

        # ── 1. Concept CE loss ────────────────────────────────
        slot_losses: dict[str, torch.Tensor] = {}
        for key in CONCEPT_KEYS_ORDERED:
            log_pred = (pred_slot[key] + 1e-8).log()   # [B, dim_k]
            target   = labels[key].long()               # [B]
            slot_losses[key] = F.nll_loss(log_pred, target)

        concept_loss = sum(
            slot_weights[k] * slot_losses[k] for k in CONCEPT_KEYS_ORDERED
        ) / sum(slot_weights.values())

        # ── 2. Reconstruction MSE ─────────────────────────────
        recon_loss = F.mse_loss(pred_cv, concept_vec.detach())

        # ── 3. Sparsity: penalize high entropy (uniform) assignment
        entropy = -(rule_assign * (rule_assign + 1e-8).log()).sum(dim=1)
        sparsity_loss = entropy.mean()

        # ── 4. Coverage: penalize unused rules ───────────────
        avg_assign    = rule_assign.mean(dim=0)              # [R]
        uniform_thr   = 0.5 / rule_assign.shape[1]
        coverage_loss = F.relu(uniform_thr - avg_assign).mean()

        # ── 5. Diversity: penalize identical prototypes ───────
        rv_norm       = F.normalize(rule_cv, dim=1)          # [R, 42]
        sim_mat       = rv_norm @ rv_norm.T                  # [R, R]
        R             = sim_mat.shape[0]
        upper         = torch.triu(
            torch.ones(R, R, device=sim_mat.device), diagonal=1
        ).bool()
        diversity_loss = sim_mat[upper].mean()

        # ── Total ─────────────────────────────────────────────
        total = (
              concept_weight   * concept_loss
            + recon_weight     * recon_loss
            + sparsity_weight  * sparsity_loss
            + coverage_weight  * coverage_loss
            + diversity_weight * diversity_loss
        )

        loss_dict = {
            "loss_total"    : total,
            "loss_concept"  : concept_loss.detach(),
            "loss_recon"    : recon_loss.detach(),
            "loss_sparsity" : sparsity_loss.detach(),
            "loss_coverage" : coverage_loss.detach(),
            "loss_diversity": diversity_loss.detach(),
            **{f"loss_slot_{k}": slot_losses[k].detach() for k in CONCEPT_KEYS_ORDERED},
        }
        return total, loss_dict

    def extra_repr(self) -> str:
        return (
            f"num_rules={self.num_rules}, "
            f"concept_dim={self.concept_dim}, "
            f"temperature={self.temperature}"
        )


# ─────────────────────────────────────────────────────────────
# Accuracy helpers
# ─────────────────────────────────────────────────────────────

def compute_system2_accuracy(
    outputs: dict[str, torch.Tensor],
    labels : dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    Per-slot accuracy + expression accuracy (tất cả slot đúng).

    Dùng pred_slot_logits từ forward() — weighted sum probs, argmax để predict.
    """
    pred_slot = outputs["pred_slot_logits"]
    B = labels["digit1"].shape[0]
    device = labels["digit1"].device

    per_slot_correct: dict[str, torch.Tensor] = {}
    all_correct = torch.ones(B, dtype=torch.bool, device=device)

    for key in CONCEPT_KEYS_ORDERED:
        preds   = pred_slot[key].argmax(dim=1)   # [B]
        targets = labels[key].long()              # [B]
        correct = (preds == targets)
        per_slot_correct[key] = correct
        all_correct = all_correct & correct

    result = {
        f"{key}_acc": per_slot_correct[key].float().mean().item()
        for key in CONCEPT_KEYS_ORDERED
    }
    result["expression_acc"] = all_correct.float().mean().item()
    result["concept_acc"] = sum(
        result[f"{k}_acc"] for k in CONCEPT_KEYS_ORDERED
    ) / len(CONCEPT_KEYS_ORDERED)
    return result


# ─────────────────────────────────────────────────────────────
# Convenience
# ─────────────────────────────────────────────────────────────

def system1_outputs_to_concept(
    s1_outputs: dict[str, torch.Tensor],
    soft: bool = True,
) -> torch.Tensor:
    if soft:
        return soft_concept_vector(s1_outputs)
    return logits_to_concept_vector(s1_outputs)