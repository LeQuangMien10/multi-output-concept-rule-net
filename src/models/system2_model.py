"""
system2_rules.py  (v2 — concept-balanced)
==========================================
Vấn đề v1: model shortcut, chỉ học "valid=0/1", bỏ qua digit/op.

Giải pháp v2 — mọi concept slot ngang hàng nhau:

Architecture thay đổi:
─────────────────────
concept_vec [B, 42]  (soft prob per slot)
    ↓  RuleMatcher
rule_scores  [B, R]
    ↓  softmax
rule_assign  [B, R]          ← distribution over rules

Mỗi rule r lưu:
  rule_concept_logits [R, 42]  ← prototype FULL concept vector
                                   (không phải chỉ "valid")

Prediction: rule_assign [B,R] x rule_concept_logits [R,42] → pred_concept [B,42]
    ↓  split theo slot
  pred_digit1 [B,10], pred_op1 [B,5], ..., pred_valid [B,2]

Loss:
  1. concept_loss  = mean CE(pred_slot, true_slot)  ← 6 slot ngang nhau
  2. recon_loss    = MSE(pred_concept, concept_vec)  ← buộc rule encode đủ thông tin
  3. sparsity_loss = entropy(rule_assign)            ← peaked assignment
  4. coverage_loss = penalize unused rules
  5. diversity_loss= penalize identical rule prototypes

Accuracy:
  - per-concept accuracy (digit1/op1/.../valid)
  - expression accuracy (tất cả 6 slot đúng)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rule_memory import (
    RuleMemory,
    CONCEPT_TOTAL_DIM,
    CONCEPT_KEYS_ORDERED,
    CONCEPT_OFFSETS,
    CONCEPT_DIMS,
    labels_to_concept_vector,
    soft_concept_vector,
    logits_to_concept_vector,
)
from src.models.rule_matching import RuleMatcher


# ─────────────────────────────────────────────────────────────
# System 2 v2
# ─────────────────────────────────────────────────────────────

class System2Rules(nn.Module):
    """
    System 2: học rule prototype cho TẤT CẢ concept slot.

    Parameters
    ----------
    num_rules       : số rule prototype
    concept_dim     : chiều concept vector (42)
    score_mode      : "dot" | "weighted" | "cosine"
    hard_threshold  : threshold khi inference
    temperature     : softmax temperature cho rule assignment
    """

    def __init__(
        self,
        num_rules: int = 64,
        concept_dim: int = CONCEPT_TOTAL_DIM,
        score_mode: str = "weighted",
        hard_threshold: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.num_rules    = num_rules
        self.concept_dim  = concept_dim
        self.temperature  = temperature

        # ── Rule memory (mask) ───────────────────────────────
        self.memory = RuleMemory(num_rules=num_rules, concept_dim=concept_dim)

        # ── Rule matcher ─────────────────────────────────────
        self.matcher = RuleMatcher(
            memory=self.memory,
            score_mode=score_mode,
            hard_threshold=hard_threshold,
        )

        # ── Rule concept prototype ────────────────────────────
        # Mỗi rule r học một prototype đầy đủ cho từng slot.
        # Lưu dưới dạng logit riêng theo từng slot để softmax độc lập.
        self.rule_slot_logits = nn.ParameterDict({
            key: nn.Parameter(torch.zeros(num_rules, dim))
            for key, dim in CONCEPT_DIMS.items()
        })
        # rule_slot_logits[key]: [R, dim_key]

    # ── Slot-wise softmax trên prototype ─────────────────────

    def get_rule_slot_probs(self) -> dict[str, torch.Tensor]:
        """
        Returns softmax probs cho từng slot của mỗi rule.
        key → [R, dim_key]
        """
        return {
            key: F.softmax(self.rule_slot_logits[key], dim=-1)
            for key in CONCEPT_KEYS_ORDERED
        }

    def get_rule_concept_vec(self) -> torch.Tensor:
        """
        Nối các slot probs thành vector đầy đủ [R, 42].
        """
        parts = [
            F.softmax(self.rule_slot_logits[key], dim=-1)
            for key in CONCEPT_KEYS_ORDERED
        ]
        return torch.cat(parts, dim=1)   # [R, 42]

    # ── Forward ──────────────────────────────────────────────

    def forward(self, concept_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        concept_vec : FloatTensor[B, 42]  — soft concept (softmax probs)

        Returns
        -------
        dict với:
          "rule_scores"      : [B, R]
          "rule_assignment"  : [B, R]   softmax weights
          "pred_concept"     : [B, 42]  weighted sum của rule prototypes
          "pred_slot_logits" : dict key→[B, dim]  per-slot logits để tính CE loss
          "rule_concept_vec" : [R, 42]  prototype của tất cả rules
        """
        # 1. Match scores [B, R]
        rule_scores  = self.matcher(concept_vec)

        # 2. Assignment [B, R]
        rule_assign  = F.softmax(rule_scores / self.temperature, dim=1)

        # 3. Rule prototype [R, 42]
        rule_cv = self.get_rule_concept_vec()   # [R, 42]

        # 4. Predicted concept = weighted sum [B, 42]
        pred_concept = rule_assign @ rule_cv    # [B, R] x [R, 42] → [B, 42]

        # 5. Per-slot logits: reconstruct từ rule_slot_logits được weight
        slot_probs = self.get_rule_slot_probs()   # key → [R, dim]
        pred_slot_logits: dict[str, torch.Tensor] = {}
        for key in CONCEPT_KEYS_ORDERED:
            # [B, R] x [R, dim] → [B, dim]
            pred_slot_logits[key] = rule_assign @ slot_probs[key]
            # Đây là weighted sum của probs → dùng log để tính NLL loss

        return {
            "rule_scores"      : rule_scores,
            "rule_assignment"  : rule_assign,
            "pred_concept"     : pred_concept,
            "pred_slot_logits" : pred_slot_logits,
            "rule_concept_vec" : rule_cv,
        }

    # ── Inference ────────────────────────────────────────────

    @torch.no_grad()
    def infer(self, concept_vec_hard: torch.Tensor) -> dict:
        """
        Hard inference: kích hoạt rule theo threshold.

        Returns
        -------
        dict:
          "activated_rules"   : list[list[int]]   per sample
          "match_ratios"      : [B, R]
          "pred_slot"         : dict key→[B]      argmax prediction per slot
          "rule_strings"      : list[list[str]]
          "best_rule_idx"     : LongTensor[B]     rule có score cao nhất
        """
        rule_list, ratios = self.matcher.predict(concept_vec_hard)

        B = concept_vec_hard.shape[0]

        # argmax score → best rule per sample (fallback khi không có rule active)
        rule_scores  = self.matcher(concept_vec_hard)         # [B, R]
        best_rule    = rule_scores.argmax(dim=1)              # [B]

        # Chọn rule để decode: ưu tiên first active rule, fallback best score
        chosen_rules = []
        for b, rules in enumerate(rule_list):
            chosen_rules.append(rules[0] if rules else best_rule[b].item())

        chosen_idx = torch.tensor(chosen_rules, dtype=torch.long,
                                  device=concept_vec_hard.device)   # [B]

        # Decode từng slot từ rule prototype được chọn
        slot_probs = self.get_rule_slot_probs()   # key → [R, dim]
        pred_slot: dict[str, torch.Tensor] = {}
        for key in CONCEPT_KEYS_ORDERED:
            probs = slot_probs[key]                     # [R, dim]
            pred_slot[key] = probs[chosen_idx].argmax(dim=1)  # [B]

        rule_strings: list[list[str]] = [
            [self.memory.decode_rule(r) for r in rules] if rules else ["(no rule)"]
            for rules in rule_list
        ]

        return {
            "activated_rules" : rule_list,
            "match_ratios"    : ratios,
            "pred_slot"       : pred_slot,
            "rule_strings"    : rule_strings,
            "best_rule_idx"   : best_rule,
        }

    # ── Loss ─────────────────────────────────────────────────

    @staticmethod
    def compute_loss(
        outputs     : dict[str, torch.Tensor],
        concept_vec : torch.Tensor,
        labels      : dict[str, torch.Tensor],
        concept_weight   : float = 1.0,
        recon_weight     : float = 0.5,
        sparsity_weight  : float = 0.05,
        coverage_weight  : float = 0.05,
        diversity_weight : float = 0.01,
        slot_weights     : dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Multi-objective loss đảm bảo mọi concept slot ngang nhau.

        Loss components
        ───────────────
        1. concept_loss   : CE cho từng slot với weight bằng nhau
                            (digit1/op1/digit2/op2/digit3/valid)
        2. recon_loss     : MSE giữa predicted concept và input concept vec
                            → buộc rule prototype encode đủ thông tin
        3. sparsity_loss  : neg-entropy của rule_assignment
                            → mỗi ảnh map rõ sang ít rule
        4. coverage_loss  : penalize rule không được dùng
        5. diversity_loss : penalize rule prototype giống nhau quá

        Parameters
        ----------
        slot_weights : weight riêng cho từng slot (None = đều nhau)
                       ví dụ {"valid": 1.0, "digit1": 1.0, ...}
        """
        pred_slot  = outputs["pred_slot_logits"]   # key → [B, dim]
        rule_assign= outputs["rule_assignment"]     # [B, R]
        pred_cv    = outputs["pred_concept"]        # [B, 42]
        rule_cv    = outputs["rule_concept_vec"]    # [R, 42]

        if slot_weights is None:
            slot_weights = {k: 1.0 for k in CONCEPT_KEYS_ORDERED}

        # ── 1. Concept reconstruction CE (per slot) ──────────
        slot_losses: dict[str, torch.Tensor] = {}
        for key in CONCEPT_KEYS_ORDERED:
            # pred_slot[key]: [B, dim] — weighted sum of probs
            # Dùng NLL loss với log(pred + eps)
            log_pred = (pred_slot[key] + 1e-8).log()       # [B, dim]
            target   = labels[key].long()                  # [B]
            ce       = F.nll_loss(log_pred, target)
            slot_losses[key] = ce

        # Weighted mean — tất cả slot ngang nhau (weight=1.0)
        concept_loss = sum(
            slot_weights[k] * slot_losses[k]
            for k in CONCEPT_KEYS_ORDERED
        ) / sum(slot_weights.values())

        # ── 2. Reconstruction loss (MSE) ─────────────────────
        # concept_vec là soft prob input [B, 42], pred_cv cũng [B, 42]
        recon_loss = F.mse_loss(pred_cv, concept_vec.detach())

        # ── 3. Sparsity loss: khuyến khích peaked assignment ──
        # Maximize entropy → thay bằng penalize high entropy
        entropy = -(rule_assign * (rule_assign + 1e-8).log()).sum(dim=1)
        sparsity_loss = entropy.mean()

        # ── 4. Coverage loss: penalize unused rules ───────────
        avg_assign = rule_assign.mean(dim=0)       # [R]
        uniform    = 1.0 / rule_assign.shape[1]
        coverage_loss = F.relu(0.5 * uniform - avg_assign).mean()

        # ── 5. Diversity loss: penalize identical prototypes ──
        # rule_cv: [R, 42] → cosine similarity matrix [R, R]
        rv_norm       = F.normalize(rule_cv, dim=1)          # [R, 42]
        sim_matrix    = rv_norm @ rv_norm.T                  # [R, R]
        R             = sim_matrix.shape[0]
        # Lấy upper triangle (bỏ diagonal)
        mask          = torch.triu(torch.ones(R, R, device=sim_matrix.device), diagonal=1).bool()
        pairwise_sim  = sim_matrix[mask]
        diversity_loss = pairwise_sim.mean()   # → minimize similarity

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
# Accuracy helpers (dùng trong train/eval loop)
# ─────────────────────────────────────────────────────────────

def compute_system2_accuracy(
    outputs : dict[str, torch.Tensor],
    labels  : dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    Tính accuracy per-slot và expression accuracy.

    Sử dụng pred_slot_logits từ forward() output.

    Returns
    -------
    dict:
      "<key>_acc"      : per-slot accuracy
      "expression_acc" : tất cả slot đúng
      "concept_acc"    : mean accuracy trên 6 slot
    """
    pred_slot = outputs["pred_slot_logits"]   # key → [B, dim]
    B = labels["digit1"].shape[0]

    per_slot_correct = {}
    all_correct = torch.ones(B, dtype=torch.bool,
                             device=labels["digit1"].device)

    for key in CONCEPT_KEYS_ORDERED:
        # argmax trên weighted-sum probs
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
    result["concept_acc"]    = sum(
        result[f"{k}_acc"] for k in CONCEPT_KEYS_ORDERED
    ) / len(CONCEPT_KEYS_ORDERED)

    return result


# ─────────────────────────────────────────────────────────────
# Convenience: build concept vector from System1 outputs
# ─────────────────────────────────────────────────────────────

def system1_outputs_to_concept(
    s1_outputs: dict[str, torch.Tensor],
    soft: bool = True,
) -> torch.Tensor:
    if soft:
        return soft_concept_vector(s1_outputs)
    else:
        return logits_to_concept_vector(s1_outputs)