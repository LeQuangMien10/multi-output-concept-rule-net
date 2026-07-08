"""
icrl_rule_memory.py — Incremental Concept-driven Rule Learning
==============================================================

Thay thế preset-init System2 bằng rule memory tự xây dựng từ data:
  - Rule memory bắt đầu RỖNG
  - Mỗi concept vector được MATCH vào rule gần nhất (cosine > θ)
    hoặc CREATE rule mới nếu không có rule nào đủ gần
  - Centroid được UPDATE theo running mean, weight bởi S1 confidence
  - Sau mỗi epoch: PRUNE rules yếu và MERGE rules trùng lặp

Mỗi Rule r:
    μ_r   : FloatTensor[D]  — running mean concept vector (centroid)
    σ_r   : FloatTensor[D]  — running std (coherence proxy)
    y_r   : int             — majority vote label
    n_r   : int             — số ảnh đã assign vào rule
    conf_r: float           — coherence × accuracy ∈ [0,1]

General: hoạt động với bất kỳ D-dim concept vector nào
  MNIST Math  : D=40, softmax probs, labels ∈ {0..9}
  Fitzpatrick : D=48, sigmoid probs, labels ∈ {0,1}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


class ICRLRuleMemory:
    """
    Incremental Concept-driven Rule Learning memory.

    Không phải nn.Module — không có gradient.
    Sống bên ngoài training loop, được build từ frozen S1 concept vectors.

    Parameters
    ----------
    concept_dim : int
        Số chiều của concept vector (D).
    theta : float
        Similarity threshold để CREATE rule mới.
        Nếu max_sim(cv, existing_rules) < theta → create.
    theta_merge : float
        Similarity threshold để MERGE hai rules với nhau.
        Thường theta_merge > theta.
    n_min : int
        Số ảnh tối thiểu để rule survive sau prune.
    conf_min : float
        Confidence tối thiểu để rule survive sau prune.
    device : str
        'cpu' hoặc 'cuda'.
    """

    def __init__(
        self,
        concept_dim:    int,
        theta:          float = 0.85,
        theta_merge:    float = 0.98,
        n_min:          int   = 5,
        conf_min:       float = 0.1,
        device:         str   = "cpu",
        match_offsets:  list | None = None,
    ):
        """
        match_offsets : list of (start, end) index pairs trong concept vector
            dùng cho MATCH / CREATE / MERGE similarity.
            None → flat cosine trên toàn bộ vector.

            MNIST Math — chỉ input slots (bỏ digit3 target, op2 trivial):
                match_offsets = [(0,10),(10,15),(15,25)]
                → digit1(10) + op1(5) + digit2(10) = 25 dims

            Fitzpatrick — tất cả concepts:
                match_offsets = None  (hoặc [(0,48)])
        """
        self.concept_dim   = concept_dim
        self.theta         = theta
        self.theta_merge   = theta_merge
        self.n_min         = n_min
        self.conf_min      = conf_min
        self.device        = device
        self.match_offsets = match_offsets

        # Rule storage (Python lists — dynamic size)
        self._mu:         list[torch.Tensor] = []   # [D] each
        self._m2:         list[torch.Tensor] = []   # running sum of squared diff (Welford)
        self._labels:     list[list[int]]    = []   # all labels seen in cluster
        self._n:          list[int]          = []   # count
        self._correct:    list[int]          = []   # correct predictions count
        self._total_pred: list[int]          = []   # total predictions count

    # ── Properties ──────────────────────────────────────────

    @property
    def num_rules(self) -> int:
        return len(self._mu)

    @property
    def is_empty(self) -> bool:
        return self.num_rules == 0

    def get_centroids(self) -> torch.Tensor:
        """Stack all centroids → [R, D]"""
        if self.is_empty:
            return torch.zeros(0, self.concept_dim, device=self.device)
        return torch.stack(self._mu, dim=0)  # [R, D]

    def get_confidences(self) -> list[float]:
        return [self._compute_conf(i) for i in range(self.num_rules)]

    def get_labels(self) -> list[int]:
        """Majority vote label per rule"""
        result = []
        for labels in self._labels:
            if not labels:
                result.append(-1)
                continue
            from collections import Counter
            result.append(Counter(labels).most_common(1)[0][0])
        return result

    # ── Core operations ─────────────────────────────────────

    def process_batch(
        self,
        concept_vecs:   torch.Tensor,           # [B, D]
        labels:         torch.Tensor,           # [B]  int
        s1_confidences: Optional[torch.Tensor] = None,  # [B] float ∈ (0,1]
    ) -> dict[str, int]:
        """
        Process một batch concept vectors:
          - Mỗi cv: MATCH nếu max_sim > theta, else CREATE
          - UPDATE centroid với confidence-weighted running mean

        Returns dict với stats: created, matched, total
        """
        B = concept_vecs.shape[0]
        if s1_confidences is None:
            s1_confidences = torch.ones(B, device=self.device)

        concept_vecs = concept_vecs.to(self.device)
        labels       = labels.to(self.device)
        s1_confidences = s1_confidences.to(self.device)

        stats = {"created": 0, "matched": 0, "total": B}

        for i in range(B):
            cv   = concept_vecs[i]        # [D]
            y    = int(labels[i].item())
            w    = float(s1_confidences[i].item())

            if self.is_empty:
                self._create_rule(cv, y, w)
                stats["created"] += 1
                continue

            # Compute cosine similarity với tất cả centroids
            centroids = self.get_centroids()    # [R, D]
            sims = self._match_sim(cv.unsqueeze(0), centroids).squeeze(0)  # [R]
            best_sim, best_r = sims.max(dim=0)
            best_sim = best_sim.item()
            best_r   = best_r.item()

            if best_sim >= self.theta:
                self._update_rule(best_r, cv, y, w)
                stats["matched"] += 1
            else:
                self._create_rule(cv, y, w)
                stats["created"] += 1

        return stats

    def update_accuracy(
        self,
        concept_vecs: torch.Tensor,   # [B, D]
        labels:       torch.Tensor,   # [B]
        predictions:  torch.Tensor,   # [B]  predicted labels
    ) -> None:
        """
        Sau mỗi epoch: cập nhật accuracy cho từng rule
        dựa trên predictions của prediction head.
        """
        if self.is_empty:
            return

        concept_vecs = concept_vecs.to(self.device)
        labels       = labels.to(self.device)
        predictions  = predictions.to(self.device)

        centroids = self.get_centroids()   # [R, D]
        sims      = self._cosine(concept_vecs, centroids)  # [B, R]
        rule_ids  = sims.argmax(dim=1)     # [B]

        for i in range(len(labels)):
            r    = int(rule_ids[i].item())
            y    = int(labels[i].item())
            pred = int(predictions[i].item())
            self._correct[r]    += int(pred == y)
            self._total_pred[r] += 1

    def prune(self, verbose: bool = True) -> dict[str, int]:
        """
        Loại bỏ rules yếu và merge rules trùng lặp.

        Returns stats: removed_weak, removed_duplicate, merged, final_count
        """
        initial = self.num_rules
        stats   = {"removed_weak": 0, "removed_duplicate": 0,
                   "merged": 0, "final_count": 0}

        # ── 1. Mark yếu (n < n_min hoặc conf < conf_min) ──
        keep_mask = []
        for i in range(self.num_rules):
            n    = self._n[i]
            conf = self._compute_conf(i)
            keep = (n >= self.n_min) and (conf >= self.conf_min)
            keep_mask.append(keep)
            if not keep:
                stats["removed_weak"] += 1

        surviving = [i for i, k in enumerate(keep_mask) if k]
        self._compact(surviving)

        # ── 2. Merge duplicates ────────────────────────────
        if self.num_rules > 1:
            centroids = self.get_centroids()   # [R, D]
            sims = self._match_sim(centroids, centroids)  # [R, R]

            merged_into: dict[int, int] = {}   # rule_i → rule_j (j survives)

            for i in range(self.num_rules):
                if i in merged_into:
                    continue
                for j in range(i + 1, self.num_rules):
                    if j in merged_into:
                        continue
                    if sims[i, j].item() >= self.theta_merge:
                        # Merge j into i (i has more samples typically)
                        survivor = i if self._n[i] >= self._n[j] else j
                        victim   = j if survivor == i else i
                        self._merge_rules(survivor, victim)
                        merged_into[victim] = survivor
                        stats["merged"] += 1

            surviving = [i for i in range(self.num_rules)
                         if i not in merged_into]
            self._compact(surviving)

        stats["removed_duplicate"] = initial - stats["removed_weak"] - self.num_rules + stats["merged"]
        stats["final_count"] = self.num_rules

        if verbose:
            print(f"  [Prune] {initial} → {self.num_rules} rules | "
                  f"weak={stats['removed_weak']} merged={stats['merged']}")

        return stats

    # ── Rule creation & update ───────────────────────────────

    def _create_rule(
        self, cv: torch.Tensor, y: int, w: float
    ) -> None:
        self._mu.append(cv.clone())
        self._m2.append(torch.zeros_like(cv))
        self._labels.append([y])
        self._n.append(1)
        self._correct.append(0)
        self._total_pred.append(0)

    def _update_rule(
        self, r: int, cv: torch.Tensor, y: int, w: float
    ) -> None:
        """Welford online mean update, weighted by S1 confidence w."""
        n_old       = self._n[r]
        n_new       = n_old + 1
        delta       = cv - self._mu[r]
        # Weighted update: w=1 → standard Welford, w<1 → uncertain sample contributes less
        self._mu[r] = self._mu[r] + (w / n_new) * delta
        self._m2[r] = self._m2[r] + w * delta * (cv - self._mu[r])
        self._n[r]  = n_new
        self._labels[r].append(y)

    def _merge_rules(self, survivor: int, victim: int) -> None:
        """Merge victim into survivor (weighted mean by count)."""
        n_s = self._n[survivor]
        n_v = self._n[victim]
        total = n_s + n_v
        self._mu[survivor] = (n_s * self._mu[survivor] + n_v * self._mu[victim]) / total
        self._n[survivor]  = total
        self._labels[survivor].extend(self._labels[victim])
        self._correct[survivor]    += self._correct[victim]
        self._total_pred[survivor] += self._total_pred[victim]

    def _compact(self, surviving_indices: list[int]) -> None:
        """Keep only surviving rules."""
        self._mu          = [self._mu[i]          for i in surviving_indices]
        self._m2          = [self._m2[i]          for i in surviving_indices]
        self._labels      = [self._labels[i]      for i in surviving_indices]
        self._n           = [self._n[i]           for i in surviving_indices]
        self._correct     = [self._correct[i]     for i in surviving_indices]
        self._total_pred  = [self._total_pred[i]  for i in surviving_indices]

    # ── Similarity ──────────────────────────────────────────

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Flat cosine. a: [M,D], b: [N,D] → [M,N]"""
        return F.normalize(a, dim=1) @ F.normalize(b, dim=1).T

    def _match_sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Similarity dùng cho MATCH / CREATE / MERGE.

        match_offsets=None : flat cosine (toàn bộ vector)
        match_offsets set  : slot-wise cosine trên các slots được chọn,
                             mỗi slot đóng góp đều nhau.

        Dùng input slots (bỏ target slot digit3) giúp clustering theo
        (d1, op, d2) pattern thay vì theo digit3 — tránh merge nhầm
        các expressions khác d1/op/d2 nhưng cùng kết quả.

        a: [M,D], b: [N,D] → [M,N]
        """
        if self.match_offsets is None:
            return self._cosine(a, b)

        total   = torch.zeros(a.shape[0], b.shape[0], device=a.device)
        n_slots = len(self.match_offsets)
        for (s, e) in self.match_offsets:
            a_s = F.normalize(a[:, s:e], dim=1)
            b_s = F.normalize(b[:, s:e], dim=1)
            total += a_s @ b_s.T
        return total / n_slots

    # ── Confidence ──────────────────────────────────────────

    def _compute_conf(self, r: int) -> float:
        coherence = self._compute_coherence(r)
        accuracy  = self._compute_accuracy(r)
        return coherence * accuracy

    def _compute_coherence(self, r: int) -> float:
        """
        coherence = exp(−mean_distance_to_centroid)
        Cao khi cluster compact, thấp khi cluster tản ra.
        """
        n = self._n[r]
        if n <= 1:
            return 1.0   # single-sample rule: perfectly coherent by definition
        # mean squared distance proxy from Welford M2
        variance_per_dim = self._m2[r] / max(n - 1, 1)
        mean_dist = variance_per_dim.mean().item() ** 0.5
        import math
        return math.exp(-mean_dist)

    def _compute_accuracy(self, r: int) -> float:
        total = self._total_pred[r]
        if total == 0:
            return 0.5   # no prediction yet → neutral
        return self._correct[r] / total

    # ── Inference ───────────────────────────────────────────

    @torch.no_grad()
    def match(
        self,
        concept_vecs: torch.Tensor,   # [B, D]
        return_scores: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Inference: match concept_vecs → best rule indices.

        Returns
        -------
        best_rule_ids : LongTensor[B]
        scores        : FloatTensor[B, R] if return_scores else None
        """
        if self.is_empty:
            raise RuntimeError("Rule memory is empty. Run build() first.")

        concept_vecs = concept_vecs.to(self.device)
        centroids    = self.get_centroids()           # [R, D]
        sims         = self._match_sim(concept_vecs, centroids)  # [B, R]
        best_ids     = sims.argmax(dim=1)             # [B]

        return best_ids, (sims if return_scores else None)

    # ── Decode ──────────────────────────────────────────────

    def decode_rule(
        self,
        rule_id:       int,
        concept_keys:  list[str],
        concept_offsets: dict[str, int],
        concept_dims:  dict[str, int],
        id_to_symbol:  Optional[dict[int, str]] = None,
        threshold:     float = 0.5,
    ) -> dict:
        """
        Decode rule r thành human-readable dict.

        MNIST Math:
            digit1=3, op1=+, digit2=5 → "3 + 5 = ?"
        Fitzpatrick:
            Erythema=present (μ[0]=0.87), Plaque=present (μ[1]=0.92), ...
        """
        mu = self._mu[rule_id]   # [D]
        slots = {}

        for key in concept_keys:
            s  = concept_offsets[key]
            e  = s + concept_dims[key]
            sv = mu[s:e]

            if concept_dims[key] > 1:
                # Categorical (MNIST Math): argmax
                idx   = sv.argmax().item()
                conf  = sv.max().item()
                label = id_to_symbol.get(idx, str(idx)) if id_to_symbol and key in ("op1","op2") else str(int(idx))
            else:
                # Binary (Fitzpatrick): threshold
                prob  = sv.item()
                label = "present" if prob >= threshold else "absent"
                conf  = prob if prob >= threshold else 1 - prob

            slots[key] = {"value": label, "confidence": round(float(conf), 4)}

        from collections import Counter
        label_counts = Counter(self._labels[rule_id])
        majority_label = label_counts.most_common(1)[0][0] if self._labels[rule_id] else -1

        return {
            "rule_id":    rule_id,
            "slots":      slots,
            "label":      majority_label,
            "n":          self._n[rule_id],
            "confidence": round(self._compute_conf(rule_id), 4),
            "coherence":  round(self._compute_coherence(rule_id), 4),
            "accuracy":   round(self._compute_accuracy(rule_id), 4),
        }

    # ── Save / Load ─────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "concept_dim":   self.concept_dim,
            "theta":         self.theta,
            "theta_merge":   self.theta_merge,
            "n_min":         self.n_min,
            "conf_min":      self.conf_min,
            "match_offsets": self.match_offsets,
            "num_rules":     self.num_rules,
            "mu":          [m.cpu().tolist() for m in self._mu],
            "m2":          [m.cpu().tolist() for m in self._m2],
            "labels":      self._labels,
            "n":           self._n,
            "correct":     self._correct,
            "total_pred":  self._total_pred,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ICRLRuleMemory":
        state = torch.load(path, map_location=device, weights_only=False)
        mem = cls(
            concept_dim   = state["concept_dim"],
            theta         = state["theta"],
            theta_merge   = state["theta_merge"],
            n_min         = state["n_min"],
            conf_min      = state["conf_min"],
            match_offsets = state.get("match_offsets", None),
            device        = device,
        )
        mem._mu          = [torch.tensor(m, device=device) for m in state["mu"]]
        mem._m2          = [torch.tensor(m, device=device) for m in state["m2"]]
        mem._labels      = state["labels"]
        mem._n           = state["n"]
        mem._correct     = state["correct"]
        mem._total_pred  = state["total_pred"]
        return mem

    def __repr__(self) -> str:
        confs = self.get_confidences()
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        slots_info = (f"slots={len(self.match_offsets)}"
                      if self.match_offsets else "full_vec")
        return (
            f"ICRLRuleMemory("
            f"num_rules={self.num_rules}, "
            f"θ={self.theta}, sim={slots_info}, "
            f"avg_conf={avg_conf:.3f})"
        )