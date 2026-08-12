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
        concept_dim:   int,
        theta:         float = 0.85,
        theta_merge:   float = 0.95,
        n_min:         int   = 5,
        conf_min:      float = 0.1,
        device:        str   = "cpu",
        cluster_dims:  tuple[int, int] | None = None,
    ):
        """
        cluster_dims : (start, end) — slice của concept vector dùng cho
            MATCH / CREATE / MERGE similarity. None → dùng toàn bộ vector.

            MNIST Math — chỉ input slots, bỏ op2(trivial) và digit3(target):
                cluster_dims = (0, 25)   # digit1(10) + op1(5) + digit2(10)
                → d3 noise không ảnh hưởng clustering
                → mỗi (d1,op,d2) combo có rule riêng

            Fitzpatrick — tất cả concept features:
                cluster_dims = None      # dùng toàn bộ 48 dims
        """
        self.concept_dim  = concept_dim
        self.theta        = theta
        self.theta_merge  = theta_merge
        self.n_min        = n_min
        self.conf_min     = conf_min
        self.device       = device
        self.cluster_dims = cluster_dims   # None = full vector

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
            sims = self._cluster_sim(cv.unsqueeze(0), centroids).squeeze(0)  # [R]
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
        sims      = self._cluster_sim(concept_vecs, centroids)  # [B, R]
        rule_ids  = sims.argmax(dim=1)     # [B]

        for i in range(len(labels)):
            r    = int(rule_ids[i].item())
            y    = int(labels[i].item())
            pred = int(predictions[i].item())
            self._correct[r]    += int(pred == y)
            self._total_pred[r] += 1

    def prune(self, verbose: bool = True,
             conf_min_override: float | None = None) -> dict[str, int]:
        """
        Loại bỏ rules yếu và merge rules trùng lặp.

        conf_min_override : ghi đè conf_min tạm thời.
            0.0 = chỉ prune theo n_min, bỏ qua accuracy signal.
            Dùng cho early epochs khi accuracy signal chưa đáng tin.

        Returns stats: removed_weak, removed_duplicate, merged, final_count
        """
        conf_threshold = (conf_min_override if conf_min_override is not None
                          else self.conf_min)
        initial = self.num_rules
        stats   = {"removed_weak": 0, "removed_duplicate": 0,
                   "merged": 0, "final_count": 0}

        # ── 1. Mark yếu (n < n_min hoặc conf < conf_threshold) ──
        keep_mask = []
        for i in range(self.num_rules):
            n    = self._n[i]
            conf = self._compute_conf(i)
            keep = (n >= self.n_min) and (conf >= conf_threshold)
            keep_mask.append(keep)
            if not keep:
                stats["removed_weak"] += 1

        surviving = [i for i, k in enumerate(keep_mask) if k]
        self._compact(surviving)

        # ── 2. Merge duplicates ────────────────────────────
        if self.num_rules > 1:
            centroids = self.get_centroids()   # [R, D]
            sims = self._cluster_sim(centroids, centroids)  # [R, R]

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
            print(f"  [Prune] {initial} -> {self.num_rules} rules | "
                  f"weak={stats['removed_weak']} merged={stats['merged']}")

        return stats

    def dedupe_by_decoded_pattern(
        self,
        concept_keys:    list[str],
        concept_offsets: dict[str, int],
        concept_dims:    dict[str, int],
        exclude_keys:    Optional[set[str]] = None,
        threshold:       float = 0.5,
        verbose:         bool = True,
    ) -> dict[str, int]:
        """
        Post-hoc cleanup theo DISPLAY pattern (present concepts, threshold
        0.5) thay vi cosine similarity tho tren toan bo vector continuous --
        bat duoc 2 thu ma theta_merge bo lot vi no so full vector chu khong
        phai chuoi rule con nguoi doc:

          - "circular": cung 1 pattern hien thi nhung khac majority label
            giua cac rule_id -- mau thuan that su, KHONG the chon 1 ben
            thang neu khong co them bang chung, nen loai bo toan bo group.
          - "duplicate": cung pattern hien thi VA cung label, tach thanh
            nhieu rule_id chi vi centroid continuous cua chung nam sat nhau
            nhung chua du gan de vuot theta_merge -- MERGE thanh 1 rule duy
            nhat (mu weighted-mean theo n, n/correct/total_pred cong don),
            thay vi bi dem thua thanh nhieu "rule" rieng biet.

        exclude_keys: cac concept key bo qua khi build pattern de group (vd.
            slot s1_label_pred an) -- phai KHOP voi tap export_rules() dung
            de loc "present_concepts" khi xuat icrl_rules.json, neu khong
            group se lech voi nhung gi nguoi dung thay tren hien thi.
        """
        exclude_keys = exclude_keys or set()
        if self.is_empty:
            return {"removed_circular": 0, "merged_duplicate_groups": 0,
                    "rules_merged_away": 0, "final_count": 0}

        from collections import Counter, defaultdict

        def pattern_of(r: int) -> tuple:
            present = []
            for key in concept_keys:
                if key in exclude_keys:
                    continue
                s = concept_offsets[key]
                e = s + concept_dims[key]
                if self._mu[r][s:e].item() >= threshold:
                    present.append(key)
            return tuple(present)

        def majority_label(r: int) -> int:
            return (Counter(self._labels[r]).most_common(1)[0][0]
                    if self._labels[r] else -1)

        groups: dict[tuple, list[int]] = defaultdict(list)
        for r in range(self.num_rules):
            groups[pattern_of(r)].append(r)

        new_mu, new_m2, new_labels = [], [], []
        new_n, new_correct, new_total_pred = [], [], []
        n_circular_removed = 0
        n_merged_groups = 0
        n_rules_merged_away = 0

        for rule_ids in groups.values():
            labels_seen = set(majority_label(r) for r in rule_ids)
            if len(labels_seen) > 1:
                n_circular_removed += len(rule_ids)
                continue

            if len(rule_ids) == 1:
                r = rule_ids[0]
                new_mu.append(self._mu[r]); new_m2.append(self._m2[r])
                new_labels.append(self._labels[r]); new_n.append(self._n[r])
                new_correct.append(self._correct[r])
                new_total_pred.append(self._total_pred[r])
                continue

            n_total = sum(self._n[r] for r in rule_ids)
            mu = sum(self._n[r] * self._mu[r] for r in rule_ids) / n_total
            labels: list[int] = []
            for r in rule_ids:
                labels.extend(self._labels[r])
            new_mu.append(mu); new_m2.append(self._m2[rule_ids[0]])
            new_labels.append(labels); new_n.append(n_total)
            new_correct.append(sum(self._correct[r] for r in rule_ids))
            new_total_pred.append(sum(self._total_pred[r] for r in rule_ids))
            n_merged_groups += 1
            n_rules_merged_away += len(rule_ids) - 1

        self._mu, self._m2 = new_mu, new_m2
        self._labels, self._n = new_labels, new_n
        self._correct, self._total_pred = new_correct, new_total_pred

        stats = {
            "removed_circular": n_circular_removed,
            "merged_duplicate_groups": n_merged_groups,
            "rules_merged_away": n_rules_merged_away,
            "final_count": self.num_rules,
        }
        if verbose:
            print(f"  [Dedupe] circular removed={n_circular_removed}  "
                  f"duplicate groups merged={n_merged_groups} (-{n_rules_merged_away} rules)  "
                  f"final={self.num_rules}")
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
        """Flat cosine. a:[M,D] b:[N,D] → [M,N]"""
        return F.normalize(a, dim=1) @ F.normalize(b, dim=1).T

    def _cluster_sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Similarity dùng cho MATCH / CREATE / MERGE.
        cluster_dims=(s,e) → cosine trên a[:,s:e] và b[:,s:e] only.
        None → flat cosine trên toàn bộ vector.
        """
        if self.cluster_dims is None:
            return self._cosine(a, b)
        s, e = self.cluster_dims
        return self._cosine(a[:, s:e], b[:, s:e])

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

        Dùng _cluster_sim (tôn trọng cluster_dims) chứ không phải _cosine thô —
        phải nhất quán với process_batch/prune (Stage 2), nếu không rule được
        gộp theo cluster_dims giới hạn lúc build nhưng lại match theo toàn bộ
        vector lúc infer, gây lệch: ảnh có thể bị match sang rule khác chỉ vì
        khác nhau ở phần bị loại khỏi cluster_dims (vd. slot nhãn S1 tự đoán).

        Returns
        -------
        best_rule_ids : LongTensor[B]
        scores        : FloatTensor[B, R] if return_scores else None
        """
        if self.is_empty:
            raise RuntimeError("Rule memory is empty. Run build() first.")

        concept_vecs = concept_vecs.to(self.device)
        centroids    = self.get_centroids()           # [R, D]
        sims         = self._cluster_sim(concept_vecs, centroids)  # [B, R]
        best_ids     = sims.argmax(dim=1)             # [B]

        return best_ids, (sims if return_scores else None)

    @torch.no_grad()
    def predict_weighted_activation(
        self,
        concept_vecs:    torch.Tensor,
        concept_keys:    list[str],
        concept_offsets: dict[str, int],
        concept_dims:    dict[str, int],
        num_classes:     int,
        exclude_keys:    Optional[set[str]] = None,
        threshold:       float = 0.5,
        min_total_weight: float = 1e-6,
        hard_gate:       bool = True,
        gate_threshold:  Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Du doan bang CACH KHAC voi match()+head(): thay vi chi 1 rule "gan
        nhat" thang tuyet doi, MOI rule co present_concepts la tap con cua
        cac concept dang "on" o anh nay deu duoc coi la KICH HOAT, voi
        firing_strength = soft-AND (product t-norm, giong ConjunctionLayer
        cua CRL/RRL that -- xem src/models/baselines/crl_rrl_components.py)
        = tich xac suat S1 cua tung concept trong present_concepts. Nhan cuoi
        cung = weighted vote giua cac rule kich hoat, trong so =
        firing_strength * confidence cua rule -- khong dung head() (khong
        can train lai gi ca, thuan tuy dua tren thong ke rule co san).

        hard_gate=True (mac dinh): mot rule CHI thuc su kich hoat neu TAT CA
        present_concepts cua no co xac suat S1 > gate_threshold o anh nay
        (nhi phan hoa truoc, giong dung "concept co mat/khong" thay vi cho
        moi rule mot chut firing_strength nho giot du chi 1-2 concept yeu) --
        firing_strength van la soft-AND (product) NHUNG bi ep ve 0 cho moi
        rule khong qua gate. Muc dich: tranh "activate tran lan" (hang chuc
        rule cung co firing_strength > 0 dong thoi, pha loang tin hieu).
        hard_gate=False: giu nguyen soft-AND thuan tuy (khong gate), moi
        rule deu co firing_strength > 0 it nhieu mien present_concepts
        khong rong.

        Fallback: anh nao KHONG rule nao kich hoat dang ke (tong trong so <
        min_total_weight) se dung match() (nearest-rule) thay the.

        Returns dict: pred [B], used_fallback [B] bool, n_fired [B] so rule
        co firing_strength > 0.01 (chi de chan doan/debug).
        """
        gate_threshold = threshold if gate_threshold is None else gate_threshold
        exclude_keys = exclude_keys or set()
        device = concept_vecs.device
        R = self.num_rules
        B = concept_vecs.shape[0]

        from collections import Counter

        # present_concepts membership mask [R, num_concept_keys] -- tinh 1
        # lan, khong phu thuoc anh.
        keys_used = [k for k in concept_keys if k not in exclude_keys]
        mask = torch.zeros(R, len(keys_used), device=device)
        rule_labels = torch.zeros(R, dtype=torch.long, device=device)
        for r in range(R):
            mu = self._mu[r]
            for j, key in enumerate(keys_used):
                s = concept_offsets[key]
                e = s + concept_dims[key]
                if mu[s:e].item() >= threshold:
                    mask[r, j] = 1.0
            rule_labels[r] = (Counter(self._labels[r]).most_common(1)[0][0]
                               if self._labels[r] else -1)
        confidences = torch.tensor(self.get_confidences(), device=device)  # [R]

        concept_probs = torch.stack(
            [concept_vecs[:, concept_offsets[k]:concept_offsets[k] + concept_dims[k]].squeeze(-1)
             if concept_dims[k] == 1 else concept_vecs[:, concept_offsets[k]:concept_offsets[k] + concept_dims[k]].max(dim=1).values
             for k in keys_used], dim=1
        )  # [B, num_concept_keys]
        log_probs = torch.log(concept_probs.clamp(min=1e-6))  # [B, num_concept_keys]
        log_firing = log_probs @ mask.T                        # [B, R] -- sum of logs over masked concepts
        firing_strength = torch.exp(log_firing)                # [B, R], empty-mask rule -> 1.0 (vacuously true)

        if hard_gate:
            concept_on = (concept_probs > gate_threshold).float()          # [B, num_concept_keys]
            required_count = mask.sum(dim=1)                                # [R]
            matched_count = concept_on @ mask.T                             # [B, R]
            gate = (matched_count >= required_count.unsqueeze(0) - 1e-6)    # [B, R] bool
            firing_strength = firing_strength * gate.float()

        weight = firing_strength * confidences.unsqueeze(0)    # [B, R]
        class_scores = torch.zeros(B, num_classes, device=device)
        for r in range(R):
            y = int(rule_labels[r].item())
            if y >= 0:
                class_scores[:, y] += weight[:, r]

        total_weight = weight.sum(dim=1)                       # [B]
        pred = class_scores.argmax(dim=1)
        used_fallback = total_weight < min_total_weight
        if used_fallback.any():
            fallback_ids, _ = self.match(concept_vecs[used_fallback])
            fallback_labels = rule_labels[fallback_ids]
            pred[used_fallback] = fallback_labels

        n_fired = (firing_strength > 0.01).sum(dim=1)

        return {"pred": pred, "used_fallback": used_fallback, "n_fired": n_fired,
                "class_scores": class_scores, "firing_strength": firing_strength}

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
            "concept_dim":  self.concept_dim,
            "theta":        self.theta,
            "theta_merge":  self.theta_merge,
            "n_min":        self.n_min,
            "conf_min":     self.conf_min,
            "cluster_dims": self.cluster_dims,
            "num_rules":    self.num_rules,
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
            cluster_dims  = state.get("cluster_dims", None),
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
        cd = (f"dims={self.cluster_dims[0]}:{self.cluster_dims[1]}"
              if self.cluster_dims else "full_vec")
        return (
            f"ICRLRuleMemory("
            f"num_rules={self.num_rules}, "
            f"θ={self.theta}, sim={cd}, "
            f"avg_conf={avg_conf:.3f})"
        )