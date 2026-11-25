"""
eval_coverage_normalized.py - Chuẩn hóa bằng chứng của rule theo độ phủ (c_r)
=============================================================================

Thí nghiệm của notes/contribution_framing_explainability.md, mục "Chuẩn hóa
theo độ phủ". Phiếu vote deployed dùng weight(r,x) = fire(r,x)·conf(r), trong đó
fire là tích xác suất các concept có mặt của rule; fire không phải likelihood đã
chuẩn hóa theo x, nên rule ngắn/chung chung (fire cao trên nhiều ảnh) áp đảo
phiếu. Biến thể chuẩn hóa:

    c_r = (Σ_{x∈val} fire(r,x) + 1) / (N_val + 1)        (làm trơn Laplace)
    weight(r,x) = fire(r,x) / c_r · conf(r)

c_r ước lượng trên VAL (tập hiệu chỉnh, giống conf(r) và g(x)), không chạm test.
Mọi thứ khác giữ đúng cấu hình deployed multi/soft/learned (gate, fallback qua
head, feat_agg5, learned_soft hiệu chỉnh trên val). Cấu hình deployed không
chuẩn hóa được assert khớp với learned_soft() gốc.

Chỉ số báo cáo (trên ảnh test có rule fire):
  len_w     số concept trung bình có trọng số P(r|x) của các rule đóng góp
  short_w   tỷ trọng phiếu đến từ rule có <= 1 concept (gồm rule rỗng)
  cùng S2 acc, S2 balanced acc, ensemble acc, tỷ lệ 4 trường hợp a/b/c/d.

Usage (local, inference only, dùng cache có sẵn):
    python -m src.scripts.eval_coverage_normalized
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.scripts.eval_ablation_crlmatched import learned_soft
from src.scripts.eval_explanation_coverage import (
    CASES, SCOPES as FITZ_SCOPES, assemble, classify_cases, learned_soft_pred,
)
from src.scripts.fitzpatrick.train_icrl import build_full_concept_layout
from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY

OUT = Path("outputs/explanation_coverage/coverage_normalized_results.json")
CUB = {
    "cub_class": dict(cache="outputs/cub_outputs/s1_cache.pt",
                      icrl="outputs/cub_icrl_theta/theta_0.95_confmin0"),
    "cub_instance": dict(cache="outputs/cub_instance_outputs/outputs/cub_instance/s1_cache.pt",
                         icrl="outputs/cub_instance_icrl/t0.95_cm0"),
}
GATES = [0.5, None]


def rule_meta(memory, keys_used, offsets, dims, threshold=0.5):
    labels = torch.tensor([collections.Counter(l).most_common(1)[0][0] if l else -1
                           for l in memory._labels], dtype=torch.long)
    length = torch.tensor([sum(1 for k in keys_used
                               if memory._mu[r][offsets[k]:offsets[k] + dims[k]].max().item() >= threshold)
                           for r in range(memory.num_rules)], dtype=torch.float32)
    return labels, length


def firing(memory, cv, layout, num_classes, gate):
    keys, offsets, dims, exclude = layout
    hard = gate is not None
    return memory.predict_weighted_activation(
        cv, keys, offsets, dims, num_classes=num_classes, exclude_keys=exclude,
        hard_gate=hard, gate_threshold=gate if hard else None)["firing_strength"]


def vote(memory, head, cv, fs, conf, labels, length, num_classes, coverage):
    """Trả về dict cùng khóa với vote_variant() của eval_explanation_coverage, cộng
    len_w/short_w. coverage=None: deployed; ngược lại tensor c_r [R]."""
    valid = labels >= 0
    raw = fs * conf.unsqueeze(0)
    raw[:, ~valid] = 0.0
    w = raw / coverage.unsqueeze(0) if coverage is not None else raw
    cs = torch.zeros(cv.shape[0], num_classes)
    cs.index_add_(1, labels[valid], w[:, valid])
    tw = cs.sum(dim=1)
    low = raw.sum(dim=1) < 1e-6            # "không rule nào fire": theo bằng chứng thô, như deployed
    p2 = cs / tw.clamp(min=1e-12).unsqueeze(-1)
    if low.any():
        rule_ids, _ = memory.match(cv)
        p2[low] = F.softmax(head(memory.get_centroids()[rule_ids]), dim=-1)[low]
    pr = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    empty = length == 0
    vacuous = (~low) & (raw[:, ~empty].sum(dim=1) < 1e-6)
    eff_n = torch.exp(-(pr * pr.clamp(min=1e-12).log()).sum(dim=1))
    return dict(p2m=p2, tw=tw, n_fired=(fs > 0.01).sum(dim=1).float(), low=low, vacuous=vacuous,
                eff_n=eff_n, len_w=(pr * length.unsqueeze(0)).sum(dim=1),
                short_w=(pr * (length <= 1).float().unsqueeze(0)).sum(dim=1))


@torch.no_grad()
def run_one(memory, head, val, test, layout, num_classes, gate, normalize, assert_deployed=False):
    keys, offsets, dims, exclude = layout
    labels, length = rule_meta(memory, [k for k in keys if k not in exclude], offsets, dims)
    conf = torch.tensor(memory.get_confidences(), dtype=torch.float32)
    fs_val = firing(memory, val["cv"], layout, num_classes, gate)
    fs_test = firing(memory, test["cv"], layout, num_classes, gate)
    coverage = (fs_val.sum(dim=0) + 1.0) / (fs_val.shape[0] + 1.0) if normalize else None
    vv = vote(memory, head, val["cv"], fs_val, conf, labels, length, num_classes, coverage)
    vt = vote(memory, head, test["cv"], fs_test, conf, labels, length, num_classes, coverage)
    a_cal, a_test = assemble(val["y"], val["p1"], vv), assemble(test["y"], test["p1"], vt)
    final = learned_soft_pred(a_cal, a_test)
    if assert_deployed:
        ref = learned_soft(a_cal, a_test, "p2m", "feat_agg5")
        assert abs(ref - float((final == test["y"]).float().mean())) < 1e-9
    y, p2 = test["y"], a_test["pred2m"]
    fired = ~(vt["low"] | vt["vacuous"])
    rec = [float(((p2 == y) & (y == k)).sum() / max(1, int((y == k).sum()))) for k in range(num_classes)
           if (y == k).any()]
    case = classify_cases(a_test, final.numpy())
    return dict(n=len(y), correct_ens=int((final == y).sum()), correct_s1=int((a_test["pred1"] == y).sum()),
                correct_s2=int((p2 == y).sum()), s2_bal=float(np.mean(rec)),
                case_counts={c: int((case == c).sum()) for c in CASES},
                len_w_sum=float(vt["len_w"][fired].sum()), short_w_sum=float(vt["short_w"][fired].sum()),
                n_fired_imgs=int(fired.sum()))


def merge(parts):
    n = sum(p["n"] for p in parts)
    nf = max(1, sum(p["n_fired_imgs"] for p in parts))
    return dict(
        ens=100 * sum(p["correct_ens"] for p in parts) / n,
        ens_foldmean=100 * float(np.mean([p["correct_ens"] / p["n"] for p in parts])),
        s1=100 * sum(p["correct_s1"] for p in parts) / n,
        s2=100 * sum(p["correct_s2"] for p in parts) / n,
        s2_bal=100 * float(np.mean([p["s2_bal"] for p in parts])),
        len_w=sum(p["len_w_sum"] for p in parts) / nf,
        short_w=100 * sum(p["short_w_sum"] for p in parts) / nf,
        cases={c: 100 * sum(p["case_counts"][c] for p in parts) / n for c in CASES},
    )


def fitz_inputs(scope, k):
    cfg = FITZ_SCOPES[scope]
    tag = scope if k is None else f"{scope}_fold{k}"
    blob = torch.load(f"outputs/explanation_coverage/cache_{tag}.pt", weights_only=False)
    num_classes = 2 if scope == "crlmatched" else 3
    keys, offsets, dims, _ = build_full_concept_layout(blob["concept_names"], cfg["label_names"])
    mem = ICRLRuleMemory.load(cfg["icrl"](k) / "icrl_rule_memory.pt", device="cpu")
    head = nn.Linear(mem.concept_dim, num_classes)
    head.load_state_dict(torch.load(cfg["icrl"](k) / "prediction_head.pt", map_location="cpu", weights_only=False))
    head.eval()
    return mem, head, blob["val"], blob["test"], (keys, offsets, dims, {S1_LABEL_CONCEPT_KEY}), num_classes


def cub_inputs(name):
    cfg = CUB[name]
    b = torch.load(cfg["cache"], weights_only=False)

    def split(s):
        out = {"concepts": b[s]["concepts"], "label": b[s]["label"]}
        return {"y": b[s]["y"], "p1": F.softmax(out["label"], dim=-1), "cv": soft_concept_vector(out)}

    num_classes = len(b["label_names"])
    keys, offsets, dims, _ = build_full_concept_layout(b["concept_names"], b["label_names"])
    mem = ICRLRuleMemory.load(Path(cfg["icrl"]) / "icrl_rule_memory.pt", device="cpu")
    head = nn.Linear(mem.concept_dim, num_classes)
    head.load_state_dict(torch.load(Path(cfg["icrl"]) / "prediction_head.pt", map_location="cpu",
                                    weights_only=False))
    head.eval()
    return mem, head, split("val"), split("test"), (keys, offsets, dims, {S1_LABEL_CONCEPT_KEY}), num_classes


def main():
    settings = [(s, FITZ_SCOPES[s]["folds"], fitz_inputs) for s in ("crlmatched", "fulldata", "concept_only")]
    settings += [(c, [None], lambda name, _k: cub_inputs(name)) for c in CUB]
    results = {}
    for name, folds, loader in settings:
        inputs = [loader(name, k) for k in folds]
        for gate in GATES:
            for normalize in (False, True):
                parts = [run_one(*inp, gate=gate, normalize=normalize,
                                 assert_deployed=(gate == 0.5 and not normalize and name in FITZ_SCOPES))
                         for inp in inputs]
                key = f"{name}/gate={'off' if gate is None else gate}/{'norm' if normalize else 'deployed'}"
                results[key] = r = merge(parts)
                ens = r["ens_foldmean"] if name == "crlmatched" else r["ens"]
                cs = " ".join(f"{r['cases'][c]:5.1f}" for c in CASES)
                print(f"{key:<38s} ens {ens:6.2f} S1 {r['s1']:6.2f} S2 {r['s2']:6.2f} S2bal {r['s2_bal']:6.2f} | "
                      f"len_w {r['len_w']:5.2f} short_w {r['short_w']:5.1f}% | {cs}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {OUT}  (cases: {' '.join(CASES)})")


if __name__ == "__main__":
    main()
