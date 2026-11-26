"""
analyze_icrl_latest.py - Ablation thanh phan va phan tich bo tro S1/S2 cho ban ICRL moi nhat.

Ablation (tren cung artifact outputs/icrl_latest/, g(x) va (a,b) hoc lai tren val
cho tung bien the): thay doi cach tinh trong so rule w(r,x) cua System 2
    full        fire / c_r * conf, khong gate                (ban moi nhat)
    no_norm     fire * conf, khong gate                      (bo chuan hoa do phu)
    no_conf     fire / c_r, khong gate                       (bo do tin cay rule)
    gate        fire / c_r * conf, hard gate 0.5             (them gate)
    gate_no_norm fire * conf, hard gate 0.5                  (cach tinh trong so cua ban cu)
Bien the "full" duoc assert trung voi icrl_inference.s2_vote.

Bo tro (chi bien the full): ty le bat dong, oracle (S1 hoac S2 dung), tren ca bat
dong: S1 dung / S2 dung / ca hai sai, va ensemble chon dung bao nhieu trong cac ca
chi mot he dung.

Usage:
    python -m src.scripts.analyze_icrl_latest
Output: outputs/icrl_latest/ablation.json, outputs/icrl_latest/complementarity.json
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import icrl_inference as ii
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.scripts.eval_icrl_latest import DATASETS, LATEST, merge

VARIANTS = {
    "full": dict(gate=None, norm=True, conf=True),
    "no_norm": dict(gate=None, norm=False, conf=True),
    "no_conf": dict(gate=None, norm=True, conf=False),
    "gate": dict(gate=0.5, norm=True, conf=True),
    "gate_no_norm": dict(gate=0.5, norm=False, conf=True),
}
ABL_DATASETS = ["crlmatched", "fulldata", "concept_only", "cub_class"]


@torch.no_grad()
def vote_variant(mem, head, cv, layout, num_classes, gate, norm, conf, coverage):
    """Ban tong quat cua ii.s2_vote. coverage: c_r uoc luong tren val VOI CUNG gate."""
    keys, offsets, dims, exclude = layout
    fs = mem.predict_weighted_activation(cv, keys, offsets, dims, num_classes=num_classes, exclude_keys=exclude,
                                         hard_gate=gate is not None, gate_threshold=gate)["firing_strength"]
    labels = ii.rule_labels(mem)
    valid = labels >= 0
    c = torch.tensor(mem.get_confidences(), dtype=torch.float32) if conf else torch.ones(mem.num_rules)
    raw = fs * c.unsqueeze(0)
    raw[:, ~valid] = 0.0
    w = raw / coverage.unsqueeze(0) if norm else raw
    cs = torch.zeros(cv.shape[0], num_classes)
    cs.index_add_(1, labels[valid], w[:, valid])
    tw = cs.sum(dim=1)
    p2 = cs / tw.clamp(min=1e-12).unsqueeze(-1)
    low = raw.sum(dim=1) < ii.MIN_TOTAL_WEIGHT
    if low.any():
        rule_ids, _ = mem.match(cv)
        p2[low] = F.softmax(head(mem.get_centroids()[rule_ids]), dim=-1)[low]
    post = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    eff_n = torch.exp(-(post * post.clamp(min=1e-12).log()).sum(dim=1))
    return dict(p2=p2, total_weight=tw, n_fired=(fs > 0.01).sum(dim=1).float(), no_rule=low,
                rule_post=post, eff_n=eff_n)


def load_part(part):
    _, _, val, test, layout, num_classes = part["load"]()
    d = LATEST / part["tag"]
    mem = ICRLRuleMemory.load(d / "icrl_rule_memory.pt", device="cpu")
    head = nn.Linear(mem.concept_dim, num_classes)
    head.load_state_dict(torch.load(d / "prediction_head.pt", map_location="cpu", weights_only=False))
    head.eval()
    return mem, head, val, test, layout, num_classes


@torch.no_grad()
def eval_variant(mem, head, val, test, layout, num_classes, gate, norm, conf, check=False):
    keys, offsets, dims, exclude = layout
    fs_val = mem.predict_weighted_activation(val["cv"], keys, offsets, dims, num_classes=num_classes,
                                             exclude_keys=exclude, hard_gate=gate is not None,
                                             gate_threshold=gate)["firing_strength"]
    coverage = (fs_val.sum(dim=0) + 1.0) / (fs_val.shape[0] + 1.0)
    vv = vote_variant(mem, head, val["cv"], layout, num_classes, gate, norm, conf, coverage)
    vt = vote_variant(mem, head, test["cv"], layout, num_classes, gate, norm, conf, coverage)
    if check:
        ref = ii.s2_vote(mem, head, test["cv"], layout, num_classes)
        assert torch.allclose(ref["p2"], vt["p2"], atol=1e-6) and torch.equal(ref["no_rule"], vt["no_rule"])
    a_val, a_test = ii.assemble(val["y"], val["p1"], vv), ii.assemble(test["y"], test["p1"], vt)
    final, w2 = ii.Combiner().fit(a_val).predict(a_test)
    return a_test, vt, final, w2


def summarize(mem, layout, a_test, vt, final, w2, num_classes):
    keys, offsets, dims, exclude = layout
    length = mem.present_mask(keys, offsets, dims, exclude).sum(dim=1)
    y = a_test["y"]
    case = ii.explanation_cases(a_test, final.numpy())
    fired = ~vt["no_rule"]
    post = vt["rule_post"][fired]
    correct = (final == y).numpy()
    return dict(n=len(y), n_rules=mem.num_rules, rule_label_hist=[0] * num_classes,
                rule_len_median=float(length.median()), rule_len_mean=float(length.mean()),
                correct_s1=int((a_test["pred1"] == y).sum()), correct_s2=int((a_test["pred2"] == y).sum()),
                correct_ens=int(correct.sum()),
                bal_s1=ii.balanced_accuracy(a_test["pred1"], y, num_classes),
                bal_s2=ii.balanced_accuracy(a_test["pred2"], y, num_classes),
                bal_ens=ii.balanced_accuracy(final, y, num_classes),
                val_acc_ens=0.0, val_acc_s2=0.0,
                case_counts={c: int((case == c).sum()) for c in ii.CASES},
                case_correct={c: int(correct[case == c].sum()) for c in ii.CASES},
                n_fired_imgs=int(fired.sum()),
                len_w_sum=float((post * length.unsqueeze(0)).sum()),
                short_w_sum=float((post * (length <= 1).float().unsqueeze(0)).sum()),
                top1_share_sum=float(post.max(dim=1).values.sum()) if len(post) else 0.0,
                eff_n=vt["eff_n"][fired].tolist(),
                w2_disagree_mean=float(w2[a_test["pred1"] != a_test["pred2"]].mean())
                if (a_test["pred1"] != a_test["pred2"]).any() else None)


def complementarity(a_test, final):
    y, p1, p2 = a_test["y"].numpy(), a_test["pred1"].numpy(), a_test["pred2"].numpy()
    f = final.numpy()
    dis = p1 != p2
    s1_only = dis & (p1 == y) & (p2 != y)
    s2_only = dis & (p2 == y) & (p1 != y)
    return dict(n=len(y), n_dis=int(dis.sum()), oracle=int(((p1 == y) | (p2 == y)).sum()),
                s1_only=int(s1_only.sum()), s2_only=int(s2_only.sum()), both_wrong_dis=int((dis & (p1 != y) & (p2 != y)).sum()),
                ens_kept_s1_when_s1_only=int((s1_only & (f == p1)).sum()),
                ens_took_s2_when_s2_only=int((s2_only & (f == p2)).sum()))


def merge_comp(parts):
    s = {k: sum(p[k] for p in parts) for k in parts[0]}
    n, nd = s["n"], max(1, s["n_dis"])
    fold_oracle = [p["oracle"] / p["n"] for p in parts]
    return dict(disagreement=100 * s["n_dis"] / n, oracle=100 * float(np.mean(fold_oracle)),
                dis_s1_right=100 * s["s1_only"] / nd, dis_s2_right=100 * s["s2_only"] / nd,
                dis_both_wrong=100 * s["both_wrong_dis"] / nd,
                ens_recovers_s2_only=100 * s["ens_took_s2_when_s2_only"] / max(1, s["s2_only"]),
                ens_keeps_s1_only=100 * s["ens_kept_s1_when_s1_only"] / max(1, s["s1_only"]),
                counts=s)


def main():
    ablation, comp = {}, {}
    for ds in ABL_DATASETS:
        parts = [load_part(p) for p in DATASETS[ds]["parts"]]
        ablation[ds] = {}
        for vname, v in VARIANTS.items():
            summ, comps = [], []
            for mem, head, val, test, layout, K in parts:
                a_test, vt, final, w2 = eval_variant(mem, head, val, test, layout, K, check=(vname == "full"), **v)
                summ.append(summarize(mem, layout, a_test, vt, final, w2, K))
                if vname == "full":
                    comps.append(complementarity(a_test, final))
            m = merge(summ)
            ablation[ds][vname] = m
            cs = " ".join(f"{c}:{m['cases_pct'][c]:5.1f}" for c in ii.CASES)
            print(f"{ds:<13s} {vname:<13s} S2 {m['s2']:6.2f} bal {m['bal_s2']:5.1f} ens {m['ens']:6.2f} "
                  f"balEns {m['bal_ens']:5.1f} | len_w {m['len_w']:5.2f} short {m['short_w']:5.1f}% "
                  f"effN {m['eff_n_median']:5.1f} top1 {m['top1_share']:5.1f}% | {cs}", flush=True)
            if comps:
                comp[ds] = c = merge_comp(comps)
                print(f"   complementarity: disagree {c['disagreement']:.1f}%  oracle {c['oracle']:.2f}  "
                      f"on disagreement S1 right {c['dis_s1_right']:.1f}% / S2 right {c['dis_s2_right']:.1f}% / "
                      f"both wrong {c['dis_both_wrong']:.1f}%  | ensemble keeps S1-only {c['ens_keeps_s1_only']:.1f}%, "
                      f"takes S2-only {c['ens_recovers_s2_only']:.1f}%", flush=True)
    with open(LATEST / "ablation.json", "w") as f:
        json.dump(ablation, f, indent=2)
    with open(LATEST / "complementarity.json", "w") as f:
        json.dump(comp, f, indent=2)
    print("Saved -> ablation.json, complementarity.json")


if __name__ == "__main__":
    main()
