"""
eval_explanation_coverage.py - Do do phu cua giai thich bang rule cho cau hinh
ensemble DEPLOYED (multi/soft/learned), va thu 2 huong thiet ke cho truong hop
"khong rule nao fire" (notes/contribution_framing_explainability.md).

Moi anh test duoc xep vao 1 truong hop:
  d_none     khong rule nao fire (total_weight < 1e-6) -> fallback
  d_vacuous  chi co rule RONG (khong concept dieu kien nao) dong gop
  a          S1 va S2 dong thuan, co rule (khong rong) fire
  b          bat dong, ket qua cuoi theo S2
  c          bat dong, ket qua cuoi theo S1
  e          bat dong, ket qua cuoi la nhan thu 3 (chi co the khi >= 3 lop)

Luoi bien the (chi inference, khong train lai gi):
  gate     in {0.5 (deployed), 0.4, 0.3, 0.2, off}   -- nguong hard gate cua
           predict_weighted_activation(); off = soft-AND thuan tuy
  fallback in {head (deployed), rule}                -- khi khong rule nao fire:
           head = softmax(aux head(centroid rule gan nhat)) nhu build_p2_multi();
           rule = point mass tren nhan da so cua rule gan nhat (fallback goc
           cua chinh predict_weighted_activation())

g(x) va (a,b) duoc hieu chinh lai cho TUNG bien the, dung cong thuc
learned_soft() deployed. Bien the deployed (gate 0.5, head) duoc assert khop
voi learned_soft() goc de dam bao tai tao dung so lieu.

Usage (local, inference only):
    python -m src.scripts.eval_explanation_coverage --scopes crlmatched,fulldata,concept_only
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.ensemble_strategies import entropy_confidence
from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY
from src.scripts.eval_ablation_crlmatched import learned_soft, strip_empty_rules
from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders, build_full_concept_layout

IMG_DIR = Path("data/fitzpatrick17k/data/finalfitz17k")
OUT_DIR = Path("outputs/explanation_coverage")
GATES = [0.5, 0.3, None]
FALLBACKS = ["head", "rule"]
CASES = ["a", "b", "c", "e", "d_none", "d_vacuous"]
FIT_NAMES = ["benign", "malignant", "non-neoplastic"]

# Cau hinh DEPLOYED, khop voi eval_ablation_crlmatched.py / eval_ablation_scope.py.
SCOPES = {
    "crlmatched": dict(
        folds=[0, 1, 2, 3, 4],
        system1=lambda k: Path(f"outputs/crlmatched_cv_gated/fold_{k}/system1/best_model.pt"),
        icrl=lambda k: Path(f"outputs/crlmatched_cv_gated_confmin03/fold_{k}/icrl"),
        data=lambda k: Path(f"data/fitzpatrick17k_crl_matched_cv_official/fold_{k}"),
        label_names=["label_0", "label_1"], expected=0.8053,
    ),
    "fulldata": dict(
        folds=[None],
        system1=lambda k: Path("outputs/scope_rebuild/fulldata/system1/best_model.pt"),
        icrl=lambda k: Path("outputs/scope_rebuild/fulldata/icrl_nolabelslot"),
        data=lambda k: Path("data/fitzpatrick17k_prepared"),
        label_names=FIT_NAMES, expected=0.8298,
    ),
    "concept_only": dict(
        folds=[None],
        system1=lambda k: Path("outputs/scope_rebuild/concept_only/system1/best_model.pt"),
        icrl=lambda k: Path("outputs/scope_rebuild/concept_only/icrl_nolabelslot"),
        data=lambda k: Path("data/fitzpatrick17k_prepared_concept_only"),
        label_names=FIT_NAMES, expected=0.7773,
    ),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scopes", type=str, default="crlmatched,fulldata,concept_only")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def cache_split(loader, system1, device):
    ys, cvs, p1s = [], [], []
    for images, labels in loader:
        out = system1(images.to(device))
        cvs.append(soft_concept_vector(out).cpu())
        p1s.append(F.softmax(out["label"], dim=-1).cpu())
        ys.append(labels["label"].cpu())
    return dict(y=torch.cat(ys), cv=torch.cat(cvs), p1=torch.cat(p1s))


def load_or_cache(scope, k, cfg, batch_size, num_workers, device):
    tag = f"{scope}" if k is None else f"{scope}_fold{k}"
    path = OUT_DIR / f"cache_{tag}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    system1, image_size = load_system1(cfg["system1"](k), device)
    system1.eval()
    _, val_loader, test_loader = make_loaders(cfg["data"](k), IMG_DIR, image_size, batch_size, num_workers)
    concept_names = test_loader.dataset.concept_names
    print(f"  [{tag}] S1 inference val ({len(val_loader.dataset)}) + test ({len(test_loader.dataset)})...")
    blob = dict(val=cache_split(val_loader, system1, device),
                test=cache_split(test_loader, system1, device),
                concept_names=concept_names)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    return blob


def empty_rule_mask(memory, keys_used, offsets, dims, threshold=0.5):
    """True cho rule khong co present concept nao (cung tieu chi voi
    predict_weighted_activation: mu >= threshold tren cac key khong bi loai)."""
    empty = torch.ones(memory.num_rules, dtype=torch.bool)
    for r in range(memory.num_rules):
        mu = memory._mu[r]
        for key in keys_used:
            s, e = offsets[key], offsets[key] + dims[key]
            if mu[s:e].item() >= threshold:
                empty[r] = False
                break
    return empty


def rule_majority_labels(memory):
    """Cung tieu chi voi predict_weighted_activation(): nhan da so, -1 neu rong."""
    return torch.tensor([collections.Counter(memory._labels[r]).most_common(1)[0][0]
                         if memory._labels[r] else -1 for r in range(memory.num_rules)], dtype=torch.long)


def vote_variant(memory, head, centroids, cv, layout, num_classes, gate, fallback, empty, prior=None):
    """prior: None (deployed) hoac tensor [C] tan suat lop; khi co, trong so
    moi rule chia cho prior[nhan cua rule] (prior can bang trong cach doc Bayes
    cua Proposition 1)."""
    keys, offsets, dims, exclude = layout
    hard = gate is not None
    v = memory.predict_weighted_activation(
        cv, keys, offsets, dims, num_classes=num_classes, exclude_keys=exclude,
        hard_gate=hard, gate_threshold=gate if hard else None)
    conf = torch.tensor(memory.get_confidences(), dtype=torch.float32)
    labels = rule_majority_labels(memory)
    w = v["firing_strength"] * conf.unsqueeze(0)
    if prior is None:
        cs = v["class_scores"]
    else:
        valid = labels >= 0
        w = w.clone()
        w[:, valid] = w[:, valid] / prior[labels[valid]].unsqueeze(0)
        w[:, ~valid] = 0.0
        cs = torch.zeros(cv.shape[0], num_classes)
        cs.index_add_(1, labels[valid], w[:, valid])
    tw = cs.sum(dim=1)
    low = tw < 1e-6
    p2 = cs / tw.clamp(min=1e-8).unsqueeze(-1)
    if low.any():
        rule_ids, _ = memory.match(cv)
        if fallback == "head":
            p2s = F.softmax(head(centroids[rule_ids]), dim=-1)
            p2[low] = p2s[low]
        else:
            p2[low] = F.one_hot(labels[rule_ids][low].clamp(min=0), num_classes).float()
    w_nonempty = w[:, ~empty].sum(dim=1)
    vacuous = (~low) & (w_nonempty < 1e-6)
    pr = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    eff_n = torch.exp(-(pr * pr.clamp(min=1e-12).log()).sum(dim=1))
    eff_n[low] = 0.0
    return dict(p2m=p2, tw=tw, n_fired=v["n_fired"].float(), low=low, vacuous=vacuous, eff_n=eff_n)


def assemble(y, p1, var):
    p2m = var["p2m"]
    pred1, pred2m = p1.argmax(1), p2m.argmax(1)

    def margin(p, a, b):
        logp = p.clamp(min=1e-8).log()
        idx = torch.arange(len(p))
        return logp[idx, a] - logp[idx, b]

    feat_agg5 = torch.stack([margin(p1, pred1, pred2m), margin(p2m, pred2m, pred1), var["tw"],
                              var["n_fired"], entropy_confidence(p1)], dim=1)
    return dict(y=y, p1=p1, p2m=p2m, pred1=pred1, pred2m=pred2m, feat_agg5=feat_agg5,
                low=var["low"], vacuous=var["vacuous"], eff_n=var["eff_n"])


def learned_soft_pred(a_cal, a_test):
    """Ban sao CHINH XAC cua eval_ablation_crlmatched.learned_soft(), tra ve du
    doan thay vi accuracy."""
    dis_cal = (a_cal["pred1"] != a_cal["pred2m"]).numpy()
    X_cal = a_cal["feat_agg5"][dis_cal].numpy()
    y_cal_label = (a_cal["pred2m"][dis_cal] == a_cal["y"][dis_cal]).long().numpy()
    pred = a_test["pred1"].clone()
    if len(np.unique(y_cal_label)) < 2 or len(X_cal) < 6:
        return pred
    scaler = StandardScaler().fit(X_cal)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(X_cal), y_cal_label)
    z_cal = torch.tensor(clf.decision_function(scaler.transform(X_cal)), dtype=torch.float32)
    p1_cal_dis, p2_cal_dis = a_cal["p1"][dis_cal], a_cal["p2m"][dis_cal]
    y_cal_dis = a_cal["y"][dis_cal]
    log_p1, log_p2 = p1_cal_dis.clamp(min=1e-8).log(), p2_cal_dis.clamp(min=1e-8).log()
    best_a, best_b, best_acc = 1.0, 0.0, -1.0
    for aa in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        for bb in (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0):
            w2 = torch.sigmoid(aa * z_cal + bb).unsqueeze(-1)
            p = ((1.0 - w2) * log_p1 + w2 * log_p2).argmax(dim=-1)
            acc = float((p == y_cal_dis).float().mean())
            if acc > best_acc:
                best_acc, best_a, best_b = acc, aa, bb
    dis_test = (a_test["pred1"] != a_test["pred2m"]).numpy()
    if dis_test.sum() > 0:
        z_t = torch.tensor(clf.decision_function(scaler.transform(a_test["feat_agg5"][dis_test].numpy())),
                           dtype=torch.float32)
        w2 = torch.sigmoid(best_a * z_t + best_b).unsqueeze(-1)
        p1_t, p2_t = a_test["p1"][torch.from_numpy(dis_test)], a_test["p2m"][torch.from_numpy(dis_test)]
        pred_dis = ((1.0 - w2) * p1_t.clamp(min=1e-8).log() + w2 * p2_t.clamp(min=1e-8).log()).argmax(dim=-1)
        pred[np.where(dis_test)[0]] = pred_dis
    return pred


def classify_cases(a, final):
    agree = a["pred1"] == a["pred2m"]
    case = np.empty(len(final), dtype=object)
    for i in range(len(final)):
        if a["low"][i]:
            case[i] = "d_none"
        elif a["vacuous"][i]:
            case[i] = "d_vacuous"
        elif agree[i]:
            case[i] = "a"
        elif final[i] == a["pred2m"][i]:
            case[i] = "b"
        elif final[i] == a["pred1"][i]:
            case[i] = "c"
        else:
            case[i] = "e"
    return case


def run_scope(scope, args, device):
    cfg = SCOPE_CONFIGS_LOCAL[scope]
    num_classes = len(cfg["label_names"]) if scope != "crlmatched" else 2
    per_variant = {}
    for k in cfg["folds"]:
        blob = load_or_cache(scope, k, cfg, args.batch_size, args.num_workers, device)
        keys, offsets, dims, _ = build_full_concept_layout(blob["concept_names"], cfg["label_names"])
        exclude = {S1_LABEL_CONCEPT_KEY}
        layout = (keys, offsets, dims, exclude)
        icrl_dir = cfg["icrl"](k)
        base_memory = ICRLRuleMemory.load(icrl_dir / "icrl_rule_memory.pt", device="cpu")
        head = nn.Linear(base_memory.concept_dim, num_classes)
        head.load_state_dict(torch.load(icrl_dir / "prediction_head.pt", map_location="cpu", weights_only=False))
        head.eval()
        keys_used = [kk for kk in keys if kk not in exclude]
        y_val = blob["val"]["y"]
        val_prior = torch.bincount(y_val, minlength=num_classes).float() / len(y_val)

        for strip in (False, True):
            memory = copy.deepcopy(base_memory)
            if strip:
                strip_empty_rules(memory, keys, offsets, dims, exclude, verbose=False)
            centroids = memory.get_centroids().cpu()
            empty = empty_rule_mask(memory, keys_used, offsets, dims)
            for use_prior in (False, True):
                prior = val_prior if use_prior else None
                for gate in GATES:
                    for fb in FALLBACKS:
                        name = (f"strip={'Y' if strip else 'N'}/prior={'Y' if use_prior else 'N'}/"
                                f"gate={'off' if gate is None else gate}/fb={fb}")
                        with torch.no_grad():
                            a_cal = assemble(y_val, blob["val"]["p1"],
                                             vote_variant(memory, head, centroids, blob["val"]["cv"], layout,
                                                          num_classes, gate, fb, empty, prior))
                            a_test = assemble(blob["test"]["y"], blob["test"]["p1"],
                                              vote_variant(memory, head, centroids, blob["test"]["cv"], layout,
                                                           num_classes, gate, fb, empty, prior))
                        final = learned_soft_pred(a_cal, a_test)
                        acc = float((final == a_test["y"]).float().mean())
                        if not strip and not use_prior and gate == 0.5 and fb == "head":
                            ref = learned_soft(a_cal, a_test, "p2m", "feat_agg5")
                            assert abs(ref - acc) < 1e-9, f"{scope} fold {k}: copy {acc} != deployed {ref}"
                        case = classify_cases(a_test, final.numpy())
                        correct = (final == a_test["y"]).numpy()
                        rec = per_variant.setdefault(name, dict(
                            accs=[], s1_accs=[], s2_accs=[], n=0,
                            counts={c: 0 for c in CASES}, correct={c: 0 for c in CASES},
                            eff_n=[], n_rules=[],
                            s2_pred=torch.zeros(num_classes), s2_hit=torch.zeros(num_classes),
                            y_count=torch.zeros(num_classes)))
                        rec["accs"].append(acc)
                        rec["s1_accs"].append(float((a_test["pred1"] == a_test["y"]).float().mean()))
                        rec["s2_accs"].append(float((a_test["pred2m"] == a_test["y"]).float().mean()))
                        rec["n"] += len(final)
                        for c in CASES:
                            m = case == c
                            rec["counts"][c] += int(m.sum())
                            rec["correct"][c] += int(correct[m].sum())
                        fired = ~(a_test["low"] | a_test["vacuous"])
                        rec["eff_n"].extend(a_test["eff_n"][fired].tolist())
                        rec["n_rules"].append(memory.num_rules)
                        yt, p2 = a_test["y"], a_test["pred2m"]
                        rec["s2_pred"] += torch.bincount(p2, minlength=num_classes).float()
                        rec["y_count"] += torch.bincount(yt, minlength=num_classes).float()
                        rec["s2_hit"] += torch.bincount(yt[p2 == yt], minlength=num_classes).float()

    summary = {}
    for name, rec in per_variant.items():
        summary[name] = dict(
            acc=float(np.mean(rec["accs"])), per_fold=rec["accs"],
            s1_acc=float(np.mean(rec["s1_accs"])), s2_acc=float(np.mean(rec["s2_accs"])),
            n_test=rec["n"],
            case_pct={c: 100.0 * rec["counts"][c] / rec["n"] for c in CASES},
            case_acc={c: (100.0 * rec["correct"][c] / rec["counts"][c] if rec["counts"][c] else None)
                      for c in CASES},
            eff_rules_median=float(np.median(rec["eff_n"])) if rec["eff_n"] else None,
            n_rules=rec["n_rules"],
            s2_pred_pct=(100.0 * rec["s2_pred"] / rec["n"]).tolist(),
            s2_recall=(100.0 * rec["s2_hit"] / rec["y_count"].clamp(min=1)).tolist(),
        )
    dep = summary["strip=N/prior=N/gate=0.5/fb=head"]["acc"]
    print(f"\n[{scope}] deployed multi/soft/learned acc = {dep*100:.2f}% (expected {cfg['expected']*100:.2f}%)")
    return summary


SCOPE_CONFIGS_LOCAL = SCOPES


def print_table(scope, summary):
    print(f"\n=== {scope} ===")
    hdr = (f"{'variant':<38s} {'acc':>6s} {'S2':>6s} {'S2bal':>6s} | "
           + " ".join(f"{c:>6s}" for c in CASES) + " | effN | S2 pred% / S2 recall%")
    print(hdr)
    for name, s in summary.items():
        cells = " ".join(f"{s['case_pct'][c]:6.1f}" for c in CASES)
        en = s["eff_rules_median"]
        bal = float(np.mean(s["s2_recall"]))
        pp = "/".join(f"{x:.0f}" for x in s["s2_pred_pct"])
        rc = "/".join(f"{x:.0f}" for x in s["s2_recall"])
        print(f"{name:<38s} {s['acc']*100:6.2f} {s['s2_acc']*100:6.2f} {bal:6.2f} | {cells} | "
              f"{'-' if en is None else f'{en:4.1f}'} | {pp} / {rc}")
    print(f"(S1 alone = {next(iter(summary.values()))['s1_acc']*100:.2f}%)")


def main():
    args = parse_args()
    device = torch.device("cpu")
    results = {}
    for scope in args.scopes.split(","):
        results[scope] = run_scope(scope, args, device)
        print_table(scope, results[scope])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
