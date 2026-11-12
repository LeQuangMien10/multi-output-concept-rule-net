"""
eval_ablation_crlmatched.py - Thi nghiem factorial day du, crlmatched, de
tach bach cac truc khac nhau giua baseline (gated_override dang deploy) va
v1/v2+pool (notes/s1s2_formalization.md), theo yeu cau nguoi dung: "chay
het cac hoan vi, dam bao khong bo sot thu nghiem nao".

4 truc nhi phan:
  - p2_source  in {single, multi}   -- match() 1-nearest, hay predict_weighted_activation() nhieu rule
  - form       in {hard, soft}      -- chon cung (override nhi phan) hay pooling lien tuc
  - signal     in {simple, learned} -- tin hieu THO kieu baseline (1-2 gia tri tay) hay g(x) hoc (6/5 dac trung)
  - protocol   in {per_fold, lofo}  -- hieu chinh CHI tren val cua chinh fold (nhu baseline), hay
                                        leave-one-fold-out (hoc tren 4 fold KHAC, nhu v1/v2+pool hien tai)

16 to hop. Da biet truoc 7 trong 16 (tu cac thi nghiem truoc), con lai la
MOI, chua tung do. Script nay TINH LAI CA 16 trong cung 1 lan chay, cung
1 pipeline du lieu, de dam bao so sanh nhat quan (khong lay lai so cu tu
cac script khac nhau).

Usage (local, inference only, khong train lai gi):
    python -m src.scripts.eval_ablation_crlmatched --device cpu
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.scripts.measure_contingency_multiscope import DEFAULT_IMG_DIR
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.ensemble_strategies import entropy_confidence
from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY

NUM_FOLDS = 5


def strip_empty_rules(memory, concept_keys, concept_offsets, concept_dims, exclude_keys, verbose=True):
    """Loai bo HAN cac rule "rong" -- khong co concept dieu kien nao (vacuously
    fire moi anh, bat ke gan/xa) -- khoi memory, dung _compact() co san (cung
    co che dedupe_by_decoded_pattern() dung). Tra ve so rule da bi loai."""
    surviving = []
    n_removed = 0
    for r in range(memory.num_rules):
        d = memory.decode_rule(r, concept_keys, concept_offsets, concept_dims)
        present = [k for k in concept_keys if k not in exclude_keys and d["slots"][k]["value"] == "present"]
        if present:
            surviving.append(r)
        else:
            n_removed += 1
    if n_removed > 0:
        memory._compact(surviving)
    if verbose:
        print(f"    strip_empty_rules: loai {n_removed}/{n_removed+len(surviving)} rule rong, "
              f"con lai {memory.num_rules} rule")
    return n_removed


@torch.no_grad()
def collect_fold(k, cv_root, system1_root, data_root, img_dir, device, batch_size=64, num_workers=0,
                  remove_empty_rules=False):
    from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders, build_full_concept_layout

    fold_dir = cv_root / f"fold_{k}"
    system1, image_size = load_system1(system1_root / f"fold_{k}" / "system1" / "best_model.pt", device)
    _, val_loader, test_loader = make_loaders(data_root / f"fold_{k}", img_dir, image_size, batch_size, num_workers)
    concept_names = test_loader.dataset.concept_names
    concept_keys, concept_offsets, concept_dims, _ = build_full_concept_layout(
        concept_names, ["label_0", "label_1"])
    exclude_keys = {S1_LABEL_CONCEPT_KEY}

    memory = ICRLRuleMemory.load(fold_dir / "icrl" / "icrl_rule_memory.pt", device=str(device))
    if remove_empty_rules:
        strip_empty_rules(memory, concept_keys, concept_offsets, concept_dims, exclude_keys)
    head = nn.Linear(memory.concept_dim, 2).to(device)
    head.load_state_dict(torch.load(fold_dir / "icrl" / "prediction_head.pt",
                                     map_location=device, weights_only=False))
    head.eval()
    centroids = memory.get_centroids().to(device)
    confidences = torch.tensor(memory.get_confidences(), device=device)
    rule_n = torch.tensor([float(memory._n[r]) for r in range(memory.num_rules)], device=device)
    rule_coh = torch.tensor([float(memory._compute_coherence(r)) for r in range(memory.num_rules)], device=device)

    def run(loader):
        ys, p1s, p2s_, rcs, rns, rcohs, css, tws, nfs = [], [], [], [], [], [], [], [], []
        for images, labels in loader:
            images = images.to(device)
            out = system1(images)
            cv = soft_concept_vector(out)
            y = labels["label"].to(device)
            p1 = F.softmax(out["label"], dim=-1)

            rule_ids, _ = memory.match(cv)
            p2s_single = F.softmax(head(centroids[rule_ids]), dim=-1)

            vote = memory.predict_weighted_activation(
                cv, concept_keys, concept_offsets, concept_dims, num_classes=2, exclude_keys=exclude_keys)

            ys.append(y); p1s.append(p1); p2s_.append(p2s_single)
            rcs.append(confidences[rule_ids]); rns.append(rule_n[rule_ids]); rcohs.append(rule_coh[rule_ids])
            css.append(vote["class_scores"]); tws.append(vote["class_scores"].sum(dim=1))
            nfs.append(vote["n_fired"].float())
        return dict(y=torch.cat(ys), p1=torch.cat(p1s), p2s=torch.cat(p2s_), rc=torch.cat(rcs),
                    rule_n=torch.cat(rns), rule_coh=torch.cat(rcohs), class_scores=torch.cat(css),
                    total_weight=torch.cat(tws), n_fired=torch.cat(nfs))

    return {"val": run(val_loader), "test": run(test_loader)}


def build_p2_multi(d):
    cs, tw, fb = d["class_scores"], d["total_weight"], d["p2s"]
    p2 = cs / tw.clamp(min=1e-8).unsqueeze(-1)
    low = tw < 1e-6
    p2[low] = fb[low]
    return p2


def assemble(d):
    p1, p2s = d["p1"], d["p2s"]
    p2m = build_p2_multi(d)
    pred1, pred2s, pred2m = p1.argmax(1), p2s.argmax(1), p2m.argmax(1)
    y = d["y"]

    def margin(p, a, b):
        logp = p.clamp(min=1e-8).log()
        idx = torch.arange(len(p))
        return logp[idx, a] - logp[idx, b]

    feat6 = torch.stack([margin(p1, pred1, pred2s), margin(p2s, pred2s, pred1), d["rc"],
                          torch.log1p(d["rule_n"]), d["rule_coh"], entropy_confidence(p1)], dim=1)
    feat_agg5 = torch.stack([margin(p1, pred1, pred2m), margin(p2m, pred2m, pred1), d["total_weight"],
                              d["n_fired"], entropy_confidence(p1)], dim=1)

    return dict(y=y, p1=p1, p2s=p2s, p2m=p2m, pred1=pred1, pred2s=pred2s, pred2m=pred2m,
                rc=d["rc"], total_weight=d["total_weight"], s1_conf=p1.max(1).values,
                feat6=feat6, feat_agg5=feat_agg5)


def acc(pred, y):
    return float((pred == y).float().mean())


# ---------- simple signal (baseline-style, khong hoc) ----------

def simple_hard(a_cal, a_test, p2_key, cal_signal_key):
    """AND-gate 2 nguong (s1_conf, cal_signal) -- dung cal_signal_key='rc'
    (single) hoac 'total_weight' (multi), GIONG CAU TRUC gated_override_grid
    cua baseline, chi doi tin hieu S2."""
    s1c_cal, sig_cal, y_cal = a_cal["s1_conf"], a_cal[cal_signal_key], a_cal["y"]
    pred1_cal, pred2_cal = a_cal["pred1"], a_cal["pred2s" if p2_key == "p2s" else "pred2m"]
    thresh_grid = np.linspace(float(s1c_cal.min()), float(s1c_cal.max()), 9)
    sig_grid = np.linspace(float(sig_cal.min()), float(sig_cal.max()), 6)
    best_combo, best_acc = None, -1.0
    for t1 in thresh_grid:
        for t2 in sig_grid:
            override = (s1c_cal < t1) & (sig_cal > t2)
            pred = pred1_cal.clone(); pred[override] = pred2_cal[override]
            a = acc(pred, y_cal)
            if a > best_acc:
                best_acc, best_combo = a, (t1, t2)
    t1, t2 = best_combo
    s1c_t, sig_t = a_test["s1_conf"], a_test[cal_signal_key]
    pred1_t, pred2_t = a_test["pred1"], a_test["pred2s" if p2_key == "p2s" else "pred2m"]
    override_t = (s1c_t < t1) & (sig_t > t2)
    pred = pred1_t.clone(); pred[override_t] = pred2_t[override_t]
    return acc(pred, a_test["y"])


def simple_soft(a_cal, a_test, p2_key, cal_signal_key):
    """w2(x) = clamp(signal*scale, 0, 1), dung CHINH CONG THUC rc_weighted_pool/
    fit_rc_weighted_scale (don hang, khong AND-gate)."""
    sig_cal, y_cal = a_cal[cal_signal_key], a_cal["y"]
    p1_cal, p2_cal = a_cal["p1"], a_cal[p2_key]
    grid = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    best_scale, best_acc = 1.0, -1.0
    for scale in grid:
        w2 = (sig_cal * scale).clamp(0.0, 1.0).unsqueeze(-1)
        w1 = 1.0 - w2
        pred = (w1 * p1_cal.clamp(min=1e-8).log() + w2 * p2_cal.clamp(min=1e-8).log()).argmax(dim=-1)
        a = acc(pred, y_cal)
        if a > best_acc:
            best_acc, best_scale = a, scale
    sig_t, p1_t, p2_t = a_test[cal_signal_key], a_test["p1"], a_test[p2_key]
    w2 = (sig_t * best_scale).clamp(0.0, 1.0).unsqueeze(-1)
    w1 = 1.0 - w2
    pred = (w1 * p1_t.clamp(min=1e-8).log() + w2 * p2_t.clamp(min=1e-8).log()).argmax(dim=-1)
    return acc(pred, a_test["y"])


# ---------- learned signal g(x) ----------

def learned_hard(a_cal, a_test, p2_key, feat_key):
    pred2_key = "pred2s" if p2_key == "p2s" else "pred2m"
    dis_cal = (a_cal["pred1"] != a_cal[pred2_key]).numpy()
    X_cal = a_cal[feat_key][dis_cal].numpy()
    y_cal = (a_cal[pred2_key][dis_cal] == a_cal["y"][dis_cal]).long().numpy()
    pred1_test = a_test["pred1"].clone()
    if len(np.unique(y_cal)) < 2 or len(X_cal) < 6:
        return acc(pred1_test, a_test["y"])  # qua it mau, khong fit duoc -> fallback S1
    scaler = StandardScaler().fit(X_cal)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(X_cal), y_cal)
    proba_cal = clf.predict_proba(scaler.transform(X_cal))[:, 1]
    best_t, best_acc = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        a = float(((proba_cal >= t).astype(int) == y_cal).mean())
        if a > best_acc:
            best_acc, best_t = a, float(t)

    dis_test = (a_test["pred1"] != a_test[pred2_key]).numpy()
    if dis_test.sum() > 0:
        proba_t = clf.predict_proba(scaler.transform(a_test[feat_key][dis_test].numpy()))[:, 1]
        idx = np.where(dis_test)[0]
        override = proba_t >= best_t
        pred1_test[idx[override]] = a_test[pred2_key][idx[override]]
    return acc(pred1_test, a_test["y"])


def learned_soft(a_cal, a_test, p2_key, feat_key):
    pred2_key = "pred2s" if p2_key == "p2s" else "pred2m"
    dis_cal = (a_cal["pred1"] != a_cal[pred2_key]).numpy()
    X_cal = a_cal[feat_key][dis_cal].numpy()
    y_cal_label = (a_cal[pred2_key][dis_cal] == a_cal["y"][dis_cal]).long().numpy()
    pred1_test = a_test["pred1"].clone()
    if len(np.unique(y_cal_label)) < 2 or len(X_cal) < 6:
        return acc(pred1_test, a_test["y"])
    scaler = StandardScaler().fit(X_cal)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(X_cal), y_cal_label)

    z_cal = clf.decision_function(scaler.transform(X_cal))
    z_cal = torch.tensor(z_cal, dtype=torch.float32)
    p1_cal_dis, p2_cal_dis = a_cal["p1"][dis_cal], a_cal[p2_key][dis_cal]
    y_cal_dis = a_cal["y"][dis_cal]
    log_p1, log_p2 = p1_cal_dis.clamp(min=1e-8).log(), p2_cal_dis.clamp(min=1e-8).log()
    best_a, best_b, best_acc = 1.0, 0.0, -1.0
    for aa in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        for bb in (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0):
            w2 = torch.sigmoid(aa * z_cal + bb).unsqueeze(-1)
            w1 = 1.0 - w2
            pred = (w1 * log_p1 + w2 * log_p2).argmax(dim=-1)
            a = acc(pred, y_cal_dis)
            if a > best_acc:
                best_acc, best_a, best_b = a, aa, bb

    dis_test = (a_test["pred1"] != a_test[pred2_key]).numpy()
    if dis_test.sum() > 0:
        z_t = clf.decision_function(scaler.transform(a_test[feat_key][dis_test].numpy()))
        z_t = torch.tensor(z_t, dtype=torch.float32)
        w2 = torch.sigmoid(best_a * z_t + best_b).unsqueeze(-1)
        w1 = 1.0 - w2
        p1_t, p2_t = a_test["p1"][torch.from_numpy(dis_test)], a_test[p2_key][torch.from_numpy(dis_test)]
        pred_dis = (w1 * p1_t.clamp(min=1e-8).log() + w2 * p2_t.clamp(min=1e-8).log()).argmax(dim=-1)
        idx = np.where(dis_test)[0]
        pred1_test[idx] = pred_dis
    return acc(pred1_test, a_test["y"])


def concat_assembled(assembled_list):
    keys = assembled_list[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.cat([a[k] for a in assembled_list], dim=0)
    return out


def main():
    device = torch.device("cpu")
    cv_root = Path("outputs/crlmatched_cv_gated_confmin03")
    system1_root = Path("outputs/crlmatched_cv_gated")
    data_root = Path("data/fitzpatrick17k_crl_matched_cv_official")

    print("Thu thap du lieu 5 fold...")
    fold_raw = [collect_fold(k, cv_root, system1_root, data_root, DEFAULT_IMG_DIR, device) for k in range(5)]
    fold_a = [{"val": assemble(fr["val"]), "test": assemble(fr["test"])} for fr in fold_raw]

    cells = [
        ("single", "hard", "simple",  "p2s", "rc",           simple_hard,  None),
        ("single", "hard", "learned", "p2s", "feat6",        learned_hard, None),
        ("single", "soft", "simple",  "p2s", "rc",           simple_soft,  None),
        ("single", "soft", "learned", "p2s", "feat6",        learned_soft, None),
        ("multi",  "hard", "simple",  "p2m", "total_weight", simple_hard,  None),
        ("multi",  "hard", "learned", "p2m", "feat_agg5",    learned_hard, None),
        ("multi",  "soft", "simple",  "p2m", "total_weight", simple_soft,  None),
        ("multi",  "soft", "learned", "p2m", "feat_agg5",    learned_soft, None),
    ]

    results = {}
    for p2_source, form, signal, p2_key, sig_or_feat_key, fn, _ in cells:
        for protocol in ["per_fold", "lofo"]:
            accs = []
            for k in range(NUM_FOLDS):
                if protocol == "per_fold":
                    a_cal = fold_a[k]["val"]
                else:
                    a_cal = concat_assembled([fold_a[j]["val"] for j in range(NUM_FOLDS) if j != k])
                a_test = fold_a[k]["test"]
                a = fn(a_cal, a_test, p2_key, sig_or_feat_key)
                accs.append(a)
            mean_acc = sum(accs) / len(accs)
            name = f"{p2_source}/{form}/{signal}/{protocol}"
            results[name] = {"mean_acc": mean_acc, "per_fold": accs}
            print(f"{name:<35s} acc={mean_acc*100:.2f}%  per_fold={[round(x*100,2) for x in accs]}")

    out_path = Path("outputs/concept_rule_balancing/ablation_crlmatched.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
