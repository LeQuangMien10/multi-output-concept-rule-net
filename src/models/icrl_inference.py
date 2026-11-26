"""
icrl_inference.py - Suy luan cua ban ICRL moi nhat (chot 2026-11-26).

System 2, cho moi anh x va moi rule r con lai trong memory:
    present(r)  = {j : mu_rj >= 0.5}                 (memory khong con rule rong)
    fire(r,x)   = prod_{j in present(r)} c_j(x)      soft-AND, khong hard gate
    c_r         = (sum_{x' in val} fire(r,x') + 1) / (N_val + 1)   (memory.fit_coverage)
    w(r,x)      = fire(r,x) / c_r * conf(r)
    P(r|x)      = w(r,x) / sum_r' w(r',x)
    P2'(y|x)    = sum_r P(r|x) 1[y_r = y]
Fallback (tong bang chung tho sum_r fire*conf < 1e-6, chi xay ra khi tich bi
underflow): softmax(head(centroid cua rule gan nhat)).

Ket hop voi System 1:
    dong thuan (argmax P1 = argmax P2')   -> giu nhan cua S1 (Corollary 1);
    bat dong -> z(x) = logistic regression (class_weight=balanced) tren 5 dac
    trung feat_agg5 [margin1, margin2, tong trong so, so rule fire, entropy S1],
    w2 = sigmoid(a*z + b) voi (a,b) chon tren luoi theo accuracy tren val,
    P = softmax((1-w2) log P1 + w2 log P2')  (log-opinion pool, Proposition 2).

Ban nay khop CHINH XAC voi eval_coverage_normalized.run_one(gate=None,
normalize=True) chay tren memory da bo rule rong (kiem chung boi
src/scripts/finalize_icrl_latest.py).
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.ensemble_strategies import entropy_confidence

MIN_TOTAL_WEIGHT = 1e-6
LOG_CLAMP = 1e-8
GRID_A = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
GRID_B = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)
CASES = ("a", "b", "c", "e", "d")   # e: bat dong, nhan cuoi khac ca hai he (chi co khi >= 3 lop)


def rule_labels(memory) -> torch.Tensor:
    return torch.tensor([collections.Counter(l).most_common(1)[0][0] if l else -1
                         for l in memory._labels], dtype=torch.long)


@torch.no_grad()
def s2_vote(memory, head, cv, layout, num_classes):
    """layout = (concept_keys, offsets, dims, exclude_keys). Tra ve P2' va cac dai
    luong giai thich cho tung anh."""
    keys, offsets, dims, exclude = layout
    cv = cv.to(memory.device)
    fs = memory.predict_weighted_activation(
        cv, keys, offsets, dims, num_classes=num_classes, exclude_keys=exclude,
        hard_gate=False)["firing_strength"].cpu()
    labels = rule_labels(memory)
    valid = labels >= 0
    conf = torch.tensor(memory.get_confidences(), dtype=torch.float32)
    raw = fs * conf.unsqueeze(0)
    raw[:, ~valid] = 0.0
    w = raw / memory.get_coverage().cpu().unsqueeze(0)
    class_scores = torch.zeros(cv.shape[0], num_classes)
    class_scores.index_add_(1, labels[valid], w[:, valid])
    total_weight = class_scores.sum(dim=1)
    p2 = class_scores / total_weight.clamp(min=1e-12).unsqueeze(-1)
    low = raw.sum(dim=1) < MIN_TOTAL_WEIGHT
    if low.any():
        rule_ids, _ = memory.match(cv)
        p2[low] = F.softmax(head(memory.get_centroids()[rule_ids]).cpu(), dim=-1)[low]
    rule_post = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)          # P(r|x)
    eff_n = torch.exp(-(rule_post * rule_post.clamp(min=1e-12).log()).sum(dim=1))
    return dict(p2=p2, total_weight=total_weight, n_fired=(fs > 0.01).sum(dim=1).float(),
                no_rule=low, rule_post=rule_post, eff_n=eff_n)


def _margin(p, a, b):
    logp = p.clamp(min=LOG_CLAMP).log()
    idx = torch.arange(len(p))
    return logp[idx, a] - logp[idx, b]


def assemble(y, p1, vote):
    p2 = vote["p2"]
    pred1, pred2 = p1.argmax(1), p2.argmax(1)
    feat = torch.stack([_margin(p1, pred1, pred2), _margin(p2, pred2, pred1), vote["total_weight"],
                        vote["n_fired"], entropy_confidence(p1)], dim=1)
    return dict(y=y, p1=p1, p2=p2, pred1=pred1, pred2=pred2, feat=feat, no_rule=vote["no_rule"])


def _pool(p1, p2, w2):
    return (1.0 - w2) * p1.clamp(min=LOG_CLAMP).log() + w2 * p2.clamp(min=LOG_CLAMP).log()


@dataclass
class Combiner:
    """g(x) va (a,b), hoc tren cac ca bat dong cua tap val."""
    scaler: StandardScaler | None = None
    clf: LogisticRegression | None = None
    a: float = 1.0
    b: float = 0.0
    fitted: bool = field(default=False)

    def fit(self, a_val):
        dis = (a_val["pred1"] != a_val["pred2"]).numpy()
        X = a_val["feat"][dis].numpy()
        t = (a_val["pred2"][dis] == a_val["y"][dis]).long().numpy()
        if len(np.unique(t)) < 2 or len(X) < 6:
            return self   # qua it ca bat dong: luon giu nhan S1
        self.scaler = StandardScaler().fit(X)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(self.scaler.transform(X), t)
        z = torch.tensor(self.clf.decision_function(self.scaler.transform(X)), dtype=torch.float32)
        p1, p2, y = a_val["p1"][dis], a_val["p2"][dis], a_val["y"][dis]
        best = -1.0
        for aa in GRID_A:
            for bb in GRID_B:
                w2 = torch.sigmoid(aa * z + bb).unsqueeze(-1)
                acc = float((_pool(p1, p2, w2).argmax(-1) == y).float().mean())
                if acc > best:
                    best, self.a, self.b = acc, aa, bb
        self.fitted = True
        return self

    def predict(self, a_test):
        """Tra ve (nhan cuoi [B], w2 [B]); w2 = 0 tren ca dong thuan."""
        pred = a_test["pred1"].clone()
        w2_all = torch.zeros(len(pred))
        dis = (a_test["pred1"] != a_test["pred2"]).numpy()
        if not self.fitted or dis.sum() == 0:
            return pred, w2_all
        z = torch.tensor(self.clf.decision_function(self.scaler.transform(a_test["feat"][dis].numpy())),
                         dtype=torch.float32)
        w2 = torch.sigmoid(self.a * z + self.b)
        m = torch.from_numpy(dis)
        pred[m] = _pool(a_test["p1"][m], a_test["p2"][m], w2.unsqueeze(-1)).argmax(-1)
        w2_all[m] = w2
        return pred, w2_all


def balanced_accuracy(pred, y, num_classes):
    rec = [float(((pred == y) & (y == k)).sum() / (y == k).sum()) for k in range(num_classes) if (y == k).any()]
    return float(np.mean(rec))


@torch.no_grad()
def s2_balanced_accuracy(memory, head, cv, y, layout, num_classes):
    """Tieu chi chon (theta, n_min) cua ban moi nhat: balanced accuracy cua P2'
    tren val (chot 2026-11-26, src/scripts/select_theta_nmin.py)."""
    if memory.num_rules == 0:
        return 0.0
    p2 = s2_vote(memory, head, cv, layout, num_classes)["p2"]
    return balanced_accuracy(p2.argmax(1), y.cpu(), num_classes)


def explanation_cases(a, final):
    """(a) dong thuan, (b) bat dong va nhan cuoi theo S2, (c) theo S1,
    (e) nhan thu ba, (d) khong rule nao dong gop (fallback)."""
    agree = (a["pred1"] == a["pred2"]).numpy()
    final = np.asarray(final)
    case = np.where(agree, "a", np.where(final == a["pred2"].numpy(), "b",
                                         np.where(final == a["pred1"].numpy(), "c", "e"))).astype(object)
    case[a["no_rule"].numpy()] = "d"
    return case
