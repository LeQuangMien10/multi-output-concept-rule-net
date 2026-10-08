"""
ensemble_strategies.py - Cac chien luoc ket hop System~1 + System~2 (ICRL),
bo sung ben canh 3 chien luoc da co (weighted-average, confidence-pick,
gated-override trong eval_ensemble.py / eval_crlmatched_metrics.py), theo
goi y cua advisor ve information theory (xem memory
project_infotheory_ensemble_plan.md):

  1. Temperature scaling: hieu chinh lai softmax cua MOI SYSTEM rieng le
     truoc khi ket hop -- buoc nen, khong phai mot chien luoc ket hop.
  2. Entropy confidence: thay max-softmax (confidence tho) bang entropy
     chuan hoa 1 - H(p)/log(K) lam tin hieu tin cay trong gated-override.
  3. Log-linear pooling (product of experts): P(y) ~ P1(y)^w1 * P2(y)^w2,
     la cach ket hop toi thieu hoa KL-divergence, khac voi weighted-average
     (cong tuyen tinh, khong co co so ly thuyet thong tin ro rang).
  4. Naive-Bayes evidence fusion: P(y|S1,S2) ~ P(y) * P(S1=s1|y) * P(S2=s2|y),
     dung confusion matrix cua tung system do tren val lam likelihood --
     doi xung, moi system co the "thang" neu bang chung cua no manh hon,
     khong bi khoa mot chieu nhu gated-override (mac dinh S1, chi doi sang
     S2 khi S1 tu nhan la khong tu tin).

KHONG sua eval_ensemble.py / eval_crlmatched_metrics.py -- day la module moi,
chi dung ben trong eval_ensemble_infotheory.py, de khong anh huong so lieu
Table 1 da bao cao.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 1. Temperature scaling
# ─────────────────────────────────────────────────────────────

def fit_temperature(logits: torch.Tensor, y: torch.Tensor, max_iter: int = 200) -> float:
    """Fit 1 scalar T toi thieu hoa NLL tren (logits, y) -- Guo et al. 2017.
    T > 1 lam "mem" xac suat (giam qua tu tin), T < 1 lam "sac" hon."""
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        T = log_T.exp()
        loss = F.cross_entropy(logits / T, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp().item())


def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    return F.softmax(logits / T, dim=-1)


# ─────────────────────────────────────────────────────────────
# 2. Entropy confidence
# ─────────────────────────────────────────────────────────────

def entropy_confidence(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - H(p)/log(K) trong [0, 1]: 1 = chac chan tuyet doi (1 lop xac suat 1),
    0 = phan phoi deu (khong biet gi). K = so lop."""
    K = probs.shape[-1]
    h = -(probs.clamp(min=eps) * probs.clamp(min=eps).log()).sum(dim=-1)
    h_max = torch.log(torch.tensor(float(K)))
    return 1.0 - h / h_max


# ─────────────────────────────────────────────────────────────
# 3. Log-linear pooling (product of experts)
# ─────────────────────────────────────────────────────────────

def log_linear_pool(p1: torch.Tensor, p2: torch.Tensor, w1: float, w2: float,
                     eps: float = 1e-8) -> torch.Tensor:
    """P(y) ~ P1(y)^w1 * P2(y)^w2, chuan hoa lai thanh phan phoi hop le.
    w1=w2=1 la product-of-experts thuan; w1=1,w2=0 la chi dung S1."""
    log_p = w1 * p1.clamp(min=eps).log() + w2 * p2.clamp(min=eps).log()
    return F.softmax(log_p, dim=-1)


def fit_log_linear_weights(p1_val: torch.Tensor, p2_val: torch.Tensor, y_val: torch.Tensor,
                            grid: list[float] | None = None) -> tuple[float, float]:
    """Grid-search w1 in [0,1] (w2 = 1-w1) toi da hoa accuracy tren val --
    cung tinh than voi alpha cua weighted-average, chi khac cong thuc ket hop."""
    if grid is None:
        grid = [i / 20 for i in range(21)]  # 0.00, 0.05, ..., 1.00
    best_w1, best_acc = 0.5, -1.0
    for w1 in grid:
        pred = log_linear_pool(p1_val, p2_val, w1, 1.0 - w1).argmax(dim=-1)
        a = (pred == y_val).float().mean().item()
        if a > best_acc:
            best_acc, best_w1 = a, w1
    return best_w1, 1.0 - best_w1


# ─────────────────────────────────────────────────────────────
# 4. Naive-Bayes evidence fusion
# ─────────────────────────────────────────────────────────────

def fit_confusion_likelihood(pred: torch.Tensor, y: torch.Tensor, num_classes: int,
                              eps: float = 1e-2) -> torch.Tensor:
    """P(pred=i | y=j) tren val, lam likelihood table [K_pred, K_true].
    Laplace-smoothing (eps) de tranh xac suat 0 cho cap (i,j) chua xuat hien
    trong val (hay gap voi so mau val nho ~180 anh/fold)."""
    K = num_classes
    table = torch.full((K, K), eps)
    for i, j in zip(pred.tolist(), y.tolist()):
        table[i, j] += 1.0
    table = table / table.sum(dim=0, keepdim=True)  # chuan hoa theo tung lop y=j
    return table


def fit_class_prior(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.tensor([float((y == c).sum()) for c in range(num_classes)])
    return counts / counts.sum()


def naive_bayes_fusion(pred1: torch.Tensor, pred2: torch.Tensor, prior: torch.Tensor,
                        likelihood1: torch.Tensor, likelihood2: torch.Tensor) -> torch.Tensor:
    """P(y | S1=pred1, S2=pred2) ~ P(y) * P(S1=pred1|y) * P(S2=pred2|y), doc
    tung mau. Doi xung: neu mot system co likelihood manh (rat phan biet
    dung/sai theo class) con system kia yeu, system manh se chi phoi ket qua
    bat ke la S1 hay S2 -- khong bi khoa mot chieu nhu gated-override."""
    N, K = len(pred1), len(prior)
    posterior = torch.zeros(N, K)
    for n in range(N):
        p = prior * likelihood1[pred1[n]] * likelihood2[pred2[n]]
        posterior[n] = p / p.sum().clamp(min=1e-12)
    return posterior


# ─────────────────────────────────────────────────────────────
# 5. Rule-confidence-only override / rule-confidence-weighted pooling
# -- suy ra tu phan tich pattern (analyze_disagreement_patterns.py):
# confidence cua S1 KHONG phai tin hieu tot de biet khi nao nen doi sang S2
# (AUROC~0.40 -- S1 thuong tu tin HON chinh o cac ca S1 sai), chi rule_conf
# (Wilson) co tin hieu dung huong, du yeu (AUROC~0.55). Vi vay bo dieu kien
# "S1 khong tu tin" ra khoi quyet dinh, chi dung rule_conf.
# ─────────────────────────────────────────────────────────────

def rule_conf_override(p1: torch.Tensor, p2: torch.Tensor, rc: torch.Tensor,
                        thresh: float) -> torch.Tensor:
    """Doi sang S2 chi can rc > thresh -- KHONG xet confidence cua S1 (khac
    gated_override, bo han dieu kien da chung minh la nguoc huong)."""
    override = rc > thresh
    pred = p1.argmax(dim=-1).clone()
    pred[override] = p2.argmax(dim=-1)[override]
    return pred


def fit_rule_conf_override_thresh(p1_val, p2_val, rc_val, y_val, n_grid: int = 25) -> float:
    lo, hi = float(rc_val.min()), float(rc_val.max())
    grid = torch.linspace(lo, hi, n_grid)
    best_t, best_acc = float(grid[0]), -1.0
    for t in grid:
        pred = rule_conf_override(p1_val, p2_val, rc_val, float(t))
        a = (pred == y_val).float().mean().item()
        if a > best_acc:
            best_acc, best_t = a, float(t)
    return best_t


def rc_weighted_pool(p1: torch.Tensor, p2: torch.Tensor, rc: torch.Tensor,
                      scale: float, eps: float = 1e-8) -> torch.Tensor:
    """Log-linear pooling nhung trong so cua S2 la MOI MAU (w2 = rc*scale,
    kep trong [0,1]), khong phai 1 trong so chung cho toan bo fold -- tan
    dung tin hieu yeu nhung dung huong cua rule_conf thay vi 1 gia tri alpha
    co dinh."""
    w2 = (rc * scale).clamp(0.0, 1.0).unsqueeze(-1)
    w1 = 1.0 - w2
    log_p = w1 * p1.clamp(min=eps).log() + w2 * p2.clamp(min=eps).log()
    return F.softmax(log_p, dim=-1)


def fit_rc_weighted_scale(p1_val, p2_val, rc_val, y_val, grid: list[float] | None = None) -> float:
    if grid is None:
        grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    best_scale, best_acc = 1.0, -1.0
    for scale in grid:
        pred = rc_weighted_pool(p1_val, p2_val, rc_val, scale).argmax(dim=-1)
        a = (pred == y_val).float().mean().item()
        if a > best_acc:
            best_acc, best_scale = a, scale
    return best_scale


# ─────────────────────────────────────────────────────────────
# Bidirectional correction accounting (dung chung cho MOI chien luoc,
# ca 3 chien luoc cu va 3 chien luoc moi, de so sanh dong bo)
# ─────────────────────────────────────────────────────────────

def bidirectional_correction_counts(pred1: torch.Tensor, pred2: torch.Tensor,
                                     pred_combined: torch.Tensor, y: torch.Tensor) -> dict:
    """Voi bai toan 2 lop: khi pred1 != pred2, LUON co dung 1 trong 2 dung
    (vi y chi co 2 gia tri) -- nen moi truong hop bat dong la mot co hoi
    "sua loi" ro rang theo 1 chieu. Dem xem combined co chon dung huong do
    khong, ca 2 chieu (S2 sua S1, S1 sua S2)."""
    s1_ok, s2_ok, comb_ok = (pred1 == y), (pred2 == y), (pred_combined == y)
    disagree = pred1 != pred2

    s1_wrong_s2_right = disagree & (~s1_ok) & s2_ok
    s2_wrong_s1_right = disagree & (~s2_ok) & s1_ok
    both_right_agree = (~disagree) & s1_ok
    both_wrong_agree = (~disagree) & (~s1_ok)

    return {
        "n": int(len(y)),
        "disagree_count": int(disagree.sum()),
        "s1_wrong_s2_right_count": int(s1_wrong_s2_right.sum()),
        "s2_fixes_s1_count": int((s1_wrong_s2_right & comb_ok).sum()),
        "s2_wrong_s1_right_count": int(s2_wrong_s1_right.sum()),
        "s1_fixes_s2_count": int((s2_wrong_s1_right & comb_ok).sum()),
        "both_agree_right_count": int(both_right_agree.sum()),
        "both_agree_right_kept_count": int((both_right_agree & comb_ok).sum()),
        "both_agree_wrong_count": int(both_wrong_agree.sum()),
        "both_agree_wrong_fixed_count": int((both_wrong_agree & comb_ok).sum()),
        "combined_accuracy": float(comb_ok.float().mean()),
    }
