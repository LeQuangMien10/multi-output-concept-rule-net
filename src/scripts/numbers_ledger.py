"""
numbers_ledger.py - R6: so cai so lieu cho draft main_v2.tex (ban ICRL moi nhat).

Gom MOI con so se dua vao paper tu cac output da kiem chung, kem nguon cua tung
muc. Khong chep so tu paper/main.tex cu. Muc chua chot (y 4, y 5, baseline CUB)
duoc ghi ro la "pending".

Nguon:
  outputs/icrl_latest/{eval_results,bootstrap,ablation,complementarity}.json, build_info.json
  outputs/s1_cache/*.pt, outputs/cub_outputs/s1_cache.pt           (kich thuoc split)
  outputs/crlmatched_cv_gated_confmin03/crlmatched_cv_fold*.json     (concept metric cua S1)
  outputs/crlmatched_cv_{basci,cmr}/cv_summary.json                  (baseline tu chay lai)
  Gao et al. 2025, arXiv 2505.14049v2, Table 1 (baseline cong bo, doi chieu 2026-11-27)

Usage:
    python -m src.scripts.numbers_ledger
Output: outputs/icrl_latest/numbers_ledger.json, notes/numbers_ledger.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
import torch

from src.scripts.bootstrap_icrl_latest import predictions
from src.scripts.eval_icrl_latest import DATASETS, LATEST

OUT_JSON = LATEST / "numbers_ledger.json"
OUT_MD = Path("notes/numbers_ledger.md")
CACHES = {**{f"crlmatched_fold{k}": f"outputs/s1_cache/crlmatched_fold{k}.pt" for k in range(5)},
          "fulldata": "outputs/s1_cache/fulldata.pt", "concept_only": "outputs/s1_cache/concept_only.pt",
          "cub_class": "outputs/cub_outputs/s1_cache.pt"}
# Quyet dinh 2026-11-27: paper dung CUB theo thiet lap cua Koh (concept theo loai);
# CUB concept theo anh bi bo khoi paper. Ca hai scope Fitzpatrick 3 lop deu trinh bay.
PAPER_DATASETS = ["crlmatched", "fulldata", "concept_only", "cub_class"]
# Gao et al. 2025 (CRL), Table 1, F17k, 5-fold CV mean ± std, doi chieu arXiv 2505.14049v2 ngay 2026-11-27.
CRL_PUBLISHED = {
    "CBM": [91.41, 0.19, 59.44, 0.97, 76.37, 3.45, 76.30, 3.45],
    "align-CBM": [89.77, 0.52, 58.98, 1.41, 75.93, 2.39, 75.83, 2.41],
    "CEM": [91.85, 0.55, 59.11, 2.06, 76.26, 2.59, 76.17, 2.52],
    "evi-CEM": [92.04, 0.80, 58.65, 1.71, 76.47, 1.69, 76.39, 1.66],
    "hard-CBM": [86.36, 1.09, 49.57, 0.48, 73.51, 2.94, 73.43, 2.92],
    "DCR": [91.03, 0.80, 49.40, 1.00, 75.05, 2.12, 74.97, 2.12],
    "CRL": [92.80, 0.47, 52.39, 0.28, 75.95, 3.09, 75.90, 3.08],
}
METRIC_KEYS = ["concept_acc", "concept_acc_std", "concept_f1", "concept_f1_std",
               "diag_acc", "diag_acc_std", "diag_f1", "diag_f1_std"]


def pm(values):
    """Trung binh va do lech chuan quan the qua fold (%), cung quy uoc cv_summary.json."""
    v = 100 * np.asarray(values, dtype=float)
    return round(float(v.mean()), 2), round(float(v.std()), 2)


def parse_pm(s):
    m, sd = s.rstrip("%").split("+-")
    return float(m), float(sd)


def dataset_stats():
    out = {}
    for name, path in CACHES.items():
        b = torch.load(path, weights_only=False)
        out[name] = dict(n_train=len(b["train"]["y"]), n_val=len(b["val"]["y"]), n_test=len(b["test"]["y"]),
                         n_concepts=len(b["concept_names"]), n_classes=len(b["label_names"]), source=path)
    return out


def crlmatched_table():
    rows = {k: dict(zip(METRIC_KEYS, v), source="Gao et al. 2025, Table 1 (published)")
            for k, v in CRL_PUBLISHED.items()}
    for name, d in (("Basci et al.", "basci"), ("CMR", "cmr")):
        s = json.load(open(f"outputs/crlmatched_cv_{d}/cv_summary.json"))["summary"]
        vals = []
        for key in ("concept_acc", "concept_f1", "diagnosis_acc", "diagnosis_f1"):
            vals += list(parse_pm(s[key]))
        rows[name] = dict(zip(METRIC_KEYS, vals), source=f"outputs/crlmatched_cv_{d}/cv_summary.json (our rerun)")
    s1 = [json.load(open(f"outputs/crlmatched_cv_gated_confmin03/crlmatched_cv_fold{k}.json")) for k in range(5)]
    folds = [predictions(p) for p in DATASETS["crlmatched"]["parts"]]
    accs = [float((f["ens"] == f["y"]).mean()) for f in folds]
    f1s = [f1_score(f["y"], f["ens"], average="macro") for f in folds]
    vals = [*pm([r["concept_acc"] for r in s1]), *pm([r["concept_f1"] for r in s1]), *pm(accs), *pm(f1s)]
    rows["ICRL (ours)"] = dict(zip(METRIC_KEYS, vals),
                               source="concept: S1, crlmatched_cv_gated_confmin03/crlmatched_cv_fold*.json; "
                                      "diagnosis: ensemble, outputs/icrl_latest (macro F1)")
    return rows


def main():
    ev = json.load(open(LATEST / "eval_results.json"))
    boot = json.load(open(LATEST / "bootstrap.json"))["results"]
    abl = json.load(open(LATEST / "ablation.json"))
    comp = json.load(open(LATEST / "complementarity.json"))
    hyper = {n: json.load(open(LATEST / n / "build_info.json")) for n in CACHES}
    main_tab = {}
    for ds in PAPER_DATASETS:
        r = ev[ds]
        main_tab[ds] = {k: r[k] for k in ("s1", "s1_std", "s2", "ens", "ens_std", "bal_s1", "bal_s2", "bal_ens",
                                          "cases_pct", "case_acc", "len_w", "short_w", "eff_n_median",
                                          "top1_share", "n_rules", "w2_disagree_mean")}
        main_tab[ds]["ci95"] = {k: boot[ds][k] for k in ("acc_ens", "acc_ens_minus_s1", "bal_ens_minus_s1")}
    ledger = dict(
        version="ICRL latest (2026-11-26): no gate, fire/c_r*conf, no empty rules, conf_min 0, "
                "(theta, n_min) by val S2 balanced accuracy",
        datasets=dataset_stats(), hyperparameters=hyper, main=main_tab,
        ablation={ds: {v: {k: m[k] for k in ("s2", "bal_s2", "ens", "bal_ens", "len_w", "short_w", "eff_n_median",
                                             "top1_share", "cases_pct")} for v, m in d.items()}
                  for ds, d in abl.items()},
        complementarity=comp, crlmatched_baselines=crlmatched_table(),
        decisions={"fitzpatrick_3class": "both full-data and concept-only in the main results; "
                                         "balanced accuracy not required in the paper (kept here for reference)",
                   "cub": "class-level concepts as in Koh et al. 2020; instance-level variant dropped"},
        pending={"cub_baselines": "published CUB numbers (e.g. Koh et al. CBM) not yet sourced and verified"},
    )
    with open(OUT_JSON, "w") as f:
        json.dump(ledger, f, indent=2)
    write_md(ledger)
    print(f"Saved -> {OUT_JSON}, {OUT_MD}")


def write_md(L):
    names = {"crlmatched": "CRL-matched (5-fold)", "fulldata": "Full-data", "concept_only": "Concept-only",
             "cub_class": "CUB (class-level, Koh)"}
    lines = ["# Numbers ledger (bản ICRL mới nhất)", "",
             "Sinh tự động bởi `python -m src.scripts.numbers_ledger`; nguồn duy nhất cho số liệu của",
             "`paper/main_v2.tex`. Không sửa tay.", "", f"Phiên bản: {L['version']}.", "",
             "## Dữ liệu và hyperparameter", "",
             "| Setting | Train / val / test | Concepts | Classes | θ | n_min | conf_min | Rules |",
             "|---|---|---|---|---|---|---|---|"]
    for n, d in L["datasets"].items():
        h = L["hyperparameters"][n]
        th = f"{h['theta']:.2f}"
        lines.append(f"| {n} | {d['n_train']} / {d['n_val']} / {d['n_test']} | {d['n_concepts']} | {d['n_classes']} | "
                     f"{th} | {h['n_min']} | {h.get('conf_min', 0.0)} | {h['num_rules']} |")
    lines += ["", "## Kết quả chính", "",
              "| Dataset | S1 | S2 | Ensemble [CI 95%] | Ens − S1 [CI 95%] | Bal. S1 / S2 / ens | (a) / (b) / (c) / (e) / (d) | len_w | Rule ngắn | effN |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, r in L["main"].items():
        ci, dd = r["ci95"]["acc_ens"], r["ci95"]["acc_ens_minus_s1"]
        ens = f"{r['ens']:.2f}" + (f" ± {r['ens_std']:.2f}" if ds == "crlmatched" else "")
        cs = " / ".join(f"{r['cases_pct'][c]:.1f}" for c in ("a", "b", "c", "e", "d"))
        lines.append(f"| {names[ds]} | {r['s1']:.2f} | {r['s2']:.2f} | {ens} [{ci['ci95'][0]:.2f}, {ci['ci95'][1]:.2f}] | "
                     f"{dd['point']:+.2f} [{dd['ci95'][0]:+.2f}, {dd['ci95'][1]:+.2f}] | "
                     f"{r['bal_s1']:.1f} / {r['bal_s2']:.1f} / {r['bal_ens']:.1f} | {cs} | {r['len_w']:.2f} | "
                     f"{r['short_w']:.1f}% | {r['eff_n_median']:.1f} |")
    lines += ["", "## Bảng CRL-matched với baseline", "",
              "| Method | Concept acc | Concept F1 | Diagnosis acc | Diagnosis F1 | Nguồn |", "|---|---|---|---|---|---|"]
    for m, r in L["crlmatched_baselines"].items():
        lines.append(f"| {m} | {r['concept_acc']:.2f} ± {r['concept_acc_std']:.2f} | {r['concept_f1']:.2f} ± "
                     f"{r['concept_f1_std']:.2f} | {r['diag_acc']:.2f} ± {r['diag_acc_std']:.2f} | "
                     f"{r['diag_f1']:.2f} ± {r['diag_f1_std']:.2f} | {r['source']} |")
    lines += ["", "## Ablation trọng số rule (cùng artifact, g(x) học lại)", "",
              "| Dataset | Biến thể | S2 | Bal. S2 | Ensemble | len_w | Rule ngắn | effN | (d) không rule |",
              "|---|---|---|---|---|---|---|---|---|"]
    for ds, d in L["ablation"].items():
        for v, m in d.items():
            lines.append(f"| {names[ds]} | {v} | {m['s2']:.2f} | {m['bal_s2']:.1f} | {m['ens']:.2f} | {m['len_w']:.2f} | "
                         f"{m['short_w']:.1f}% | {m['eff_n_median']:.1f} | {m['cases_pct']['d']:.1f}% |")
    lines += ["", "## Bổ trợ S1/S2", "",
              "| Dataset | Bất đồng | Oracle | Khi bất đồng: S1 đúng / S2 đúng / cả hai sai | Ens giữ S1 khi chỉ S1 đúng | Ens theo S2 khi chỉ S2 đúng |",
              "|---|---|---|---|---|---|"]
    for ds, c in L["complementarity"].items():
        lines.append(f"| {names[ds]} | {c['disagreement']:.1f}% | {c['oracle']:.2f} | {c['dis_s1_right']:.1f} / "
                     f"{c['dis_s2_right']:.1f} / {c['dis_both_wrong']:.1f} | {c['ens_keeps_s1_only']:.1f}% | "
                     f"{c['ens_recovers_s2_only']:.1f}% |")
    lines += ["", "## Quyết định", ""] + [f"- {k}: {v}" for k, v in L["decisions"].items()]
    lines += ["", "## Chưa chốt", ""] + [f"- {k}: {v}" for k, v in L["pending"].items()] + [""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
