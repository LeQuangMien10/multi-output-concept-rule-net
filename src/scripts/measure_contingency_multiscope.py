"""
measure_contingency_multiscope.py - Buoc 1 cua ke hoach S1/S2 coupling (xem
memory project_s1s2_coupling_plan.md): do lai contingency breakdown
(both_correct / s1_only_correct / s2_only_correct / both_wrong) VA quy mo
circular-rule bucket (so anh test co pattern trung voi 1 group circular da
bi dedupe xoa), tren nhieu scope, TRUOC khi sua gi o System~2. Chi inference,
khong train lai, dung lai checkpoint da co.

Scope duoc do (xem CAVEAT khac nhau tung scope):
  - crlmatched   : Fitzpatrick CRL-matched, 5-fold CV, pipeline HIEN TAI
                   (Wilson score + lazy-collapse fix + conf_min=0.3).
                   SACH -- day la so lieu da biet, dung de doi chieu.
  - fulldata     : Fitzpatrick full-data, 1 split, checkpoint thang 8/2026
                   (TRUOC lazy-collapse fix, thang 10/2026).
  - concept_only : Fitzpatrick concept-only, 1 split, checkpoint thang 8/2026.
  - pad_ufes20   : PAD-UFES-20, 1 split, checkpoint thang 9/2026, 6 lop.

CAVEAT QUAN TRONG cho fulldata/concept_only/pad_ufes20: checkpoint System~1
cua 3 scope nay deu TRUOC fix lazy-collapse (xem project_crlmatched_5fold_cv_plan.md),
nen so lieu o day KHONG sach -- neu pattern "22%"/circular-rule VAN xuat hien
thi la bang chung YEU rang day la 1 ICRL property chung (vi van thay du o
checkpoint cu); neu KHONG xuat hien thi KHONG KET LUAN DUOC gi (co the chi vi
S1 con collapse ve lam methep het moi thu), can rerun S1 voi
--monitor label_acc_gated cho scope do truoc khi tin ket qua.

Usage (local, inference only):
    python -m src.scripts.measure_contingency_multiscope \\
        --output_json outputs/contingency_multiscope/contingency_multiscope.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description="Contingency breakdown + circular-rule bucket, nhieu scope.")
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--scopes", type=str, default="crlmatched,fulldata,concept_only,pad_ufes20",
                    help="Comma-separated subset cua 4 scope, de chay rieng tung cai khi debug.")
    return p.parse_args()


def decode_pattern_from_vec(cv: torch.Tensor, concept_keys, concept_offsets, concept_dims,
                             exclude_keys, threshold: float = 0.5) -> tuple:
    """Giong pattern_of() trong ICRLRuleMemory.dedupe_by_decoded_pattern(), nhung
    ap dung cho 1 concept vector bat ky (anh test), khong phai cho 1 rule's mu."""
    present = []
    for key in concept_keys:
        if key in exclude_keys:
            continue
        s = concept_offsets[key]
        e = s + concept_dims[key]
        if cv[s:e].item() >= threshold:
            present.append(key)
    return tuple(present)


def find_circular_patterns(memory, concept_keys, concept_offsets, concept_dims, exclude_keys):
    """Non-destructive (chi doc, khong goi dedupe_by_decoded_pattern() nen
    khong mutate memory): group cac rule theo pattern hien thi (dung
    decode_rule() -- API cong khai), tra ve tap pattern nao co >= 2 majority
    label khac nhau (= dung group se bi dedupe xoa vi circular).

    QUAN TRONG: `memory` phai la trang thai NGAY TRUOC dedupe_by_decoded_pattern()
    se duoc goi trong pipeline that -- nghia la DA prune qua n_min roi conf_min
    (xem get_pre_dedupe_memory() ben duoi), KHONG phai icrl_rule_memory_base.pt
    tho (hang tram micro-rule chua prune) -- dung base tho se dem thua rat
    nhieu "circular" gia (2 rule n=1 trung pattern tinh co, nhung se bi n_min
    loai truoc khi dedupe thay trong pipeline that)."""
    groups = defaultdict(set)
    for r in range(memory.num_rules):
        d = memory.decode_rule(r, concept_keys, concept_offsets, concept_dims)
        present = tuple(k for k in concept_keys if k not in exclude_keys
                         and d["slots"][k]["value"] == "present")
        groups[present].add(d["label"])
    circular = {pattern for pattern, labels in groups.items() if len(labels) > 1}
    return circular


def get_pre_dedupe_memory(base_path, n_min, conf_min, system1, head, val_loader, device,
                           record_rule_accuracy):
    """Replay dung cac buoc pipeline that lam TRUOC khi goi dedupe_by_decoded_pattern()
    (xem train_icrl.py::run_stage3_onward / apply_conf_min_cv.py::main): load
    base tho -> prune theo n_min (bo qua accuracy signal) -> record_rule_accuracy
    tren val -> prune theo conf_min thuc. KHONG goi dedupe, KHONG save -- chi
    de co dung trang thai memory ma dedupe se nhin thay, phuc vu
    find_circular_patterns() o tren.

    `record_rule_accuracy` duoc truyen vao (khong import cung) vi fitzpatrick/
    pad_ufes20 co 2 ban train_icrl.py rieng -- goi dung ban khop voi module
    dang dung o moi scope."""
    from src.models.icrl_rule_memory import ICRLRuleMemory

    memory = ICRLRuleMemory.load(base_path, device=str(device))
    memory.n_min = n_min
    memory.prune(verbose=False, conf_min_override=0.0)
    record_rule_accuracy(system1, head, memory, val_loader, device)
    memory.conf_min = conf_min
    memory.prune(verbose=False)
    return memory


@torch.no_grad()
def collect_split(loader, system1, memory, head, centroids, device,
                   concept_keys, concept_offsets, concept_dims, exclude_keys):
    """Tra ve y, pred S1, pred S2, va so anh match circular pattern cho 1 split."""
    from src.models.fitzpatrick.system1 import soft_concept_vector

    ys, s1_preds, s2_preds, patterns = [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        cv = soft_concept_vector(out)
        rule_ids, _ = memory.match(cv)

        ys.append(labels["label"].to(device))
        s1_preds.append(F.softmax(out["label"], dim=-1).argmax(-1))
        s2_preds.append(F.softmax(head(centroids[rule_ids]), dim=-1).argmax(-1))

        for i in range(cv.shape[0]):
            patterns.append(decode_pattern_from_vec(cv[i], concept_keys, concept_offsets,
                                                      concept_dims, exclude_keys))
    return (torch.cat(ys).cpu().numpy(), torch.cat(s1_preds).cpu().numpy(),
            torch.cat(s2_preds).cpu().numpy(), patterns)


def contingency(y, s1_pred, s2_pred, circular_patterns, patterns) -> dict:
    s1_correct = (s1_pred == y)
    s2_correct = (s2_pred == y)
    n = len(y)
    n_circular = sum(1 for p in patterns if p in circular_patterns)
    return {
        "n": int(n),
        "both_correct": int((s1_correct & s2_correct).sum()),
        "s1_only_correct": int((s1_correct & ~s2_correct).sum()),
        "s2_only_correct": int((~s1_correct & s2_correct).sum()),
        "both_wrong": int((~s1_correct & ~s2_correct).sum()),
        "s1_acc": float(s1_correct.mean()),
        "s2_acc": float(s2_correct.mean()),
        "n_circular_pattern_images": int(n_circular),
        "pct_circular_pattern_images": float(n_circular / n) if n else 0.0,
    }


def run_single_split_scope(name, module, data_dir, img_dir, system1_ckpt, icrl_dir,
                            num_labels, batch_size, num_workers, device, caveat):
    if module == "fitzpatrick":
        from src.scripts.fitzpatrick.train_icrl import (
            load_system1, make_loaders, build_full_concept_layout, record_rule_accuracy,
        )
        from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY
    else:
        from src.scripts.pad_ufes20.train_icrl import (
            load_system1, make_loaders, build_full_concept_layout, record_rule_accuracy,
        )
        from src.utils.pad_ufes20_concepts import S1_LABEL_CONCEPT_KEY

    from src.models.icrl_rule_memory import ICRLRuleMemory

    system1, image_size = load_system1(Path(system1_ckpt), device)
    _, val_loader, test_loader = make_loaders(Path(data_dir), img_dir, image_size,
                                               batch_size, num_workers)
    concept_names = test_loader.dataset.concept_names
    # build_full_concept_layout chi dung len(label_names) de xac dinh so chieu
    # cua slot s1_label_pred, khong dung noi dung ten -- placeholder du dai la du.
    concept_keys, concept_offsets, concept_dims, _ = build_full_concept_layout(
        concept_names, [f"label_{i}" for i in range(num_labels)],
    )
    exclude_keys = {S1_LABEL_CONCEPT_KEY}

    memory = ICRLRuleMemory.load(Path(icrl_dir) / "icrl_rule_memory.pt", device=str(device))
    head = nn.Linear(memory.concept_dim, num_labels).to(device)
    head.load_state_dict(torch.load(Path(icrl_dir) / "prediction_head.pt",
                                     map_location=device, weights_only=False))
    head.eval()
    centroids = memory.get_centroids().to(device)

    # circular-pattern check PHAI chay tren trang thai memory NGAY TRUOC dedupe
    # (da prune qua n_min roi conf_min), khong phai icrl_rule_memory_base.pt tho
    # -- xem docstring get_pre_dedupe_memory(). n_min/conf_min doc tu metrics.json
    # da luu (dung dung gia tri pipeline that da dung, khong doan lai).
    base_path = Path(icrl_dir) / "icrl_rule_memory_base.pt"
    metrics_path = Path(icrl_dir) / "metrics.json"
    if base_path.exists() and metrics_path.exists():
        saved_metrics = json.load(open(metrics_path))
        rec_n_min = saved_metrics["n_min"]
        rec_conf_min = saved_metrics["args"]["conf_min"]
        pre_dedupe_memory = get_pre_dedupe_memory(base_path, rec_n_min, rec_conf_min,
                                                   system1, head, val_loader, device,
                                                   record_rule_accuracy)
        circular_patterns = find_circular_patterns(pre_dedupe_memory, concept_keys, concept_offsets,
                                                     concept_dims, exclude_keys)
    else:
        circular_patterns = set()
        print(f"[WARN] {name}: thieu icrl_rule_memory_base.pt hoac metrics.json, "
              f"bo qua circular-pattern check.")

    results = {}
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        y, s1p, s2p, patterns = collect_split(loader, system1, memory, head, centroids,
                                                device, concept_keys,
                                                concept_offsets, concept_dims, exclude_keys)
        results[split_name] = contingency(y, s1p, s2p, circular_patterns, patterns)

    out = {
        "scope": name, "kind": "single_split", "caveat": caveat,
        "num_rules": memory.num_rules, "num_circular_patterns_in_base": len(circular_patterns),
        **results,
    }
    t = results["test"]
    print(f"[{name}] rules={memory.num_rules} circular_patterns={len(circular_patterns)} | "
          f"test: n={t['n']} both_correct={t['both_correct']} ({t['both_correct']/t['n']*100:.1f}%) "
          f"s1_only={t['s1_only_correct']} ({t['s1_only_correct']/t['n']*100:.1f}%) "
          f"s2_only={t['s2_only_correct']} ({t['s2_only_correct']/t['n']*100:.1f}%) "
          f"both_wrong={t['both_wrong']} ({t['both_wrong']/t['n']*100:.1f}%) | "
          f"circular_pattern_images={t['n_circular_pattern_images']} "
          f"({t['pct_circular_pattern_images']*100:.2f}%)")
    return out


def run_crlmatched_cv_scope(cv_root, system1_root, data_root, num_folds,
                             batch_size, num_workers, device, caveat):
    from src.scripts.fitzpatrick.train_icrl import (
        load_system1, make_loaders, build_full_concept_layout, record_rule_accuracy,
    )
    from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY
    from src.models.icrl_rule_memory import ICRLRuleMemory

    cv_root, system1_root, data_root = Path(cv_root), Path(system1_root), Path(data_root)
    exclude_keys = {S1_LABEL_CONCEPT_KEY}

    pooled_test = {"y": [], "s1p": [], "s2p": []}
    per_fold = {}

    for k in range(num_folds):
        fold_dir = cv_root / f"fold_{k}"
        system1, image_size = load_system1(system1_root / f"fold_{k}" / "system1" / "best_model.pt", device)
        _, val_loader, test_loader = make_loaders(data_root / f"fold_{k}", DEFAULT_IMG_DIR, image_size,
                                                    batch_size, num_workers)
        concept_names = test_loader.dataset.concept_names
        num_labels = 2
        concept_keys, concept_offsets, concept_dims, _ = build_full_concept_layout(
            concept_names, [f"label_{i}" for i in range(num_labels)],
        )

        memory = ICRLRuleMemory.load(fold_dir / "icrl" / "icrl_rule_memory.pt", device=str(device))
        head = nn.Linear(memory.concept_dim, num_labels).to(device)
        head.load_state_dict(torch.load(fold_dir / "icrl" / "prediction_head.pt",
                                         map_location=device, weights_only=False))
        head.eval()
        centroids = memory.get_centroids().to(device)

        # icrl_rule_memory_base.pt co the KHONG nam trong cv_root (vd.
        # apply_conf_min_cv.py chi doc base tu system1_root/cv_root GOC, khong
        # copy lai vao output_root moi vi re-tune conf_min khong doi clustering)
        # -- fallback sang system1_root truoc khi bo qua han. Dung memory.n_min/
        # memory.conf_min DA LUU trong icrl_rule_memory.pt (final) de replay dung
        # chinh xac recipe pipeline that dung, khong doan lai tu file khac --
        # xem get_pre_dedupe_memory()/find_circular_patterns() ve ly do phai
        # prune truoc moi kiem tra circular (khong dung base tho).
        base_path = fold_dir / "icrl" / "icrl_rule_memory_base.pt"
        if not base_path.exists():
            base_path = system1_root / f"fold_{k}" / "icrl" / "icrl_rule_memory_base.pt"
        if base_path.exists():
            pre_dedupe_memory = get_pre_dedupe_memory(base_path, memory.n_min, memory.conf_min,
                                                       system1, head, val_loader, device,
                                                       record_rule_accuracy)
            circular_patterns = find_circular_patterns(pre_dedupe_memory, concept_keys, concept_offsets,
                                                         concept_dims, exclude_keys)
        else:
            circular_patterns = set()
            print(f"[WARN] fold_{k}: khong tim thay icrl_rule_memory_base.pt o ca "
                  f"{fold_dir / 'icrl'} va {system1_root / f'fold_{k}' / 'icrl'} -- "
                  f"bo qua circular-pattern check cho fold nay.")

        y, s1p, s2p, patterns = collect_split(test_loader, system1, memory, head, centroids,
                                                device, concept_keys,
                                                concept_offsets, concept_dims, exclude_keys)
        per_fold[f"fold_{k}"] = contingency(y, s1p, s2p, circular_patterns, patterns)
        per_fold[f"fold_{k}"]["num_rules"] = memory.num_rules
        per_fold[f"fold_{k}"]["num_circular_patterns_in_base"] = len(circular_patterns)

        pooled_test["y"].append(y)
        pooled_test["s1p"].append(s1p)
        pooled_test["s2p"].append(s2p)

    # QUAN TRONG: "circular" la khai niem GAN VOI RULE MEMORY RIENG CUA TUNG
    # FOLD (moi fold tu train/dedupe rieng) -- khong the pool bang cach check
    # tat ca anh pool lai voi UNION cac circular_pattern cua moi fold (se bi
    # "leak": anh fold_3 trung ngau nhien 1 pattern tung circular o fold_1,
    # trong khi memory CUA FOLD_3 chua bao gio flag pattern do la circular,
    # dem thua). Phai cong truc tiep so circular-image DA DEM DUNG tung fold.
    n_circular_pooled = sum(per_fold[f"fold_{k}"]["n_circular_pattern_images"] for k in range(num_folds))

    import numpy as np
    y_all = np.concatenate(pooled_test["y"])
    s1p_all = np.concatenate(pooled_test["s1p"])
    s2p_all = np.concatenate(pooled_test["s2p"])
    pooled = contingency(y_all, s1p_all, s2p_all, set(), [])
    pooled["n_circular_pattern_images"] = n_circular_pooled
    pooled["pct_circular_pattern_images"] = n_circular_pooled / pooled["n"] if pooled["n"] else 0.0

    out = {"scope": "crlmatched", "kind": "5fold_cv", "caveat": caveat,
           "per_fold": per_fold, "pooled_test": pooled}
    print(f"[crlmatched] pooled test: n={pooled['n']} "
          f"both_correct={pooled['both_correct']} ({pooled['both_correct']/pooled['n']*100:.1f}%) "
          f"s1_only={pooled['s1_only_correct']} ({pooled['s1_only_correct']/pooled['n']*100:.1f}%) "
          f"s2_only={pooled['s2_only_correct']} ({pooled['s2_only_correct']/pooled['n']*100:.1f}%) "
          f"both_wrong={pooled['both_wrong']} ({pooled['both_wrong']/pooled['n']*100:.1f}%) | "
          f"circular_pattern_images={pooled['n_circular_pattern_images']} "
          f"({pooled['pct_circular_pattern_images']*100:.2f}%)")
    return out


DEFAULT_IMG_DIR = "data/fitzpatrick17k/data/finalfitz17k"


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    scopes = set(args.scopes.split(","))

    results = {}

    if "crlmatched" in scopes:
        results["crlmatched"] = run_crlmatched_cv_scope(
            cv_root="outputs/crlmatched_cv_gated_confmin03",
            system1_root="outputs/crlmatched_cv_gated",
            data_root="data/fitzpatrick17k_crl_matched_cv_official",
            num_folds=5, batch_size=args.batch_size, num_workers=args.num_workers,
            device=device,
            caveat="SACH: pipeline hien tai (Wilson + lazy-collapse fix + conf_min=0.3).",
        )

    if "fulldata" in scopes:
        results["fulldata"] = run_single_split_scope(
            name="fulldata", module="fitzpatrick",
            data_dir="data/fitzpatrick17k_prepared", img_dir=DEFAULT_IMG_DIR,
            system1_ckpt="outputs/fitzpatrick_system1/best_model.pt",
            icrl_dir="outputs/fitzpatrick_icrl_calibrated",
            num_labels=3, batch_size=args.batch_size, num_workers=args.num_workers,
            device=device,
            caveat="CHUA SACH: checkpoint S1 thang 8/2026, TRUOC lazy-collapse fix (thang 10/2026).",
        )

    if "concept_only" in scopes:
        results["concept_only"] = run_single_split_scope(
            name="concept_only", module="fitzpatrick",
            data_dir="data/fitzpatrick17k_prepared_concept_only", img_dir=DEFAULT_IMG_DIR,
            system1_ckpt="outputs/fitzpatrick_system1_concept_only/best_model.pt",
            icrl_dir="outputs/fitzpatrick_icrl_concept_only_calibrated",
            num_labels=3, batch_size=args.batch_size, num_workers=args.num_workers,
            device=device,
            caveat="CHUA SACH: checkpoint S1 thang 8/2026, TRUOC lazy-collapse fix (thang 10/2026).",
        )

    if "pad_ufes20" in scopes:
        results["pad_ufes20"] = run_single_split_scope(
            name="pad_ufes20", module="pad_ufes20",
            data_dir="data/PAD-UFES-20_prepared", img_dir="data/PAD-UFES-20/images",
            system1_ckpt="outputs/pad_ufes20_system1/best_model.pt",
            icrl_dir="outputs/pad_ufes20_icrl",
            num_labels=6, batch_size=args.batch_size, num_workers=args.num_workers,
            device=device,
            caveat="CHUA SACH: checkpoint S1 thang 9/2026, chua tung ap lazy-collapse fix (scope nay bi parked).",
        )

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved -> {args.output_json}")


if __name__ == "__main__":
    main()
