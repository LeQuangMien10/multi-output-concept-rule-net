"""
concept_rule_validity_table.py - Buoc 1 cua ke hoach "Concept-Rule
Balancing" (ghi chu thay gui, paper/slides_and_notes (1).pdf, 2026-10-17):
ap dung khung 2x2 (True/False cua S1 x Satisfactory/Violation cua S2) len
du lieu CHECKPOINT DA CO (khong train/build lai gi), lap "Appendix:
Preliminary Results" cua ghi chu.

Dinh nghia (theo dung hinh thuc hoa trong ghi chu):
  True/False   : du doan NHAN rieng cua System~1 dung/sai (nhu truoc gio).
  Satisfactory : HARD concept pattern cua System~1 (threshold 0.5, loai bo
                 slot nhan S1 tu du doan) TRUNG KHOP CHINH XAC voi pattern
                 cua it nhat 1 rule CON SONG SOT (khong circular) trong
                 rule memory CUOI CUNG -- nghia la ton tai 1 cong thuc
                 r_i trong R giai thich DUNG cho assignment nay (C |= R).
  Violation    : khong co rule nao trong R co pattern trung khop chinh xac
                 -- hoac vi pattern do nam trong nhom circular da bi xoa
                 (mau thuan, khong the chon 1 nhan ma khong co them bang
                 chung), hoac vi khong rule nao tung thay pattern nay.

QUAN TRONG: day la khai niem KHAC voi "S2 dung/sai nhan" ma tuan truoc da
do (contingency table s1_only/s2_only) -- Satisfactory/Violation la ve
TINH HOP LE LOGIC cua chinh concept pattern S1 tu du doan, khong phai ve
viec rule duoc match (qua continuous nearest-centroid, luon tra ve 1 ket
qua bat ke gan hay xa) co dung nhan hay khong.

Usage (local, inference only, dung checkpoint da co):
    python -m src.scripts.concept_rule_validity_table \\
        --output_json outputs/concept_rule_balancing/validity_table.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from src.scripts.run_scope_rebuild import SCOPE_CONFIGS
from src.scripts.measure_contingency_multiscope import decode_pattern_from_vec, DEFAULT_IMG_DIR
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.fitzpatrick.system1 import soft_concept_vector


def parse_args():
    p = argparse.ArgumentParser(description="Bang 2x2 True/False x Satisfactory/Violation, 4 scope.")
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def build_valid_pattern_set(final_memory, concept_keys, concept_offsets, concept_dims, exclude_keys):
    """R, theo nghia 'tap pattern co 1 rule KHONG-CIRCULAR giai thich dung'
    -- doc truc tiep tu rule memory CUOI CUNG (da qua prune+dedupe), dung
    API cong khai decode_rule()."""
    patterns = set()
    for r in range(final_memory.num_rules):
        d = final_memory.decode_rule(r, concept_keys, concept_offsets, concept_dims)
        present = tuple(k for k in concept_keys if k not in exclude_keys
                         and d["slots"][k]["value"] == "present")
        patterns.add(present)
    return patterns


@torch.no_grad()
def collect_2x2(loader, system1, valid_patterns, concept_keys, concept_offsets,
                 concept_dims, exclude_keys, device):
    counts = defaultdict(int)
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        cv = soft_concept_vector(out)
        y = labels["label"].to(device)
        s1_pred = out["label"].argmax(dim=-1)
        s1_correct = (s1_pred == y)
        for i in range(cv.shape[0]):
            pattern = decode_pattern_from_vec(cv[i], concept_keys, concept_offsets,
                                               concept_dims, exclude_keys)
            satisfactory = pattern in valid_patterns
            key = ("True" if s1_correct[i].item() else "False",
                   "Satisfactory" if satisfactory else "Violation")
            counts[key] += 1
    return counts


def run_single_split(name, module, data_dir, img_dir, system1_ckpt, icrl_dir,
                      num_labels, batch_size, num_workers, device):
    if module == "fitzpatrick":
        from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders, record_rule_accuracy, build_full_concept_layout
        from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY
    else:
        from src.scripts.pad_ufes20.train_icrl import load_system1, make_loaders, record_rule_accuracy, build_full_concept_layout
        from src.utils.pad_ufes20_concepts import S1_LABEL_CONCEPT_KEY

    system1, image_size = load_system1(Path(system1_ckpt), device)
    _, val_loader, test_loader = make_loaders(Path(data_dir), img_dir, image_size, batch_size, num_workers)
    concept_names = test_loader.dataset.concept_names
    concept_keys, concept_offsets, concept_dims, _ = build_full_concept_layout(
        concept_names, [f"label_{i}" for i in range(num_labels)])
    exclude_keys = {S1_LABEL_CONCEPT_KEY}

    # Dung rule memory CUOI CUNG (da qua prune+dedupe) truc tiep -- day
    # chinh la R duoc deploy, khong can replay lai pre-dedupe state.
    final_memory = ICRLRuleMemory.load(Path(icrl_dir) / "icrl_rule_memory.pt", device=str(device))
    valid_patterns = build_valid_pattern_set(final_memory, concept_keys, concept_offsets,
                                               concept_dims, exclude_keys)

    counts = collect_2x2(test_loader, system1, valid_patterns, concept_keys,
                          concept_offsets, concept_dims, exclude_keys, device)
    n = sum(counts.values())
    result = {"n": n, "num_valid_patterns": len(valid_patterns),
              "counts": {f"{a}/{b}": c for (a, b), c in counts.items()},
              "pct": {f"{a}/{b}": c / n for (a, b), c in counts.items()}}
    print(f"[{name}] n={n} valid_patterns={len(valid_patterns)}")
    for (a, b), c in sorted(counts.items()):
        print(f"  {a:>5s} / {b:<12s} {c:>5d}  ({c/n*100:.1f}%)")
    return result


def run_crlmatched(device, batch_size, num_workers):
    from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders, build_full_concept_layout
    from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY

    cv_root = Path("outputs/crlmatched_cv_gated_confmin03")
    system1_root = Path("outputs/crlmatched_cv_gated")
    data_root = Path("data/fitzpatrick17k_crl_matched_cv_official")
    exclude_keys = {S1_LABEL_CONCEPT_KEY}

    total_counts = defaultdict(int)
    per_fold = {}
    for k in range(5):
        fold_dir = cv_root / f"fold_{k}"
        system1, image_size = load_system1(system1_root / f"fold_{k}" / "system1" / "best_model.pt", device)
        _, val_loader, test_loader = make_loaders(data_root / f"fold_{k}", DEFAULT_IMG_DIR, image_size,
                                                    batch_size, num_workers)
        concept_names = test_loader.dataset.concept_names
        concept_keys, concept_offsets, concept_dims, _ = build_full_concept_layout(
            concept_names, ["label_0", "label_1"])

        final_memory = ICRLRuleMemory.load(fold_dir / "icrl" / "icrl_rule_memory.pt", device=str(device))
        valid_patterns = build_valid_pattern_set(final_memory, concept_keys, concept_offsets,
                                                   concept_dims, exclude_keys)
        counts = collect_2x2(test_loader, system1, valid_patterns, concept_keys,
                              concept_offsets, concept_dims, exclude_keys, device)
        n = sum(counts.values())
        per_fold[f"fold_{k}"] = {"n": n, "num_valid_patterns": len(valid_patterns),
                                   "counts": {f"{a}/{b}": c for (a, b), c in counts.items()}}
        for key, c in counts.items():
            total_counts[key] += c
        print(f"[crlmatched fold_{k}] n={n} valid_patterns={len(valid_patterns)}")

    n = sum(total_counts.values())
    result = {"n": n, "per_fold": per_fold,
              "counts": {f"{a}/{b}": c for (a, b), c in total_counts.items()},
              "pct": {f"{a}/{b}": c / n for (a, b), c in total_counts.items()}}
    print(f"[crlmatched pooled] n={n}")
    for (a, b), c in sorted(total_counts.items()):
        print(f"  {a:>5s} / {b:<12s} {c:>5d}  ({c/n*100:.1f}%)")
    return result


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

    results = {}
    results["crlmatched"] = run_crlmatched(device, args.batch_size, args.num_workers)
    results["pad_ufes20"] = run_single_split(
        "pad_ufes20", "pad_ufes20",
        "data/PAD-UFES-20_prepared", "data/PAD-UFES-20/images",
        "outputs/scope_rebuild/pad_ufes20/system1/best_model.pt",
        "outputs/scope_rebuild/pad_ufes20/icrl",
        6, args.batch_size, args.num_workers, device)
    results["fulldata"] = run_single_split(
        "fulldata", "fitzpatrick",
        "data/fitzpatrick17k_prepared", DEFAULT_IMG_DIR,
        "outputs/scope_rebuild/fulldata/system1/best_model.pt",
        "outputs/scope_rebuild/fulldata/icrl_nolabelslot",
        3, args.batch_size, args.num_workers, device)
    results["concept_only"] = run_single_split(
        "concept_only", "fitzpatrick",
        "data/fitzpatrick17k_prepared_concept_only", DEFAULT_IMG_DIR,
        "outputs/scope_rebuild/concept_only/system1/best_model.pt",
        "outputs/scope_rebuild/concept_only/icrl_nolabelslot",
        3, args.batch_size, args.num_workers, device)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
