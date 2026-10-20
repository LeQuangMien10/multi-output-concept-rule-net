"""
greedy_constrained_match.py - Buoc 2 cua ke hoach "Concept-Rule Balancing"
(ghi chu thay gui, paper/slides_and_notes (1).pdf muc "Ideas 1"): cai dat
thuat toan tim kiem tham lam (greedy local search) de tim concept
assignment C hop le (C |= R) GAN NHAT voi du doan doc lap cua System~1,
thay the cach match hien tai cua System~2 (nearest-centroid tren continuous
similarity -- luon tra ve 1 ket qua bat ke gan hay xa).

Thuat toan (dung nguyen vi spec trong ghi chu, danh cho concept nhi phan
present/absent -- tat ca concept trong du an nay deu nhi phan):
  1. Gan moi concept theo xac suat cao nhat cua chinh no (= dung cach S1
     lam hien tai, hard-decode threshold 0.5).
  2. Kiem tra assignment co nam trong R (tap pattern cua cac rule KHONG
     circular con song sot trong rule memory) hay khong.
  3. Neu co -> dung, tra ve (0 flip, Satisfactory).
  4. Neu khong -> trong so cac concept CHUA flip, chon concept co xac suat
     GAN 0.5 NHAT (= "next highest P(ckj'|x)" trong ghi chu -- doi voi
     concept nhi phan, do chinh la concept re nhat de lat sang gia tri con
     lai) va lat gia tri cua no.
  5. Quay lai buoc 2, lap toi khi tim duoc assignment hop le HOAC vuot qua
     --max_flips (ngan chan no bung to hop khi K lon, vd 48 concept o
     crlmatched) -- vuot qua ma khong tim duoc thi bao Violation.

So sanh voi cach match hien tai (ICRLRuleMemory.match(), continuous nearest-
centroid): (a) ty le anh S1 da hop le ngay tu dau (0 flip) -- trung voi cot
Satisfactory cua Buoc 1; (b) trong so con lai, ty le repair thanh cong
trong ngan sach va so flip trung binh can thiet; (c) neu dung nhan cua rule
tim duoc qua repair lam du doan System~2 moi, so sanh accuracy + Hamming
distance toi pattern GOC cua S1 , doi chieu voi cach match hien tai (de
kiem tra gia thuyet: repair giu GAN pattern goc cua S1 hon, co the giai
thich vi sao concept-fidelity cua S2 hien tai kem hon S1).

Usage (local, inference only, dung checkpoint da co):
    python -m src.scripts.greedy_constrained_match \\
        --max_flips 5 \\
        --output_json outputs/concept_rule_balancing/greedy_repair.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.scripts.measure_contingency_multiscope import decode_pattern_from_vec, DEFAULT_IMG_DIR
from src.scripts.concept_rule_validity_table import build_valid_pattern_set
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.fitzpatrick.system1 import soft_concept_vector


def parse_args():
    p = argparse.ArgumentParser(description="Greedy constrained-repair matching, doi chieu voi match() hien tai.")
    p.add_argument("--max_flips", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_json", type=str, required=True)
    return p.parse_args()


def greedy_repair(cv_row, valid_patterns, keys, concept_offsets, concept_dims,
                   max_flips, threshold=0.5):
    """cv_row: 1-D tensor (concept_dim,) -- concept vector THO cua 1 anh
    (chua gom slot nhan). Tra ve (pattern_cuoi_cung, so_flip, thanh_cong)."""
    present = {}
    cost = {}
    for k in keys:
        s, e = concept_offsets[k], concept_offsets[k] + concept_dims[k]
        prob = cv_row[s:e].item()
        present[k] = prob >= threshold
        cost[k] = abs(prob - threshold)

    def pattern():
        return tuple(k for k in keys if present[k])

    if pattern() in valid_patterns:
        return pattern(), 0, True

    order = sorted(keys, key=lambda k: cost[k])  # re nhat (gan 0.5 nhat) truoc
    n_flips = 0
    for k in order:
        if n_flips >= max_flips:
            break
        present[k] = not present[k]
        n_flips += 1
        if pattern() in valid_patterns:
            return pattern(), n_flips, True
    return pattern(), n_flips, False


@torch.no_grad()
def evaluate_scope(name, module, data_dir, img_dir, system1_ckpt, icrl_dir,
                    num_labels, max_flips, batch_size, num_workers, device):
    if module == "fitzpatrick":
        from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders, build_full_concept_layout
        from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY
    else:
        from src.scripts.pad_ufes20.train_icrl import load_system1, make_loaders, build_full_concept_layout
        from src.utils.pad_ufes20_concepts import S1_LABEL_CONCEPT_KEY

    system1, image_size = load_system1(Path(system1_ckpt), device)
    _, val_loader, test_loader = make_loaders(Path(data_dir), img_dir, image_size, batch_size, num_workers)
    concept_names = test_loader.dataset.concept_names
    concept_keys, concept_offsets, concept_dims, _ = build_full_concept_layout(
        concept_names, [f"label_{i}" for i in range(num_labels)])
    exclude_keys = {S1_LABEL_CONCEPT_KEY}
    keys = [k for k in concept_keys if k not in exclude_keys]

    memory = ICRLRuleMemory.load(Path(icrl_dir) / "icrl_rule_memory.pt", device=str(device))
    head = nn.Linear(memory.concept_dim, num_labels).to(device)
    head.load_state_dict(torch.load(Path(icrl_dir) / "prediction_head.pt",
                                     map_location=device, weights_only=False))
    head.eval()
    centroids = memory.get_centroids().to(device)

    valid_patterns = build_valid_pattern_set(memory, concept_keys, concept_offsets, concept_dims, exclude_keys)
    # pattern -> nhan da so cua rule tuong ung (de tra cuu nhan sau khi repair)
    pattern_to_label = {}
    for r in range(memory.num_rules):
        d = memory.decode_rule(r, concept_keys, concept_offsets, concept_dims)
        pat = tuple(k for k in keys if d["slots"][k]["value"] == "present")
        pattern_to_label[pat] = d["label"]

    n = 0
    n_zero_flip = 0
    n_repaired = 0  # can flip nhung thanh cong trong ngan sach
    n_failed = 0    # vuot ngan sach, khong tim duoc
    total_flips_when_needed = 0
    n_y_correct_current_match = 0
    n_y_correct_repair = 0
    hamming_current_match = 0
    hamming_repair = 0

    for images, labels in test_loader:
        images = images.to(device)
        out = system1(images)
        cv = soft_concept_vector(out)
        y = labels["label"].to(device)
        rule_ids, _ = memory.match(cv)
        s2_pred_current = head(centroids[rule_ids]).argmax(dim=-1)

        for i in range(cv.shape[0]):
            n += 1
            s1_pattern = decode_pattern_from_vec(cv[i], concept_keys, concept_offsets,
                                                  concept_dims, exclude_keys)
            repaired_pattern, n_flips, success = greedy_repair(
                cv[i], valid_patterns, keys, concept_offsets, concept_dims, max_flips)

            if n_flips == 0:
                n_zero_flip += 1
            elif success:
                n_repaired += 1
                total_flips_when_needed += n_flips
            else:
                n_failed += 1

            # Hamming distance cua pattern MATCH hien tai (centroid cua rule
            # da khop) toi pattern GOC cua S1 -- vs Hamming cua pattern sau
            # repair toi pattern goc (theo dinh nghia, <= max_flips).
            matched_rule_decoded = memory.decode_rule(rule_ids[i].item(), concept_keys,
                                                        concept_offsets, concept_dims)
            matched_rule_pattern = tuple(k for k in keys
                                          if matched_rule_decoded["slots"][k]["value"] == "present")
            hamming_current_match += len(set(s1_pattern) ^ set(matched_rule_pattern))
            hamming_repair += len(set(s1_pattern) ^ set(repaired_pattern))

            if int(s2_pred_current[i].item()) == int(y[i].item()):
                n_y_correct_current_match += 1

            if repaired_pattern in pattern_to_label:
                repair_label = pattern_to_label[repaired_pattern]
            else:
                repair_label = int(s2_pred_current[i].item())  # fallback: giu cach match hien tai
            if repair_label == int(y[i].item()):
                n_y_correct_repair += 1

    result = {
        "n": n,
        "max_flips": max_flips,
        "num_valid_patterns": len(valid_patterns),
        "pct_zero_flip_already_valid": n_zero_flip / n,
        "pct_repaired_within_budget": n_repaired / n,
        "pct_failed_exceed_budget": n_failed / n,
        "avg_flips_when_repair_needed": total_flips_when_needed / max(1, n_repaired),
        "avg_hamming_current_match_to_s1": hamming_current_match / n,
        "avg_hamming_repair_to_s1": hamming_repair / n,
        "s2_acc_current_match": n_y_correct_current_match / n,
        "s2_acc_after_repair": n_y_correct_repair / n,
    }
    print(f"[{name}] n={n}  0-flip(Satisfactory)={result['pct_zero_flip_already_valid']*100:.1f}%  "
          f"repaired={result['pct_repaired_within_budget']*100:.1f}%  "
          f"failed={result['pct_failed_exceed_budget']*100:.1f}%  "
          f"avg_flips={result['avg_flips_when_repair_needed']:.2f}")
    print(f"  Hamming toi pattern goc S1: match_hien_tai={result['avg_hamming_current_match_to_s1']:.3f}  "
          f"repair={result['avg_hamming_repair_to_s1']:.3f}")
    print(f"  S2 acc: match_hien_tai={result['s2_acc_current_match']*100:.2f}%  "
          f"sau_repair={result['s2_acc_after_repair']*100:.2f}%")
    return result


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

    results = {}
    results["pad_ufes20"] = evaluate_scope(
        "pad_ufes20", "pad_ufes20",
        "data/PAD-UFES-20_prepared", "data/PAD-UFES-20/images",
        "outputs/scope_rebuild/pad_ufes20/system1/best_model.pt",
        "outputs/scope_rebuild/pad_ufes20/icrl",
        6, args.max_flips, args.batch_size, args.num_workers, device)
    results["fulldata"] = evaluate_scope(
        "fulldata", "fitzpatrick",
        "data/fitzpatrick17k_prepared", DEFAULT_IMG_DIR,
        "outputs/scope_rebuild/fulldata/system1/best_model.pt",
        "outputs/scope_rebuild/fulldata/icrl_nolabelslot",
        3, args.max_flips, args.batch_size, args.num_workers, device)
    results["concept_only"] = evaluate_scope(
        "concept_only", "fitzpatrick",
        "data/fitzpatrick17k_prepared_concept_only", DEFAULT_IMG_DIR,
        "outputs/scope_rebuild/concept_only/system1/best_model.pt",
        "outputs/scope_rebuild/concept_only/icrl_nolabelslot",
        3, args.max_flips, args.batch_size, args.num_workers, device)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
