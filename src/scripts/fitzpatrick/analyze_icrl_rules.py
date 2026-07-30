"""
analyze_icrl_rules.py - Kiem tra chat luong rule ICRL cho Fitzpatrick17k
============================================================================

Cung phuong phap da dung voi MultiConcept: KHONG tin so lieu tinh san trong
icrl_rules.json, chay lai inference THAT tren tap du lieu de:

  1. Do LIVE PURITY tung rule (trong so anh that su roi vao rule do, bao
     nhieu % dung nhan da luu). Ly do phai do lai: field "accuracy" trong
     icrl_rules.json luon = 0.5000 cho CA 61 rule -- kiem tra code cho thay
     ICRLRuleMemory.update_accuracy() (ham duy nhat cap nhat _correct/
     _total_pred) KHONG BAO GIO duoc goi trong train_icrl.py (ca ban
     MultiConcept lan Fitzpatrick) -- day la 1 lo hong that trong pipeline,
     khong phai loi rieng cua Fitzpatrick. Confidence xuat ra = coherence*0.5
     voi moi rule, khong mang thong tin gi ngoai coherence. Script nay dung
     LIVE PURITY (do truc tiep) thay the, khong phu thuoc field bi loi do.

  2. Contradiction theo pattern: nhom rule theo dung present_concepts (bo qua
     s1_label_pred slot), kiem tra bao nhieu pattern bi gan nhieu nhan khac
     nhau -- dung cau hoi da hoi voi MultiConcept ("rule 0,5,6 ket luan even
     co hop ly khong?").

Usage:
    python -m src.scripts.fitzpatrick.analyze_icrl_rules \\
        --data_dir data/fitzpatrick17k_prepared \\
        --img_dir data/fitzpatrick17k/data/finalfitz17k \\
        --system1_ckpt outputs/fitzpatrick_system1/best_model.pt \\
        --icrl_dir outputs/fitzpatrick_icrl \\
        --split val
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.fitzpatrick.fitzpatrick_dataset import FitzpatrickDataset, build_transforms
from src.models.fitzpatrick.system1 import FitzpatrickSystem1, soft_concept_vector, hard_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.utils.fitzpatrick_concepts import LABEL_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Live rule-quality check for Fitzpatrick ICRL.")
    p.add_argument("--data_dir", type=str, default="data/fitzpatrick17k_prepared")
    p.add_argument("--img_dir", type=str, default="data/fitzpatrick17k/data/finalfitz17k")
    p.add_argument("--system1_ckpt", type=str, default="outputs/fitzpatrick_system1/best_model.pt")
    p.add_argument("--icrl_dir", type=str, default="outputs/fitzpatrick_icrl")
    p.add_argument("--output_path", type=str, default="outputs/fitzpatrick_audit/icrl_dashboard.png")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--use_hard_cv", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def run_live_stats(system1, memory, loader, device, use_hard=False):
    centroids = memory.get_centroids().to(device)
    stored_labels = memory.get_labels()

    per_rule_true_labels = defaultdict(list)   # rule_id -> [true_label_idx, ...]

    for images, labels in tqdm(loader, desc="Live infer"):
        images = images.to(device)
        s1_out = system1(images)
        cv = hard_concept_vector(s1_out) if use_hard else soft_concept_vector(s1_out)
        y = labels["label"].to(device)

        rule_ids, sims = memory.match(cv)
        for rid, true_y in zip(rule_ids.tolist(), y.tolist()):
            per_rule_true_labels[rid].append(true_y)

    rule_stats = []
    for rid in range(memory.num_rules):
        true_ys = per_rule_true_labels.get(rid, [])
        n_live = len(true_ys)
        if n_live == 0:
            purity = None
            majority_live = None
        else:
            counts = Counter(true_ys)
            majority_live, majority_count = counts.most_common(1)[0]
            purity = majority_count / n_live
        rule_stats.append({
            "rule_id": rid,
            "stored_label": stored_labels[rid],
            "n_live": n_live,
            "live_majority_label": majority_live,
            "live_purity": purity,
            "stored_matches_live_majority": (majority_live == stored_labels[rid]) if n_live > 0 else None,
        })
    return rule_stats


def pattern_contradiction_stats(rules_json: list[dict]) -> dict:
    by_pattern = defaultdict(list)
    for r in rules_json:
        pattern = frozenset(r["present_concepts"])
        by_pattern[pattern].append(r)

    total_mass = sum(r["n"] for r in rules_json)
    contra_patterns = []
    for pattern, rs in by_pattern.items():
        labels = set(r["label_name"] for r in rs)
        mass = sum(r["n"] for r in rs)
        if len(labels) > 1:
            contra_patterns.append({
                "pattern": sorted(pattern) or ["(none)"],
                "n_rules": len(rs),
                "mass": mass,
                "labels": sorted(labels),
            })
    contra_patterns.sort(key=lambda x: -x["mass"])
    contra_mass = sum(p["mass"] for p in contra_patterns)

    return {
        "n_distinct_patterns": len(by_pattern),
        "n_contradictory_patterns": len(contra_patterns),
        "total_mass": total_mass,
        "contradictory_mass": contra_mass,
        "contradictory_mass_pct": round(contra_mass / total_mass * 100, 1) if total_mass else 0,
        "top_contradictory_patterns": contra_patterns[:10],
    }


def plot_dashboard(rule_stats, rules_json, contra_stats, split_name, save_path):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Fitzpatrick ICRL - Kiem tra chat luong rule (live, split '{split_name}')",
                 fontsize=14, y=0.98)
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.28, top=0.90, bottom=0.08, left=0.08, right=0.97)

    live_stats = [r for r in rule_stats if r["n_live"] > 0]
    purities = [r["live_purity"] for r in live_stats]
    sizes = [r["n_live"] for r in live_stats]

    # (a) live purity distribution
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(purities, bins=np.linspace(0, 1, 21), color="#2980b9")
    mean_p = np.mean(purities) if purities else 0
    ax.axvline(mean_p, color="#c0392b", linestyle="--", label=f"mean={mean_p:.3f}")
    ax.set_xlabel("Live purity (ti le anh dung nhan da luu / rule)")
    ax.set_ylabel("So rule")
    ax.set_title(f"Phan bo purity tren {len(live_stats)} rule co anh trong split nay")
    ax.legend(fontsize=9)

    # (b) purity vs size (weighted view -- big rules matter more)
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(sizes, purities, alpha=0.6, color="#27ae60")
    ax.set_xlabel(f"So anh {split_name} roi vao rule (live)")
    ax.set_ylabel("Live purity")
    ax.set_xscale("log")
    ax.set_title("Purity co giam o rule lon khong?")

    # (c) contradiction breakdown (top patterns by mass in TRAIN, xem docstring)
    ax = fig.add_subplot(gs[1, 0])
    top = contra_stats["top_contradictory_patterns"][:8]
    names = ["+".join(p["pattern"])[:25] for p in top]
    masses = [p["mass"] for p in top]
    ax.barh(range(len(names)), masses, color="#c0392b")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("So anh (train, cong don 3 epoch build)")
    ax.set_title(f"Top pattern co nhieu nhan khac nhau ({contra_stats['contradictory_mass_pct']}% "
                 f"tong khoi luong mau la contradictory)")

    # (d) summary text
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    n_rules_no_data = sum(1 for r in rule_stats if r["n_live"] == 0)
    n_mismatch = sum(1 for r in rule_stats if r["stored_matches_live_majority"] is False)
    txt = (
        f"Tong so rule: {len(rule_stats)}\n"
        f"Rule khong co anh nao trong split '{split_name}': {n_rules_no_data}\n"
        f"Rule co live-majority-label KHAC voi nhan da luu: {n_mismatch}\n\n"
        f"Live purity trung binh (weighted theo size): "
        f"{np.average(purities, weights=sizes) if purities else 0:.3f}\n"
        f"Live purity trung binh (khong weight): {mean_p:.3f}\n\n"
        f"Pattern (theo present_concepts) trong dataset train:\n"
        f"  {contra_stats['n_distinct_patterns']} pattern rieng biet\n"
        f"  {contra_stats['n_contradictory_patterns']} pattern bi CONTRADICT "
        f"(cung pattern, khac nhan)\n"
        f"  {contra_stats['contradictory_mass_pct']}% khoi luong mau roi vao pattern contradictory\n\n"
        f"LUU Y: field 'accuracy' trong icrl_rules.json luon = 0.5 (bug -- "
        f"update_accuracy() khong duoc goi trong train_icrl.py). Dung live "
        f"purity o day thay the, dang tin cay hon."
    )
    ax.text(0.0, 1.0, txt, transform=ax.transAxes, fontsize=9.5, va="top", family="monospace")

    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.system1_ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    system1 = FitzpatrickSystem1(
        backbone_name=ckpt_args["backbone"], pretrained=False,
        num_concepts=ckpt_args["num_concepts"], num_labels=ckpt_args["num_labels"],
    ).to(device)
    system1.load_state_dict(ckpt["model_state_dict"])
    system1.eval()
    for p in system1.parameters():
        p.requires_grad_(False)

    memory = ICRLRuleMemory.load(Path(args.icrl_dir) / "icrl_rule_memory.pt", device=str(device))
    print(f"[INFO] Loaded rule memory: {memory.num_rules} rules")

    dataset = FitzpatrickDataset(
        Path(args.data_dir) / f"{args.split}.csv", args.img_dir,
        build_transforms("val", ckpt_args.get("image_size", 224)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"[INFO] {len(dataset)} images in '{args.split}' split.")

    rule_stats = run_live_stats(system1, memory, loader, device, args.use_hard_cv)

    with open(Path(args.icrl_dir) / "icrl_rules.json", encoding="utf-8") as f:
        rules_json = json.load(f)
    contra_stats = pattern_contradiction_stats(rules_json)

    live_stats = [r for r in rule_stats if r["n_live"] > 0]
    purities = [r["live_purity"] for r in live_stats]
    sizes = [r["n_live"] for r in live_stats]
    n_mismatch = sum(1 for r in rule_stats if r["stored_matches_live_majority"] is False)

    print(f"\n[SUMMARY] Live purity (unweighted mean): {np.mean(purities):.3f}" if purities else "[SUMMARY] no live data")
    print(f"[SUMMARY] Live purity (size-weighted mean): {np.average(purities, weights=sizes):.3f}" if purities else "")
    print(f"[SUMMARY] Rules where live majority label != stored label: {n_mismatch}/{len(rule_stats)}")
    print(f"[SUMMARY] Contradictory patterns: {contra_stats['n_contradictory_patterns']}/{contra_stats['n_distinct_patterns']} "
          f"({contra_stats['contradictory_mass_pct']}% of train sample-mass)")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    plot_dashboard(rule_stats, rules_json, contra_stats, args.split, args.output_path)

    result = {
        "split": args.split,
        "rule_stats": rule_stats,
        "contradiction_stats": contra_stats,
        "live_purity_unweighted_mean": float(np.mean(purities)) if purities else None,
        "live_purity_weighted_mean": float(np.average(purities, weights=sizes)) if purities else None,
    }
    result_path = Path(args.output_path).with_suffix(".json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[DONE] Saved {args.output_path} and {result_path}")


if __name__ == "__main__":
    main()
