"""
train_icrl.py - ICRL Stage 2/3 cho PAD-UFES-20
=====================================================

Đa số hàm trong file này (build_rule_memory, train_head, evaluate, export_rules,
run_stage3_onward, ...) đã dataset-agnostic sẵn trong bản Fitzpatrick17k gốc
(src/scripts/fitzpatrick/train_icrl.py) — concept/label schema được suy ra tại
runtime từ train_loader.dataset.concept_names + --label_names, không hardcode
số lượng. Bản này chỉ khác 3 chỗ: import PadUfes20Dataset thay FitzpatrickDataset
(FitzpatrickSystem1 và build_transforms dùng lại nguyên xi, không đổi), import
pad_ufes20_concepts thay fitzpatrick_concepts, và default CLI path.

Khác biệt cần lưu ý khi chạy (không phải bug, là đặc điểm dataset mới):
  - --theta mặc định "auto" (không phải 1 hằng số đã đo trước như 0.886 của
    Fitzpatrick) — hằng số đó đo trên concept vector GROUND-TRUTH 35-chiều
    của Fitzpatrick, không áp dụng được cho 6 concept của PAD-UFES-20.
  - --n_min mặc định giữ 15 (giống Fitzpatrick) NHƯNG train split ở đây chỉ
    ~1,606 ảnh (gần quy mô concept-only 3,227 hơn là full-data 16,577 nơi
    giá trị 15 được calibrate) — nên dùng --n_min_sweep để tìm giá trị phù
    hợp thay vì tin thẳng default, đúng cách đã làm khi chuyển sang các scope
    nhỏ hơn của Fitzpatrick (CRL-matched/concept-only).
  - 6 concept ở đây là triệu chứng bệnh nhân tự khai, không phải concept hình
    thái nhìn từ ảnh — nếu concept vector S1 dự đoán kém tin cậy hơn hẳn
    Fitzpatrick, cluster/rule ở Stage 2 cũng sẽ phản ánh đúng điều đó (xem
    ghi chú dự án trước khi triển khai Part 3).

Usage:
    python -m src.scripts.pad_ufes20.train_icrl \\
        --data_dir data/PAD-UFES-20_prepared --img_dir data/PAD-UFES-20/images \\
        --system1_ckpt outputs/pad_ufes20_system1/best_model.pt \\
        --output_dir outputs/pad_ufes20_icrl \\
        --theta auto --n_min_sweep 5,10,15,20,30 \\
        --epochs 3 --head_epochs 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.fitzpatrick.fitzpatrick_dataset import build_transforms
from src.datasets.pad_ufes20.pad_ufes20_dataset import PadUfes20Dataset
from src.models.fitzpatrick.system1 import FitzpatrickSystem1, soft_concept_vector, hard_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.scripts.fitzpatrick.train_icrl_gt_ablation import measure_theta
from src.utils.seed import set_seed
from src.utils.pad_ufes20_concepts import (
    LABEL_NAMES as DEFAULT_LABEL_NAMES, S1_LABEL_CONCEPT_KEY,
)


def build_full_concept_layout(concept_names: list[str], label_names: list[str]):
    """FULL concept vector layout (concepts + s1_label_pred slot), derived at
    runtime from whatever concept_names/label_names this run actually uses."""
    full_keys = list(concept_names) + [S1_LABEL_CONCEPT_KEY]
    dims = {name: 1 for name in concept_names}
    dims[S1_LABEL_CONCEPT_KEY] = len(label_names)
    offsets, off = {}, 0
    for name in full_keys:
        offsets[name] = off
        off += dims[name]
    return full_keys, offsets, dims, off


# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build ICRL rule memory cho PAD-UFES-20.")

    p.add_argument("--data_dir", type=str, default="data/PAD-UFES-20_prepared")
    p.add_argument("--img_dir", type=str, default="data/PAD-UFES-20/images")
    p.add_argument("--system1_ckpt", type=str, default="outputs/pad_ufes20_system1/best_model.pt")
    p.add_argument("--output_dir", type=str, default="outputs/pad_ufes20_icrl")

    p.add_argument("--theta", type=str, default="auto",
                    help="Float, hoac 'auto' (mac dinh) de do truoc Stage 2 tren dung concept "
                         "vector S1 THAT du doan cho dataset nay -- khong co hang so tien-do "
                         "san nhu Fitzpatrick (0.886 do rieng tren ground-truth 35-concept cua "
                         "dataset do, khong ap dung duoc o day).")
    p.add_argument("--theta_percentile", type=float, default=95.0,
                    help="Percentile dung khi --theta auto (measure_theta). Mac dinh 95, ke thua "
                         "tu Fitzpatrick, CHUA re-validate rieng cho scope 6-concept/6-lop nay -- "
                         "thu percentile khac (vd 90, 99) neu nghi ngo theta hien tai qua cao/thap "
                         "khien rule memory khong phu du 6 lop (xem outputs/pad_ufes20_icrl: chi "
                         "3/6 lop co rule).")
    p.add_argument("--theta_merge", type=float, default=0.93)
    p.add_argument("--n_min", type=int, default=15,
                    help="Gia tri copy tu Fitzpatrick full-data (16,577 anh) -- CHUA calibrate "
                         "rieng cho quy mo ~1,606 anh train cua PAD-UFES-20. Dung --n_min_sweep "
                         "de tim gia tri phu hop thay vi tin thang default nay.")
    p.add_argument("--n_min_sweep", type=str, default=None,
                    help="Danh sach n_min cach nhau boi dau phay, vd '5,10,15,20,30'. Neu duoc set, "
                         "Stage 2 chi build MOT LAN, sau do moi gia tri duoc prune+train head+danh "
                         "gia RIENG (khong build lai). Ket qua tung gia tri luu vao "
                         "output_dir/n_min_<N>/, kem 1 de xuat tu dong copy len output_dir/.")
    p.add_argument("--conf_min", type=float, default=0.5)

    p.add_argument("--epochs", type=int, default=3,
                    help="So lan pass qua training set de build rule memory (Stage 2).")
    p.add_argument("--use_hard_cv", action="store_true",
                    help="Dung hard (nhi phan, threshold 0.5) concept vector thay vi soft "
                         "(sigmoid/softmax probs) cho MATCH/CREATE/MERGE.")
    p.add_argument("--exclude_label_slot", action="store_true",
                    help="Bo s1_label_pred khoi vector dung de MATCH/CREATE/MERGE "
                         "(cluster_dims=(0,NUM_CONCEPTS) thay vi None).")

    p.add_argument("--head_epochs", type=int, default=20)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--head_steps_per_epoch", type=int, default=250)
    p.add_argument("--label_names", type=str, default=",".join(DEFAULT_LABEL_NAMES),
                    help="Comma-separated class names, in label_idx order.")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────

def load_system1(ckpt_path: Path, device: torch.device) -> FitzpatrickSystem1:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    model = FitzpatrickSystem1(
        backbone_name=saved_args.get("backbone", "resnet50"),
        pretrained=False,   # trong so se duoc load tu checkpoint ngay ben duoi
        num_concepts=saved_args.get("num_concepts"),
        num_labels=saved_args.get("num_labels", len(DEFAULT_LABEL_NAMES)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, saved_args.get("image_size", 224)


def make_loaders(data_dir: Path, img_dir: Path, image_size: int, batch_size: int, num_workers: int):
    # "val" transform (khong augment) cho CA train/val/test -- xem docstring
    # tuong ung trong ban Fitzpatrick: augmentation ngau nhien chi hop ly khi
    # TRAIN S1, khong hop ly khi trich concept vector de cluster.
    clean_transform = build_transforms("val", image_size)

    def _loader(split, shuffle):
        ds = PadUfes20Dataset(data_dir / f"{split}.csv", img_dir, clean_transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=num_workers, pin_memory=True)

    return _loader("train", True), _loader("val", False), _loader("test", False)


@torch.no_grad()
def collect_concept_vectors(system1, loader, device, use_hard=False, cluster_dims=None) -> torch.Tensor:
    vecs = []
    for images, _ in tqdm(loader, desc="  Collecting concept vectors (for --theta auto)", leave=False):
        images = images.to(device)
        s1_out = system1(images)
        cv = hard_concept_vector(s1_out) if use_hard else soft_concept_vector(s1_out)
        if cluster_dims is not None:
            cv = cv[:, cluster_dims[0]:cluster_dims[1]]
        vecs.append(cv.cpu())
    return torch.cat(vecs, dim=0)


# ─────────────────────────────────────────────────────────────
# Stage 2: Build rule memory
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def build_rule_memory(system1, loader, memory, device, use_hard=False, epoch_label="Epoch"):
    total_stats = {"created": 0, "matched": 0, "total": 0}

    for images, labels in tqdm(loader, desc=f"  Build [{epoch_label}]", leave=False):
        images = images.to(device)
        s1_out = system1(images)

        cv = hard_concept_vector(s1_out) if use_hard else soft_concept_vector(s1_out)

        concept_probs = torch.sigmoid(s1_out["concepts"])
        concept_conf = (2.0 * (concept_probs - 0.5).abs()).mean(dim=1)
        label_conf = F.softmax(s1_out["label"], dim=-1).max(dim=1).values
        s1_conf = (concept_conf + label_conf) / 2.0

        y = labels["label"].to(device)

        stats = memory.process_batch(cv, y, s1_conf)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    return total_stats


# ─────────────────────────────────────────────────────────────
# Stage 3: Train prediction head truc tiep tren R rule centroids
# ─────────────────────────────────────────────────────────────

def train_head(system1, memory, val_loader, num_classes, epochs, lr, device,
                use_hard=False, steps_per_epoch=250):
    head = nn.Linear(memory.concept_dim, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)

    best_val = 0.0
    best_state = None

    centroids = memory.get_centroids().to(device)
    rule_labels = torch.tensor(memory.get_labels(), dtype=torch.long, device=device)

    print(f"\n[Stage 3] Train prediction head tren {memory.num_rules} rule centroids "
          f"({epochs} epochs x {steps_per_epoch} steps)")

    for epoch in range(1, epochs + 1):
        head.train()
        for _ in range(steps_per_epoch):
            logits = head(centroids)
            loss = F.cross_entropy(logits, rule_labels)
            opt.zero_grad(); loss.backward(); opt.step()
        train_acc = (logits.argmax(dim=1) == rule_labels).float().mean().item()

        head.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                s1_out = system1(images)
                cv = hard_concept_vector(s1_out) if use_hard else soft_concept_vector(s1_out)
                y = labels["label"].to(device)
                rule_ids, _ = memory.match(cv)
                rule_cvs = centroids[rule_ids]
                preds = head(rule_cvs).argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += len(y)

        val_acc = val_correct / val_total
        print(f"  Ep {epoch:2d}/{epochs}: rule_train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    if best_state:
        head.load_state_dict(best_state)
    print(f"  Best val_acc = {best_val:.4f}")
    return head


@torch.no_grad()
def record_rule_accuracy(system1, head, memory, loader, device, use_hard=False):
    centroids = memory.get_centroids().to(device)
    for images, labels in tqdm(loader, desc="  Record rule accuracy", leave=False):
        images = images.to(device)
        s1_out = system1(images)
        cv = hard_concept_vector(s1_out) if use_hard else soft_concept_vector(s1_out)
        y = labels["label"].to(device)
        rule_ids, _ = memory.match(cv)
        rule_cvs = centroids[rule_ids]
        preds = head(rule_cvs).argmax(dim=1)
        memory.update_accuracy(cv, y, preds)


# ─────────────────────────────────────────────────────────────
# Evaluate & export
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(system1, head, loader, device, memory, split="test", use_hard=False):
    head.eval()
    correct, total = 0, 0
    centroids = memory.get_centroids().to(device)

    for images, labels in tqdm(loader, desc=f"  Eval {split}", leave=False):
        images = images.to(device)
        s1_out = system1(images)
        cv = hard_concept_vector(s1_out) if use_hard else soft_concept_vector(s1_out)
        y = labels["label"].to(device)
        rule_ids, _ = memory.match(cv)
        rule_cvs = centroids[rule_ids]
        preds = head(rule_cvs).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)

    return {"accuracy": correct / total, "correct": correct, "total": total}


def summarize_rules(rules_data: list[dict], concept_names: list[str], label_names: list[str]) -> dict:
    entries = []
    pattern_groups: dict[tuple, list[int]] = {}
    for r in rules_data:
        present = r["present_concepts"]
        rule_string = " AND ".join(present) if present else "(no concept -- default/catch-all rule)"
        pattern_groups.setdefault(tuple(present), []).append(len(entries))
        entries.append({
            "rule_id": r["rule_id"],
            "label": r["label_name"],
            "rule": rule_string,
            "circular": False,  # filled below
            "_confidence": r["confidence"],  # sort key only, stripped before output
        })

    n_circular = 0
    for idx_list in pattern_groups.values():
        if len(idx_list) < 2:
            continue
        labels_here = set(entries[i]["label"] for i in idx_list)
        if len(labels_here) > 1:
            for i in idx_list:
                entries[i]["circular"] = True
            n_circular += len(idx_list)

    entries.sort(key=lambda x: -x["_confidence"])
    for e in entries:
        del e["_confidence"]

    return {
        "num_rules": len(rules_data),
        "num_effective_rules": len(rules_data) - n_circular,
        "circular_rate": round(n_circular / len(rules_data), 4) if rules_data else 0.0,
        "rules": entries,
    }


def export_rules(memory, output_dir, full_concept_keys, full_concept_offsets,
                  full_concept_dims, label_names, n_show=20):
    num_labels = len(label_names)
    rules_data = []
    for r in range(memory.num_rules):
        decoded = memory.decode_rule(
            rule_id=r,
            concept_keys=full_concept_keys,
            concept_offsets=full_concept_offsets,
            concept_dims=full_concept_dims,
            id_to_symbol=None,
        )
        present = [k for k, v in decoded["slots"].items()
                   if k != S1_LABEL_CONCEPT_KEY and v["value"] == "present"]

        s1_label_idx = int(decoded["slots"][S1_LABEL_CONCEPT_KEY]["value"])
        s1_label_conf = decoded["slots"][S1_LABEL_CONCEPT_KEY]["confidence"]

        decoded["label_name"] = label_names[decoded["label"]] if 0 <= decoded["label"] < num_labels else "?"
        decoded["s1_label_guess_name"] = label_names[s1_label_idx]
        decoded["s1_label_guess_confidence"] = s1_label_conf
        decoded["s1_label_agrees_with_truth"] = (label_names[s1_label_idx] == decoded["label_name"])
        decoded["present_concepts"] = present
        rules_data.append(decoded)

    rules_data.sort(key=lambda x: -x["confidence"])

    json_path = output_dir / "icrl_rules.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rules_data, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] {len(rules_data)} rules exported to {json_path}")

    concept_names = [k for k in full_concept_keys if k != S1_LABEL_CONCEPT_KEY]
    summary = summarize_rules(rules_data, concept_names, label_names)
    summary_path = output_dir / "icrl_rules_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Compact summary ({summary['num_effective_rules']}/{summary['num_rules']} rules "
          f"non-circular, circular_rate={summary['circular_rate']:.2%}) exported to {summary_path}")

    n_agree = sum(1 for r in rules_data if r["s1_label_agrees_with_truth"])
    print(f"[INFO] S1's own label guess agrees with ground-truth majority label: "
          f"{n_agree}/{len(rules_data)} rules")

    print(f"\n[INFO] Top {min(n_show, len(rules_data))} rules (sorted by confidence):")
    for r in rules_data[:n_show]:
        bar = "#" * int(r["confidence"] * 20)
        concepts_str = "+".join(r["present_concepts"]) or "(none)"
        agree = "=" if r["s1_label_agrees_with_truth"] else "!="
        print(f"  [{r['confidence']:.3f}] {r['label_name']:15s} (S1 {agree} {r['s1_label_guess_name']:15s})  "
              f"n={r['n']:4d}  coh={r['coherence']:.3f}  {concepts_str[:50]:50s}  {bar}")

    return summary


# ─────────────────────────────────────────────────────────────
# Stage 3 onward: head train -> record accuracy -> final prune -> eval -> export
# ─────────────────────────────────────────────────────────────

def _collapsed_result(memory, output_dir, full_concept_keys, full_concept_offsets,
                       full_concept_dims, label_names, args) -> dict:
    memory.save(output_dir / "icrl_rule_memory.pt")
    rule_summary = export_rules(memory, output_dir, full_concept_keys, full_concept_offsets,
                                 full_concept_dims, label_names)
    result = {
        "n_min": memory.n_min,
        "val_accuracy": 0.0,
        "test_accuracy": 0.0,
        "num_rules": 0,
        "num_effective_rules": rule_summary["num_effective_rules"],
        "circular_rate": rule_summary["circular_rate"],
        "rule_confidence_stats": {"mean": 0.0, "min": 0.0, "max": 0.0},
        "collapsed": True,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({**result, "args": vars(args)}, f, indent=2)
    return result


def run_stage3_onward(memory, system1, val_loader, test_loader, label_names,
                       full_concept_keys, full_concept_offsets, full_concept_dims,
                       output_dir: Path, args, conf_min: float | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    conf_min = args.conf_min if conf_min is None else conf_min
    device = torch.device(memory.device)

    if memory.num_rules == 0:
        print(f"\n[WARN] Rule memory is empty going into Stage 3 (n_min={memory.n_min} pruned "
              f"away every rule) -- skipping head training/eval, recording as a collapsed candidate.")
        return _collapsed_result(memory, output_dir, full_concept_keys, full_concept_offsets,
                                  full_concept_dims, label_names, args)

    head = train_head(
        system1, memory, val_loader,
        num_classes=len(label_names),
        epochs=args.head_epochs,
        lr=args.head_lr,
        device=device,
        use_hard=args.use_hard_cv,
        steps_per_epoch=args.head_steps_per_epoch,
    )
    torch.save(head.state_dict(), output_dir / "prediction_head.pt")

    print(f"\n[Stage 3.5] Recording rule accuracy on val split (n_min={memory.n_min})")
    record_rule_accuracy(system1, head, memory, val_loader, device, args.use_hard_cv)

    print(f"\n[INFO] Final prune using real accuracy signal (conf_min={conf_min})")
    memory.conf_min = conf_min
    memory.prune(verbose=True)
    print(f"  After final prune: {memory.num_rules} rules")

    print(f"\n[INFO] Dedupe by decoded display pattern (drop circular, merge exact duplicates)")
    dedupe_stats = memory.dedupe_by_decoded_pattern(
        full_concept_keys, full_concept_offsets, full_concept_dims,
        exclude_keys={S1_LABEL_CONCEPT_KEY}, verbose=True,
    )
    print(f"  After dedupe: {memory.num_rules} rules")

    memory_path = output_dir / "icrl_rule_memory.pt"
    memory.save(memory_path)

    if memory.num_rules == 0:
        print(f"\n[WARN] Final accuracy-based prune (conf_min={conf_min}) removed every remaining "
              f"rule -- skipping evaluation, recording as a collapsed candidate.")
        return _collapsed_result(memory, output_dir, full_concept_keys, full_concept_offsets,
                                  full_concept_dims, label_names, args)

    print("\n[INFO] Evaluating...")
    test_metrics = evaluate(system1, head, test_loader, device, memory, "test", args.use_hard_cv)
    val_metrics = evaluate(system1, head, val_loader, device, memory, "val", args.use_hard_cv)
    print(f"  val_accuracy  = {val_metrics['accuracy']:.4f}")
    print(f"  test_accuracy = {test_metrics['accuracy']:.4f}")

    rule_summary = export_rules(memory, output_dir, full_concept_keys, full_concept_offsets,
                                 full_concept_dims, label_names)

    result = {
        "n_min": memory.n_min,
        "val_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "num_rules": memory.num_rules,
        "num_effective_rules": rule_summary["num_effective_rules"],
        "circular_rate": rule_summary["circular_rate"],
        "dedupe_stats": dedupe_stats,
        "rule_confidence_stats": {
            "mean": sum(memory.get_confidences()) / max(1, memory.num_rules),
            "min": min(memory.get_confidences()) if memory.num_rules else 0,
            "max": max(memory.get_confidences()) if memory.num_rules else 0,
        },
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({**result, "args": vars(args)}, f, indent=2)
    return result


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" \
             else torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_values = [int(v) for v in args.n_min_sweep.split(",")] if args.n_min_sweep else None

    print(f"[INFO] Device: {device}")
    theta_merge_display = "auto (recomputed after theta measurement)" if args.theta == "auto" else args.theta_merge
    print(f"[INFO] ICRL params: theta={args.theta}  theta_merge={theta_merge_display}  "
          f"n_min={args.n_min if sweep_values is None else sweep_values}  conf_min={args.conf_min}")

    system1, image_size = load_system1(Path(args.system1_ckpt), device)
    print(f"[INFO] System1 loaded (frozen): {args.system1_ckpt}  image_size={image_size}")

    train_loader, val_loader, test_loader = make_loaders(
        Path(args.data_dir), args.img_dir, image_size, args.batch_size, args.num_workers,
    )
    print(f"[INFO] Data: {args.data_dir}")

    label_names = args.label_names.split(",")
    concept_names = train_loader.dataset.concept_names
    num_concepts = len(concept_names)
    full_concept_keys, full_concept_offsets, full_concept_dims, full_cv_dim = \
        build_full_concept_layout(concept_names, label_names)
    print(f"[INFO] {num_concepts} concepts, {len(label_names)} classes ({label_names}), "
          f"full_cv_dim={full_cv_dim}")

    cluster_dims = (0, num_concepts) if args.exclude_label_slot else None
    print(f"[INFO] cluster_dims={cluster_dims} (exclude_label_slot={args.exclude_label_slot})")

    if args.theta == "auto":
        print("\n[INFO] --theta auto: measuring theta on this run's actual S1-predicted "
              "concept vectors (train split, respecting cluster_dims/--use_hard_cv exactly "
              "as configured for this run)...")
        train_cv = collect_concept_vectors(system1, train_loader, device, args.use_hard_cv, cluster_dims)
        theta = measure_theta(train_cv, percentile=args.theta_percentile)
        theta_merge = min(theta + 0.04, 0.999)
        print(f"[INFO] Measured theta={theta:.4f}  theta_merge={theta_merge:.4f} (theta+0.04)")
    else:
        theta = float(args.theta)
        theta_merge = args.theta_merge

    build_n_min = 1 if sweep_values is not None else args.n_min

    memory = ICRLRuleMemory(
        concept_dim=full_cv_dim,
        theta=theta,
        theta_merge=theta_merge,
        n_min=build_n_min,
        conf_min=args.conf_min,
        cluster_dims=cluster_dims,
        device=str(device),
    )

    print(f"\n[Stage 2] Building rule memory ({args.epochs} epochs)")
    for epoch in range(1, args.epochs + 1):
        print(f"\n  Epoch {epoch}/{args.epochs}")
        stats = build_rule_memory(
            system1, train_loader, memory, device,
            use_hard=args.use_hard_cv, epoch_label=f"{epoch}/{args.epochs}",
        )
        print(f"  Created={stats['created']}  Matched={stats['matched']}  "
              f"Rules so far={memory.num_rules}")

        memory.prune(verbose=True, conf_min_override=0.0)
        print(f"  After prune: {memory.num_rules} rules")
        cohs = [memory._compute_coherence(i) for i in range(memory.num_rules)]
        print(f"  Coherence: mean={sum(cohs)/max(1,len(cohs)):.3f}  "
              f"min={min(cohs) if cohs else 0:.3f}  max={max(cohs) if cohs else 0:.3f}")

    base_memory_path = output_dir / ("icrl_rule_memory_base.pt" if sweep_values is not None
                                      else "icrl_rule_memory.pt")
    memory.save(base_memory_path)
    print(f"\n[INFO] Rule memory saved: {base_memory_path}  ({memory.num_rules} rules, "
          f"n_min={build_n_min} at build time)")

    if sweep_values is None:
        run_stage3_onward(memory, system1, val_loader, test_loader, label_names,
                           full_concept_keys, full_concept_offsets, full_concept_dims,
                           output_dir, args)
        print(f"\n[INFO] Results saved to {output_dir}/metrics.json")
        return

    print(f"\n[Stage 2.5] n_min sweep: {sweep_values}")
    sweep_results = []
    for n_min in sweep_values:
        print(f"\n{'='*60}\n[Sweep] n_min={n_min}\n{'='*60}")
        candidate = ICRLRuleMemory.load(base_memory_path, device=str(device))
        candidate.n_min = n_min
        candidate.conf_min = 0.0
        candidate.prune(verbose=True)
        sub_dir = output_dir / f"n_min_{n_min}"
        result = run_stage3_onward(candidate, system1, val_loader, test_loader, label_names,
                                    full_concept_keys, full_concept_offsets, full_concept_dims,
                                    sub_dir, args, conf_min=args.conf_min)
        sweep_results.append(result)

    print(f"\n{'='*60}\n[Sweep] Summary\n{'='*60}")
    print(f"{'n_min':>6} {'rules':>6} {'effective':>10} {'circular%':>10} {'val_acc':>8} {'test_acc':>9}")
    for r in sweep_results:
        print(f"{r['n_min']:>6} {r['num_rules']:>6} {r['num_effective_rules']:>10} "
              f"{r['circular_rate']*100:>9.1f}% {r['val_accuracy']:>8.4f} {r['test_accuracy']:>9.4f}")

    best_val = max(r["val_accuracy"] for r in sweep_results)
    in_range = [r for r in sweep_results if r["val_accuracy"] >= best_val - 0.02]
    recommended = max(in_range, key=lambda r: r["num_effective_rules"])
    print(f"\n[RECOMMENDED] n_min={recommended['n_min']} "
          f"(val_accuracy={recommended['val_accuracy']:.4f}, within 2pp of best {best_val:.4f}; "
          f"{recommended['num_effective_rules']} effective rules, "
          f"circular_rate={recommended['circular_rate']:.2%})")

    with open(output_dir / "n_min_sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "sweep_values": sweep_values,
            "results": sweep_results,
            "recommended_n_min": recommended["n_min"],
            "recommendation_rule": "max num_effective_rules among candidates with "
                                    "val_accuracy >= best_val_accuracy - 0.02",
        }, f, indent=2)

    import shutil
    rec_dir = output_dir / f"n_min_{recommended['n_min']}"
    for fname in ("icrl_rules.json", "icrl_rules_summary.json", "icrl_rule_memory.pt",
                  "prediction_head.pt", "metrics.json"):
        src = rec_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
    print(f"\n[INFO] Recommended candidate (n_min={recommended['n_min']}) copied to {output_dir}/")


if __name__ == "__main__":
    main()
