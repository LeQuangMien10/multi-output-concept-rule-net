"""
train_icrl.py - ICRL Stage 2/3 cho Fitzpatrick17k
=====================================================

Giong het tinh than train_icrl.py cua MultiConcept (nhan la 1 concept, noi vao
sau 35 concept nhi phan giong s1_label_pred cua MultiConcept/digit3 cua MNIST
Math), NHUNG khac o mot diem quan trong da xac nhan qua danh gia checkpoint S1:

  19/35 concept van con F1=0 (chua hoc duoc gi, do qua it mau/split -- xem
  outputs/fitzpatrick_audit/s1_dashboard.json). Quyet dinh: GIU NGUYEN ca 35
  concept trong vector dung de cluster (khong loai heuristic) -- logit cua
  cac concept F1=0 gan nhu hang so cho moi anh nen dong gop rat it vao cosine
  similarity, tu nhien bi "lam mo" trong qua trinh MATCH/CREATE ma khong can
  can thiep thu cong. Day la quyet dinh nhat quan voi bai hoc da rut ra o
  MultiConcept: khong loai bo concept theo heuristic khi chua co bang chung
  ro rang no lam hai (xem docstring src/models/icrl_rule_memory.py va
  src/utils/multiconcept_concepts.py).

  cluster_dims=None -- dung toan bo vector (concept + label-slot), giong
  quyet dinh cuoi cung o MultiConcept sau khi verify gioi han cluster_dims
  xoa tin hieu that ma khong loi ich tuong xung.

  theta mac dinh = 0.886 -- do THAT tren concept vector GROUND-TRUTH 35-dim
  (measure_theta.py, percentile 99.9 cua cos giua 2 pattern KHAC nhau +
  bien an toan). Day la diem khoi dau, KHONG phai gia tri cuoi cung -- sau
  khi chay xong nen kiem tra rule contradiction/purity that (giong cach da
  lam voi MultiConcept) de quyet dinh co can dieu chinh khong.

QUAN TRONG: Stage 2 (build rule memory) va Stage 3 (eval head) dung transform
"val" (resize+center-crop, KHONG augment) cho ca train/val/test -- augmentation
ngau nhien (RandomResizedCrop, xoay, lat) chi hop ly khi TRAIN S1 (buoc 4),
khong hop ly khi trich concept vector de cluster: 2 lan augment khac nhau tren
CUNG 1 anh se cho 2 concept vector khac nhau mot cach gia tao, lam nhieu qua
trinh MATCH khong lien quan gi den su khac biet that giua cac anh.

Usage (Kaggle mac dinh, override neu chay local):
    python -m src.scripts.fitzpatrick.train_icrl \\
        --data_dir /kaggle/input/datasets/lquangmin/fitzpatrick17k-prepared \\
        --img_dir /kaggle/input/datasets/lquangmin/fitzpatrick17k/data/finalfitz17k \\
        --system1_ckpt /kaggle/working/outputs/fitzpatrick_system1/best_model.pt \\
        --output_dir /kaggle/working/outputs/fitzpatrick_icrl \\
        --theta 0.886 --theta_merge 0.93 --n_min 15 --conf_min 0.5 \\
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

from src.datasets.fitzpatrick.fitzpatrick_dataset import FitzpatrickDataset, build_transforms
from src.models.fitzpatrick.system1 import FitzpatrickSystem1, soft_concept_vector, hard_concept_vector
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.utils.seed import set_seed
from src.utils.fitzpatrick_concepts import (
    LABEL_NAMES as DEFAULT_LABEL_NAMES, S1_LABEL_CONCEPT_KEY,
)


def build_full_concept_layout(concept_names: list[str], label_names: list[str]):
    """FULL concept vector layout (concepts + s1_label_pred slot), derived at
    runtime from whatever concept_names/label_names this run actually uses --
    NOT a fixed 35-concept/3-class constant, so this works unchanged for
    alternate data preparations (e.g. the 48-concept/2-class CRL-matched
    variant, see prepare_dataset_crl_matched.py)."""
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
    p = argparse.ArgumentParser(description="Build ICRL rule memory cho Fitzpatrick17k.")

    p.add_argument("--data_dir", type=str, default="/kaggle/input/datasets/lquangmin/fitzpatrick17k-prepared")
    p.add_argument("--img_dir", type=str, default="/kaggle/input/datasets/lquangmin/fitzpatrick17k/data/finalfitz17k")
    p.add_argument("--system1_ckpt", type=str, default="/kaggle/working/outputs/fitzpatrick_system1/best_model.pt")
    p.add_argument("--output_dir", type=str, default="/kaggle/working/outputs/fitzpatrick_icrl")

    p.add_argument("--theta", type=float, default=0.886,
                    help="Do tren concept vector ground-truth (measure_theta.py). Kiem tra lai "
                         "rule contradiction/purity sau khi chay de xac nhan gia tri nay con hop ly "
                         "tren concept vector S1 THAT DU DOAN (co the khac ground-truth).")
    p.add_argument("--theta_merge", type=float, default=0.93)
    p.add_argument("--n_min", type=int, default=15,
                    help="Tang tu 5 -> 15 sau khi phan tich lan chay dau: 12/61 rule co n nho "
                         "(rieng biet ma khong on dinh -- live-majority label khac nhan da luu). "
                         "n_min cao hon loc bot duoi rule nho ngay tu Stage 2.")
    p.add_argument("--conf_min", type=float, default=0.5,
                    help="Tang tu 0.1 -> 0.5 sau khi accuracy tro nen y nghia (xem update_accuracy "
                         "fix): 0.1 qua de, khong loc duoc rule gan-ngau-nhien (vd accuracy 30-50%%, "
                         "confidence ~0.28-0.47). 0.5 loai rule co accuracy <~0.6-0.7 tuy coherence.")

    p.add_argument("--epochs", type=int, default=3,
                    help="So lan pass qua training set de build rule memory (Stage 2).")
    p.add_argument("--use_hard_cv", action="store_true",
                    help="Buoc 0 (concept leakage, xem paper CRL MICCAI 2025): dung hard "
                         "(nhi phan, threshold 0.5) concept vector thay vi soft (sigmoid/softmax "
                         "probs) cho MATCH/CREATE/MERGE. Concept lien tuc co the 'leak' thong tin "
                         "ngoai y concept vao rule, lam giam generalization (bang chung thuc "
                         "nghiem cua paper tren chinh Fitzpatrick17k+SkinCon).")
    p.add_argument("--exclude_label_slot", action="store_true",
                    help="Buoc 0: bo s1_label_pred (nhan S1 tu du doan) khoi vector dung de "
                         "MATCH/CREATE/MERGE (cluster_dims=(0,NUM_CONCEPTS) thay vi None). "
                         "Kiem tra xem slot nay co dang 'leak' quyet dinh cua S1 vao rule khong "
                         "-- CHU Y: MultiConcept da thu huong nay va BAC BO (mat ~13 diem accuracy "
                         "khong co loi ich ro rang). Chay de xac nhan co nhat quan voi ket qua do "
                         "khong, khong ky vong se ap dung mac dinh neu accuracy giam tuong tu.")

    p.add_argument("--head_epochs", type=int, default=20)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--head_steps_per_epoch", type=int, default=250)
    p.add_argument("--label_names", type=str, default=",".join(DEFAULT_LABEL_NAMES),
                    help="Comma-separated class names, in label_idx order. Override for "
                         "alternate preparations, e.g. 'benign,malignant' for the CRL-matched "
                         "2-class variant (default: the 3-class benign/malignant/non-neoplastic "
                         "setup in src/utils/fitzpatrick_concepts.py).")

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
    # "val" transform (khong augment) cho CA train/val/test o day -- xem docstring dau file.
    clean_transform = build_transforms("val", image_size)

    def _loader(split, shuffle):
        ds = FitzpatrickDataset(data_dir / f"{split}.csv", img_dir, clean_transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=num_workers, pin_memory=True)

    return _loader("train", True), _loader("val", False), _loader("test", False)


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
    """Nhan train head = memory.get_labels() (majority-vote ground-truth moi
    rule) -- KHONG doc slot s1_label_pred trong centroid lam nhan, dung bai
    hoc da fix o MNIST Math/MultiConcept."""
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
    """Goi memory.update_accuracy() -- buoc nay BI THIEU trong ban dau (ca
    MultiConcept lan Fitzpatrick), khien field 'accuracy' trong icrl_rules.json
    luon = 0.5 (gia tri trung lap mac dinh khi total_pred=0) va 'confidence'
    xuat ra chi con la coherence*0.5, khong phan biet duoc rule tot/xau.

    Dung val_loader (khong dung train) de danh gia rule co du doan dung tren
    du lieu KHONG dung de build no hay khong -- trung thuc hon so voi do
    tren chinh du lieu da dung de tao rule."""
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
    """
    Build a compact, scannable summary alongside the verbose per-concept
    icrl_rules.json (which lists EVERY concept's present/absent/confidence
    per rule -- unreadable at a glance with 35-48 concepts). Modeled after
    the official CRL repo's rules.txt format: one line per rule with a
    human rule_string, support/confidence, and predicted class.

    Two fixes baked in here (previously only available as a manual
    post-processing step via reexport_rules_with_not.py):
      1. "Informative" NOT: a concept only appears as "NOT X" in a rule
         string if X is 'present' in >=1 rule anywhere in this set --
         excludes S1's dead/never-learned concepts from cluttering every
         rule with meaningless absences.
      2. s1_label_pred is shown explicitly per rule, and rules sharing the
         exact same visible (present + informative-NOT) pattern but a
         different majority label are flagged "circular" -- separated only
         by S1's own label guess, not by concept evidence a reader can see.
    """
    informative = set()
    for r in rules_data:
        informative.update(r["present_concepts"])

    entries = []
    pattern_groups: dict[tuple, list[int]] = {}
    for r in rules_data:
        present = r["present_concepts"]
        absent_informative = [
            c for c in concept_names
            if c not in present and c in informative
        ]
        rule_string = " AND ".join(present + [f"NOT {c}" for c in absent_informative])
        key = (tuple(present), tuple(absent_informative))
        pattern_groups.setdefault(key, []).append(len(entries))
        entries.append({
            "rule_id": r["rule_id"],
            "label": r["label_name"],
            "confidence": round(r["confidence"], 4),
            "coherence": round(r["coherence"], 4),
            "n": r["n"],
            "s1_label_pred": r["s1_label_guess_name"],
            "s1_label_pred_confidence": round(r["s1_label_guess_confidence"], 4),
            "rule_string": rule_string or "(no informative concept -- default/catch-all rule)",
            "num_conditions": len(present) + len(absent_informative),
            "circular_group": None,  # filled below
        })

    n_circular = 0
    for idx_list in pattern_groups.values():
        if len(idx_list) < 2:
            continue
        labels_here = set(entries[i]["label"] for i in idx_list)
        if len(labels_here) > 1:
            for i in idx_list:
                entries[i]["circular_group"] = [entries[j]["rule_id"] for j in idx_list if j != i]
            n_circular += len(idx_list)

    entries.sort(key=lambda x: -x["confidence"])

    confs = [r["confidence"] for r in rules_data] or [0.0]
    ns = [r["n"] for r in rules_data] or [0]
    return {
        "num_rules": len(rules_data),
        "num_effective_rules": len(rules_data) - n_circular,
        "circular_rules": n_circular,
        "circular_rate": round(n_circular / len(rules_data), 4) if rules_data else 0.0,
        "num_informative_concepts": len(informative),
        "informative_concepts": sorted(informative),
        "labels": label_names,
        "confidence_stats": {"mean": round(sum(confs) / len(confs), 4), "min": round(min(confs), 4), "max": round(max(confs), 4)},
        "n_stats": {"mean": round(sum(ns) / len(ns), 1), "min": min(ns), "max": max(ns)},
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

    print(f"[INFO] Device: {device}")
    print(f"[INFO] ICRL params: theta={args.theta}  theta_merge={args.theta_merge}  "
          f"n_min={args.n_min}  conf_min={args.conf_min}")

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

    memory = ICRLRuleMemory(
        concept_dim=full_cv_dim,
        theta=args.theta,
        theta_merge=args.theta_merge,
        n_min=args.n_min,
        conf_min=args.conf_min,
        cluster_dims=cluster_dims,   # None = toan bo vector; xem --exclude_label_slot help
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

    memory_path = output_dir / "icrl_rule_memory.pt"
    memory.save(memory_path)
    print(f"\n[INFO] Rule memory saved: {memory_path}  ({memory.num_rules} rules)")

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

    print("\n[Stage 3.5] Recording rule accuracy on val split "
          "(fixes update_accuracy() previously never being called)")
    record_rule_accuracy(system1, head, memory, val_loader, device, args.use_hard_cv)

    print(f"\n[INFO] Final prune using real accuracy signal (conf_min={args.conf_min})")
    memory.prune(verbose=True)   # khong override -- dung conf_min that, gio da co accuracy y nghia
    print(f"  After final prune: {memory.num_rules} rules")

    memory.save(memory_path)   # ghi de, phan anh dung trang thai cuoi cung (sau final prune)
    print(f"[INFO] Rule memory re-saved after final prune: {memory_path}  ({memory.num_rules} rules)")

    print("\n[INFO] Evaluating...")
    test_metrics = evaluate(system1, head, test_loader, device, memory, "test", args.use_hard_cv)
    val_metrics = evaluate(system1, head, val_loader, device, memory, "val", args.use_hard_cv)

    print(f"\n[DONE] Results:")
    print(f"  val_accuracy  = {val_metrics['accuracy']:.4f}")
    print(f"  test_accuracy = {test_metrics['accuracy']:.4f}")

    export_rules(memory, output_dir, full_concept_keys, full_concept_offsets,
                 full_concept_dims, label_names)

    metrics = {
        "val_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "num_rules": memory.num_rules,
        "args": vars(args),
        "rule_confidence_stats": {
            "mean": sum(memory.get_confidences()) / max(1, memory.num_rules),
            "min": min(memory.get_confidences()) if memory.num_rules else 0,
            "max": max(memory.get_confidences()) if memory.num_rules else 0,
        }
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[INFO] Results saved to {output_dir}/metrics.json")


if __name__ == "__main__":
    main()
