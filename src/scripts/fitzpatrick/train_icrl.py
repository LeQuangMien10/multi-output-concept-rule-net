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
from src.scripts.fitzpatrick.train_icrl_gt_ablation import measure_theta
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

    p.add_argument("--theta", type=str, default="0.886",
                    help="Float, hoac 'auto' de tu do truoc Stage 2 tren dung concept vector S1 "
                         "THAT du doan (tren train split, ton trong cluster_dims/use_hard_cv) -- "
                         "percentile-99.9 cosine giua cac pattern KHAC nhau + bien an toan 0.02. "
                         "Gia tri float mac dinh (0.886) do tren concept vector GROUND-TRUTH 35-dim "
                         "cua dataset goc -- KHONG con dung neu doi so concept/them label-slot "
                         "(vd. scope CRL-matched 48-concept): dung 'auto' cho cac scope khac nhau.")
    p.add_argument("--theta_merge", type=float, default=0.93)
    p.add_argument("--n_min", type=int, default=15,
                    help="Tang tu 5 -> 15 sau khi phan tich lan chay dau: 12/61 rule co n nho "
                         "(rieng biet ma khong on dinh -- live-majority label khac nhan da luu). "
                         "n_min cao hon loc bot duoi rule nho ngay tu Stage 2. Gia tri nay copy tu "
                         "lan calibrate dau tien (16.577 anh) -- KHONG tu dong phu hop voi scope "
                         "nho hon (vd. 636-3.227 anh); dung --n_min_sweep de tim gia tri phu hop.")
    p.add_argument("--n_min_sweep", type=str, default=None,
                    help="Danh sach n_min cach nhau boi dau phay, vd '5,10,15,20,30'. Neu duoc set, "
                         "Stage 2 chi build MOT LAN (memory 'goc' khong prune theo n_min), sau do "
                         "moi gia tri trong danh sach duoc prune+train head+danh gia RIENG (khong can "
                         "build lai) -- rat re vi Stage 2 (anh + S1 forward) la phan ton thoi gian "
                         "nhat. Ket qua tung gia tri luu vao output_dir/n_min_<N>/, kem 1 bang so sanh "
                         "+ 1 de xuat tu dong (uu tien so rule khong-circular cao nhat trong so cac "
                         "gia tri co val_accuracy trong pham vi 2 diem %% cua gia tri tot nhat) duoc "
                         "copy len output_dir/ (top-level) de tuong thich nguoc voi cac script khac.")
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


@torch.no_grad()
def collect_concept_vectors(system1, loader, device, use_hard=False, cluster_dims=None) -> torch.Tensor:
    """One pass over `loader`, stacking the exact concept vector Stage 2 will
    cluster on (post cluster_dims slicing) -- used by --theta auto so theta is
    measured on the real S1-predicted vectors for THIS run's concept/label
    schema, not a value borrowed from a different concept count / dataset."""
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
    Build a minimal, scannable summary alongside the verbose per-concept
    icrl_rules.json (which lists EVERY concept's present/absent/confidence
    per rule -- unreadable at a glance with 35-48 concepts). Just the rule
    itself (present concepts only, AND-joined) + predicted label + whether
    it's "circular" (see below) -- everything else (confidence, coherence,
    n, s1_label_pred, ...) lives in icrl_rules.json, cross-referenced by
    rule_id, so it isn't duplicated here.

    "circular": rules sharing the exact same visible present-concept
    pattern but a different majority label -- concept evidence alone
    doesn't explain the split (it's actually the hidden s1_label_pred slot
    doing the separating, see icrl_rules.json's "slots" field for that rule
    if you need to confirm why).
    """
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
# Factored out so --n_min_sweep can run this once per candidate n_min without
# re-doing Stage 2 (the expensive image + S1 forward-pass part) each time.
# ─────────────────────────────────────────────────────────────

def _collapsed_result(memory, output_dir, full_concept_keys, full_concept_offsets,
                       full_concept_dims, label_names, args) -> dict:
    """0 rules survived pruning -- there is nothing left to match concept
    vectors against, so head-training/evaluation is meaningless (not just
    slow to skip). Still write the same file set (empty rules list, zeroed
    metrics) so sweep summaries and the recommended-candidate copy step
    don't need to special-case this outcome."""
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

    print(f"\n[INFO] Dedupe by decoded display pattern (drop circular, merge exact "
          f"duplicates -- theta_merge misses these since it compares raw continuous "
          f"vectors, not the human-readable pattern)")
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
        # percentile=95 (not measure_theta's own 99.9 default): measured locally across all
        # 3 Fitzpatrick scopes with the exact soft/full-vector config above --
        #   p99   -> CRL-matched=0.961  concept-only=0.988  full-data=0.982  (2 of 3 still hit
        #            the 0.999 cap after +0.02 margin -- not just a CRL-matched-specific fluke)
        #   p95   -> CRL-matched=0.826  concept-only=0.972  full-data=0.955  (none hit the cap)
        # The 35-concept scopes' cosine-similarity distributions sit closer to 1.0 overall than
        # the 48-concept CRL-matched one, so even p99 wasn't low enough headroom below the 0.999
        # cap for them. p95 keeps theta below the cap for all 3 scopes while still producing a
        # meaningfully different, high-similarity bar per dataset.
        theta = measure_theta(train_cv, percentile=95)
        theta_merge = min(theta + 0.04, 0.999)
        print(f"[INFO] Measured theta={theta:.4f}  theta_merge={theta_merge:.4f} (theta+0.04)")
    else:
        theta = float(args.theta)
        theta_merge = args.theta_merge

    # Build n_min=1 during Stage 2 so nothing is discarded prematurely when
    # sweeping -- the REAL n_min filtering happens per-candidate afterward,
    # on saved copies, without re-running Stage 2 (the expensive part).
    build_n_min = 1 if sweep_values is not None else args.n_min

    memory = ICRLRuleMemory(
        concept_dim=full_cv_dim,
        theta=theta,
        theta_merge=theta_merge,
        n_min=build_n_min,
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

    # ── n_min sweep: reload the base memory once per candidate, prune to
    # that n_min, run Stage 3 onward into its own subdir ──────────────────
    print(f"\n[Stage 2.5] n_min sweep: {sweep_values}")
    sweep_results = []
    for n_min in sweep_values:
        print(f"\n{'='*60}\n[Sweep] n_min={n_min}\n{'='*60}")
        candidate = ICRLRuleMemory.load(base_memory_path, device=str(device))
        candidate.n_min = n_min
        candidate.conf_min = 0.0  # accuracy not measured yet; real conf_min applied in run_stage3_onward
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

    # Recommend: highest num_effective_rules among candidates whose val_accuracy
    # is within 2 points of the best val_accuracy in the sweep.
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

    # Copy the recommended candidate's outputs up to output_dir/ (top-level)
    # so consumers that expect output_dir/icrl_rules.json etc. keep working.
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
