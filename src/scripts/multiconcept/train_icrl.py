"""
train_icrl_multiconcept.py — ICRL Stage 2/3 cho MNIST-MultiConcept
======================================================================

Giống train_icrl.py (MNIST Math), và NHÃN LÀ MỘT CONCEPT giống hệt digit3:
  - S1 (MultiConceptSystem1) dự đoán 13 concept nhị phân ĐỘC LẬP (sigmoid)
    + 1 label 3-way (softmax) -> concept vector FULL 16 chiều (13 concept +
    3 label), nối đúng thứ tự FULL_CONCEPT_OFFSETS trong
    multiconcept_concepts.py. Centroid VẪN lưu đủ 16 chiều (dùng cho
    export_rules/so sánh s1_label_guess, Stage 3 head).

  - cluster_dims = (0, NUM_CONCEPTS) — CHỈ 13 concept thị giác tham gia
    MATCH/CREATE/MERGE, KHÔNG bao gồm slot label. Lý do (phát hiện qua kiểm
    chứng thực nghiệm — xem hội thoại thiết kế): nếu để label tham gia
    similarity, CÙNG một concept pattern has_digit_X có thể bị TÁCH thành
    nhiều rule khác nhau, vì S1 vẫn "nhìn" ảnh gốc nên dự đoán label khác
    nhau tuỳ digit nào bị lặp lại dù pattern concept giống hệt — đo được
    35% pattern bị phân mảnh khi cluster_dims=None. Giới hạn về 13 chiều
    concept đưa fraction "rule sai" (so với nhãn thật tính trực tiếp từ
    digits) từ ~19% mẫu xuống còn ~1%. Đánh đổi: accuracy cuối giảm (do
    head mất tín hiệu tinh của S1 để phân biệt sub-case) nhưng đổi lại mỗi
    rule ứng với ĐÚNG MỘT concept pattern — ưu tiên diễn giải được đúng như
    mục tiêu ban đầu của S2, không phải tối đa accuracy (S1 alone đã ~98%).

  - nhãn đích (3 lớp even/equal/odd) TẤT ĐỊNH từ digits (đếm chẵn/lẻ) —
    impurity của cluster đến từ việc concept has_digit_X không ghi nhận số
    lần lặp, không phải sample xác suất.

  - Stage 3 dùng đúng fix đã verify trên MNIST Math: train head trực tiếp
    trên R centroids với rule_labels = memory.get_labels() (majority-vote
    ground-truth NGOÀI concept vector), weight_decay=0, đủ step để đạt
    separable fit. QUAN TRỌNG: dù label đã là 1 slot trong concept vector
    (lưu trong centroid), Stage 3 vẫn KHÔNG được đọc trực tiếp slot đó
    làm nhãn train — slot mang nhiễu của S1, memory.get_labels() mới là
    sự thật. Đây chính là bài học đã fix ở MNIST Math, áp dụng lại ở đây.

Usage (Kaggle mặc định, override --data_dir/--system1_ckpt nếu chạy local):
    python -m src.scripts.multiconcept.train_icrl \\
        --data_dir /kaggle/input/mnist-multiconcept \\
        --system1_ckpt /kaggle/working/outputs/multiconcept_system1/best_model.pt \\
        --output_dir /kaggle/working/outputs/multiconcept_icrl \\
        --theta 0.93 --theta_merge 0.97 --n_min 5 --conf_min 0.1 \\
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

from src.datasets.multiconcept.mnist_multiconcept_dataset import MNISTMultiConceptPTDataset
from src.models.multiconcept.system1 import (
    MultiConceptSystem1, soft_concept_vector, hard_concept_vector,
)
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.utils.seed import set_seed
from src.utils.multiconcept_concepts import (
    CONCEPT_NAMES, NUM_CONCEPTS, LABEL_NAMES, NUM_LABELS,
    FULL_CONCEPT_KEYS, FULL_CONCEPT_OFFSETS, FULL_CONCEPT_DIMS, FULL_CV_DIM,
    S1_LABEL_CONCEPT_KEY,
)


# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build ICRL rule memory cho MNIST-MultiConcept.")

    # Kaggle-first defaults — override bằng đường dẫn local nếu cần.
    p.add_argument("--data_dir",     type=str, default="/kaggle/input/mnist-multiconcept")
    p.add_argument("--system1_ckpt", type=str, default="/kaggle/working/outputs/multiconcept_system1/best_model.pt")
    p.add_argument("--output_dir",   type=str, default="/kaggle/working/outputs/multiconcept_icrl")

    p.add_argument("--theta",        type=float, default=0.93,
                    help="Đã tăng từ 0.85 (copy từ MNIST Math) — verify thực nghiệm cho thấy "
                         "0.85 gây gộp nhầm pattern khác nhau (cos({2,3,4},{2,3,4,5})=0.866 > 0.85).")
    p.add_argument("--theta_merge",  type=float, default=0.97)
    p.add_argument("--n_min",        type=int,   default=5)
    p.add_argument("--conf_min",     type=float, default=0.1)

    p.add_argument("--epochs",       type=int,   default=3,
                    help="Số lần pass qua training set để build rule memory (Stage 2).")
    p.add_argument("--use_hard_cv",  action="store_true")

    p.add_argument("--head_epochs",  type=int,   default=20)
    p.add_argument("--head_lr",      type=float, default=1e-3)
    p.add_argument("--head_steps_per_epoch", type=int, default=250,
                    help="Full-batch gradient step mỗi epoch khi train head trên R centroids.")
    p.add_argument("--num_classes",  type=int,   default=NUM_LABELS)

    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       type=str,   default="auto")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────

def load_system1(ckpt_path: Path, device: torch.device) -> MultiConceptSystem1:
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args  = ckpt.get("args", {})
    feature_dim = saved_args.get("feature_dim", 256)
    num_concepts = saved_args.get("num_concepts", NUM_CONCEPTS)
    num_labels   = saved_args.get("num_labels", NUM_LABELS)
    model = MultiConceptSystem1(feature_dim=feature_dim, num_concepts=num_concepts, num_labels=num_labels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_loaders(data_dir: Path, batch_size: int, num_workers: int):
    def _loader(split, shuffle):
        for fname in (["val.pt", "valid.pt"] if split == "val" else [f"{split}.pt"]):
            pt = data_dir / fname
            if pt.exists():
                ds = MNISTMultiConceptPTDataset(pt)
                return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=True)
        raise FileNotFoundError(f"No {split} split found in {data_dir}")

    return _loader("train", True), _loader("val", False), _loader("test", False)


# ─────────────────────────────────────────────────────────────
# Stage 2: Build rule memory
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def build_rule_memory(
    system1: MultiConceptSystem1,
    loader: DataLoader,
    memory: ICRLRuleMemory,
    device: torch.device,
    use_hard: bool = False,
    epoch_label: str = "Epoch",
) -> dict[str, int]:
    total_stats = {"created": 0, "matched": 0, "total": 0}

    for images, labels in tqdm(loader, desc=f"  Build [{epoch_label}]", leave=False):
        images = images.to(device)

        with torch.no_grad():
            s1_out = system1(images)   # dict: {"concepts": [B,16], "label": [B,3]}

        cv = hard_concept_vector(s1_out) if use_hard else soft_concept_vector(s1_out)

        # S1 confidence per sample: trung bình của 2 khối tín hiệu —
        #   concept_conf: trung bình |sigmoid-0.5|*2 trên 16 concept nhị phân
        #   label_conf:   max-softmax-prob của label 3-way
        # (mirror "mean max-prob across slots" của MNIST Math, ở đây 2 khối
        # concept-block và label-block được coi ngang nhau).
        concept_probs = torch.sigmoid(s1_out["concepts"])
        concept_conf  = (2.0 * (concept_probs - 0.5).abs()).mean(dim=1)   # [B]
        label_conf    = F.softmax(s1_out["label"], dim=-1).max(dim=1).values  # [B]
        s1_conf = (concept_conf + label_conf) / 2.0

        y = labels["label"].to(device)

        stats = memory.process_batch(cv, y, s1_conf)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    return total_stats


# ─────────────────────────────────────────────────────────────
# Stage 3: Train prediction head trực tiếp trên R rule centroids
# ─────────────────────────────────────────────────────────────

def train_head(
    system1:     MultiConceptSystem1,
    memory:      ICRLRuleMemory,
    val_loader:  DataLoader,
    num_classes: int,
    epochs:      int,
    lr:          float,
    device:      torch.device,
    use_hard:    bool = False,
    steps_per_epoch: int = 250,
) -> nn.Linear:
    """
    Xem giải thích chi tiết trong train_icrl.py::train_head (MNIST Math) —
    cùng nguyên tắc: nhãn train head = memory.get_labels() (majority-vote
    ground-truth per rule), weight_decay=0 để đạt separable fit trong ngân
    sách step hợp lý. Ở MultiConcept, nhãn GIỜ ĐÃ nằm trong concept vector
    (slot s1_label_pred, dùng để match/cluster) — nhưng nguyên tắc vẫn giữ
    nguyên: KHÔNG đọc slot đó làm nhãn train head, vì nó mang nhiễu của S1.
    memory.get_labels() (ground-truth ngoài, tracked qua y lúc
    process_batch) mới là nguồn nhãn hợp lệ duy nhất.
    """
    head = nn.Linear(FULL_CV_DIM, num_classes).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)

    best_val   = 0.0
    best_state = None

    centroids   = memory.get_centroids().to(device)
    rule_labels = torch.tensor(memory.get_labels(), dtype=torch.long, device=device)

    print(f"\n[Stage 3] Train prediction head trực tiếp trên {memory.num_rules} rule centroids "
          f"({epochs} epochs x {steps_per_epoch} steps)")
    print(f"  Nhãn: memory.get_labels() (majority-vote ground-truth mỗi rule)")

    for epoch in range(1, epochs + 1):
        head.train()
        for _ in range(steps_per_epoch):
            logits = head(centroids)
            loss   = F.cross_entropy(logits, rule_labels)
            opt.zero_grad(); loss.backward(); opt.step()
        train_acc = (logits.argmax(dim=1) == rule_labels).float().mean().item()

        head.eval()
        val_correct = 0; val_total = 0
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
                val_total   += len(y)

        val_acc = val_correct / val_total
        print(f"  Ep {epoch:2d}/{epochs}: rule_train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    if best_state:
        head.load_state_dict(best_state)
    print(f"  Best val_acc = {best_val:.4f}")
    return head


# ─────────────────────────────────────────────────────────────
# Evaluate & export
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(system1, head, loader, device, memory, split="test", use_hard=False):
    head.eval()
    correct = 0; total = 0
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
        total   += len(y)

    return {"accuracy": correct / total, "correct": correct, "total": total}


def export_rules(memory: ICRLRuleMemory, output_dir: Path, n_show: int = 20) -> None:
    """
    decode_rule trả về 2 khái niệm "nhãn" khác nhau cho mỗi rule — dễ nhầm,
    cần phân biệt rõ khi đọc JSON:
      - label_name          : decoded["label"] = majority-vote GROUND-TRUTH
                               (memory.get_labels()) — đây là nhãn ĐÚNG, dùng
                               để train Stage-3 head.
      - s1_label_guess_name  : decoded["slots"]["s1_label_pred"] = giá trị
                               S1 tự đoán (nằm trong centroid, chỉ dùng để
                               match/cluster) — có thể SAI so với label_name,
                               đặc biệt ở rule impure (xem is_palindrome,
                               strictly_increasing trong phân tích trước).
    So sánh 2 cột này chính là cách kiểm chứng "S1 đoán nhãn tốt tới đâu"
    độc lập với "S2 sửa được bao nhiêu qua clustering".
    """
    rules_data = []
    for r in range(memory.num_rules):
        decoded = memory.decode_rule(
            rule_id=r,
            concept_keys=FULL_CONCEPT_KEYS,
            concept_offsets=FULL_CONCEPT_OFFSETS,
            concept_dims=FULL_CONCEPT_DIMS,
            id_to_symbol=None,
        )
        present = [k for k, v in decoded["slots"].items()
                   if k != S1_LABEL_CONCEPT_KEY and v["value"] == "present"]

        s1_label_idx = int(decoded["slots"][S1_LABEL_CONCEPT_KEY]["value"])
        s1_label_conf = decoded["slots"][S1_LABEL_CONCEPT_KEY]["confidence"]

        decoded["label_name"] = LABEL_NAMES[decoded["label"]] if 0 <= decoded["label"] < NUM_LABELS else "?"
        decoded["s1_label_guess_name"] = LABEL_NAMES[s1_label_idx]
        decoded["s1_label_guess_confidence"] = s1_label_conf
        decoded["s1_label_agrees_with_truth"] = (LABEL_NAMES[s1_label_idx] == decoded["label_name"])
        decoded["present_concepts"] = present
        rules_data.append(decoded)

    rules_data.sort(key=lambda x: -x["confidence"])

    json_path = output_dir / "icrl_rules.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rules_data, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] {len(rules_data)} rules exported to {json_path}")

    n_agree = sum(1 for r in rules_data if r["s1_label_agrees_with_truth"])
    print(f"[INFO] S1's own label guess (concept slot) agrees with ground-truth "
          f"majority label: {n_agree}/{len(rules_data)} rules")

    print(f"\n[INFO] Top {min(n_show, len(rules_data))} rules (sorted by confidence):")
    for r in rules_data[:n_show]:
        bar = "█" * int(r["confidence"] * 20)
        concepts_str = "+".join(r["present_concepts"]) or "(none)"
        agree = "=" if r["s1_label_agrees_with_truth"] else "≠"
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
    print(f"[INFO] ICRL params: θ={args.theta}  θ_merge={args.theta_merge}  "
          f"n_min={args.n_min}  conf_min={args.conf_min}")

    data_dir = Path(args.data_dir)
    train_loader, val_loader, test_loader = make_loaders(
        data_dir, args.batch_size, args.num_workers
    )
    print(f"[INFO] Data: {data_dir}")

    system1 = load_system1(Path(args.system1_ckpt), device)
    print(f"[INFO] System1 loaded (frozen): {args.system1_ckpt}")

    memory = ICRLRuleMemory(
        concept_dim  = FULL_CV_DIM,   # 13 concept + 3 label-slot (nối như digit3 ở MNIST Math)
        theta        = args.theta,
        theta_merge  = args.theta_merge,
        n_min        = args.n_min,
        conf_min     = args.conf_min,
        # cluster_dims chỉ dùng 13 concept đầu (KHÔNG gồm s1_label_pred) cho
        # MATCH/CREATE/MERGE. Lý do: nếu để label_pred tham gia similarity,
        # cùng 1 presence-pattern has_digit_X (vd. {2,3,4}) có thể bị TÁCH
        # thành 2+ rule khác nhau tuỳ digit nào bị lặp lại — S1 vẫn "nhìn"
        # được ảnh gốc nên dự đoán label khác nhau dù concept pattern giống
        # hệt nhau, kéo các ảnh cùng pattern rẽ sang rule khác nhau (đã đo
        # thực nghiệm: 132/377 pattern bị phân mảnh khi để None). label_pred
        # vẫn được lưu trong centroid (dùng để hiển thị/so sánh trong
        # export_rules, Stage 3 head vẫn nhận đủ FULL_CV_DIM) — chỉ loại
        # khỏi phép so khớp để giữ đúng 1 rule cho mỗi presence-pattern.
        cluster_dims = (0, NUM_CONCEPTS),
        device       = str(device),
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
        num_classes=args.num_classes,
        epochs=args.head_epochs,
        lr=args.head_lr,
        device=device,
        use_hard=args.use_hard_cv,
        steps_per_epoch=args.head_steps_per_epoch,
    )
    torch.save(head.state_dict(), output_dir / "prediction_head.pt")

    print("\n[INFO] Evaluating...")
    test_metrics = evaluate(system1, head, test_loader, device, memory, "test", args.use_hard_cv)
    val_metrics  = evaluate(system1, head, val_loader,  device, memory, "val",  args.use_hard_cv)

    print(f"\n[DONE] Results:")
    print(f"  val_accuracy  = {val_metrics['accuracy']:.4f}")
    print(f"  test_accuracy = {test_metrics['accuracy']:.4f}")

    export_rules(memory, output_dir)

    metrics = {
        "val_accuracy":  val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "num_rules":     memory.num_rules,
        "args": vars(args),
        "rule_confidence_stats": {
            "mean": sum(memory.get_confidences()) / max(1, memory.num_rules),
            "min":  min(memory.get_confidences()) if memory.num_rules else 0,
            "max":  max(memory.get_confidences()) if memory.num_rules else 0,
        }
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[INFO] Results saved to {output_dir}/metrics.json")


if __name__ == "__main__":
    main()
