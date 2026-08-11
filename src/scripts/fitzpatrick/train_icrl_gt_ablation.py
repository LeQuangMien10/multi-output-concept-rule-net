"""
train_icrl_gt_ablation.py - ICRL Stage 2/3 with GROUND-TRUTH inputs (no System 1
involved at all) -- diagnostic ablation to separate "System 1's concept/label
predictions are the bottleneck" from "System 2's clustering mechanism itself
is the bottleneck".

Concept vector = 48 REAL SkinCon concepts (hard 0/1, from CSV) concatenated
with the GROUND-TRUTH label as a one-hot vector (2-dim) -- NOT S1's predicted
label. This makes task accuracy close to trivial by construction (the label
is literally embedded in the input); the point of this run is NOT accuracy,
it's whether System 2 forms a rich, coherent, non-contradictory rule set when
given perfect information. If rules are still few/coarse/contradictory here,
that is System 2's own limitation, not a symptom of weak S1.

No image loading, no S1 forward pass -- pure tabular processing on the
prepared CSVs (data/fitzpatrick17k_crl_matched/), runs in seconds locally.

Also implements the negation-display fix discussed separately: shows
meaningful "NOT concept" conditions (filtered to exclude concepts that never
appear as present anywhere in the rule set -- the "dead concept" noise
problem), instead of hiding all absent concepts like the S1-based pipeline does.

Usage:
    python -m src.scripts.fitzpatrick.train_icrl_gt_ablation \
        --data_dir data/fitzpatrick17k_crl_matched \
        --output_dir outputs/fitzpatrick_icrl_gt_ablation
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.icrl_rule_memory import ICRLRuleMemory
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/fitzpatrick17k_crl_matched")
    p.add_argument("--output_dir", type=str, default="outputs/fitzpatrick_icrl_gt_ablation")
    p.add_argument("--theta", type=float, default=None, help="If not given, measured from data.")
    p.add_argument("--theta_merge", type=float, default=None)
    p.add_argument("--n_min", type=int, default=15)
    p.add_argument("--conf_min", type=float, default=0.5)
    p.add_argument("--head_epochs", type=int, default=20)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--head_steps_per_epoch", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_split(data_dir: Path, split: str, concept_names: list[str], label_names: list[str]):
    rows = list(csv.DictReader(open(data_dir / f"{split}.csv", encoding="utf-8")))
    concepts = np.array([[float(r[c]) for c in concept_names] for r in rows], dtype=np.float32)
    labels = np.array([int(r["label_idx"]) for r in rows], dtype=np.int64)
    num_labels = len(label_names)
    label_onehot = np.eye(num_labels, dtype=np.float32)[labels]
    cv = np.concatenate([concepts, label_onehot], axis=1)  # [N, num_concepts + num_labels]
    return torch.tensor(cv), torch.tensor(labels)


def measure_theta(cv: torch.Tensor, percentile: float = 99.9, margin: float = 0.02) -> float:
    """Same methodology as measure_theta.py: cosine among DIFFERENT full
    (concept+label) patterns, theta set just above the high percentile to
    avoid merging genuinely distinct patterns."""
    X = cv.numpy()
    uniq, idx = np.unique(X, axis=0, return_index=True)
    norm = np.linalg.norm(uniq, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    Xn = uniq / norm
    sims = Xn @ Xn.T
    n = sims.shape[0]
    off_diag = sims[~np.eye(n, dtype=bool)]
    theta = float(np.percentile(off_diag, percentile)) + margin
    return min(theta, 0.999)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    concept_names = meta["concept_names"]
    label_names = meta["label_names"]
    concept_dim = len(concept_names) + len(label_names)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_train, y_train = load_split(data_dir, "train", concept_names, label_names)
    cv_val, y_val = load_split(data_dir, "val", concept_names, label_names)
    cv_test, y_test = load_split(data_dir, "test", concept_names, label_names)
    print(f"[INFO] train={cv_train.shape} val={cv_val.shape} test={cv_test.shape}")

    theta = args.theta if args.theta is not None else measure_theta(cv_train)
    theta_merge = args.theta_merge if args.theta_merge is not None else min(theta + 0.04, 0.999)
    print(f"[INFO] theta={theta:.4f} (measured={args.theta is None})  theta_merge={theta_merge:.4f}")

    memory = ICRLRuleMemory(
        concept_dim=concept_dim, theta=theta, theta_merge=theta_merge,
        n_min=args.n_min, conf_min=args.conf_min, cluster_dims=None, device=str(device),
    )

    print("\n[Stage 2] Building rule memory (ground-truth inputs, deterministic -> 1 pass)")
    stats = memory.process_batch(cv_train, y_train, torch.ones(cv_train.shape[0]))
    print(f"  Created={stats['created']}  Matched={stats['matched']}  Rules so far={memory.num_rules}")
    memory.prune(verbose=True, conf_min_override=0.0)
    print(f"  After prune: {memory.num_rules} rules")

    # ---- Stage 3: train head on rule centroids ----
    num_labels = len(label_names)
    head = nn.Linear(concept_dim, num_labels)
    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=0.0)
    centroids = memory.get_centroids()
    rule_labels = torch.tensor(memory.get_labels(), dtype=torch.long)

    print(f"\n[Stage 3] Train head on {memory.num_rules} rule centroids")
    best_val, best_state = 0.0, None
    for epoch in range(1, args.head_epochs + 1):
        head.train()
        for _ in range(args.head_steps_per_epoch):
            logits = head(centroids)
            loss = F.cross_entropy(logits, rule_labels)
            opt.zero_grad(); loss.backward(); opt.step()

        head.eval()
        with torch.no_grad():
            rule_ids, _ = memory.match(cv_val)
            preds = head(centroids[rule_ids]).argmax(dim=1)
            val_acc = (preds == y_val).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    print(f"  Best val_acc = {best_val:.4f}")

    # ---- Stage 3.5: record real rule accuracy ----
    with torch.no_grad():
        rule_ids, _ = memory.match(cv_val)
        preds = head(centroids[rule_ids]).argmax(dim=1)
        memory.update_accuracy(cv_val, y_val, preds)
    print(f"\n[INFO] Final prune using real accuracy (conf_min={args.conf_min})")
    memory.prune(verbose=True)
    print(f"  After final prune: {memory.num_rules} rules")

    # ---- Evaluate ----
    centroids = memory.get_centroids()
    with torch.no_grad():
        rid_val, _ = memory.match(cv_val)
        val_acc = (head(centroids[rid_val]).argmax(dim=1) == y_val).float().mean().item()
        rid_test, _ = memory.match(cv_test)
        test_acc = (head(centroids[rid_test]).argmax(dim=1) == y_test).float().mean().item()
    print(f"\n[DONE] val_accuracy={val_acc:.4f}  test_accuracy={test_acc:.4f}")

    # ---- Export rules, WITH informative negations ----
    # A concept is "informative" if it's present in >=1 rule anywhere in this
    # set -- filters out concepts that never fire at all (would be dead-concept
    # noise, same issue diagnosed on the S1-based pipeline).
    present_by_rule = []
    for r in range(memory.num_rules):
        mu = memory.get_centroids()[r]
        present = [concept_names[i] for i in range(len(concept_names)) if mu[i].item() >= 0.5]
        present_by_rule.append(present)
    informative_concepts = set(c for plist in present_by_rule for c in plist)
    print(f"[INFO] {len(informative_concepts)}/{len(concept_names)} concepts are informative "
          f"(appear as 'present' in >=1 rule) -- only these are eligible to show as NOT.")

    rules_out = []
    for r in range(memory.num_rules):
        mu = memory.get_centroids()[r]
        present = present_by_rule[r]
        absent_informative = [
            concept_names[i] for i in range(len(concept_names))
            if mu[i].item() < 0.5 and concept_names[i] in informative_concepts
        ]
        gt_label_onehot = mu[len(concept_names):]
        gt_label_idx = int(gt_label_onehot.argmax().item())
        gt_label_conf = float(gt_label_onehot.max().item())
        majority_label = memory.get_labels()[r]
        rule_str = " AND ".join(present + [f"NOT {c}" for c in absent_informative])
        rules_out.append({
            "rule_id": r,
            "n": memory._n[r],
            "coherence": memory._compute_coherence(r),
            "confidence": memory._compute_conf(r),
            "accuracy": memory._correct[r] / memory._total_pred[r] if memory._total_pred[r] else None,
            "label_name": label_names[majority_label] if 0 <= majority_label < num_labels else "?",
            "gt_label_slot": label_names[gt_label_idx],
            "gt_label_slot_confidence": round(gt_label_conf, 4),
            "present_concepts": present,
            "absent_informative_concepts": absent_informative,
            "rule_string": rule_str,
        })
    rules_out.sort(key=lambda x: -x["confidence"])

    with open(out_dir / "icrl_rules.json", "w", encoding="utf-8") as f:
        json.dump(rules_out, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Top rules:")
    for r in rules_out:
        print(f"  [{r['confidence']:.3f}] n={r['n']:4d} acc={r['accuracy']}  {r['label_name']:10s}  {r['rule_string']}")

    metrics = {
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "num_rules": memory.num_rules,
        "theta": theta,
        "theta_merge": theta_merge,
        "args": vars(args),
        "rule_confidence_stats": {
            "mean": sum(memory.get_confidences()) / max(1, memory.num_rules),
            "min": min(memory.get_confidences()) if memory.num_rules else 0,
            "max": max(memory.get_confidences()) if memory.num_rules else 0,
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
