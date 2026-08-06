"""
train_classic_baselines.py — Two classic, well-established interpretable-ML
reference points, trained on the SAME precomputed concept vectors as the
CRL-RRL and ICRL comparisons (dataset-agnostic, takes the .npz produced by
extract_concepts_*.py):

  1. Decision Tree (CART, src/models/baselines/decision_tree.py) — textbook
     rule-induction baseline. Every root-to-leaf path IS an if/then rule
     (num_rules = num_leaves), giving a non-neural "how much does ICRL's
     clustering / CRL's logic layers buy you over the oldest interpretable-
     ML trick in the book" reference point.

  2. Logistic Regression (plain linear softmax over the concept vector,
     trained with gradient descent — no external deps needed) — a
     Concept-Bottleneck-Model-style control with NO rule structure at all.
     Answers "does structuring predictions into rules even help over an
     unstructured linear classifier on the same concepts?"

(scikit-learn would normally provide both, but repeated PyPI downloads on
this network kept resetting mid-transfer — both are simple enough to
reimplement directly instead of blocking on that.)

Usage:
    python -m src.scripts.baselines.train_classic_baselines \
        --concepts_dir outputs/baselines/multiconcept_v1 \
        --output_dir outputs/baselines/multiconcept_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.baselines.decision_tree import DecisionTreeClassifier
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--concepts_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--tree_max_depth", type=int, default=6)
    p.add_argument("--tree_min_samples_leaf", type=int, default=10)
    p.add_argument("--lr_epochs", type=int, default=200)
    p.add_argument("--lr_lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_split(concepts_dir: Path, split: str):
    data = np.load(concepts_dir / f"{split}_concepts.npz")
    return data["concept_vec"].astype(np.float64), data["label"].astype(np.int64)


def run_decision_tree(Xtr, ytr, Xv, yv, Xte, yte, concept_names, args) -> dict:
    tree = DecisionTreeClassifier(
        max_depth=args.tree_max_depth,
        min_samples_leaf=args.tree_min_samples_leaf,
    ).fit(Xtr, ytr)

    val_acc = float((tree.predict(Xv) == yv).mean())
    test_acc = float((tree.predict(Xte) == yte).mean())
    rules = tree.extract_rules(concept_names)
    rules.sort(key=lambda r: -r["n"])

    print(f"[Decision Tree] val_acc={val_acc:.4f}  test_acc={test_acc:.4f}  "
          f"leaves={tree.n_leaves}  depth={tree.depth()}")
    return {
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "num_rules": tree.n_leaves,
        "tree_depth": tree.depth(),
        "rules": rules,
        "args": {"max_depth": args.tree_max_depth, "min_samples_leaf": args.tree_min_samples_leaf},
    }


def run_logistic_regression(Xtr, ytr, Xv, yv, Xte, yte, num_classes, args) -> dict:
    set_seed(args.seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    model = torch.nn.Linear(Xtr.shape[1], num_classes)
    optim = torch.optim.LBFGS(model.parameters(), lr=args.lr_lr, max_iter=args.lr_epochs,
                               line_search_fn="strong_wolfe")

    def closure():
        optim.zero_grad()
        loss = F.cross_entropy(model(Xtr_t), ytr_t)
        loss.backward()
        return loss

    optim.step(closure)

    with torch.no_grad():
        val_acc = float((model(Xv_t).argmax(dim=1).numpy() == yv).mean())
        test_acc = float((model(Xte_t).argmax(dim=1).numpy() == yte).mean())

    print(f"[Logistic Regression / CBM linear probe] val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")
    return {
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "num_rules": None,  # not rule-based by design — the no-structure control
        "args": {"lr_epochs": args.lr_epochs, "lr_lr": args.lr_lr, "optimizer": "LBFGS"},
    }


def main():
    args = parse_args()
    concepts_dir = Path(args.concepts_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((concepts_dir / "meta.json").read_text(encoding="utf-8"))
    concept_names = meta["concept_names"]
    num_classes = meta["num_classes"]

    Xtr, ytr = load_split(concepts_dir, "train")
    Xv, yv = load_split(concepts_dir, "val")
    Xte, yte = load_split(concepts_dir, "test")
    print(f"[INFO] train={Xtr.shape} val={Xv.shape} test={Xte.shape} num_classes={num_classes}")

    tree_result = run_decision_tree(Xtr, ytr, Xv, yv, Xte, yte, concept_names, args)
    with open(out_dir / "decision_tree_metrics.json", "w", encoding="utf-8") as f:
        json.dump(tree_result, f, indent=2)

    lr_result = run_logistic_regression(Xtr, ytr, Xv, yv, Xte, yte, num_classes, args)
    with open(out_dir / "logistic_regression_metrics.json", "w", encoding="utf-8") as f:
        json.dump(lr_result, f, indent=2)

    print(f"[INFO] Results written to {out_dir}")


if __name__ == "__main__":
    main()
