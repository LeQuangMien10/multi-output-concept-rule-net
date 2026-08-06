"""
train_crl_rrl.py — Train + evaluate the faithful CRL/RRL logical-layer
baseline (src/models/baselines/crl_rrl_net.py) on precomputed concept
vectors from ANY dataset (dataset-agnostic — takes .npz produced by
extract_concepts_multiconcept.py / extract_concepts_fitzpatrick.py).

Mirrors official CRL's training recipe (main.py + callbacks.py):
    - AdamW, CosineAnnealingLR over max_epochs
    - loss = CE(y_pred / exp(t), y) + l2_weight * l2_penalty()   (concept_loss
      term dropped — see crl_rrl_net.py docstring: no concept head here,
      concepts come frozen from our own S1)
    - clip_weights() after every optimizer step (ClipWeights callback)
    - rule extraction: one extra forward pass over train+val with
      bi_forward(count=True) to populate node_activation_cnt/forward_tot,
      then decode_rules()

Usage:
    python -m src.scripts.baselines.train_crl_rrl \
        --concepts_dir outputs/baselines/multiconcept_v1 \
        --output_dir outputs/baselines/multiconcept_v1/crl_rrl \
        --hidden_dims 64 --epochs 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.baselines.crl_rrl_net import CRLLogicNet
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--concepts_dir", type=str, required=True,
                    help="Dir with {train,val,test}_concepts.npz + meta.json (see extract_concepts_*.py)")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--hidden_dims", type=int, nargs="+", default=[64],
                    help="Union-layer widths, e.g. --hidden_dims 64 64 for a 2-layer net.")
    p.add_argument("--use_not", action="store_true", default=True)
    p.add_argument("--no_use_not", dest="use_not", action="store_false")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--l2_weight", type=float, default=5e-6, help="Matches official skin.yaml.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_split(concepts_dir: Path, split: str):
    data = np.load(concepts_dir / f"{split}_concepts.npz")
    return torch.tensor(data["concept_vec"], dtype=torch.float32), torch.tensor(data["label"], dtype=torch.long)


def iterate_batches(X, y, batch_size, shuffle, generator=None):
    n = X.shape[0]
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        b = idx[i:i + batch_size]
        yield X[b], y[b]


@torch.no_grad()
def evaluate(model: CRLLogicNet, X: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    logits = model(X.to(device))
    preds = logits.argmax(dim=1).cpu()
    return (preds == y).float().mean().item()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    concepts_dir = Path(args.concepts_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((concepts_dir / "meta.json").read_text(encoding="utf-8"))
    concept_names = meta["concept_names"]
    num_classes = meta["num_classes"]

    X_train, y_train = load_split(concepts_dir, "train")
    X_val, y_val = load_split(concepts_dir, "val")
    X_test, y_test = load_split(concepts_dir, "test")
    print(f"[INFO] train={X_train.shape} val={X_val.shape} test={X_test.shape} "
          f"concept_dim={X_train.shape[1]} num_classes={num_classes}")

    model = CRLLogicNet(
        concept_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dims=args.hidden_dims,
        use_not=args.use_not,
    ).to(device)
    print(f"[INFO] CRLLogicNet: hidden_dims={args.hidden_dims} use_not={args.use_not} "
          f"layers={[type(l).__name__ for l in model.layer_list]}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    gen = torch.Generator().manual_seed(args.seed)
    best_val_acc, best_state = -1.0, None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in iterate_batches(X_train, y_train, args.batch_size, shuffle=True, generator=gen):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb) + args.l2_weight * model.l2_penalty()
            optim.zero_grad()
            loss.backward()
            optim.step()
            model.clip_weights()
        scheduler.step()

        val_acc = evaluate(model, X_val, y_val, device)
        history.append({"epoch": epoch, "val_acc": val_acc, "lr": scheduler.get_last_lr()[0]})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs}  val_acc={val_acc:.4f}  best={best_val_acc:.4f}")

    model.load_state_dict(best_state)
    test_acc = evaluate(model, X_test, y_test, device)
    val_acc = evaluate(model, X_val, y_val, device)
    print(f"\n[DONE] best val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    # ── Rule extraction: pass over train+val to collect activation stats ──
    model.eval()
    model.reset_activation_stats(device)
    with torch.no_grad():
        model.bi_forward(torch.cat([X_train, X_val], dim=0).to(device), count=True)
    rules = model.decode_rules(concept_names)
    rules.sort(key=lambda r: -max(abs(v) for v in r["class_weights"].values()))

    rules_path = out_dir / "crl_rrl_rules.json"
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)
    print(f"[INFO] {len(rules)} rules extracted -> {rules_path}")
    print(f"[INFO] Top 10 rules by |weight|:")
    for r in rules[:10]:
        print(f"  [support={r['support']:.3f}] class={r['predicts_class']}  {r['description']}")

    torch.save(model.state_dict(), out_dir / "crl_rrl_model.pt")
    metrics = {
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "num_rules": len(rules),
        "args": vars(args),
        "history": history,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] metrics.json written to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
