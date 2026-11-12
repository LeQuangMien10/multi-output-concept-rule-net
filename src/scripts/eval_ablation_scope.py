"""
eval_ablation_scope.py - Ban tong quat 1-split cua eval_ablation_crlmatched.py
(chi tinh 2 o quan trong nhat: baseline = single/hard/simple, va "baseline
moi" de xuat = multi/soft/learned) cho fulldata/concept_only/pad_ufes20.

Scope 1-split KHONG co truc protocol (per_fold vs lofo) -- chi co 1 val/1
test nen hieu chinh luon la "per_fold"-equivalent (khong co fold nao khac
de muon). Dung lai dung ham tu eval_ablation_crlmatched.py (simple_hard,
learned_soft) de dam bao nhat quan cong thuc voi ban crlmatched.

Usage (local, inference only):
    python -m src.scripts.eval_ablation_scope --scope fulldata \\
        --icrl_root outputs/scope_rebuild/fulldata \\
        --icrl_dir outputs/scope_rebuild/fulldata/icrl_nolabelslot \\
        --data_dir data/fitzpatrick17k_prepared \\
        --img_dir data/fitzpatrick17k/data/finalfitz17k
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scripts.run_scope_rebuild import SCOPE_CONFIGS
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.models.fitzpatrick.system1 import soft_concept_vector
from src.models.ensemble_strategies import entropy_confidence
from src.scripts.eval_ablation_crlmatched import simple_hard, learned_soft, build_p2_multi


def parse_args():
    p = argparse.ArgumentParser(description="baseline vs multi/soft/learned, 1-split, scope bat ky.")
    p.add_argument("--scope", type=str, required=True, choices=list(SCOPE_CONFIGS.keys()))
    p.add_argument("--icrl_root", type=str, required=True)
    p.add_argument("--icrl_dir", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


@torch.no_grad()
def collect(loader, system1, memory, head, centroids, confidences, rule_n, rule_coh,
            num_classes, exclude_keys, concept_keys, concept_offsets, concept_dims, device):
    ys, p1s, p2ss, rcs, rns, rcohs, css, nfs = [], [], [], [], [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        out = system1(images)
        cv = soft_concept_vector(out)
        y = labels["label"].to(device)
        p1 = F.softmax(out["label"], dim=-1)

        rule_ids, _ = memory.match(cv)
        p2s = F.softmax(head(centroids[rule_ids]), dim=-1)

        vote = memory.predict_weighted_activation(
            cv, concept_keys, concept_offsets, concept_dims, num_classes=num_classes,
            exclude_keys=exclude_keys)

        ys.append(y); p1s.append(p1); p2ss.append(p2s)
        rcs.append(confidences[rule_ids]); rns.append(rule_n[rule_ids]); rcohs.append(rule_coh[rule_ids])
        css.append(vote["class_scores"]); nfs.append(vote["n_fired"].float())
    class_scores_cat = torch.cat(css)
    return dict(y=torch.cat(ys), p1=torch.cat(p1s), p2s=torch.cat(p2ss), rc=torch.cat(rcs),
                rule_n=torch.cat(rns), rule_coh=torch.cat(rcohs), class_scores=class_scores_cat,
                total_weight=class_scores_cat.sum(dim=1), n_fired=torch.cat(nfs))


def assemble(d, num_classes):
    p1, p2s = d["p1"], d["p2s"]
    p2m = build_p2_multi({"class_scores": d["class_scores"], "total_weight": d["total_weight"], "p2s": p2s})
    pred1, pred2s, pred2m = p1.argmax(1), p2s.argmax(1), p2m.argmax(1)
    y = d["y"]

    def margin(p, a, b):
        logp = p.clamp(min=1e-8).log()
        idx = torch.arange(len(p))
        return logp[idx, a] - logp[idx, b]

    feat6 = torch.stack([margin(p1, pred1, pred2s), margin(p2s, pred2s, pred1), d["rc"],
                          torch.log1p(d["rule_n"]), d["rule_coh"], entropy_confidence(p1)], dim=1)
    feat_agg5 = torch.stack([margin(p1, pred1, pred2m), margin(p2m, pred2m, pred1), d["total_weight"],
                              d["n_fired"], entropy_confidence(p1)], dim=1)

    return dict(y=y, p1=p1, p2s=p2s, p2m=p2m, pred1=pred1, pred2s=pred2s, pred2m=pred2m,
                rc=d["rc"], total_weight=d["total_weight"], s1_conf=p1.max(1).values,
                feat6=feat6, feat_agg5=feat_agg5)


def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    cfg = SCOPE_CONFIGS[args.scope]
    module = cfg["module"]
    label_names = cfg["icrl_args"]["label_names"].split(",")
    num_classes = len(label_names)

    if module == "fitzpatrick":
        from src.scripts.fitzpatrick.train_icrl import load_system1, make_loaders, build_full_concept_layout
        from src.utils.fitzpatrick_concepts import S1_LABEL_CONCEPT_KEY
    else:
        from src.scripts.pad_ufes20.train_icrl import load_system1, make_loaders, build_full_concept_layout
        from src.utils.pad_ufes20_concepts import S1_LABEL_CONCEPT_KEY

    icrl_root, icrl_dir = Path(args.icrl_root), Path(args.icrl_dir)
    system1, image_size = load_system1(icrl_root / "system1" / "best_model.pt", device)
    _, val_loader, test_loader = make_loaders(Path(args.data_dir), args.img_dir, image_size,
                                               args.batch_size, args.num_workers)
    concept_names = test_loader.dataset.concept_names
    concept_keys, concept_offsets, concept_dims, _ = build_full_concept_layout(concept_names, label_names)
    exclude_keys = {S1_LABEL_CONCEPT_KEY}

    memory = ICRLRuleMemory.load(icrl_dir / "icrl_rule_memory.pt", device=str(device))
    head = nn.Linear(memory.concept_dim, num_classes).to(device)
    head.load_state_dict(torch.load(icrl_dir / "prediction_head.pt",
                                     map_location=device, weights_only=False))
    head.eval()
    centroids = memory.get_centroids().to(device)
    confidences = torch.tensor(memory.get_confidences(), device=device)
    rule_n = torch.tensor([float(memory._n[r]) for r in range(memory.num_rules)], device=device)
    rule_coh = torch.tensor([float(memory._compute_coherence(r)) for r in range(memory.num_rules)], device=device)

    print(f"[{args.scope}] thu thap du lieu...")
    d_val = collect(val_loader, system1, memory, head, centroids, confidences, rule_n, rule_coh,
                     num_classes, exclude_keys, concept_keys, concept_offsets, concept_dims, device)
    d_test = collect(test_loader, system1, memory, head, centroids, confidences, rule_n, rule_coh,
                      num_classes, exclude_keys, concept_keys, concept_offsets, concept_dims, device)
    a_val, a_test = assemble(d_val, num_classes), assemble(d_test, num_classes)

    acc_baseline = simple_hard(a_val, a_test, "p2s", "rc")
    acc_new = learned_soft(a_val, a_test, "p2m", "feat_agg5")

    print(f"\n[{args.scope}]")
    print(f"  baseline (single/hard/simple):     {acc_baseline*100:.2f}%")
    print(f"  moi (multi/soft/learned):          {acc_new*100:.2f}%")
    print(f"  chenh lech:                        {(acc_new-acc_baseline)*100:+.2f}pp")


if __name__ == "__main__":
    main()
