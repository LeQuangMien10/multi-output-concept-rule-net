"""
run_scope_rebuild.py - Chay lai System~1 (ap dung fix lazy-collapse:
--monitor label_acc_gated) roi ICRL Stage 2/3 (code hien tai, co Wilson
score trong ICRLRuleMemory) cho 1 trong 3 scope da bi flag la "chua sach"
trong ke hoach S1/S2 coupling (xem memory project_s1s2_coupling_plan.md,
muc "Bug #4"): fulldata, concept_only (Fitzpatrick), hoac pad_ufes20.

Ly do can script nay: artifact cu cua 3 scope nay (thang 8-9/2026) co TRUOC
Wilson score va truoc fix lazy-collapse -- khong the "replay" lai bang code
hien tai de kiem chung (cong thuc confidence da doi), chi co the BUILD LAI
tu dau moi co so lieu dang tin cho 3 scope nay.

Moi scope la 1-split (KHONG phai 5-fold CV nhu CRL-matched -- xem cau hoi
"co nen lam 5-fold cho 3 scope nay khong" trong cung memory, khuyen nghi
KHONG lam vi day la scope phu, chua co script chia 5-fold cho chung), nen
chi can 1 lan train S1 + 1 lan build ICRL moi scope.

Giu NGUYEN het muc cac hyperparameter S1/ICRL da dung truoc day cho tung
scope (doc lai tu *_calibrated/metrics.json cu da luu), CHI doi phan S1 lien
quan truc tiep den fix lazy-collapse (--monitor, --min_concept_diversity) --
de so sanh cong bang "co fix" vs "khong fix", khong doi nhieu bien cung luc
(dung logic nhu khi tune conf_min=0.3 cho CRL-matched tuan truoc: chi doi 1
thu, giu moi thu khac nguyen).

Chay "lan luot" tung scope (--scope, khong phai --scopes) -- moi lenh ung
voi 1 Kaggle session rieng, theo thu tu de nghi: pad_ufes20 (nho nhat,
~1.6k anh train) -> concept_only (~3.2k) -> fulldata (~16.6k, lau nhat).
Output luon nam duoi <output_root>/<scope>/{system1,icrl} (co them 1 cap
thu muc theo TEN SCOPE) -- dung CHUNG 1 --output_root cho ca 3 lan chay la
AN TOAN, khong con ghi de len nhau nhu truoc (da sua bug nay).

Usage (Kaggle):
    python -m src.scripts.run_scope_rebuild --scope pad_ufes20 \\
        --output_root /kaggle/working/outputs/scope_rebuild
    python -m src.scripts.run_scope_rebuild --scope concept_only \\
        --output_root /kaggle/working/outputs/scope_rebuild
    python -m src.scripts.run_scope_rebuild --scope fulldata \\
        --output_root /kaggle/working/outputs/scope_rebuild

--exclude_label_slot: build THEM bien the "no label-slot" (xem memory
project_s1s2_coupling_plan.md muc "Bug #6") -- day la bien the CHINH THUC
paper dung lam System~2 cho ensemble, KHONG phai bien the mac dinh o tren
(dung de so sanh trong paper's tab:concept-metrics, khong deploy). Dung
--skip_s1 de tai dung S1 da train (buoc 1, --output_root phai TRUNG voi
lan chay truoc), chi build lai ICRL -- vd sau khi da chay lenh mac dinh
phia tren roi moi nhan ra can them bien the nay:
    python -m src.scripts.run_scope_rebuild --scope fulldata --skip_s1 \\
        --exclude_label_slot --output_root /kaggle/working/outputs/scope_rebuild
    python -m src.scripts.run_scope_rebuild --scope concept_only --skip_s1 \\
        --exclude_label_slot --output_root /kaggle/working/outputs/scope_rebuild

Usage (local, test nhanh truoc khi len Kaggle -- --s1_epochs it de kiem tra
script chay duoc, KHONG dung ket qua nay de ket luan gi):
    python -m src.scripts.run_scope_rebuild --scope pad_ufes20 \\
        --data_dir data/PAD-UFES-20_prepared --img_dir data/PAD-UFES-20/images \\
        --output_root outputs/scope_rebuild_test --s1_epochs 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ── Hyperparameter goc cua tung scope (doc lai tu outputs/*/metrics.json cu
#    da luu -- xem memory project_s1s2_coupling_plan.md de biet nguon) ──────

SCOPE_CONFIGS = {
    "fulldata": {
        "module": "fitzpatrick",
        "data_dir": "/kaggle/input/datasets/lquangmin/fitzpatrick17k-prepared",
        "img_dir": "/kaggle/input/datasets/lquangmin/fitzpatrick17k/data/finalfitz17k",
        "s1_args": {
            "backbone": "resnet50", "image_size": 224, "epochs": 30,
            "batch_size": 32, "lr": 0.0001, "backbone_lr_scale": 0.1,
            "grad_clip_norm": 1.0, "weight_decay": 0.0001, "lr_schedule": "none",
            "num_concepts": 35, "num_labels": 3,
        },
        "icrl_args": {
            "theta": "auto", "n_min_sweep": "5,10,25,50,75,100", "conf_min": 0.5,
            "head_epochs": 20, "head_lr": 0.001, "head_steps_per_epoch": 250,
            "label_names": "benign,malignant,non-neoplastic",
        },
    },
    "concept_only": {
        "module": "fitzpatrick",
        "data_dir": "/kaggle/input/datasets/lquangmin/fitzpatrick17k-prepared-concept-only",
        "img_dir": "/kaggle/input/datasets/lquangmin/fitzpatrick17k/data/finalfitz17k",
        "s1_args": {
            "backbone": "resnet50", "image_size": 224, "epochs": 30,
            "batch_size": 32, "lr": 0.0001, "backbone_lr_scale": 0.1,
            "grad_clip_norm": 1.0, "weight_decay": 0.01, "lr_schedule": "none",
            "num_concepts": 35, "num_labels": 3,
        },
        "icrl_args": {
            "theta": "auto", "n_min_sweep": "5,10,15,30,50", "conf_min": 0.5,
            "head_epochs": 20, "head_lr": 0.001, "head_steps_per_epoch": 250,
            "label_names": "benign,malignant,non-neoplastic",
        },
    },
    "pad_ufes20": {
        "module": "pad_ufes20",
        "data_dir": "/kaggle/working/PAD-UFES-20_prepared",
        "img_dir": "/kaggle/input/datasets/lquangmin/pad-ufes-20/images",
        "s1_args": {
            "backbone": "resnet50", "image_size": 224, "epochs": 30,
            "batch_size": 32, "lr": 0.0001, "backbone_lr_scale": 0.1,
            "grad_clip_norm": 1.0, "weight_decay": 0.01, "lr_schedule": "cosine",
            "num_concepts": 6, "num_labels": 6,
        },
        "icrl_args": {
            "theta": "auto", "n_min_sweep": "5,10,15,20,30", "conf_min": 0.5,
            "head_epochs": 20, "head_lr": 0.001, "head_steps_per_epoch": 250,
            "label_names": "ACK,BCC,MEL,NEV,SCC,SEK",
        },
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Rebuild S1 (fix lazy-collapse) + ICRL (Wilson score) cho 1 scope.")
    p.add_argument("--scope", type=str, required=True, choices=list(SCOPE_CONFIGS.keys()))
    p.add_argument("--data_dir", type=str, default=None,
                    help="Mac dinh = duong dan Kaggle goc cua scope (xem SCOPE_CONFIGS).")
    p.add_argument("--img_dir", type=str, default=None)
    p.add_argument("--output_root", type=str, required=True,
                    help="S1 -> <output_root>/system1, ICRL -> <output_root>/icrl.")
    p.add_argument("--min_concept_diversity", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    # Cho phep rut ngan epoch khi test local -- KHONG dung so nay tren Kaggle.
    p.add_argument("--s1_epochs", type=int, default=None,
                    help="Ghi de so epoch S1 (mac dinh = gia tri goc cua scope, 30). "
                         "Chi dung de test nhanh script chay duoc, khong dung de ket luan.")
    p.add_argument("--exclude_label_slot", action="store_true",
                    help="Build bien the ICRL KHONG dung label-slot S1 du doan luc cluster -- "
                         "day la bien the CHINH THUC paper dung lam System~2 cho ensemble o "
                         "ca 3 scope (xem tab:ensemble-strategies), KHONG phai bien the "
                         "mac dinh/--icrl_args cua SCOPE_CONFIGS (do la bien the so sanh, "
                         "khong phai bien the deploy). ICRL se ghi vao "
                         "<output_root>/<scope>/icrl_nolabelslot/ thay vi icrl/.")
    p.add_argument("--skip_s1", action="store_true",
                    help="Bo qua buoc 1 (train S1), chi build lai ICRL -- dung khi S1 hien tai "
                         "da co san (vd da train xong tu lan chay truoc trong cung output_root).")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    cfg = SCOPE_CONFIGS[args.scope]
    data_dir = args.data_dir or cfg["data_dir"]
    img_dir = args.img_dir or cfg["img_dir"]
    # Thu muc con PHAI theo ten scope -- neu khong, 3 lan chay voi cung
    # --output_root se ghi de len nhau (vd chay fulldata sau khi da chay
    # concept_only se xoa mat ket qua concept_only).
    output_root = Path(args.output_root) / args.scope
    s1_dir = output_root / "system1"
    icrl_dir = output_root / ("icrl_nolabelslot" if args.exclude_label_slot else "icrl")
    py = sys.executable
    module = cfg["module"]

    s1 = cfg["s1_args"]
    epochs = args.s1_epochs if args.s1_epochs is not None else s1["epochs"]

    print(f"{'=' * 70}\n[SCOPE={args.scope}] module={module}  exclude_label_slot={args.exclude_label_slot}\n"
          f"data_dir={data_dir}\nimg_dir={img_dir}\noutput_root={output_root}\n{'=' * 70}")

    # ── Buoc 1: train lai System~1 voi fix lazy-collapse (bo qua neu --skip_s1) ──
    if not args.skip_s1:
        run([py, "-m", f"src.scripts.{module}.train_system1_baseline",
             "--data_dir", data_dir, "--img_dir", img_dir,
             "--output_dir", str(s1_dir),
             "--backbone", s1["backbone"], "--image_size", str(s1["image_size"]),
             "--epochs", str(epochs), "--batch_size", str(s1["batch_size"]),
             "--lr", str(s1["lr"]), "--backbone_lr_scale", str(s1["backbone_lr_scale"]),
             "--grad_clip_norm", str(s1["grad_clip_norm"]), "--weight_decay", str(s1["weight_decay"]),
             "--lr_schedule", s1["lr_schedule"],
             "--num_concepts", str(s1["num_concepts"]), "--num_labels", str(s1["num_labels"]),
             "--monitor", "label_acc_gated", "--min_concept_diversity", str(args.min_concept_diversity),
             "--seed", str(args.seed)])
    else:
        print(f"[INFO] --skip_s1: tai dung S1 co san tai {s1_dir / 'best_model.pt'}")

    # ── Buoc 2: build lai ICRL Stage 2/3 bang code hien tai (Wilson score) ──
    icrl = cfg["icrl_args"]
    icrl_cmd = [py, "-m", f"src.scripts.{module}.train_icrl",
         "--data_dir", data_dir, "--img_dir", img_dir,
         "--system1_ckpt", str(s1_dir / "best_model.pt"),
         "--output_dir", str(icrl_dir),
         "--theta", icrl["theta"], "--n_min_sweep", icrl["n_min_sweep"],
         "--conf_min", str(icrl["conf_min"]), "--label_names", icrl["label_names"],
         "--head_epochs", str(icrl["head_epochs"]), "--head_lr", str(icrl["head_lr"]),
         "--head_steps_per_epoch", str(icrl["head_steps_per_epoch"]),
         "--seed", str(args.seed)]
    if args.exclude_label_slot:
        icrl_cmd.append("--exclude_label_slot")
    run(icrl_cmd)

    print(f"\n[DONE] {args.scope}: S1 -> {s1_dir}, ICRL -> {icrl_dir}")


if __name__ == "__main__":
    main()
