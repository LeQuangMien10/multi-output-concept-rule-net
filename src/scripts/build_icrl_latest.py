"""
build_icrl_latest.py - Dung artifact chinh thuc cua ban ICRL moi nhat (chot 2026-11-26).

Ban moi nhat: soft-AND khong gate, trong so fire/c_r * conf, khong rule rong,
conf_min = 0, (theta, n_min) chon theo MOT quy tac chung cho moi bo du lieu:
balanced accuracy cua phieu vote S2 tren val cao nhat, hoa thi it rule hon,
tren luoi theta in {0.80, 0.85, 0.90, 0.95, 0.97, 0.99} x n_min in {2, 3, 5, 10, 15, 25}
(CRL-matched: mot cap chung cho 5 fold). Luoi va lua chon do
src/scripts/select_theta_nmin.py tao (build tu logit S1 da cache, tat dinh).

Script nay chi chep ung vien duoc chon sang outputs/icrl_latest/<setting>/.
CUB theo anh nam ngoai luoi (quyet dinh co dua vao paper hay khong con mo):
artifact cua no giu nguyen tu lan dung truoc (theta 0.95, n_min 2, conf_min 0).

Usage:
    python -m src.scripts.select_theta_nmin --phase build
    python -m src.scripts.select_theta_nmin --phase select
    python -m src.scripts.build_icrl_latest
Output: outputs/icrl_latest/<setting>/{icrl_rule_memory.pt, prediction_head.pt, build_info.json}
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.scripts.select_theta_nmin import DATASETS, ROOT, tdir

OUT_ROOT = Path("outputs/icrl_latest")
CRITERION = "s2_bal"


def main():
    selection = json.load(open(ROOT / "selection.json"))
    manifest = {}
    for ds, names in DATASETS.items():
        theta, n_min = selection[ds]["picks"][CRITERION]
        cand = selection[ds]["grid"][f"{theta:.2f}/{n_min}"]
        for name in names:
            src = tdir(name, theta) / f"n_min_{n_min}"
            dst = OUT_ROOT / name
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("icrl_rule_memory.pt", "prediction_head.pt"):
                shutil.copy2(src / f, dst / f)
            sweep = json.load(open(tdir(name, theta) / "n_min_sweep_summary.json"))
            info = dict(source=str(src), theta=theta, theta_merge=min(theta + 0.04, 0.999), n_min=n_min,
                        conf_min=0.0, selection_rule="max val S2 balanced accuracy, ties -> fewer rules",
                        dataset_val_s2_bal=cand["val_s2_bal"],
                        num_rules=next(r["num_rules"] for r in sweep["results"] if r["n_min"] == n_min))
            with open(dst / "build_info.json", "w") as f:
                json.dump(info, f, indent=2)
            manifest[name] = info
            print(f"{name:<18s} theta={theta:.2f} n_min={n_min:<3d} rules={info['num_rules']:<5d} <- {src}")
    with open(OUT_ROOT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved -> {OUT_ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
