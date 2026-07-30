"""
measure_theta.py — Buoc 2: do theta (nguong match) cho ICRL tren concept vector that
====================================================================================

KHONG doan theta -- do truc tiep tren du lieu, dung cach da lam voi MNIST-MultiConcept
(phat hien theta=0.85 qua thap vi cos({2,3,4},{2,3,4,5})=0.866 > 0.85, phai nang len 0.93).

Chi dung cac anh co concept_mask=1 (SkinCon that) trong train.csv da xuat boi
prepare_dataset.py. Concept vector o day la 35-dim nhi phan (khong noi them
S1 label slot vi chua co model S1 nao chay that -- day la uoc luong tien-training,
theta cuoi cung nen do lai tren concept vector THAT SU S1 du doan sau khi train xong,
giong cach FULL_CV_DIM cua MultiConcept dung ca S1 label slot).

Logic: voi 2 anh CO CUNG concept-presence pattern -> cos luon = 1, khong thong tin.
Voi 2 anh KHAC pattern -> cos cang cao thi nguy co MATCH/MERGE nham cang lon.
=> theta nen dat CAO HON max(cos) cua các cap KHAC pattern, de tranh gop nham
2 pattern khac nhau vao 1 rule.

Usage:
    python -m src.scripts.fitzpatrick.measure_theta --data_dir data/fitzpatrick17k_prepared
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.fitzpatrick_concepts import CONCEPT_NAMES

OUT_DIR = Path("outputs/fitzpatrick_audit")


def parse_args():
    p = argparse.ArgumentParser(description="Measure ICRL theta on real Fitzpatrick concept vectors.")
    p.add_argument("--data_dir", type=str, default="data/fitzpatrick17k_prepared")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    return p.parse_args()


def load_concept_matrix(csv_path: Path) -> np.ndarray:
    with open(csv_path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["concept_mask"] == "1"]
    return np.array([[float(r[c]) for c in CONCEPT_NAMES] for r in rows], dtype=np.float32)


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    Xn = X / norm
    return Xn @ Xn.T


def main():
    args = parse_args()
    csv_path = Path(args.data_dir) / f"{args.split}.csv"
    X = load_concept_matrix(csv_path)
    n = X.shape[0]
    print(f"[INFO] {n} concept-labeled images in {args.split} split, {X.shape[1]} concepts.")

    # Bo anh all-zero concept (khong co concept nao active) -- cos khong xac dinh/vo nghia.
    nonzero_mask = X.sum(axis=1) > 0
    n_zero = (~nonzero_mask).sum()
    if n_zero:
        print(f"[INFO] Dropping {n_zero} images with all-zero concept vector (undefined cosine).")
    X = X[nonzero_mask]
    n = X.shape[0]

    cos = cosine_matrix(X)
    # Pattern giong het nhau (exact presence-pattern match) -> loai khoi phan tich collision.
    patterns = [tuple(row) for row in (X > 0.5).astype(int)]
    same_pattern = np.zeros((n, n), dtype=bool)
    pattern_to_idx: dict[tuple, list[int]] = {}
    for i, p in enumerate(patterns):
        pattern_to_idx.setdefault(p, []).append(i)
    for idx_list in pattern_to_idx.values():
        for i in idx_list:
            for j in idx_list:
                same_pattern[i, j] = True

    iu = np.triu_indices(n, k=1)
    cos_pairs = cos[iu]
    same_pairs = same_pattern[iu]

    cross_pattern_cos = cos_pairs[~same_pairs]
    same_pattern_cos = cos_pairs[same_pairs]

    print(f"[INFO] {len(pattern_to_idx)} distinct concept patterns among {n} images.")
    print(f"[INFO] cross-pattern pairs: {len(cross_pattern_cos)}, same-pattern pairs: {len(same_pattern_cos)}")

    percentiles = [50, 90, 95, 99, 99.5, 99.9, 100]
    pct_values = {p: float(np.percentile(cross_pattern_cos, p)) for p in percentiles}
    for p, v in pct_values.items():
        print(f"  cross-pattern cos percentile {p:>5}: {v:.4f}")

    # De xuat theta: ngay tren muc collision cao nhat quan sat duoc (percentile 99.9),
    # cong them bien an toan nho -- cung logic da dung cho MultiConcept (theta=0.93
    # dat tren muc collision do duoc 0.866).
    recommended_theta = min(0.999, pct_values[99.9] + 0.02)

    print(f"\n[RECOMMENDATION] theta khoi diem de xuat: {recommended_theta:.3f}")
    print("  (dat tren percentile 99.9 cua cos cross-pattern quan sat duoc + bien an toan 0.02)")
    print("  LUU Y: day la uoc luong TIEN-training tren concept GROUND-TRUTH 35-dim.")
    print("  Sau khi co S1 that, nen do lai tren concept vector S1 DU DOAN (co nhieu hon,")
    print("  co the can theta khac) -- giong cach theta=0.93 cua MultiConcept duoc chon")
    print("  dua tren du lieu that, khong phai gia dinh ly thuyet.")

    # Histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    bin_edges = np.linspace(0.0, 1.01, 51)   # fixed range -- tránh lỗi khi 1 nhóm có range gần 0 (vd. same-pattern luôn ~1.0)
    ax.hist(same_pattern_cos, bins=bin_edges, alpha=0.5, label=f"Same pattern (n={len(same_pattern_cos)}, luon ~1.0)", color="#7f8c8d")
    ax.hist(cross_pattern_cos, bins=bin_edges, alpha=0.7, label=f"Cross pattern (n={len(cross_pattern_cos)})", color="#2980b9")
    ax.axvline(recommended_theta, color="#c0392b", linestyle="--", label=f"Recommended theta={recommended_theta:.3f}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("So cap anh")
    ax.set_title(f"Fitzpatrick17k ({args.split}) — Phan bo cosine similarity giua concept vector that")
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "theta_measurement.png", dpi=150)
    plt.close(fig)

    result = {
        "split": args.split,
        "n_images": n,
        "n_zero_concept_dropped": int(n_zero),
        "n_distinct_patterns": len(pattern_to_idx),
        "n_cross_pattern_pairs": int(len(cross_pattern_cos)),
        "n_same_pattern_pairs": int(len(same_pattern_cos)),
        "cross_pattern_cos_percentiles": pct_values,
        "recommended_theta": recommended_theta,
        "note": "Do tren concept vector GROUND-TRUTH 35-dim (chua co S1 label slot). "
                "Do lai sau khi co S1 that truoc khi chot theta cuoi cung cho ICRL.",
    }
    with open(OUT_DIR / "theta_measurement.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[DONE] Saved outputs/fitzpatrick_audit/theta_measurement.{{json,png}}")


if __name__ == "__main__":
    main()
