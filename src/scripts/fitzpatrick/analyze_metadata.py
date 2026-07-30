"""
analyze_metadata.py — Khảo sát metadata Fitzpatrick17k + SkinCon trước khi triển khai
=======================================================================================

Mục đích: trả lời bằng SỐ LIỆU THẬT các câu hỏi đã nêu khi đánh giá rủi ro chuyển
từ MNIST-MultiConcept sang Fitzpatrick17k, KHÔNG cần train model:

  - Concept (SkinCon) có mất cân bằng tần suất mạnh không? Bao nhiêu concept hiếm?
  - Nhãn (three/nine/fine-grained) mất cân bằng tới mức nào?
  - 22% ảnh có SkinCon concept-label có phải mẫu ngẫu nhiên không, hay lệch theo
    nhãn/màu da so với toàn bộ dataset?
  - Có bao nhiêu ảnh trùng lặp/gần giống nhau (nguy cơ leak train/val/test)?
  - Cờ chất lượng (qc) của tác giả gốc đáng chú ý gì (vd. "Wrongly labelled")?

Chỉ đọc CSV + ảnh cục bộ tại data/fitzpatrick17k/, KHÔNG cần mạng, KHÔNG cần train.
Input:
    data/fitzpatrick17k/fitzpatrick17k.csv   (16,577 dòng: md5hash, fitzpatrick_scale,
                                               label, nine/three_partition_label, qc, url)
    data/fitzpatrick17k/skincon.csv          (3,690 dòng: 48 concept nhị phân/ảnh +
                                               cờ "Do not consider this image")
    data/fitzpatrick17k/data/finalfitz17k/*.jpg  (ảnh, tên file = md5hash.jpg)
Output:
    outputs/fitzpatrick_audit/metadata_audit.json   (toàn bộ số liệu, máy đọc được)
    outputs/fitzpatrick_audit/metadata_audit.png    (dashboard trực quan)
"""
from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.fitzpatrick_dedup import (
    DUP_HAMMING_LOOSE,
    DUP_HAMMING_STRICT,
    compute_dhashes,
    find_near_duplicates,
)

DATA_DIR = Path("data/fitzpatrick17k")
IMG_DIR = DATA_DIR / "data" / "finalfitz17k"
OUT_DIR = Path("outputs/fitzpatrick_audit")


# ─────────────────────────────────────────────────────────────
# 1) Load CSV
# ─────────────────────────────────────────────────────────────

def load_fitzpatrick_csv() -> dict[str, dict]:
    with open(DATA_DIR / "fitzpatrick17k.csv", encoding="utf-8") as f:
        return {r["md5hash"]: r for r in csv.DictReader(f)}


def load_skincon_csv() -> tuple[list[dict], list[str]]:
    with open(DATA_DIR / "skincon.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    concept_cols = [c for c in fieldnames if c not in ("", "ImageID", "Do not consider this image")]
    return rows, concept_cols


# ─────────────────────────────────────────────────────────────
# 2) Thống kê nhãn / concept / qc / coverage bias
# ─────────────────────────────────────────────────────────────

def analyze_labels(fp_rows: dict[str, dict]) -> dict:
    fine = Counter(r["label"] for r in fp_rows.values())
    three = Counter(r["three_partition_label"] for r in fp_rows.values())
    nine = Counter(r["nine_partition_label"] for r in fp_rows.values())
    fz = Counter(r["fitzpatrick_scale"] for r in fp_rows.values())
    qc = Counter(r["qc"] for r in fp_rows.values())

    fine_counts = sorted(fine.values())
    wrong_hashes = [h for h, r in fp_rows.items() if r["qc"].startswith("3 Wrongly")]

    return {
        "n_total": len(fp_rows),
        "fine_label": {
            "num_classes": len(fine),
            "min_count": fine_counts[0],
            "max_count": fine_counts[-1],
            "median_count": fine_counts[len(fine_counts) // 2],
            "imbalance_ratio_max_over_min": round(fine_counts[-1] / fine_counts[0], 1),
            "top10": fine.most_common(10),
            "bottom10": fine.most_common()[-10:],
        },
        "three_partition_label": dict(three),
        "nine_partition_label": dict(nine),
        "fitzpatrick_scale": dict(sorted(fz.items())),
        "qc_flags": dict(qc),
        "wrongly_labelled_hashes": wrong_hashes,
    }


def analyze_concepts(skincon_rows: list[dict], concept_cols: list[str]) -> dict:
    dnc_flagged = [r for r in skincon_rows if r["Do not consider this image"] == "1"]
    usable = [r for r in skincon_rows if r["Do not consider this image"] != "1"]

    freq = Counter()
    per_image_count = []
    for r in usable:
        c = 0
        for col in concept_cols:
            if r[col] == "1":
                freq[col] += 1
                c += 1
        per_image_count.append(c)

    freq_sorted = freq.most_common()
    rare = [(name, n) for name, n in freq_sorted if n < 20]

    return {
        "n_skincon_rows_total": len(skincon_rows),
        "n_flagged_do_not_consider": len(dnc_flagged),
        "n_usable": len(usable),
        "coverage_pct_of_full_dataset": round(len(usable) / 16577 * 100, 2),
        "num_concepts": len(concept_cols),
        "concept_freq_sorted_desc": freq_sorted,
        "concepts_with_lt20_images": rare,
        "n_rare_concepts": len(rare),
        "avg_concepts_per_image": round(float(np.mean(per_image_count)), 3),
        "median_concepts_per_image": float(np.median(per_image_count)),
        "per_image_concept_count_histogram": dict(Counter(per_image_count)),
    }


def analyze_coverage_bias(fp_rows: dict[str, dict], skincon_rows: list[dict]) -> dict:
    """SkinCon chỉ phủ ~22% ảnh — kiểm tra tập được phủ có lệch nhãn/màu da so với
    toàn bộ dataset không (nếu lệch, concept head sẽ học trên phân bố khác với
    phân bố suy luận thực tế)."""
    covered_hashes = {
        r["ImageID"].replace(".jpg", "")
        for r in skincon_rows
        if r["Do not consider this image"] != "1"
    }
    covered_hashes &= fp_rows.keys()

    def dist(hashes, key):
        c = Counter(fp_rows[h][key] for h in hashes)
        total = sum(c.values())
        return {k: round(v / total * 100, 2) for k, v in c.items()}

    all_hashes = set(fp_rows.keys())
    return {
        "n_covered": len(covered_hashes),
        "three_partition_label_full_pct": dist(all_hashes, "three_partition_label"),
        "three_partition_label_covered_pct": dist(covered_hashes, "three_partition_label"),
        "fitzpatrick_scale_full_pct": dist(all_hashes, "fitzpatrick_scale"),
        "fitzpatrick_scale_covered_pct": dist(covered_hashes, "fitzpatrick_scale"),
    }


# ─────────────────────────────────────────────────────────────
# 3) Near-duplicate detection (average hash, Hamming distance)
# ─────────────────────────────────────────────────────────────

def analyze_near_duplicates(fp_rows: dict[str, dict]) -> dict:
    image_files = sorted(p.name for p in IMG_DIR.glob("*.jpg"))
    n_files_on_disk = len(image_files)
    n_csv_rows = len(fp_rows)

    t0 = time.time()
    hashes = compute_dhashes(IMG_DIR, image_files)
    t1 = time.time()

    strict_pairs, loose_pairs = find_near_duplicates(hashes)
    t2 = time.time()

    def to_examples(pairs, k=15):
        out = []
        for i, j, d in sorted(pairs, key=lambda p: p[2])[:k]:
            fn_i, fn_j = image_files[i], image_files[j]
            h_i, h_j = fn_i.replace(".jpg", ""), fn_j.replace(".jpg", "")
            li = fp_rows.get(h_i, {}).get("label", "?")
            lj = fp_rows.get(h_j, {}).get("label", "?")
            out.append({"a": fn_i, "b": fn_j, "hamming_dist_256bit": d, "label_a": li, "label_b": lj,
                        "same_label": li == lj})
        return out

    def pct_involved(pairs):
        ids = {i for i, j, d in pairs} | {j for i, j, d in pairs}
        return round(len(ids) / n_files_on_disk * 100, 2)

    return {
        "n_files_on_disk": n_files_on_disk,
        "n_csv_rows": n_csv_rows,
        "files_match_csv_rows": n_files_on_disk == n_csv_rows,
        "hash_compute_seconds": round(t1 - t0, 1),
        "pairwise_compare_seconds": round(t2 - t1, 1),
        "hash_method": "dHash 16x16 (256-bit), thay cho aHash 8x8 ban đầu (xem docstring compute_dhashes)",
        "strict_threshold": DUP_HAMMING_STRICT,
        "loose_threshold": DUP_HAMMING_LOOSE,
        "n_strict_pairs": len(strict_pairs),
        "n_loose_pairs": len(loose_pairs),
        "pct_images_in_strict_dup": pct_involved(strict_pairs),
        "pct_images_in_loose_or_strict_dup": pct_involved(strict_pairs + loose_pairs),
        "example_strict_pairs_lowest_distance": to_examples(strict_pairs),
        "example_loose_pairs_lowest_distance": to_examples(loose_pairs),
    }


# ─────────────────────────────────────────────────────────────
# 4) Dashboard
# ─────────────────────────────────────────────────────────────

def plot_dashboard(labels: dict, concepts: dict, coverage: dict, dups: dict, save_path: Path) -> None:
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Fitzpatrick17k + SkinCon — Khảo sát metadata trước khi triển khai", fontsize=14, y=0.995)
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.28, top=0.93, bottom=0.06, left=0.08, right=0.97)

    # (a) concept frequency
    ax = fig.add_subplot(gs[0, :])
    names_counts = concepts["concept_freq_sorted_desc"]
    names = [n for n, _ in names_counts]
    counts = [c for _, c in names_counts]
    colors = ["#c0392b" if c < 20 else "#2980b9" for c in counts]
    ax.bar(range(len(names)), counts, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("số ảnh có concept")
    ax.set_title(f"Tần suất 48 concept SkinCon (đỏ = <20 ảnh, {concepts['n_rare_concepts']}/48 concept hiếm)")
    ax.set_yscale("log")

    # (b) label distribution full vs covered
    ax = fig.add_subplot(gs[1, 0])
    keys = sorted(coverage["three_partition_label_full_pct"].keys())
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w / 2, [coverage["three_partition_label_full_pct"][k] for k in keys], w, label="Toàn bộ (n=16577)", color="#7f8c8d")
    ax.bar(x + w / 2, [coverage["three_partition_label_covered_pct"][k] for k in keys], w, label="Có SkinCon (n=%d)" % coverage["n_covered"], color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("% ảnh")
    ax.set_title("Nhãn 3 lớp: toàn bộ vs. tập có concept-label")
    ax.legend(fontsize=8)

    # (c) fitzpatrick scale full vs covered
    ax = fig.add_subplot(gs[1, 1])
    fkeys = sorted(coverage["fitzpatrick_scale_full_pct"].keys(), key=lambda v: int(v))
    x = np.arange(len(fkeys))
    ax.bar(x - w / 2, [coverage["fitzpatrick_scale_full_pct"][k] for k in fkeys], w, label="Toàn bộ", color="#7f8c8d")
    ax.bar(x + w / 2, [coverage["fitzpatrick_scale_covered_pct"][k] for k in fkeys], w, label="Có SkinCon", color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels(["?" if k == "-1" else k for k in fkeys])
    ax.set_xlabel("Fitzpatrick skin type (-1 = không rõ)")
    ax.set_ylabel("% ảnh")
    ax.set_title("Màu da: toàn bộ vs. tập có concept-label")
    ax.legend(fontsize=8)

    # (d) fine label distribution (114-way) sorted
    ax = fig.add_subplot(gs[2, 0])
    # dùng min/median/max thay vì phân phối đầy đủ (không lưu full trong json nhẹ)
    stats = [labels["fine_label"]["min_count"], labels["fine_label"]["median_count"], labels["fine_label"]["max_count"]]
    x = np.arange(3)
    ax.bar(x, stats, width=0.5, color=["#c0392b", "#f39c12", "#2980b9"])
    ax.set_xticks(x)
    ax.set_xticklabels(["min", "median", "max"], fontsize=10)
    ax.set_xlim(-0.6, 2.6)
    ax.set_title(f"Phân bố 114 nhãn chi tiết (tỉ lệ max/min = {labels['fine_label']['imbalance_ratio_max_over_min']}x)")
    ax.set_ylabel("số ảnh / lớp")
    for i, v in enumerate(stats):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    # (e) near-dup summary as text tile
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    txt = (
        f"Ảnh trên đĩa: {dups['n_files_on_disk']} / CSV: {dups['n_csv_rows']} "
        f"({'khớp' if dups['files_match_csv_rows'] else 'LỆCH'})\n\n"
        f"dHash 256-bit, strict <= {dups['strict_threshold']}, loose <= {dups['loose_threshold']}\n"
        f"Cặp strict (gần chắc chắn trùng): {dups['n_strict_pairs']}\n"
        f"Cặp loose (đáng xem lại thủ công): {dups['n_loose_pairs']}\n"
        f"% ảnh dính strict: {dups['pct_images_in_strict_dup']}%\n"
        f"% ảnh dính strict+loose: {dups['pct_images_in_loose_or_strict_dup']}%\n\n"
        f"SkinCon coverage: {concepts['coverage_pct_of_full_dataset']}% "
        f"(usable={concepts['n_usable']}, do-not-consider={concepts['n_flagged_do_not_consider']})\n"
        f"Avg concept/ảnh: {concepts['avg_concepts_per_image']} (median {concepts['median_concepts_per_image']})\n\n"
        f"Ảnh bị tác giả gốc gắn cờ 'Wrongly labelled': {len(labels['wrongly_labelled_hashes'])}"
    )
    ax.text(0.0, 1.0, txt, transform=ax.transAxes, fontsize=10, va="top", family="monospace")
    ax.set_title("Tóm tắt rủi ro vận hành", loc="left")

    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CSVs...")
    fp_rows = load_fitzpatrick_csv()
    skincon_rows, concept_cols = load_skincon_csv()

    print("Analyzing labels/qc...")
    labels = analyze_labels(fp_rows)
    print("Analyzing concepts...")
    concepts = analyze_concepts(skincon_rows, concept_cols)
    print("Analyzing coverage bias...")
    coverage = analyze_coverage_bias(fp_rows, skincon_rows)
    print("Analyzing near-duplicates (hashing all images, may take a few minutes)...")
    dups = analyze_near_duplicates(fp_rows)

    summary = {"labels": labels, "concepts": concepts, "coverage_bias": coverage, "near_duplicates": dups}
    with open(OUT_DIR / "metadata_audit.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Plotting dashboard...")
    plot_dashboard(labels, concepts, coverage, dups, OUT_DIR / "metadata_audit.png")

    print(f"\nDone. Saved to {OUT_DIR}/metadata_audit.json and metadata_audit.png")


if __name__ == "__main__":
    main()
