"""
visualize_rule_contrib_cards.py - Inference card so sánh rule đóng góp vào phiếu
vote của System 2: cấu hình deployed (gate 0.5, fire thô) và cấu hình đề xuất
(tắt gate + chuẩn hóa theo độ phủ c_r, xem eval_coverage_normalized.py).

Mỗi thiết lập lấy 2 ảnh test:
  - "most changed": ảnh có rule fire ở cả hai cấu hình và tỷ trọng phiếu của rule
    ngắn (<= 1 concept) giảm nhiều nhất;
  - "agreement": ảnh ngẫu nhiên (seed 0) mà S1 và phiếu vote của cấu hình đề xuất
    cùng đoán đúng nhãn thật.

Usage (local, dùng cache có sẵn):
    python -m src.scripts.visualize_rule_contrib_cards
Output: outputs/explanation_coverage/cards/<setting>_<kind>.png
"""
from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.scripts.eval_ablation_crlmatched import strip_empty_rules
from src.scripts.eval_coverage_normalized import (
    CUB, FITZ_SCOPES, cub_inputs, firing, fitz_inputs, rule_meta,
)

OUT_DIR = Path("outputs/explanation_coverage/cards")
FITZ_IMG = Path("data/fitzpatrick17k/data/finalfitz17k")
CUB_IMG = Path("data/CUB/CUB_200_2011/images")
SETTINGS = {
    "crlmatched_fold0": dict(kind="fitz", scope="crlmatched", fold=0,
                             csv="data/fitzpatrick17k_crl_matched_cv_official/fold_0/test.csv"),
    "fulldata": dict(kind="fitz", scope="fulldata", fold=None, csv="data/fitzpatrick17k_prepared/test.csv"),
    "concept_only": dict(kind="fitz", scope="concept_only", fold=None,
                         csv="data/fitzpatrick17k_prepared_concept_only/test.csv"),
    "cub_class": dict(kind="cub", name="cub_class", csv="data/CUB_prepared/test.csv"),
    "cub_instance": dict(kind="cub", name="cub_instance", csv="data/CUB_prepared_instance/test.csv"),
}
TOP_K = 5


def short_name(c: str) -> str:
    return c.replace("has_", "").replace("::", ":")


def rule_text(r, share, labels, label_names, present, length):
    conc = ", ".join(short_name(c) for c in present[r][:6]) + (f" +{len(present[r]) - 6}" if len(present[r]) > 6 else "")
    if not present[r]:
        conc = "(no condition)"
    return f"{share * 100:5.1f}%  -> {label_names[int(labels[r])]}  [len {int(length[r])}]  {conc}"


def panel_lines(title, w_row, labels, label_names, present, length):
    total = w_row.sum()
    if total < 1e-6:
        return [title, "no rule fires -> fallback (nearest rule + aux head)"]
    pr = w_row / total
    vote = np.zeros(len(label_names))
    np.add.at(vote, labels.numpy(), pr.numpy())
    short = float(pr[length <= 1].sum())
    lines = [title,
             f"S2 vote: {label_names[int(vote.argmax())]} ({vote.max() * 100:.1f}%)",
             f"short-rule share: {short * 100:.1f}%   rules firing: {int((pr > 0.001).sum())}", ""]
    for r in torch.argsort(pr, descending=True)[:TOP_K].tolist():
        if pr[r] < 1e-4:
            break
        lines += textwrap.wrap(rule_text(r, float(pr[r]), labels, label_names, present, length), 70,
                               subsequent_indent="        ")
    return lines


def draw(path, img_path, header, left, right):
    fig = plt.figure(figsize=(17, 5.2))
    ax0 = fig.add_axes([0.01, 0.05, 0.22, 0.8])
    ax0.imshow(Image.open(img_path).convert("RGB"))
    ax0.axis("off")
    fig.text(0.01, 0.95, header, fontsize=10, va="top", family="monospace")
    for x, lines in ((0.25, left), (0.62, right)):
        fig.text(x, 0.86, "\n".join(lines), fontsize=8.2, va="top", family="monospace")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, s in SETTINGS.items():
        if s["kind"] == "fitz":
            mem, head, val, test, layout, num_classes = fitz_inputs(s["scope"], s["fold"])
            img_root = FITZ_IMG
        else:
            mem, head, val, test, layout, num_classes = cub_inputs(s["name"])
            img_root = CUB_IMG
        rows = list(csv.DictReader(open(s["csv"], encoding="utf-8")))
        assert [int(r["label_idx"]) for r in rows] == test["y"].tolist(), f"{name}: cache/test.csv order mismatch"
        label_names = [None] * num_classes
        for r in rows:
            label_names[int(r["label_idx"])] = r["label"] if "label" in r and r["label"] else None
        if None in label_names:
            from src.utils.cub_concepts import load_schema
            label_names = load_schema(Path(s["csv"]).parent)["label_names"]

        keys, offsets, dims, exclude = layout
        keys_used = [k for k in keys if k not in exclude]
        # Bỏ hoàn toàn rule rỗng (không concept điều kiện) ở CẢ hai cột, theo quyết
        # định 2026-11-25; sau này sẽ bỏ ngay trong quá trình build rule.
        n_before = mem.num_rules
        strip_empty_rules(mem, keys, offsets, dims, exclude, verbose=False)
        print(f"[{name}] removed {n_before - mem.num_rules} empty rule(s), {mem.num_rules} left")
        labels, length = rule_meta(mem, keys_used, offsets, dims)
        present = [[k for k in keys_used if mem._mu[r][offsets[k]].item() >= 0.5] for r in range(mem.num_rules)]
        conf = torch.tensor(mem.get_confidences(), dtype=torch.float32)
        valid = (labels >= 0).float()

        with torch.no_grad():
            w_dep = firing(mem, test["cv"], layout, num_classes, 0.5) * conf * valid
            fs_val = firing(mem, val["cv"], layout, num_classes, None)
            c_r = (fs_val.sum(dim=0) + 1.0) / (fs_val.shape[0] + 1.0)
            w_new = firing(mem, test["cv"], layout, num_classes, None) * conf * valid / c_r

        def short_share(w):
            return (w * (length <= 1).float()).sum(1) / w.sum(1).clamp(min=1e-12)

        both = (w_dep.sum(1) > 1e-6) & (w_new.sum(1) > 1e-6)
        drop = torch.where(both, short_share(w_dep) - short_share(w_new), torch.tensor(-1.0))
        vote_new = torch.zeros(len(test["y"]), num_classes).index_add_(1, labels.clamp(min=0), w_new).argmax(1)
        s1_pred = test["p1"].argmax(1)
        agree = ((s1_pred == test["y"]) & (vote_new == test["y"]) & both).nonzero().flatten().tolist()
        picks = [("most_changed", int(drop.argmax()))]
        if agree:
            picks.append(("agreement", int(rng.choice(agree))))

        for kind, i in picks:
            p1 = test["p1"][i]
            header = (f"{name}  |  test image #{i}  |  true: {label_names[int(test['y'][i])]}  |  "
                      f"S1: {label_names[int(p1.argmax())]} ({p1.max() * 100:.1f}%)  |  {kind}")
            left = panel_lines("DEPLOYED  (hard gate 0.5, raw fire x conf; no empty rules)", w_dep[i], labels,
                               label_names, present, length)
            right = panel_lines("PROPOSED  (no gate, fire / c_r x conf; no empty rules)", w_new[i], labels,
                                label_names, present, length)
            out = OUT_DIR / f"{name}_{kind}.png"
            draw(out, img_root / rows[i]["filename"], header, left, right)
            print(f"{out}  drop in short-rule share: {float(drop[i]) * 100:.1f} pp")


if __name__ == "__main__":
    main()
