"""
visualize_inference_cards.py - R5: inference card cua ban ICRL moi nhat.

Moi thiet lap lay mot anh test cho moi truong hop giai thich (seed 0):
  (a) agreement, nhan cuoi dung;
  (b) disagreement, nhan cuoi theo S2 va dung;
  (c) disagreement, nhan cuoi theo S1 va dung;
  (d) khong rule nao dong gop (chi co tren CUB).
Card: anh, nhan that, S1, phieu vote S2, nhan cuoi, w2, va top rule theo P(r|x).

Usage:
    python -m src.scripts.visualize_inference_cards
Output: outputs/icrl_latest/cards/<setting>_<case>.png
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
import torch.nn as nn
from PIL import Image

from src.models import icrl_inference as ii
from src.models.icrl_rule_memory import ICRLRuleMemory
from src.scripts.eval_coverage_normalized import cub_inputs, fitz_inputs
from src.scripts.eval_icrl_latest import LATEST

OUT_DIR = LATEST / "cards"
FITZ_IMG = Path("data/fitzpatrick17k/data/finalfitz17k")
CUB_IMG = Path("data/CUB/CUB_200_2011/images")
SETTINGS = {
    "crlmatched_fold0": dict(load=lambda: fitz_inputs("crlmatched", 0), img=FITZ_IMG,
                             csv="data/fitzpatrick17k_crl_matched_cv_official/fold_0/test.csv",
                             title="Fitzpatrick17k, CRL-matched (fold 0)"),
    "fulldata": dict(load=lambda: fitz_inputs("fulldata", None), img=FITZ_IMG,
                     csv="data/fitzpatrick17k_prepared/test.csv", title="Fitzpatrick17k, full-data"),
    "concept_only": dict(load=lambda: fitz_inputs("concept_only", None), img=FITZ_IMG,
                         csv="data/fitzpatrick17k_prepared_concept_only/test.csv",
                         title="Fitzpatrick17k, concept-only"),
    "cub_class": dict(load=lambda: cub_inputs("cub_class"), img=CUB_IMG, csv="data/CUB_prepared/test.csv",
                      title="CUB-200-2011"),
}
CASE_TITLE = {"a": "(a) agreement", "b": "(b) disagreement, final follows the rules",
              "c": "(c) disagreement, final follows System 1", "d": "(d) no rule contributes"}
TOP_K = 5
# Anh khong dung lam minh hoa (vd. vung co the te nhi): (setting, case) -> chi so anh test.
EXCLUDE = {("crlmatched_fold0", "b"): {143, 126}}
COLORS = plt.get_cmap("tab10")


def short(c):
    return c.replace("has_", "").replace("::", ": ").replace("_", " ")


def label_names_for(name, rows, num_classes):
    names = [None] * num_classes
    for r in rows:
        if r.get("label"):
            names[int(r["label_idx"])] = r["label"]
    if None in names:
        from src.utils.cub_concepts import load_schema
        names = load_schema(Path(SETTINGS[name]["csv"]).parent)["label_names"]
    return [n.split(".", 1)[-1].replace("_", " ") for n in names]


def draw(path, img_path, title, info, rules, label_names):
    fig = plt.figure(figsize=(13, 4.6))
    ax_img = fig.add_axes([0.01, 0.06, 0.25, 0.80])
    ax_img.imshow(Image.open(img_path).convert("RGB"))
    ax_img.axis("off")
    fig.text(0.01, 0.95, title, fontsize=12, weight="bold", va="top")
    fig.text(0.28, 0.86, "\n".join(info), fontsize=9.5, va="top", family="monospace", linespacing=1.5)

    ax = fig.add_axes([0.55, 0.10, 0.20, 0.72])
    shares = [r["share"] for r in rules][::-1]
    cols = [COLORS(r["label"] % 10) for r in rules][::-1]
    ax.barh(range(len(rules)), shares, color=cols)
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels([f"r{r['id']}" for r in rules][::-1], fontsize=8.5)
    ax.set_xlim(0, max(shares) * 1.15 if shares else 1)
    ax.set_xlabel("contribution P(r | x)", fontsize=9)
    ax.set_title(f"top {len(rules)} rules", fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    lines = []
    for r in rules:
        conc = ", ".join(short(c) for c in r["concepts"][:5]) + (f" +{len(r['concepts']) - 5}" if len(r["concepts"]) > 5 else "")
        head = f"r{r['id']} -> {label_names[r['label']]}  ({r['share'] * 100:.1f}%, {len(r['concepts'])} concepts)"
        lines += [head] + textwrap.wrap("IF " + conc, 46, initial_indent="   ", subsequent_indent="   ")
    fig.text(0.77, 0.86, "\n".join(lines), fontsize=8, va="top", family="monospace", linespacing=1.35)
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, s in SETTINGS.items():
        _, _, val, test, layout, num_classes = s["load"]()
        d = LATEST / name
        mem = ICRLRuleMemory.load(d / "icrl_rule_memory.pt", device="cpu")
        head = nn.Linear(mem.concept_dim, num_classes)
        head.load_state_dict(torch.load(d / "prediction_head.pt", map_location="cpu", weights_only=False))
        head.eval()
        rows = list(csv.DictReader(open(s["csv"], encoding="utf-8")))
        assert [int(r["label_idx"]) for r in rows] == test["y"].tolist(), f"{name}: cache/test.csv order mismatch"
        names = label_names_for(name, rows, num_classes)

        keys, offsets, dims, exclude = layout
        keys_used = [k for k in keys if k not in exclude]
        mask = mem.present_mask(keys, offsets, dims, exclude)
        rlabels = ii.rule_labels(mem)
        a_val = ii.assemble(val["y"], val["p1"], ii.s2_vote(mem, head, val["cv"], layout, num_classes))
        vt = ii.s2_vote(mem, head, test["cv"], layout, num_classes)
        a_test = ii.assemble(test["y"], test["p1"], vt)
        final, w2 = ii.Combiner().fit(a_val).predict(a_test)
        case = ii.explanation_cases(a_test, final.numpy())
        y = test["y"].numpy()
        correct = final.numpy() == y

        for c in ("a", "b", "c", "d"):
            pool = np.where((case == c) & (correct | (c == "d")))[0]
            pool = np.array([j for j in pool if j not in EXCLUDE.get((name, c), set())], dtype=int)
            if len(pool) == 0:
                continue
            i = int(rng.choice(pool))
            p1, p2 = a_test["p1"][i], a_test["p2"][i]
            info = [CASE_TITLE[c], "",
                    f"true label : {names[y[i]]}",
                    f"System 1   : {names[int(p1.argmax())]} ({p1.max() * 100:.1f}%)",
                    f"rule vote  : {names[int(p2.argmax())]} ({p2.max() * 100:.1f}%)",
                    f"final      : {names[int(final[i])]}",
                    f"w2         : {float(w2[i]):.2f}" + ("  (agreement: S1 label kept)" if c == "a" else "")]
            rules = []
            if c != "d":
                post = vt["rule_post"][i]
                info.append(f"rules with P(r|x) > 1%: {int((post > 0.01).sum())}")
                for r in torch.argsort(post, descending=True)[:TOP_K].tolist():
                    if post[r] < 1e-3:
                        break
                    rules.append(dict(id=r, share=float(post[r]), label=int(rlabels[r]),
                                      concepts=[keys_used[j] for j in torch.nonzero(mask[r]).flatten().tolist()]))
            else:
                info.append("rule vote from the nearest rule (fallback)")
            out = OUT_DIR / f"{name}_{c}.png"
            draw(out, s["img"] / rows[i]["filename"], f"{s['title']}  |  test image #{i}", info, rules, names)
            print(f"{out}  image #{i}  {rows[i]['filename']}", flush=True)


if __name__ == "__main__":
    main()
