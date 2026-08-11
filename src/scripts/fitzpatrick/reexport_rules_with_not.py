"""
reexport_rules_with_not.py - Re-decode an ALREADY-BUILT ICRL rule memory
(icrl_rule_memory.pt) with informative negations (NOT) shown, and the
s1_label_pred slot made explicit instead of hidden -- the fix discussed for
the "same concept pattern -> different label" interpretability problem.

No re-clustering, no S1 needed -- pure post-processing on the saved .pt, runs
in under a second. Point this at the S1-v3-based ICRL run
(outputs/fitzpatrick_icrl_crlmatched/, 10 rules, 79.4% test acc, built with
the better concept_macro_f1 S1 checkpoint) to see whether showing NOT
actually adds real distinguishing information now that concept detection
is richer than the original run.

Two fixes applied vs. the original export_rules():
  1. "Informative" negation filter: NOT concept only shown if that concept
     is 'present' in >=1 rule anywhere in the set (excludes S1's dead/never-
     learned concepts from cluttering every rule string with meaningless
     absences).
  2. s1_label_pred is shown explicitly (value + confidence) instead of
     stripped from the display -- and rules sharing the exact same visible
     concept+NOT pattern but different s1_label_pred/label are flagged as
     "circular" so a reader isn't misled into thinking 2 such rules
     contradict each other on morphology.

Usage:
    python -m src.scripts.fitzpatrick.reexport_rules_with_not \
        --rule_memory outputs/fitzpatrick_icrl_crlmatched/icrl_rule_memory.pt \
        --data_dir data/fitzpatrick17k_crl_matched \
        --output_dir outputs/fitzpatrick_icrl_crlmatched
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.models.icrl_rule_memory import ICRLRuleMemory

S1_LABEL_CONCEPT_KEY = "s1_label_pred"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rule_memory", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True,
                    help="Dir with meta.json (concept_names, label_names) matching this rule memory.")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def build_full_layout(concept_names: list[str], label_names: list[str]):
    full_keys = list(concept_names) + [S1_LABEL_CONCEPT_KEY]
    dims = {name: 1 for name in concept_names}
    dims[S1_LABEL_CONCEPT_KEY] = len(label_names)
    offsets, off = {}, 0
    for name in full_keys:
        offsets[name] = off
        off += dims[name]
    return full_keys, offsets, dims, off


def main():
    args = parse_args()
    meta = json.loads((Path(args.data_dir) / "meta.json").read_text(encoding="utf-8"))
    concept_names = meta["concept_names"]
    label_names = meta["label_names"]
    full_keys, full_offsets, full_dims, full_dim = build_full_layout(concept_names, label_names)

    memory = ICRLRuleMemory.load(args.rule_memory, device="cpu")
    assert memory.concept_dim == full_dim, f"concept_dim mismatch: memory={memory.concept_dim} vs meta={full_dim}"
    print(f"[INFO] Loaded {memory.num_rules} rules from {args.rule_memory}")

    # ---- Pass 1: decode every rule, find which concepts are ever "present" ----
    decoded_all = []
    for r in range(memory.num_rules):
        d = memory.decode_rule(rule_id=r, concept_keys=full_keys,
                                concept_offsets=full_offsets, concept_dims=full_dims)
        decoded_all.append(d)

    informative = set()
    for d in decoded_all:
        for k, v in d["slots"].items():
            if k != S1_LABEL_CONCEPT_KEY and v["value"] == "present":
                informative.add(k)
    print(f"[INFO] {len(informative)}/{len(concept_names)} concepts are informative "
          f"(present in >=1 rule): {sorted(informative)}")

    # ---- Pass 2: build rule strings with informative NOT + explicit label slot ----
    rules_out = []
    pattern_groups = defaultdict(list)  # (present tuple, not tuple) -> [rule indices]
    for r, d in enumerate(decoded_all):
        present = [k for k in concept_names if d["slots"][k]["value"] == "present"]
        absent_informative = [k for k in concept_names
                               if d["slots"][k]["value"] == "absent" and k in informative]
        s1_slot = d["slots"][S1_LABEL_CONCEPT_KEY]
        s1_idx = int(s1_slot["value"])
        s1_name = label_names[s1_idx] if 0 <= s1_idx < len(label_names) else "?"

        rule_string = " AND ".join(present + [f"NOT {k}" for k in absent_informative])
        majority_label = label_names[d["label"]] if 0 <= d["label"] < len(label_names) else "?"

        key = (tuple(present), tuple(absent_informative))
        pattern_groups[key].append(r)

        rules_out.append({
            "rule_id": r,
            "n": d["n"],
            "coherence": d["coherence"],
            "confidence": d["confidence"],
            "label_name": majority_label,
            "s1_label_pred": s1_name,
            "s1_label_pred_confidence": round(s1_slot["confidence"], 4),
            "present_concepts": present,
            "absent_informative_concepts": absent_informative,
            "rule_string": rule_string,
            "_pattern_key": key,
        })

    # ---- Flag circular pairs: same visible pattern, different label ----
    for rules_in_group in pattern_groups.values():
        if len(rules_in_group) < 2:
            continue
        labels_in_group = set(rules_out[i]["label_name"] for i in rules_in_group)
        if len(labels_in_group) > 1:
            for i in rules_in_group:
                rules_out[i]["circular_group"] = [j for j in rules_in_group if j != i]

    for r in rules_out:
        r.pop("_pattern_key")

    rules_out.sort(key=lambda x: -x["confidence"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "icrl_rules_with_not.json", "w", encoding="utf-8") as f:
        json.dump(rules_out, f, indent=2, ensure_ascii=False)

    n_circular = sum(1 for r in rules_out if "circular_group" in r)
    print(f"\n[INFO] {n_circular}/{len(rules_out)} rules flagged as part of a 'circular' "
          f"(same visible pattern, different label) group -- separated only by s1_label_pred.")

    print(f"\n[INFO] Full rule list:")
    for r in rules_out:
        flag = "  <== CIRCULAR (S1-label-driven, not concept-driven)" if "circular_group" in r else ""
        print(f"  [{r['confidence']:.3f}] n={r['n']:4d} label={r['label_name']:10s} "
              f"(S1_label_pred={r['s1_label_pred']}, conf={r['s1_label_pred_confidence']:.3f})  "
              f"{r['rule_string']}{flag}")

    print(f"\n[INFO] Saved to {out_dir / 'icrl_rules_with_not.json'}")


if __name__ == "__main__":
    main()
