"""Post hoc: at operand positions, does the write transfer the source's operand, which the model then computes with?

Qwen3-4B (Study 1, 36 answer-only development pairs under 12 donors). For every source condition at every
whole-vector site, counts outputs equal to the recipient's own two-step program run with the source's first operand
(first-operand sites) or second operand (second-operand site). Pairs in which the source operand equals the
recipient's, or in which the substituted program gives the recipient's own answer, are left out of that count.
"""
from __future__ import annotations

import collections

from common import HV, jsonl, rate, run_program, write

SITES = [
    ("Block 20, first operand", HV / "competence-v3/results/qwen4b-fewshot/controls.jsonl", 20, "a_end", "a"),
    ("Block 9, first operand", HV / "competence-v3/results/qwen4b-fewshot/controls.jsonl", 9, "a_end", "a"),
    ("Block 10, first operand", HV / "competence-v3/results/qwen4b-fewshot/controls.jsonl", 10, "a_end", "a"),
    ("Block 14, second operand", HV / "competence-v3/results/eligible-sites/controls.jsonl", 14, "b_end", "b"),
    ("Block 20, first-stage end", HV / "competence-v3/results/eligible-sites/controls.jsonl", 20, "first_stage_end", "a"),
    ("Block 23, answer prefix", HV / "competence-v3/results/eligible-sites/controls.jsonl", 23, "prompt_end", "a"),
]
SOURCES = ["donor", "same_z_decomposition", "same_z_different_answer", "different_z_same_answer",
           "same_leading_different_value", "random_source"]
CONTROLS = ["isotropic_delta", "unpatched", "same_example"]


def substituted(rec, source, operand):
    task = rec["recipient_task"]
    return run_program(task, **{operand: source[operand]})


def count(records, operand, source_of):
    """Matches to the substituted program among eligible pairs, with per-donor clusters."""
    clusters = collections.defaultdict(lambda: [0, 0])
    beyond_first_digit = 0
    for r in records:
        src = source_of(r)
        task = r["recipient_task"]
        pred = substituted(r, src, operand)
        if src[operand] == task[operand] or pred == task["y"]:
            continue
        hit = r["answer"] == pred
        clusters[r["group"]][0] += hit
        clusters[r["group"]][1] += 1
        if hit:
            beyond_first_digit += str(pred)[1:] != str(task["y"])[1:] or len(str(pred)) != len(str(task["y"]))
    k = sum(a for a, _ in clusters.values())
    n = sum(b for _, b in clusters.values())
    out = rate(k, n, [tuple(v) for v in clusters.values()], "donor group") if n else {"k": 0, "n": 0}
    out["matches_differing_beyond_first_digit"] = beyond_first_digit
    return out


def main():
    rows = []
    for label, path, layer, role, operand in SITES:
        recs = [r for r in jsonl(path) if r["layer"] == layer and r["role"] == role]
        by_cond = collections.defaultdict(list)
        for r in recs:
            by_cond[r["condition"]].append(r)
        donor_task = {r["id"]: r["source_task"] for r in by_cond["donor"]}
        site = {"site": label, "operand": operand, "sources": {}, "controls": {}}
        for cond in SOURCES:
            site["sources"][cond] = count(by_cond[cond], operand, lambda r: r["source_task"])
        # controls carry no source of their own: score them against the original donor's operand
        for cond in CONTROLS:
            site["controls"][cond] = count(by_cond[cond], operand, lambda r: donor_task[r["id"]])
        # the original donor's complete counterfactual coincides with the substituted program in these pairs
        site["donor_prediction_equals_counterfactual"] = sum(
            substituted(r, r["source_task"], operand) == r["predictions"]["counterfactual"] for r in by_cond["donor"])
        rows.append(site)
    write("operand-substitution", {"sites": rows}, [s[1] for s in SITES])


if __name__ == "__main__":
    main()
