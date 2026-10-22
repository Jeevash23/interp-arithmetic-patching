"""Post hoc: does a text edit of the first answer digit reproduce the activation write's output?

Qwen3-4B (Study 1, answer-only development pairs): for each whole-vector site and for the rank-8 projection and the
two component writes at block 23, compares the activation output with the output after forcing the donor's first
answer digit as text (the "first-digit text edit"), token for token. Also counts how often the first emitted token
equals that digit, and what happens after it.

Qwen3-0.6B readout experiment (Study 1): how many of the activation's complete-counterfactual successes also occur
under the first-digit text edit on the same sampled path.
"""
from __future__ import annotations

import collections

from common import HV, READOUT, jsonl, write

FOURB = [
    ("whole_vector", "Block 20, first operand", HV / "competence-v3/results/qwen4b-fewshot/controls.jsonl", 20, "a_end"),
    ("whole_vector", "Block 9, first operand", HV / "competence-v3/results/qwen4b-fewshot/controls.jsonl", 9, "a_end"),
    ("whole_vector", "Block 10, first operand", HV / "competence-v3/results/qwen4b-fewshot/controls.jsonl", 10, "a_end"),
    ("whole_vector", "Block 14, second operand", HV / "competence-v3/results/eligible-sites/controls.jsonl", 14, "b_end"),
    ("whole_vector", "Block 20, first-stage end", HV / "competence-v3/results/eligible-sites/controls.jsonl", 20, "first_stage_end"),
    ("whole_vector", "Block 23, answer prefix", HV / "competence-v3/results/eligible-sites/controls.jsonl", 23, "prompt_end"),
    ("rank8_projection", "Block 23, answer prefix", HV / "subspace-v1/results/controls.jsonl", 23, "prompt_end"),
    ("attention_output", "Block 23, answer prefix", HV / "component-v1/results/attention/controls.jsonl", 23, "prompt_end"),
    ("mlp_output", "Block 23, answer prefix", HV / "component-v1/results/mlp/controls.jsonl", 23, "prompt_end"),
]
TEXT_CONTROL = HV / "competence-v3/results/eligible-sites/controls.jsonl"   # first-digit text outputs at block 23


def recipient_answer(r):
    if "predictions" in r:
        return r["predictions"]["recipient_answer"]
    return r["comparison_reference"]["recipient_answer"]


def fourb_site(write_kind, path, layer, role, text_by_id):
    by = collections.defaultdict(dict)
    for r in jsonl(path):
        if r["layer"] == layer and r["role"] == role:
            by[r["id"]][r["condition"]] = r
    c = collections.Counter()
    for pid, conds in sorted(by.items()):
        act = conds["donor"]
        text = conds.get("text_first_answer_digit") or text_by_id[pid]
        forced = text["forced_output_ids"][0]
        rec = recipient_answer(act)
        c["pairs"] += 1
        c["identical_to_text_edit"] += act["output_ids"] == text["output_ids"]
        c["answer_changed"] += act["answer"] != rec
        noise = conds.get("isotropic_delta")
        if noise is not None:
            c["noise_pairs"] += 1
            c["noise_answer_changed"] += noise["answer"] != rec
        first_follows = act["output_ids"][:1] == [forced]
        c["first_token_is_donor_digit"] += first_follows
        rec_first = str(rec)[0]
        donor_digit_differs = text["text"][:1] != rec_first
        c["donor_digit_differs_from_recipient_digit"] += donor_digit_differs
        c["first_token_is_donor_digit_where_digits_differ"] += first_follows and donor_digit_differs
        if first_follows:
            if act["output_ids"] == text["output_ids"]:
                c["first_token_follows_then_identical"] += 1
            else:
                c["first_token_follows_then_diverges"] += 1
                c["diverges_with_second_token_of_donor_answer"] += (
                    len(act["output_ids"]) > 1 and len(text["output_ids"]) > 1
                    and act["output_ids"][1] != text["output_ids"][1]
                    and act["text"][:2] == str((act.get("source_task") or act["original_donor"])["y"])[:2])
    return {"write": write_kind, **dict(c)}


def fourb():
    text_by_id = {}
    for r in jsonl(TEXT_CONTROL):
        if r["layer"] == 23 and r["role"] == "prompt_end" and r["condition"] == "text_first_answer_digit":
            text_by_id[r["id"]] = r
    out = []
    for kind, label, path, layer, role in FOURB:
        res = fourb_site(kind, path, layer, role, text_by_id)
        res["site"] = label
        out.append(res)
    return out


def readout():
    pairs = collections.defaultdict(dict)
    for seed in ("seed-20262909", "seed-20263909"):
        for r in jsonl(READOUT / f"results/{seed}/interventions.jsonl"):
            pairs[(r["id"], r["seed"])][r["condition"]] = r
    act = {k for k, v in pairs.items() if v["activation"]["success"]}
    text = {k for k, v in pairs.items() if v["text_first_digit"].get("success")}
    # width = number of digits of the donor's intermediate; for one-digit donors the first digit is the whole value
    width = collections.Counter(pairs[k]["activation"]["width"] for k in act)
    return {"records": len(pairs), "activation_successes": len(act), "first_digit_text_successes": len(text),
            "successes_under_both": len(act & text), "activation_only_successes": len(act - text),
            "activation_successes_by_donor_width": {str(w): n for w, n in sorted(width.items())},
            "activation_successes_two_digit_also_text": sum(
                1 for k in act & text if pairs[k]["activation"]["width"] == 2)}


if __name__ == "__main__":
    inputs = [site[2] for site in FOURB] + [TEXT_CONTROL]
    inputs += [READOUT / f"results/{s}/interventions.jsonl" for s in ("seed-20262909", "seed-20263909")]
    write("first-digit-equivalence", {"qwen3_4b": fourb(), "qwen3_06b_readout": readout()}, inputs)
