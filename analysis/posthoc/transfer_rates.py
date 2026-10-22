"""Complete-transfer counts in every setting, with exact intervals and cluster-level bounds.

A success is a freely generated final answer equal to the recipient's remaining operation applied to the donor's
intermediate. For each count the script gives the exact (Clopper-Pearson) 95% interval and one-sided 95% upper
bound, treating attempts as independent, and the same quantities with the shared unit as the independent unit:
programs in Study 2 (two sampled streams each), program groups in the readout experiment, donors in the Qwen3-4B
experiments. For nonzero counts it adds a cluster bootstrap interval. Only the per-program bound of Study 2 was
specified in advance; the rest are post hoc.
"""
from __future__ import annotations

import collections

from common import HV, READOUT, S2, jsonl, rate, write

MAIN = S2 / "04-Results/final/qwen06-main-v1/records.jsonl"
CONTROLS = S2 / "04-Results/final/qwen06-controls-v1/records.jsonl"
DEVELOPMENT = S2 / "04-Results/development/qwen06-development-v2/records.jsonl"
READOUT_JOURNALS = [READOUT / f"results/{s}/interventions.jsonl" for s in ("seed-20262909", "seed-20263909")]
FOURB_SITES = [
    ("Block 20, first operand", HV / "competence-v3/results/qwen4b-fewshot/controls.jsonl", 20, "a_end"),
    ("Block 9, first operand", HV / "competence-v3/results/qwen4b-fewshot/controls.jsonl", 9, "a_end"),
    ("Block 10, first operand", HV / "competence-v3/results/qwen4b-fewshot/controls.jsonl", 10, "a_end"),
    ("Block 14, second operand", HV / "competence-v3/results/eligible-sites/controls.jsonl", 14, "b_end"),
    ("Block 20, first-stage end", HV / "competence-v3/results/eligible-sites/controls.jsonl", 20, "first_stage_end"),
    ("Block 23, answer prefix", HV / "competence-v3/results/eligible-sites/controls.jsonl", 23, "prompt_end"),
]
SUBSPACE = HV / "subspace-v1/results/controls.jsonl"
COMPONENTS = {"attention_output": HV / "component-v1/results/attention/controls.jsonl",
              "mlp_output": HV / "component-v1/results/mlp/controls.jsonl"}
DIAGNOSTIC = HV / "subspace-v1/value-digit-qualified-sources/results/controls.jsonl"


def clustered(items, unit):
    """items: iterable of (cluster id, success); returns rate() with clusters."""
    clusters = collections.defaultdict(lambda: [0, 0])
    for cid, ok in items:
        clusters[cid][0] += bool(ok)
        clusters[cid][1] += 1
    k = sum(a for a, _ in clusters.values())
    n = sum(b for _, b in clusters.values())
    return rate(k, n, [tuple(v) for v in clusters.values()], unit)


def study2():
    out = {}
    main = [r for r in jsonl(MAIN) if r["kind"] == "intervention" and r["condition"] == "activation"]
    for site in ("label_z-L20", "label_z-L26"):
        rs = [r for r in main if r["site"] == site]
        out[f"heldout/{site}"] = clustered(((r["group"], r["outcome"]["counterfactual"]) for r in rs), "program")
    both = [r for r in main if r["site"] in ("label_z-L20", "label_z-L26")]
    out["heldout/both_sites"] = clustered(((r["group"], r["outcome"]["counterfactual"]) for r in both), "program")
    ctrl = [r for r in jsonl(CONTROLS) if r["kind"] == "intervention"]
    for site in ("first_z-L26", "label_z-L26"):
        for cond in ("activation", "text_digit", "text_full"):
            rs = [r for r in ctrl if r["site"] == site and r["condition"] == cond]
            out[f"controls/{site}/{cond}"] = clustered(((r["group"], r["outcome"]["counterfactual"]) for r in rs),
                                                       "program")
    dev = [r for r in jsonl(DEVELOPMENT) if r["kind"] == "intervention" and r["condition"] == "activation"]
    for site in sorted({r["site"] for r in dev}):
        rs = [r for r in dev if r["site"] == site]
        out[f"development/{site}"] = clustered(((r["group"], r["outcome"]["counterfactual"]) for r in rs), "program")
    return out


def readout():
    rows = [r for path in READOUT_JOURNALS for r in jsonl(path)]
    out = {}
    for cond in ("activation", "text_first_digit", "text_full_value"):
        rs = [r for r in rows if r["condition"] == cond]
        out[cond] = clustered(((r["group"], r.get("success")) for r in rs), "program group")
    full = [r for r in rows if r["condition"] == "text_full_value"]
    out["text_full_value"]["generated"] = sum(r["status"] == "generated" for r in full)
    return out


def fourb():
    out = {}
    for label, path, layer, role in FOURB_SITES:
        rs = [r for r in jsonl(path) if r["layer"] == layer and r["role"] == role and r["condition"] == "donor"]
        out[f"whole_vector/{label}"] = clustered(((r["group"], r["counterfactual_success"]) for r in rs), "donor")
    rs = [r for r in jsonl(SUBSPACE) if r["condition"] == "donor"]
    out["rank8_projection/Block 23, answer prefix"] = clustered(((r["group"], r["counterfactual_success"]) for r in rs),
                                                                "donor")
    for name, path in COMPONENTS.items():
        rs = [r for r in jsonl(path) if r["layer"] == 23 and r["role"] == "prompt_end" and r["condition"] == "donor"]
        out[f"{name}/Block 23, answer prefix"] = clustered(((r["group"], r["counterfactual_success"]) for r in rs),
                                                           "donor")
    diag = jsonl(DIAGNOSTIC)
    for cond in ("donor", "same_value_coupled"):
        rs = [r for r in diag if r["condition"] == cond and r["discriminative"]]
        out[f"diagnostic/{cond}/discriminating"] = clustered(((r["donor"]["z"], r["full_success"]) for r in rs),
                                                             "donor value")
        rs = [r for r in diag if r["condition"] == cond]
        out[f"diagnostic/{cond}/coarse_all"] = clustered(((r["donor"]["z"], r["coarse_success"]) for r in rs),
                                                         "donor value")
    rs = [r for r in diag if r["condition"] == "text_actual_first_token"]
    out["diagnostic/text_actual_first_token/coarse_all"] = clustered(((r["donor"]["z"], r["coarse_success"])
                                                                      for r in rs), "donor value")
    return out


if __name__ == "__main__":
    inputs = [MAIN, CONTROLS, DEVELOPMENT, SUBSPACE, DIAGNOSTIC, *READOUT_JOURNALS, *COMPONENTS.values()]
    inputs += [p for _, p, _, _ in FOURB_SITES]
    write("transfer-rates", {"study2_qwen3_06b": study2(), "readout_qwen3_06b": readout(), "qwen3_4b": fourb()},
          inputs)
