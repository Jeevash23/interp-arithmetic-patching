"""Post hoc: what drives the log-ratio screen statistic in Qwen3-4B?

The screen (Study 1, 36 blocks x 4 prompt positions x 36 development pairs) ranked sites by the change in
log P(target) - log P(recipient answer) caused by the donor write, where the target is the complete counterfactual
answer and both probabilities are teacher-forced. That change splits exactly into the target's gain,
Delta log P(target), and the suppression of the recipient's answer, -Delta log P(recipient answer). This script
computes both parts for all 144 sites (means of group means, as in the original screen), the mean patched
probabilities, and each site's rank by the screen statistic.
"""
from __future__ import annotations

import collections
import math
import statistics

from common import HV, jsonl, write

SCREEN = HV / "competence-v3/results/qwen4b-fewshot/screen.jsonl"
TESTED = [(20, "a_end"), (9, "a_end"), (10, "a_end"), (14, "b_end"), (20, "first_stage_end"), (23, "prompt_end")]


def group_mean(rows, key):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["group"]].append(key(r))
    return statistics.mean(statistics.mean(v) for v in groups.values())


def main():
    by_site = collections.defaultdict(list)
    worst_identity = 0.0
    for r in jsonl(SCREEN):
        by_site[(r["layer"], r["role"])].append(r)
        gain = r["counterfactual"]["log_probability"] - r["unpatched_counterfactual"]["log_probability"]
        supp = r["unpatched_recipient"]["log_probability"] - r["recipient"]["log_probability"]
        worst_identity = max(worst_identity, abs(gain + supp - r["delta_log_ratio"]))
    sites = []
    for (layer, role), rows in by_site.items():
        sites.append({
            "layer": layer, "role": role, "pairs": len(rows),
            "delta_log_ratio": group_mean(rows, lambda r: r["delta_log_ratio"]),
            "target_gain": group_mean(rows, lambda r: r["counterfactual"]["log_probability"]
                                      - r["unpatched_counterfactual"]["log_probability"]),
            "recipient_suppression": group_mean(rows, lambda r: r["unpatched_recipient"]["log_probability"]
                                                - r["recipient"]["log_probability"]),
            "mean_p_target": statistics.mean(math.exp(r["counterfactual"]["log_probability"]) for r in rows),
            "mean_p_recipient": statistics.mean(math.exp(r["recipient"]["log_probability"]) for r in rows),
            "mean_p_target_unpatched": statistics.mean(math.exp(r["unpatched_counterfactual"]["log_probability"])
                                                       for r in rows),
            "pairs_p_target_above_half": sum(math.exp(r["counterfactual"]["log_probability"]) > 0.5 for r in rows),
        })
    sites.sort(key=lambda s: (-s["delta_log_ratio"], s["layer"]))
    for i, s in enumerate(sites, 1):
        s["rank_by_screen_statistic"] = i
        s["suppression_share"] = s["recipient_suppression"] / s["delta_log_ratio"] if s["delta_log_ratio"] else None
    top = sites[:3]
    big = [s for s in sites if s["delta_log_ratio"] > 10]
    summary = {
        "sites": len(sites),
        "identity_max_abs_error": worst_identity,
        "top3": [{k: s[k] for k in ("layer", "role", "delta_log_ratio", "target_gain", "recipient_suppression",
                                    "mean_p_target", "rank_by_screen_statistic")} for s in top],
        "tested_sites": [{k: s[k] for k in ("layer", "role", "rank_by_screen_statistic", "delta_log_ratio",
                                            "target_gain", "recipient_suppression", "mean_p_target")}
                         for s in sites if (s["layer"], s["role"]) in TESTED],
        "sites_with_ratio_above_10": len(big),
        "min_suppression_share_above_10": min(s["suppression_share"] for s in big),
        "max_mean_p_target_blocks_0_11": max(s["mean_p_target"] for s in sites if s["layer"] <= 11),
    }
    write("screen-decomposition", {"summary": summary, "sites": sites}, [SCREEN])


if __name__ == "__main__":
    main()
