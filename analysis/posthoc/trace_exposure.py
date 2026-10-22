"""Post hoc: had the model already written the intermediate before the readout experiment's patched position?

Qwen3-0.6B readout experiment (Study 1). The write sits immediately before the labelled intermediate that follows
the thinking text. For every saved donor and recipient baseline trace (two sampling seeds), this counts whether the
thinking text (before </think>) contains the program's intermediate z, and its final answer y, as a standalone
integer. A literal occurrence shows exposure only; it does not show where the value was first computed.

Some baseline records reuse the generation of an earlier record with the identical prompt (field
reused_exact_prompt_from): each recipient serves two donors, and a few donor problems occur twice. Counts are given
per record and per distinct trace (reused records left out).
"""
from __future__ import annotations

import collections
import re

from common import READOUT, jsonl, write

SEEDS = ("seed-20262909", "seed-20263909")


def literal(value, text):
    return re.search(r"(?<!\d)%d(?!\d)" % value, text) is not None


def main():
    c = collections.Counter()
    distinct = collections.Counter()
    for seed in SEEDS:
        for r in jsonl(READOUT / f"results/{seed}/baselines.jsonl"):
            role = {"recipient_full": "recipient", "donor_full": "donor"}.get(r["condition"])
            if role is None:
                continue
            counters = [c] if r.get("reused_exact_prompt_from") else [c, distinct]
            for k in counters:
                k[f"{role}_traces"] += 1
            if "</think>" not in r["text"]:
                for k in counters:
                    k[f"{role}_unclosed_thinking"] += 1
                continue
            thinking = r["text"].split("</think>")[0]
            hz, hy = literal(r["task"]["z"], thinking), literal(r["task"]["y"], thinking)
            for k in counters:
                k[f"{role}_closed"] += 1
                k[f"{role}_z_in_thinking"] += hz
                k[f"{role}_y_in_thinking"] += hy
                k[f"{role}_both_in_thinking"] += hz and hy
    write("trace-exposure", {"readout": dict(c), "readout_distinct_traces": dict(distinct)},
          [READOUT / f"results/{s}/baselines.jsonl" for s in SEEDS])


if __name__ == "__main__":
    main()
