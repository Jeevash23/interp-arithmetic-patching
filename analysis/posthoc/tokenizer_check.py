"""Post hoc: how does the Qwen3 tokenizer split numbers, and do the studied checkpoints share it?

Reads the shipped tokenizer.json of Qwen3-0.6B, Qwen3-1.7B and Qwen3-4B-Instruct-2507-FP8 (kept in Study 1 after
the weights were removed). Records the pre-tokenizer pattern, every vocabulary entry that contains a digit, and
whether the three files are byte-identical.
"""
from __future__ import annotations

import json
import re

from common import S1, rel, sha256, write

FILES = {
    "Qwen3-0.6B": S1 / "models/Qwen3-0.6B/tokenizer.json",
    "Qwen3-1.7B": S1 / "followup-readout/checkpoint-replication/model/tokenizer.json",
    "Qwen3-4B-Instruct-2507-FP8": S1 / "hidden-value-transfer/competence-v2/models/qwen4b/tokenizer.json",
}


def main():
    out = {"files": {}, "identical": None}
    hashes = set()
    for name, path in FILES.items():
        tok = json.loads(path.read_text())
        vocab = tok["model"]["vocab"]
        added = [t["content"] for t in tok.get("added_tokens", [])]
        pattern = next(p["pattern"]["Regex"] for p in tok["pre_tokenizer"]["pretokenizers"] if p["type"] == "Split")
        with_digit = sorted((k for k in list(vocab) + added if re.search(r"[0-9]", k)), key=lambda k: vocab.get(k, -1))
        multi = [k for k in with_digit if re.fullmatch(r"Ġ?[0-9]{2,}", k)]
        out["files"][name] = {"path": rel(path), "sha256": sha256(path), "vocabulary": len(vocab),
                              "added_tokens": len(added), "split_pattern": pattern,
                              "digits_isolated": "|\\p{N}|" in pattern,
                              "entries_containing_a_digit": with_digit, "multi_digit_entries": len(multi)}
        hashes.add(out["files"][name]["sha256"])
    out["identical"] = len(hashes) == 1
    write("tokenizer-check", out, FILES.values())


if __name__ == "__main__":
    main()
