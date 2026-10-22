"""Rerun each study's saved-output analysis and compare its outputs with the shipped files, byte for byte.

    python tools/replay_analyses.py          # after tools/unpack.py; exit status 1 if any output differs

The analyses read saved records only (no model, no network). Each runs in a temporary copy of the folders it
reads, so the release itself is never written: Studies 3, 5 and 6, the Study 1 journals from which Study 3's
development reference is computed, and the files the adapted Study 5 verifier needs to check freeze records.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
S3, S5, S6 = "studies/3-heldout-4b-confirmation", "studies/5-direction-and-span", "studies/6-olmo2-7b-multidigit"
COPY = [S3, S5, S6, "06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer",
        "tools", "release-changes.json"]
STEPS = [  # (study folder, command, outputs relative to the study folder)
    (S3, ["code/analyze.py", "development"], ["results/development-reference.json"]),
    (S3, ["code/analyze.py", "box"], ["results/box/analysis.json"]),
    (S3, ["code/analyze.py", "machine"], ["results/machine/analysis.json"]),
    (S3, ["code/analyze.py", "summary"], ["results/summary.json"]),
    (S5, ["code/analyze.py", "development"], ["results/development-analysis.json"]),
    (S5, ["code/analyze.py", "test"], ["results/test-analysis.json"]),
    (S5, ["code/analyze.py", "summary"], ["results/summary.json"]),
    (S5, ["code/posthoc.py"], ["results/posthoc.json"]),
    (S5, ["verify.py"], ["results/verification.json"]),
    (S6, ["code/gate_summary.py"], ["results/summary.json"]),
]


def main():
    missing = [s for s, _, outs in STEPS for o in outs if not (RELEASE / s / o).exists()]
    if missing:
        sys.exit("saved outputs missing; run tools/unpack.py (make unpack) first")
    failed = 0
    with tempfile.TemporaryDirectory(prefix="replay-") as tmp:
        work = Path(tmp)
        for rel in COPY:
            src, target = RELEASE / rel, work / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, target, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(src, target)
        for study, cmd, outs in STEPS:
            run = subprocess.run([sys.executable, "-B", *cmd], cwd=work / study, capture_output=True, text=True)
            if run.returncode != 0:
                failed += 1
                print(f"FAIL  {study}: {' '.join(cmd)} exited with {run.returncode}")
                print("      " + (run.stdout + run.stderr).strip().replace("\n", "\n      ")[-2000:])
                continue
            for o in outs:
                same = (work / study / o).read_bytes() == (RELEASE / study / o).read_bytes()
                failed += not same
                print(f"{'identical' if same else 'DIFFERENT'}  {study}/{o}  ({' '.join(cmd)})")
    print(f"{len(STEPS)} analyses rerun from the saved records: "
          + ("every output identical to the shipped file" if not failed else f"{failed} problem(s)"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
