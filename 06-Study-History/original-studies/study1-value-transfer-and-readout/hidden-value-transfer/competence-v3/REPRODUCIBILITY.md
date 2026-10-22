# Reproducing the current hidden-value development study

This is the active original-question branch. The earlier readout paper and failed no-scratchpad configurations are separate preserved studies. Consult `RESEARCH-LOG.md` for current completion status; a prepared command is not evidence that its phase completed.

The primary and eligible-site whole-vector controls are complete. After the primary completion audit, `eligible_sites.py` reproduces the declared three-site revision and its 1,080 controls in a separate result folder. It inherits exact baseline and screen records with hashes and does not overwrite the original selection. `plot_selection.py` compares the two selection statistics. The subsequent single learned-method experiment and its separate reproduction instructions are in [subspace-v1](../subspace-v1/REPRODUCIBILITY.md).

## Execution environment and exact models

Use the existing project environment through `../../.venv/bin/python` from this directory. The measured runtime uses PyTorch 2.13.0, Transformers 5.16.1, CUDA 13 and a 6 GB NVIDIA GPU (6,141 MiB total). The selected model is Qwen3-4B-Instruct-2507-FP8, revision `8591804019c8b22094c3b5b4454e0edc05dffc98`. It uses the official FP8 weights and the pinned finegrained-FP8 v4 kernel, revision `8c178950e8710e26a2210e4e909cd02dc16c8715`, through isolated kernels 0.17.1. Full runtime metadata is saved with each phase.

Model files are in `../competence-v2/models`; manifests there include weight-provider checksum verification. The added Python dependency and kernel source are isolated in `../competence-v2/.packages` and `.cache`. The shared environment was not changed. The model-loading metadata correction is described in `../competence-v2/AMENDMENT-DENSE-LOADER.md`; it changes no arithmetic kernel or model weight. Offline kernel resolution requires the complete pinned kernel snapshot, not only its CUDA subdirectory.

The baseline comparison also used Qwen2.5-1.5B-Instruct, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, in bfloat16. It failed the initial competence gate. Do not silently replace checkpoints, precision or kernels and call the result an exact replication. Another execution stack requires its own instrumentation validation.

## Data, source and qualification

`development-with-source-controls.jsonl` is an exact copy of the augmented 36-pair development data in competence-v2. The parent `tasks.py` contains the deterministic arithmetic generator and strict parser; `prepare_sources.py` adds the same-leading-digit source. Both the older validation set and the 72-pair reserve remain untested.

`freeze-qwen4b-fewshot.json` hashes the selected scientific source and input files. `selected-configuration.json` records the first passing entry in the prospectively bounded calibration list. Initial system-only baseline outcomes for both candidates are under competence-v2. The selected calibration requires those prior failures and stops the list after its first success.

All model runs are local GPU work. From this directory, the reference commands are:

```bash
../../.venv/bin/python study_formats.py --candidate qwen4b --format fewshot --phase baseline
../../.venv/bin/python study_formats.py --candidate qwen4b --format fewshot --phase screen
../../.venv/bin/python study_formats.py --candidate qwen4b --format fewshot --phase controls
../../.venv/bin/python analyze_formats.py --candidate qwen4b-fewshot
../../.venv/bin/python audit_controls.py
../../.venv/bin/python actual_digit_control.py
../../.venv/bin/python check_results.py
```

These commands resume unique-key journals. Reused rows are not new independent runs. Preserve the original outputs; use a separately declared output directory/source version for an independent rerun. Do not delete prior results or alter a freeze to hide a change.

## Optional, validated screen optimization

The full reference screen is sufficient. The optimized screen reuses unaffected upstream block outputs for identical complete token prefixes, while recomputing all blocks downstream of the original intervention. It does not introduce a KV-cache intervention. Verification requires an already-recorded complete first reference pair:

```bash
../../.venv/bin/python fast_screen.py --verify
../../.venv/bin/python fast_screen.py --run
```

`fast-equivalence.json` and `fast-execution-freeze.json` record equivalence and source provenance. The original run retained 601 reference rows and resumed the same journal after exact equivalence passed. The fresh generation controls use the full reference path.

## Required output checks and interpretation

A completed screen has 5,184 unique records. Its primary control journal has 1,080 records (36 pairs × three selected sites × ten conditions), and constructed-source baselines have 432 records. There are 864 saved intervention-vector hashes and 216 unpatched/self generations that must equal their baseline token sequences. `check_results.py` audits these counts, score arithmetic, vector hashes, write locality and final-block causal invariants.

`analysis/summary.json` and `conditions.csv` report the original frozen analysis. `supplemental-interpretation.json` adds joint source-control success and an explicit origin ledger; see `CONTROL-INTERPRETATION-ADDENDUM.md` for the inherited raw journal's reference-field convention. `actual-donor-digit-summary.json` tests the actual first donor output token, including the one incorrectly answered donor. Identical textual-control conditions are explicitly reused.

Inference is exploratory: the same 12 donor groups support site selection and controls, and the bootstrap is not selection-adjusted. Competence and a successful activation write are prerequisites, not mechanism results. No conclusion about held-out performance, broad model families or language transfer is justified by these development outputs alone.
