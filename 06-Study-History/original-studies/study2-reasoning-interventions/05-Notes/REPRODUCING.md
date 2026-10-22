# Reproduction instructions

Work from the AI-04 project root. Do not invoke earlier studies' runners or inspect AI-02's reserved validation. Saved-data analysis needs no GPU. Qwen3 inference on this host requires supported host access to expose the NVIDIA device; do not change drivers.

## Environment and assets

Python 3.12.14 and `uv sync --frozen` recreate the dedicated `.venv` from `uv.lock`. A hash-pinned export is in `06-Study-History/environment/requirements.lock.txt`. No global installation is required. Direct packages include PyTorch 2.13.0, Transformers 5.16.1, NumPy 2.5.2 and SciPy 1.18.1.

Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca` is read from `../study1-value-transfer-and-readout/models/Qwen3-0.6B`. Its file hashes are in `06-Study-History/environment/model-files.json`. DeepSeek-R1-Distill-Qwen-1.5B revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` is at `03-Data/raw/models/DeepSeek-R1-Distill-Qwen-1.5B`; its manifest is beside the Qwen manifest. Both weight files were checked against public LFS hashes. For relocation, provide those exact assets or explicitly change a new reproduction workspace's checkpoint path. Never mutate an existing journal's settings.

## Saved-data reproduction

Each destination must be fresh. These commands create additional analysis outputs, without new model inference or changes to the saved estimates.

```bash
PYTHONPATH=02-Code .venv/bin/python -m pytest -q 02-Code/tests
.venv/bin/python 02-Code/final_analysis_v2.py --output /tmp/ai04-primary-reanalysis-new
.venv/bin/python 02-Code/secondary_analysis.py --output /tmp/ai04-secondary-reanalysis-new
.venv/bin/python 02-Code/verify_run.py final/qwen06-main-v1 --output /tmp/ai04-main-integrity-new.json
.venv/bin/python 02-Code/validate_study.py --output /tmp/ai04-full-validation-new
```

The final analysis asserts complete journals and planned counts, preserving program, site and stream pairing. The full validator independently recalculates selectors and the primary mean using standard-library arithmetic, checks deterministic splits and model hashes, verifies reused observations, and repeats both analyses in new paths. It expects the preserved original pilot norm failure and requires the qualified rerun to pass. This is internal computational reproduction, not independent external scientific review.

The executed record is `04-Results/analysis/validation-v1/checks.json`. The report's numerical inputs are `04-Results/analysis/final-v2/results.json` and `secondary-v1/secondary.json`. `make_figures.py` and `build_manuscript.py` consume those completed observations; neither selects sites.

## New inference from an executed snapshot

Do not run changed working-tree code into an old journal. Each `freeze.json` hashes configuration, source and declared inputs. Frozen files are under that run's `source/`. A snapshot derives its project root from its own file location, so running it directly without a prepared workspace would use the wrong data/model root.

The preparation helper verifies hashes, copies the snapshot into a new local workspace, supplies declared inputs, and links the checkpoint read only. It assigns a new run ID and regenerates baselines rather than reusing old observations. It prints the exact command, without executing inference.

```bash
.venv/bin/python 02-Code/prepare_reproduction.py final/qwen06-main-v1 --name main-reproduction-01
.venv/bin/python 02-Code/prepare_reproduction.py final/qwen06-controls-v1 --name controls-reproduction-01
```

Run the printed commands separately. Numerical code and sampling come from the executed snapshot; intended changes are only the destination, checkpoint-link path and regeneration of reused baselines. Preparation was exercised on `pilot/qwen06-random-repair-v2` in `04-Results/reproductions/preparation-check` and the frozen main run in `04-Results/reproductions/main-preparation-check`, without generating new evidence. The second-checkpoint panel's run is `development/deepseek-panel-v1`; the helper preserves its CPU/parser/format wrapper. The original pilot's failed random operator remains in its own snapshot and should not be silently replaced by the corrected version.

For the full chronological study, follow the separate protocols under `05-Notes/protocols`: Qwen qualification, pilot, random-control correction, development, frozen selector, 64-program main, 16-program controls, eight-program donor panel, eight-program lookup scope check, and separate second-checkpoint qualification and conditional four-program panel. `sequence.py` and `second_sequence.py` record execution order, but are not general restart commands for completed studies. Failed and interrupted run IDs remain distinct.

## Evidence and semantics

Rows retain exact prompt/prefix/output tokens, program/condition/stream IDs, parser and termination results, timing, and activation/source references. Assay files save full candidate token log probabilities and exact replay prefixes. Random norms are checked in float64 against 0.0001 absolute tolerance. Cache/logit and per-token-score audits use 0.0005; native/no-op sequence checks require exact identity.

The replay assay fixes all intervening original recipient text and scores a complete integer plus native EOS. Generation changes one historical node; the 768-token total assistant budget includes the replayed prefix. First-digit and full-value text edits have different lengths and aligned sampling streams. See the report for unavailable sites, hypothesis collisions and competent subsets.

The first 64 final programs are evaluated, the first 16 of those supply controls, and the last 16 of 80 have no model outputs. Branches are not independent programs. AI-02's separate reserved validation was never used.

PDF creation uses the bundled ReportLab runtime, without adding packages to the scientific environment. Pages and figures are rendered and visually inspected. Source/license provenance is in `01-Papers/sources` and `06-Study-History/environment`. Preserve substantial AI-assistance disclosure and the absence of independent external review in derivative reports.

Final runs also use the prospective `FINAL-EXECUTION-v3.md` optimization: an unchanged candidate-score dictionary is computed once per exact prefix/candidate set and reused after checking both. An audit of 312 repeated pre-final score calls found exact equality. Distinct intervention scores still execute; the frozen final collector and protocol record this difference from development's earlier collector.

The original frozen final_analysis.py and its final-v1 output are retained. Final-analysis v2 corrects a secondary answer-agreement edge case: null or unfinished outputs do not count as semantic agreement. It leaves the primary estimator unchanged; see ANALYSIS-CORRECTION-v2.md.
