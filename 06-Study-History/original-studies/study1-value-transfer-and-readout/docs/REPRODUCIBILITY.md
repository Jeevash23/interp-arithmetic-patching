# Reproducing and auditing this project

The tested environment is the local project-04 environment, reused through project-05's `.venv` symlink. No packages were installed into or removed from that shared environment. The exact installed distribution list is `results/environment-packages.txt`; principal versions and hardware measurements are in `results/environment.json`. The previous project's lockfile was copied to `configs/environment-source.uv.lock` for provenance. It is **not** a lockfile generated for this new project. `pyproject.toml` records the main direct dependencies. A clean installation on another machine has not been tested.

The checkpoint is public Qwen/Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`, with every downloaded model file hashed in `data/model-manifest.json`. The executed backend is native Hugging Face Qwen3 with PyTorch hooks, float32 and eager attention. The inspected TransformerLens adapter is not the executed backend. All model inference runs locally. Host GPU access is needed because this desktop sandbox initially hid the working device.

## Review saved results without generating new model outputs

From the project directory:

```sh
.venv/bin/python scripts/analyze.py v3
.venv/bin/python scripts/summarize_mechanisms.py
.venv/bin/python scripts/analyze_languages.py
.venv/bin/python scripts/plan_sample_size.py
.venv/bin/python scripts/verify_artifacts.py
```

The language analysis requires the language supplement to be complete. The sample-size and audit scripts require completed English controls. The planning script simulates a statistical design; it never fabricates language-model runs. Figures and machine-readable tables go to `results/v3/analysis/`.

## Execute or resume the development stages

```sh
.venv/bin/python scripts/run.py smoke
.venv/bin/python scripts/validate_extensions.py
.venv/bin/python scripts/screen_batched.py
.venv/bin/python scripts/textual_scratchpad.py
.venv/bin/python scripts/validate_control_operators.py
.venv/bin/python scripts/controls_batched.py --n 100 --sites 1
.venv/bin/python scripts/clamp_mediation.py
.venv/bin/python scripts/digit_controls.py
.venv/bin/python scripts/recover_sources.py
.venv/bin/python scripts/multilingual_baselines.py
.venv/bin/python scripts/budget_extension.py
```

Do not run several model processes concurrently on this 6 GB GPU. Each command resumes completed journal entries. Original source vectors and prior rows are preserved when resuming. Raw scientific outputs are append-only JSONL; aggregate tables can be rebuilt. The original serial screen was interrupted after 80 complete rows to validate batching; its traceback and rows are retained. Source reconstruction, if needed, compares each reconstructed tensor hash with its original pre-intervention hash.

For an independent rerun, make a separate project/results copy before execution. Existing journals intentionally reuse outputs and therefore are **not** independent repeat measurements. Do not change scientific configuration and resume into an existing journal. Preserve the original data, model manifest and source snapshot; select a new results branch and document any amendment. The current runners are explicitly development-stage runners and do not evaluate the reserved validation set.

Initial setup used `scripts/prepare.py` to inspect hardware and download the pinned public checkpoint, and `value_transfer.tasks.write_data()` to create deterministic data. Re-running preparation can update environment records; preserve the original records first. The generator refuses to overwrite changed dataset contents. Split checksums are audited without evaluating validation prompts.

## What each generation record means

- `prompt_ids`: native templated user input.
- `prefix_ids`: recipient input plus the unpatched assistant text replayed before intervention.
- `replayed_output_ids` and `fresh_output_ids`: distinguish supplied history from genuinely generated continuation.
- `semantics`: layer, tensor hook, fixed recipient position, generation step, rule, duration, cache use and precision.
- `donor_prefix_ids`, `donor_position`, `donor_source_hash`: identify the source activation; saved vector files support reconstruction.
- `parsed`, `prediction_matches`, `category`: strict final-answer parse, all matching causal predictions and mutually exclusive display category.
- `actual_source_predictions`: score crossed/random donors against their own values, as well as the originally designated donor target.
- `truncated` and `stop_reason`: distinguish budget exhaustion and native EOS.

Unavailable semantic positions remain invalid in complete-sample success denominators. They are separated from parser failures on actually generated continuations in the analysis tables. Repeated exact prompts under the same prompt-derived seed may reuse the identical generation; provenance fields record that reuse. They are not additional independent observations.

Source snapshots preserve v1/v2 failures, v3 before screening, and v3 before controls. Later changes to analysis, source-vector resume handling and the separate four-timing smoke suite do not rewrite earlier model outputs. Development amendments distinguish generation repair, earlier equation locations, text-prefix replacement, the pre-final number control, completed-number clamping, the language-competence supplement and first-digit versus whole-value text comparisons.


## Completion snapshot

The delivered code, configuration, documentation and manuscript are copied into `results/v3/source-at-completion/`. The SHA-256 inventory in [completion-manifest.json](../results/v3/completion-manifest.json) also covers raw journals, saved vectors, analyses, logs and data. Checkpoint weight hashes remain in `data/model-manifest.json`; the artifact audit checked those weights directly. This archive was made after all outcomes and is not a preregistration or held-out freeze.

The one-time `scripts/archive_completion.py` utility creates this snapshot only after a passing audit and refuses to overwrite an existing archive. Its copy is for source provenance; use the project root with its data and model files to execute the runners. The shared environment is represented by its recorded inventory and source lockfile, rather than copied into the archive.
