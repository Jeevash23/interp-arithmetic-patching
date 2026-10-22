# Study 6: the multi-digit-token assay in OLMo-2-7B

**Question.** Study 4's question, with a larger model. Qwen3 writes one digit per token, so in Qwen3 "a write at the answer prefix sets the next token" and "a write moves the leading digit" make the same prediction. In a model whose tokenizer writes 0 to 999 as single tokens, the first account predicts copies of the donor's whole answer token, the second leading-digit splices and partial values, and a third (added from Baeumel et al., 2025) one digit changing inside the answer token. `PROTOCOL.md` explains why this is a new study rather than a fourth Study 4 candidate, and runs Study 4's frozen design (`STUDY4-PROTOCOL.md`) unchanged with one model.

**Status.** The study ended at the competence gate. OLMo-2-1124-7B-Instruct passed the tokenizer check, loaded in bfloat16 on a cloud L4, passed the instrumentation check, and failed the prespecified gate: like Study 4's two candidates, it answers each step correctly when it is asked alone and fails on the combined problem, most often when the second step is a multiplication. No development screen, `PREDICTIONS.md`, `FREEZE.json` or test inference exists. Whether a single-position write moves a whole multi-digit token, a leading digit or one digit inside the token stays open.

## What was done (17 October 2026)

- **Tokenizer check** (`code/tokenizer_check.py`, Study 4's code unchanged; `results/tokenizer-check.json`). Every integer from 0 to 999 is one token on its own, as the whole answer after the generation prompt and inside the problem sentence; all 309 numbers of the three splits are in range; all 3,600 operands of the split prompts are single tokens. The tokenizer files are byte-identical to the published ones (`results/tokenizer-files-check.json`: same git blob identifiers; three of the five files are identical to OLMo-2-0425-1B-Instruct's, already on disk).
- **Protocol** recorded at 10:57:25 UTC, before the weights were downloaded (`PROTOCOL-RECORD.json`).
- **Weights** downloaded inside Modal at revision `470b1fba1ae01581f270116362ee4aa1b97f4c84`; every file matched the published size and hash (`results/weights-check-olmo2-7b.json`). 7.30 billion bfloat16 parameters on one NVIDIA L4 (14.7 GB allocated at most).
- **Instrumentation** on the first four calibration pairs: passed (all 144 null-write logit changes 0.0; native and custom generation equal in 4 of 4).
- **Competence gate** on the 120 calibration pairs (Study 4's split, also used for Study 4's two candidates):

| Calibration measure (120 pairs) | OLMo-2-7B | Gate |
|---|---|---|
| Donor full problem | 60/120 (50.0%) | ≥ 90% |
| Recipient full problem | 53/120 (44.2%) | ≥ 90% |
| Recipient's second step with the donor's intermediate given | 120/120 | ≥ 90% |
| First-step probes, donor and recipient | 120/120 each | ≥ 90% |
| Pairs correct on all three central measures | 22/120 (18%) | ≥ 80% |

Every output parsed as an integer. Distinct calibration problems answered wrongly: 49 of 54 with a multiplication second step, 21 of 52 with addition and 17 of 53 with subtraction (Study 4: Phi-4-mini 26, 7 and 3; SmolLM3-3B 33, 19 and 14). Of the 87 wrong answers, 49 lie within 10 of the correct answer, 13 equal the intermediate and 25 are other numbers (a descriptive split defined after the gate result). `results/summary.json` collects these numbers (`code/gate_summary.py`).

## Files

- `PROTOCOL.md`, `PROTOCOL-RECORD.json`; `STUDY4-PROTOCOL.md` and `STUDY4-AMENDMENTS.md` (byte copies of Study 4's).
- `code/`: Study 4's code (byte copies, `code/COPIED-FROM-STUDY4.json`), `candidates.py` (the one new candidate), `modal_app.py` (the cloud stages), `gate_summary.py`.
- `data/`: Study 4's splits (byte copies). `tokenizers/olmo2-7b/`: the tokenizer and configuration files.
- `results/`: the tokenizer check, the Hub file list, the weights check, `olmo2-7b/calibration/` (gate, competence, instrumentation, baselines, runtime), `summary.json`, billing and transfer logs.
- `verify.py.draft`: Study 4's draft verifier, unused because no test ran.

## Reproducibility and cost

From this folder, with the Modal CLI: `~/.local/bin/modal run --detach code/modal_app.py::gate`, then `~/.local/bin/modal run code/modal_app.py::pull --stage gate` and `python3 -B code/gate_summary.py`. The container is built from AI-01's `uv.lock` (Python 3.12.10, torch 2.13.0 with CUDA 13.0, transformers 5.16.1, accelerate 1.14.0, NumPy 2.5.2). The gate took 143 seconds on the L4. The records brought back to the local machine were 0.04 MB (gzipped); the weights stayed in Modal. Cost: see the Study 5 README, which reports the spend of both cloud studies from Modal's billing report.
