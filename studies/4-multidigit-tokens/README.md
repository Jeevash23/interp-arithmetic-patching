# Study 4: the answer-only assay in a model with multi-digit number tokens

**Question.** Qwen3 writes one digit per token, so "a write at the answer prefix sets the next token" and "a write moves the leading digit" make the same prediction there. In a model whose tokenizer writes 0 to 999 as single tokens, the first account predicts copies of the donor's whole answer token, and the second predicts leading-digit splices and partial values. `PROTOCOL.md` fixes the design, the competence gate and the decision rules.

**Status.** The study ended at the competence gate. The first two candidates, Phi-4-mini-instruct and SmolLM3-3B, both loaded, passed instrumentation and failed the prespecified gate: each solves the two steps separately but not the whole problem in one answer-only reply. The third candidate, OLMo-2-0425-1B-Instruct, was not tried: on 10 October, after SmolLM3-3B failed the gate, the author decided not to try it (`AMENDMENTS.md`, decision of 10 October). No development screen, prediction file, freeze or test inference exists, so the frozen-protocol order is intact. The question remains open: two of the three candidates failed the gate, and one was not tried.

## What was done

- **Splits** (`code/make_splits.py`, `data/split-checks.json`): calibration 120 pairs (40 donor groups), development 36 (12), test 144 (48). None shares a problem with another split or with the 2,588 problems in 3,250 earlier project files.
- **Tokenizer check** (`code/tokenizer_check.py`, `results/tokenizer-check.json`). All three candidates pass on their tokenizer files alone: every integer from 0 to 999 is one token on its own, as the whole answer after the generation prompt and inside the problem sentence; every number in the splits is in range. The files match the published files at the pinned revisions (`results/hub-files.json`).
- **Protocol** recorded at 14:31 UTC and amended twice before any candidate output (`PROTOCOL-RECORD.json`, `AMENDMENTS.md`).
- **Candidates 1 and 2.** Phi-4-mini-instruct (revision `cfbefacb99257ffa30c83adab238a50856ac3083`) and SmolLM3-3B (revision `a07cc9a04f16550a088caea529712d1d335b0ac1`) both loaded with on-the-fly FP8 quantization (Phi-4-mini: 3.22 billion FP8 and 0.61 billion bfloat16 parameters, 0.77 GB of GPU memory left; SmolLM3: 2.81 and 0.26 billion, 2.3 GB left). Both passed instrumentation on four calibration pairs (all 144 null-write logit changes 0.0; native and custom generation equal), and both failed the competence gate (`results/<candidate>/calibration/`).

| Calibration measure (120 pairs) | Phi-4-mini | SmolLM3-3B | Gate |
|---|---|---|---|
| Donor full problem | 93/120 (77.5%) | 63/120 (52.5%) | ≥ 90% |
| Recipient full problem | 92/120 (76.7%) | 72/120 (60.0%) | ≥ 90% |
| Recipient's second step with the donor's intermediate given | 120/120 | 120/120 | ≥ 90% |
| First-step probes, donor and recipient | 120/120 each | 120/120 each | ≥ 90% |
| Pairs correct on all three central measures | 72/120 (60%) | 38/120 (32%) | ≥ 80% |

Every output parsed as an integer. Both models answer each step correctly when it is asked alone and fail on the combined problem, most often when the second step is a multiplication (distinct problems wrong: Phi-4-mini 26 of 54 with multiplication, 7 of 52 with addition, 3 of 53 with subtraction; SmolLM3 33 of 54, 19 of 52 and 14 of 53).

- **Weights.** Both candidates' weights were deleted after their gate failures, as was the unused partial SmolLM3 download of 9 October (`results/cleanup-2026-10-09.json`, with sizes and SHA-256). `results/summary.json` collects the numbers above.

## If the study is ever resumed

The study stops with the author's decision of 10 October. The code for the remaining steps is kept in place (`code/study4.py`, `code/analyze.py`, `code/operating_characteristics.py`, `code/freeze.py`, `code/run_phase.sh`, `code/download.sh`). With `PY` the AI-01 environment's python and `LOCK` the shared GPU lock, from this folder:

1. `PY=$PY bash code/download.sh olmo2-1b` (about 3 GB; checked against `results/hub-files.json`);
2. `GPU_LOCK=$LOCK PY=$PY bash code/run_phase.sh gate olmo2-1b`; if it fails, the study ends with all three candidates failing the gate;
3. `bash code/run_phase.sh screen <candidate>`, then `bash code/run_phase.sh develop <candidate>` (same environment variables);
4. `$PY -B code/analyze.py <candidate> development` and `$PY -B code/operating_characteristics.py <candidate>`; write `PREDICTIONS.md` from them;
5. `$PY -B code/freeze.py <candidate>`; only after the freeze, rename `verify.py.draft` to `verify.py`;
6. `bash code/run_phase.sh test <candidate>`, then `$PY -B code/analyze.py <candidate> test`, `$PY -B code/analyze.py <candidate> summary` and `$PY -B verify.py`.

## Reproducibility

The restored AI-01 environment: Python 3.12.14, torch 2.13.0 (CUDA 13.0), transformers 5.16.1; FP8 matrix kernel `kernels-community/finegrained-fp8` at revision `8c178950e8710e26a2210e4e909cd02dc16c8715` via `kernels` 0.17.1 (`code/frozen/`). One 6 GB NVIDIA GPU. The gate runs took 2.6 minutes for Phi-4-mini and 1.8 minutes for SmolLM3 (`results/<candidate>/calibration/chunks.jsonl`).
