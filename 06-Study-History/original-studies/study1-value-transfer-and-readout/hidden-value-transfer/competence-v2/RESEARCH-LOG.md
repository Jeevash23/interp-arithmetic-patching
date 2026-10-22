# Continuation log: answer the original research question

Client date: 1 October 2026. This is development work. No reserved-model outputs have been inspected.

## Decision before candidate inference

The user clarified that solving the original intermediate-computation question is the objective. The previous 1.7B configuration's failure is a reason to revise that configuration, not end the project. The frozen protocol chooses the dedicated Qwen3-4B-Instruct-2507-FP8 model first, then Qwen2.5-1.5B-Instruct if needed. Both run locally. No paid service is used.

All 216 full-problem role mappings and arithmetic/source-control invariants passed. Scientific settings were frozen before candidate inference. A cache-completion repair and the documented dense-model loader correction were needed before model loading; their failed logs and prior source freeze are retained. See `PREPARATION.md` and `AMENDMENT-DENSE-LOADER.md`.

## Candidate 1: official Qwen3-4B-Instruct-2507-FP8

The weights matched the provider SHA256. The model loaded on the 6 GB NVIDIA GPU. Four native/custom output comparisons, 144 read/identity/captured-self null comparisons and four captured-self generation checks passed. Maximum tested null-logit difference was zero. A donor write changed only the selected block-output position, with nonzero downstream logit change. Full-answer sequential scores matched identity intervention scores. The separately reported single-pass score differed from sequential scoring by up to 2.496726 nats among the eight diagnostic candidate answers; it is not used for localization.

| Baseline | Correct paired records | Distinct prompts |
|---|---:|---:|
| Donor full problem | 30/36 | 12 |
| Recipient full problem | 24/36 | 36 |
| Donor first step | 36/36 | 12 |
| Recipient first step | 36/36 | 12 |
| Recipient remaining step | 36/36 | 34 |
| Explicit donor-intermediate counterfactual | 36/36 | 36 |

All outputs were valid bare integers without truncation. Triple-competent coverage is 20/36 pairs. Examples of full-task failures include `(67-4)+5 → 78` rather than 68 and `(16+7)*3 → 129` rather than 69. These are measured failures of this particular answer-only checkpoint/precision/prompt configuration. They do not prove that this model family lacks intermediate computation. No localization or scientific activation-transfer rows were run because the declared competence gate failed.

A post-baseline numerical diagnostic tested actual layer-0 query, layer-18 MLP-up and layer-35 MLP-down matrices with varying checkpoint weight scales. Relative RMS discrepancies against a separately quantized, FP32-matmul reference were 0.004868, 0.005149 and 0.005164. Removing three trailing input rows caused zero discrepancy on the retained rows in these tests. This rules out a gross failure on these sampled matrix operations, not every numerical/backend issue or a difference from the unquantized parent. Code, inputs' seed and measurements are retained in `kernel_real_weight_check.py` and `kernel-real-weight-check.json`.

## Candidate 2: download and next evaluation

The prespecified Qwen2.5-1.5B-Instruct fallback downloaded at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, and its weight file matched the provider SHA256. The same data, no-scratchpad instruction, complete-output parser and competence thresholds applied. Its evaluation completed after the client date changed to 2 October 2026; the source declaration and download-start date remain 1 October.

Native/custom, null and positive-write instrumentation passed. Its strict accuracies were donor full 6/36, recipient full 4/36, donor first 36/36, recipient first 27/36, recipient remaining 33/36 and explicit donor-intermediate counterfactual 29/36. All 216 records were valid integers without truncation. No pair was correct in all three central conditions. It therefore fails the declared gate and has no localization/intervention outcomes.

Both initial candidates have now failed. The prospectively declared [prompt calibration](../competence-v3/PROTOCOL.md) is being executed in its fixed order, beginning with the stronger 4B model and the three existing demonstrations. This is a separate development version; the v2 results and thresholds are unchanged.

To maintain disk headroom, a redundant 975,175,680-byte failed temporary download was removed only after verifying it was byte-for-byte identical to the corresponding prefix of the completed 4B checkpoint. The full verified model and all earlier models/results remain. `redundant-download-cleanup.json` records the removed temporary prefix's hash.

## Commands and artifacts

From this directory, use the project environment at `../../.venv/bin/python`. The commands run were `download_ranges.py`, `study.py --candidate qwen4b --phase baseline`, `kernel_real_weight_check.py`, `analyze.py --candidate qwen4b`, and `download_fallback.py`. Complete execution logs are in `logs`; model manifests, source freezes, raw baseline journals and runtime configurations are retained. The screen and control phases remain guarded by instrumentation and competence checks.

The paper remains a living research draft. The existing readout result is a useful boundary/control study, not an answer to the full hidden-intermediate or multilingual question.
