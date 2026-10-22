# Progress toward testing hidden intermediate transfer

2 October 2026. The external review is mostly well supported, and the project pursued its second option: test transfer before the model writes an intermediate value. **This development round is complete. Whole-vector sites yield at most 2/36 complete counterfactuals; the learned projection and two native-component writes each yield 1/36 and fail source controls. A qualified-source diagnostic yields 0/72 full-value successes where full-value and coarse-digit predictions differ, with no joint source-invariant coarse effect.** The [living original-question draft](paper/manuscript.md) contains the measured results and limits. Reliable full-value transfer remains unestablished; the draft is not submission-ready.

## Active continuation

The [competence-v2 protocol](competence-v2/PROTOCOL.md) declares a dedicated non-thinking Qwen3-4B-Instruct-2507-FP8 candidate and a Qwen2.5-1.5B-Instruct fallback. The [continuation log](competence-v2/RESEARCH-LOG.md) records downloads, pinned dependencies, startup repairs and actual outcomes. Both models load locally and pass instrumentation. The 4B model's initial no-example prompt produces 30/36 correct donor problems and 24/36 correct recipient problems, with every isolated-operation and explicit-intermediate control correct. The fallback, evaluated on 2 October, produces 6/36 correct donor and 4/36 correct recipient problems. All 432 paired baseline records are valid integers without truncation. Both fail the frozen competence threshold, so neither initial configuration has been used for localization.

The first [bounded prompt calibration](competence-v3/PROTOCOL.md), declared before fallback outcomes, passes: Qwen3-4B-Instruct-2507-FP8 with three demonstrations obtains donor 33/36, recipient 36/36 and explicit-intermediate 36/36, with every isolated-operation baseline correct. The same data, parser, generation budget and thresholds were retained. The other three calibration candidates were not needed. The [active experiment log](competence-v3/RESEARCH-LOG.md) records the frozen configuration, 5,184-record activation grid and verified execution optimization. The previous 1.7B results below remain preserved as an earlier failed configuration.

The completed [learned-subspace branch](subspace-v1/REPRODUCIBILITY.md) and its [qualified-source diagnostic](subspace-v1/value-digit-qualified-sources/PROTOCOL.md) retain all failed competence calibrations and exact inherited outputs. Both [native-component checks](component-v1/PROTOCOL.md) pass instrumentation and fail the substantive transfer criteria, with no new learning. The original validation, reserve and multilingual interventions remain untouched. A further study should qualify behavior over a broader development distribution and explicitly revise the assay; these failures do not prove absence of intermediate computation.

Read the [checked review](review/FEEDBACK-REVIEW.md) for the verdict on individual claims and primary-source references. The September readout result remains valid within its narrow scope. It is not a completed answer to the original research question.

## Historical 1.7B branch: completed work

Created a separate [development protocol](PROTOCOL.md), 36 new donor/recipient pairs grouped under 12 donors, and 72 reserved pairs under 24 additional donors with another wording. Each development donor is paired with addition, subtraction and multiplication as the recipient's remaining operation. Same-value and same-answer source controls are saved. Neither the new reserve nor the original September validation set has been evaluated.

Reused the existing Qwen3-1.7B checkpoint and Python environment on the local 6 GB NVIDIA GPU with 6,141 MiB of GPU memory. The initial device check found 5,852 MiB free and no compute jobs. Execution uses bfloat16, eager attention, disabled thinking, deterministic greedy decoding and full-prefix recomputation without a KV cache. All 11 checkpoint files were checked against the saved download hashes. No model was downloaded or service purchased for these runs.

The native-versus-custom generation check passed on four recipient prompts. All 144 null/read/identity/captured-self comparisons had zero maximum logit difference at the tested matched input shapes. The positive write changed its intended activation and downstream logits without changing other positions at the write. These checks validate this execution path; they do not establish arithmetic competence. The first startup's tokenizer return-type error occurred before any model forward pass and is preserved with its [correction](AMENDMENT-TEMPLATE-API.md).

## Behavioral results

The first prompt asked for only a final integer but the model generated explanations. With a 16-token budget, all full-task donor and recipient outputs were truncated. This fails the intended no-scratchpad configuration, rather than measuring the model's eventual accuracy with a long explanation.

| Initial condition | Strict correct records | Unique prompts |
|---|---:|---:|
| Donor two-step problem | 0/36 | 12 |
| Recipient two-step problem | 0/36 | 36 |
| Donor first step alone | 3/36 | 12 |
| Recipient first step alone | 0/36 | 12 |
| Recipient second step with its own intermediate supplied | 18/36 | 34 |
| Recipient second step with donor intermediate supplied | 20/36 | 36 |

All 216 baseline records are retained. Exact prompt reuse is marked and must not be counted as independent runs.

A separate [bounded format pilot](FORMAT-PILOT.md) then tested the first two development donor groups under three declared formats. Each has six paired records per condition because the two donors each serve three recipient operations.

| Format | Donor two-step | Recipient two-step | Supplied donor intermediate |
|---|---:|---:|---:|
| Calculator system instruction | 0/6 | 0/6 | 6/6 |
| Calculator instruction and three worked input/output examples | 0/6 | 1/6 | 6/6 |
| Calculator instruction and parenthesized expression | 0/6 | 1/6 | 6/6 |

The examples produce valid bare integers in all 18 records, so that format's low two-step accuracy is not a parser artifact. For instance, the donor problem `(20 + 4) * 2` produces `40` rather than `48`. Other formats fail through a mixture of wrong numbers and explanatory outputs. All 54 format records are saved; no format met the predeclared 90% criterion in all three central baselines. The six supplied-intermediate controls per format were all correct, showing that these particular one-step counterfactuals are accessible when the intermediate is explicitly supplied.

These are small development checks, especially the two-donor format pilot. They do not prove that Qwen3-1.7B cannot perform direct arithmetic under any setting, and they say nothing yet about the causal organization of a competent model. Greedy decoding differs from the model card's recommended sampling for non-thinking mode; it was a declared development choice. We have not converted this failure into a hidden-mechanism negative result.

## Historical decision after the 1.7B branch

Stop localization on these failed configurations. The next prerequisite is a model/configuration that reliably produces complete correct answers without intermediate text. A stronger checkpoint that fits local memory, or a separately declared task-difficulty and decoding calibration, is justified. The present evidence does not justify spending the next run on another late layer-26 readout replication, DAS training, or cross-language transfer.

Before a new model screen, freeze the candidate list, prompt format, arithmetic range and competence threshold; retain all attempted configurations. A larger model's successful benchmark score is not a substitute for the exact task baseline. Gemma-2-2B is a plausible candidate from the review, but is not installed here and its official repository requires access terms. Qwen3-4B cannot simply replace this model at full bfloat16 on this 6 GiB GPU; placement or precision would need its own validation. Neither alternative has been run or declared successful.

Once competence passes, the prepared localization design covers all 28 Qwen3-1.7B blocks at four prompt roles. It scores complete answer sequences, then tests fresh generation with source and readout controls at at most three selected sites. Its screen and controls code has been prepared but **has not been execution-validated or used for scientific outcomes** because the behavioral gate failed. A successful controlled development result would precede a frozen held-out study, followed by cross-language work if warranted.

The research question remains open. Current progress is a verified critique, a more appropriate protocol, working baseline instrumentation, and an observed feasibility limit that prevents a misleading mechanistic conclusion.

## Reproduction

From this directory, `../.venv/bin/python tasks.py` regenerates deterministic data; `../.venv/bin/python run.py --phase baseline` resumes the initial baseline journal; `../.venv/bin/python prompt_formats.py` resumes the format pilot; and `python3 audit_feedback.py` reproduces the critique counts. Completed rows are reused, not independently rerun. The screen command is guarded by the failed competence gate. Change scientific settings only in a separate declared version.

The raw results are in [results](results), [format-pilot/results](format-pilot/results), and [review](review). Both execution logs and source/data freezes are preserved. The checkpoint and old readout records are unchanged by this work.
