# Selected configuration and completed intermediate-transfer development round

2 October 2026. Development only; reserved English examples and cross-language interventions remain unevaluated.

## Competence barrier resolved on the declared first calibration

Qwen3-4B-Instruct-2507-FP8 with three existing input/output demonstrations passes the unchanged gate. All generations are valid bare integers without truncation. No worked intermediate is provided for the current task. The demonstrations' complete prompts and answers are recorded in the runtime file.

| Baseline | Strict correct paired records | Unique prompts |
|---|---:|---:|
| Donor full problem | 33/36 | 12 |
| Recipient full problem | 36/36 | 36 |
| Donor first step | 36/36 | 12 |
| Recipient first step | 36/36 | 12 |
| Recipient remaining step | 36/36 | 34 |
| Explicit donor-intermediate counterfactual | 36/36 | 36 |

Triple-competent coverage is 33/36; all three excluded diagnostic-subset rows share the same donor group. They remain in the full-sample mechanistic analysis. The initial system-only 4B and 1.5B failures are preserved in competence-v2. The remaining three entries in the finite calibration list were not run because the first passed. This is selection on development competence, not confirmatory evidence or a controlled model-size comparison.

Native/custom generation and captured-self generation agree on four smoke recipients. All 144 same-shape read/identity/captured-self null comparisons pass, as does the positive write. `freeze-qwen4b-fewshot.json` records the source/data hashes. `selected-configuration.json` freezes this choice before localization.

## Screen execution

The frozen exploratory grid is 36 blocks × four prompt roles × 36 pairs, or 5,184 site-pair records. The ranking metric is the change in the complete-answer counterfactual-versus-recipient log-probability ratio, computed by sequential conditional scoring. Site means weight the 12 donor groups equally. At most three sites proceed to the ten-condition fresh-generation controls. A large likelihood change alone is not the scientific outcome.

The full-prefix reference implementation completed 601 unique flushed records before a deliberate interrupt for a speed check. It took about 85 seconds per complete pair. The [execution amendment](SCREEN-EXECUTION-AMENDMENT.md) introduces exact reuse of the unaffected upstream block outputs for identical input prefixes. This is not a KV-cache intervention and changes no scientific settings.

All 48 full-vocabulary logit comparisons, four fresh-generation comparisons and 144 token-score comparisons for the first complete pair passed the declared 1e-5 equivalence criteria. Measurements and executable-source hashes are in `fast-equivalence.json` and `fast-execution-freeze.json`. The original log and runtime are preserved. The optimized screen resumes the same unique-key journal. Fresh source-control generations will still use the full reference execution.

The scan is complete. It selected block 20, block 9 and block 10 at `a_end`, with mean complete-answer log-ratio changes of 30.405, 30.159 and 30.159 nats. All 1,080 controls and 432 source baselines completed. Each site's original-donor intervention produces **0/36 complete counterfactuals**, including 0/33 in the triple-competent subset. Outputs match first-operand-only substitution in 16/36, 19/36 and 19/36 respectively. These donor positions cannot yet read the first operation and second operand, so they cannot hold the arbitrary task's computed z.

Both same-z variants, the different-z/same-answer variant and the same-leading-digit variant produce zero respective counterfactual successes at all three sites. Source full-problem competence is 36/36, 36/36, 33/36 and 30/36 respectively; every auxiliary first-step and explicit-counterfactual baseline is correct. Unpatched/self generations all match baseline. The supplementary actual-donor-first-token control and the original oracle-digit control both produce 1/36 counterfactuals. No site passes the frozen progression rules.

`results/qwen4b-fewshot/completed-result-audit.json` confirms the 5,184-record grid, 864 vector hashes, 216 identity generations, write locality, finite sequential scores and final-block causal invariants. The screen and fresh-generation figures were rendered and visually inspected. Raw summaries and joint source-control results are in the result folder's `analysis` directory.

The [eligible-site follow-up](ELIGIBLE-SITE-FOLLOWUP.md) was declared after screen inspection while the original controls completed. It uses the existing absolute complete-CF probabilities to select one block for each causally eligible role: block 14/second operand, block 20/first-stage end, and block 23/answer prefix. Its 1,080-record test is complete in `results/eligible-sites`: original-donor counterfactuals are 1/36, 0/36 and 2/36 respectively; joint original-plus-both-same-z successes are zero at every site. No progression criterion set passes. At block 23, same-z/new-answer sources yield 0/36, changed-z/same-answer sources 0/36, and the original/same-leading-digit pair has zero joint correct outputs. One of its two original successes uses the originally incorrect donor, leaving 1/33 on the competent subset. An additional 864 vector hashes and 216 identity outputs pass audit. This is an explicitly adaptive development revision, not a new confirmation.

The subsequent and only learned-method branch is [subspace-v1](../subspace-v1/PROTOCOL.md), declared while these controls ran. It fits one rank-eight value-conditioned mean subspace at block 23/answer prefix, with the model unchanged. It completed 756 fitting stimuli, 1,512 behavioral records and 504 control records (360 fresh activation generations and 144 reused comparisons). Original-donor counterfactuals are 1/36, with one joint original-plus-both-same-z success. Changed-z/same-answer and same-leading-digit/new-value sources each give zero respective counterfactuals; all progression criteria fail. The independent reconstruction audit matches all 360 target vectors exactly. See its [result report](../subspace-v1/RESULTS.md), including the preserved CPU/GPU norm-audit discrepancy and exact resolution.

The isolated subspace success was investigated with a fixed 81-pair value/digit diagnostic. Two donor decompositions failed qualification (7/9 and 6/9 unique original donor prompts) and stopped before interventions. Both predefined auxiliary source families in the second grid were fully competent. A separately declared [qualified-source contrast](../subspace-v1/value-digit-qualified-sources/PROTOCOL.md) inherited those exact baseline outputs, used all 81 pairs, and completed 567 control records with the same learned parameters. On the 72 pairs where full-value and coarse-digit predictions differ, both source families yield 0/72 full-value targets, with zero joint coarse matches. Its 405 activation targets reconstruct exactly. The qualified grid also confounds source operand-leading and intermediate-leading digits, recorded in an interpretation addendum before inspecting its outputs; no partial-computation discovery is claimed.

Finally, [component-v1](../component-v1/PROTOCOL.md) tested attention and feed-forward projection outputs at the same fixed block 23 / answer-prefix node. Both pass native/custom, null, self and positive-write instrumentation. Each completes 360 control generations and yields 1/36 original-donor counterfactuals, with zero joint original-plus-both-same-z successes. Both fail all substantive progression requirements; their 576 vector hashes and 144 identity generations pass audit. No further component or learned-parameter search was performed.

This development round is complete. The [living draft](../paper/manuscript.md) includes methods, counts, descriptive figures, failed qualification attempts, limitations and the publication-readiness decision. No intervention qualifies for confirmation. The original reserve, September validation and new multilingual intervention sets remain unevaluated; a broader competence and assay study is a proposed next step, not a result already obtained.

## Interpretation checks prepared before controlled outcomes

All 36 original/same-leading-z source pairs share the intermediate's leading digit; 30/36 also share the donor final answer's leading digit. Report paired success on this latter subset separately. No primary counterfactual equals one of the three example answers (14, 18, 23).

Changing a whole donor expression and transferring its already-computed value can predict the same recipient output. Even successful controls would establish behavior consistent with the proposed z abstraction within this assay, not a uniquely identified literal numeric register or proof of when the donor computed it.

## Execution

Commands, from this folder using `../../.venv/bin/python`: `study_formats.py --candidate qwen4b --format fewshot --phase baseline`; initial reference `study_formats.py --candidate qwen4b --format fewshot --phase screen`; `fast_screen.py --verify`; then `fast_screen.py --run`. Full logs are retained in `logs`. Subsequent `--phase controls` uses the selected format and the original full-prefix engine.
