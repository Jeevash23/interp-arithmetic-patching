# Study 5: an added mean-difference direction and a span write in Qwen3-4B

**Question.** Single-position whole-vector writes in Qwen3-4B rarely transferred the donor's intermediate (Studies 1 and 3). Is the intermediate carried by a direction, or across the positions of the first-step sentence, rather than by one whole vector? Gurnee et al. (2026, Appendix A.24.1) found that adding a mean-difference direction at a single layer changes a larger model's answer to the one another intermediate value implies, in a narrow band of layers. Replacing a whole vector and adding such a direction test different things, so this study applies both the added direction and a span write to the Qwen3-4B configuration of Studies 1 and 3.

**Answer.** No, at the sites and blocks tested. On the 72 held-out pairs neither intervention carried the intermediate at its selected sites (R1 and R2 not supported). The span after the operands changed almost no answer at any block; the whole first-step span changed answers only by supplying the donor's operation word at early blocks; the added direction did nothing at the end of the first step and, at the answer prefix, moved mainly the intermediate's leading digit. The prespecified overall verdict is **inconclusive**, because the layer sweep (R4) found answer-prefix blocks 25 to 29 with 7 to 10 complete counterfactuals of 72: too many to exclude the intermediate there, too few to count as carrying it.

`PROTOCOL.md` and the run code were hashed in `PROTOCOL-FREEZE.json` (17 October 2026, 10:52:02 UTC) before any output of the new interventions existed; `AMENDMENTS.md` records three amendments (a file not mounted, descriptive outcome classes, the cost rule in the launcher), none touching the computation or the decision rules. The numeric expectations are in `PREDICTIONS.md`, frozen with the development records in `FREEZE.json` before any test output.

## Design

- **Model and execution.** Qwen3-4B-Instruct-2507-FP8 at revision `8591804019c8b22094c3b5b4454e0edc05dffc98`, exactly as in Study 3 (byte copies of Study 3's frozen code, FP8 kernel snapshot and `kernels` package in `code/frozen/`): few-shot box-wording prompt, greedy decoding, no key-value cache, the write reapplied at every step. Run on Modal, one NVIDIA L4 per worker; the weights were downloaded inside Modal and checked against the 1 October manifest.
- **Pairs.** Development: the 36 development pairs of Study 1. Test: the 72 held-out pairs of Study 3 in the development wording.
- **Added direction.** For every block and each of the four positions, the class mean of the residual vector over 24 fitting problems per intermediate value (1,488 problems, none shared with either split); the difference between the means for the donor's and the recipient's intermediates is added to the recipient's vector (α = 1, Gurnee's recipe; α = 4 on development only).
- **Span write.** The donor's vectors replace the recipient's at every position of a span of the first-step sentence at one block: the post-operand span (" to finish the first step.", 6 tokens) or the whole first-step sentence without its operand tokens (15 tokens, including the operation word).
- **Outcome.** The complete generated answer; the complete counterfactual y* = g_r(z_d) is the target. For the direction, also Gurnee's measure: the teacher-forced log probability of y* minus that of the recipient's answer.

## First-stage checks

- The frozen instrumentation passed on four development pairs (null writes 0.0 logits; native and custom generation equal), and the span hook's 48 null checks changed no logit (largest change 0.0).
- **Reproduction:** the 27 saved Study 3 records of donor group 0 (six baselines, the donor write at the six whole-vector sites, the unpatched run) were regenerated on the L4 and all 27 equal the saved records token for token; all 18 captured donor vectors are byte-identical to the saved ones (largest difference 0.0).

## Results

### Prespecified verdicts (held-out pairs, conservative 95% ranges)

| Rule | Site (selected on development) | Development (36) | Held-out (72) | Verdict |
|---|---|---|---|---|
| R1 | Direction, end of first step, block 1, α = 1 | 0/36 | 0/72 [0, 5.0%] | does not carry |
| R1 | Direction, answer prefix, block 27, α = 4 | 6/36 | 4/72 [0, 13.6%]; noise 1/72 | does not carry |
| R2 | Span after the second operand, block 2 | 0/36 | 0/72 [0, 5.0%] | does not carry |
| R2 | Span of the whole first step, block 0 | 0/36 | 0/72 [0, 5.0%] | does not carry |
| R3 | | | | not applied |
| R4 | Every block (α = 1 direction at both positions; both spans) | at most 5/36 | at most 10/72 (answer prefix, blocks 25–29: 7–10, upper bounds 20.8–25.0%) | inconclusive |
| | **Verdict** | inconclusive | | **inconclusive** |

`PREDICTIONS.md` expected R1 inconclusive overall (the answer-prefix site between the thresholds), R2 not supported, R4 inconclusive and the verdict inconclusive. R1 came out "not supported" because the answer-prefix site gave 4 of 72.

### Descriptive results

- **Span after the second operand:** at least 68 of 72 answers unchanged at every block; 0 complete counterfactuals.
- **Span of the whole first step:** at blocks 0 to 11 the answer was the recipient's operands combined with the donor's first operation in 66 to 69 of 72 pairs; from block 19 on at least 67 answers were unchanged (at block 18, 64).
- **Five other sources over either span:** their own counterfactual only in pairs where it equals the recipient's unchanged answer (12 cases; post hoc check).
- **Direction at the end of the first step (α = 1 sweep):** at least 70 of 72 answers unchanged at every block; the teacher-forced score never ranked the counterfactual first at the selected block.
- **Direction at the answer prefix (α = 1 sweep), blocks 25 to 30:** 44 to 56 of 72 answers changed; complete counterfactual 7 to 10; the recipient's second step applied to an intermediate made of the donor's leading digit and the recipient's last digit 17 to 24; the donor's leading digit spliced into the recipient's answer 12 to 16. Among multiplication recipients the intermediate splice reached 12 of 24 (post hoc).
- **Teacher-forced score (Gurnee et al.'s measure)** at the selected answer-prefix site: the counterfactual was more likely than the recipient's answer in 43 of 72 pairs (mean change 31.9 nats), while 4 generated answers were the counterfactual.
- **Operand positions** (selected on development; their sweeps were dropped from the test by the cost rule): first operand (block 13, α = 4) 1/72, second operand (block 20, α = 4) 0/72.
- Baselines: recipient 69/72 and donor 66/72 correct (the same as Study 3); 144 of 144 span self-writes reproduced the unpatched output.
- Of the five descriptive expectations in `PREDICTIONS.md`, four held in full; the one about the whole-sentence span held except at block 18 (64 of 72 unchanged, 90% expected). `code/posthoc.py` and `results/posthoc.json` evaluate them, with the post hoc checks.

### Verification

`verify.py` (written during the test run, before its records were pulled) re-parses every answer, recomputes every sweep cell and selected-site count with its exact interval from the raw records, re-derives the R1 and R2 verdicts, and checks the freeze records, the order of the stages and the first-stage checks: 16 of 16 checks pass (`results/verification.json`).

### Cost and data

Modal's billing report: $4.47 for this study's app (image build, the first stage including one restart after Amendment 1, development, the second development pass and the test); Study 6 $0.07. GPU time: about 1.9 L4-hours of generation in development and 1.6 in the test, on four workers. The cost rule triggered after development (spent $2.72 + projected test $2.56 > $4.50), so the direction sweep at the operand positions was dropped from the test. Records pulled to the local machine: 1.7 MB in all; uploads about 8 MB (code, frozen kernel package, inputs). The weights stayed in Modal.

## Files

- `PROTOCOL.md`, `PROTOCOL-FREEZE.json`, `AMENDMENTS.md`, `PROTOCOL-FREEZE-amendment-*.json`, `PREDICTIONS.md`, `FREEZE.json`.
- `code/`: `prepare.py` (inputs), `study5.py` (the interventions, run in Modal), `modal_app.py` (cloud stages), `plans.json` (work lists), `analyze.py`, `stats.py` (Study 3's estimators), `freeze.py`, `cost_rule.py`, `posthoc.py` (written after the test), `frozen/` (byte copies, `COPIED-FROM-STUDY3.json`).
- `verify.py`: the independent check.
- `data/`: the two splits, the fitting problems, the reproduction reference and vectors, `data-manifest.json`.
- `results/`: `first/checks.json` (first-stage checks), `development/`, `devsites/`, `test/` (gzipped per-pair records), `development-selection.json`, `development-sweeps.json`, `development-analysis.json`, `test-analysis.json`, `cost-decision.json`, `posthoc.json`, `verification.json`, `summary.json`, billing and transfer logs. The class means (`means.pt`, 91 MB) stay on the Modal Volume `ai0204-study5`; their SHA-256 is in `results/first/checks.json`.
- `run_stage.sh`, `watch_billing.sh`: start a stage as a systemd user unit; the billing watchdog.
