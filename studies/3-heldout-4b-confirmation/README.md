# Study 3: held-out confirmation in Qwen3-4B

**Question.** Do the development findings of the AI-02 answer-only study in Qwen3-4B-Instruct-2507-FP8 hold on pairs that played no part in choosing the sites? On 36 development pairs, single-position writes almost never produced the complete counterfactual answer, writes at the answer prefix set the first answer token, and writes at operand positions acted as operand substitution.

**Answer.** Yes for the absence of intermediate transfer, the operand substitution at two of the four operand sites, and the null effect at the end of the first step. At the answer prefix the first token followed the donor in 45 of 72 pairs (62.5%; P2 inconclusive), and the write changed later tokens in about a third of the pairs. On 72 held-out pairs in the development wording, the complete counterfactual appeared in at most 2 of 72 pairs at any of the six development sites. The 0.6B one-token account, in which a text edit of the first digit reproduces the activation output, is refuted for the 4B answer prefix: the two outputs agree in 27 of 72 pairs.

All files, the protocol and the predictions were frozen in `FREEZE.json` at 2026-10-09T14:26:43Z, before any model output on a test pair existed. Every prespecified verdict is reported below, including the inconclusive ones.

## Design

- **Model.** Qwen3-4B-Instruct-2507-FP8 (revision `8591804019c8b22094c3b5b4454e0edc05dffc98`), answer-only, in the frozen AI-02 development configuration: calculator system instruction, three box-wording demonstrations, greedy decoding, full recomputation without a key-value cache, bfloat16 activations and FP8 linear layers through the pinned `finegrained-fp8` kernel. The pipeline is a byte-identical copy of the frozen AI-02 code (`code/frozen/`).
- **Pairs.** The 72-pair reserve generated with the development pairs on 1 October 2026: 24 donor problems, each paired with three recipients whose second operation is addition, subtraction or multiplication. A scan of all 3,233 JSON files of the project found no earlier evaluation of any reserve pair (`data/reserve-checks.json`), and no reserve problem equals a development problem.
- **Wordings.** Primary run in the development wording ("A box starts with 63 counters. …"); secondary run in the reserve's declared wording ("A machine initially holds 60 units. …"), with the demonstrations still in the box wording.
- **Sites.** The six whole-vector sites of the development study, fixed without a new search: block 23 at the answer prefix; blocks 20, 9 and 10 at the last token of the first operand; block 14 at the last token of the second operand; block 20 at the end of the first-step sentence (blocks numbered from 0, of 36).
- **Conditions.** At each site and pair: the donor write, unpatched generation, a self write, norm-matched noise, an unrelated donor, four constructed sources (same intermediate with another decomposition; same intermediate with another final answer; another intermediate with the same final answer; same leading digit with another value), the first-digit text edit, and two added text edits (the donor run's own first token; the first token produced by the write itself).
- **Outcomes.** For a donor d and recipient r with programs z = f(a, b), y = g(z, k): (a) the complete counterfactual y* = g_r(z_d); (b) the first answer token equals the donor run's first answer token; (c) the activation output equals the first-digit text-edit output, and (c′) equals the output of the text edit that forces the write's own first token; (d) at operand sites, the recipient's program run on the donor's operand, on the pairs where that differs from the recipient's own answer; (e) the recipient's answer unchanged.
- **Estimators.** Counts with exact (Clopper–Pearson) and donor-cluster bootstrap 95% intervals (24 donor groups, 10,000 resamples). The decision rules in `PREDICTIONS.md` use the more conservative bound of the two at each end; the tables below give that conservative range.

## Results

### Gates

| | Box wording (primary) | Machine wording (secondary) |
|---|---|---|
| Instrumentation (4 test pairs; 144 null-write checks, native and custom generation) | passed; largest null-write logit change 0.0 | passed; 0.0 |
| Donor full problem | 66/72 | 66/72 |
| Recipient full problem | 69/72 | 69/72 |
| Recipient's second step with the donor's intermediate given | 72/72 | 72/72 |
| Pairs correct on all three (gate 80%) | 63/72 | 64/72 |
| Competence gate (90% each, 80% jointly) | passed | passed |

### Prespecified verdicts

| Rule (PREDICTIONS.md) | Expected | Box wording (primary) | Machine wording (secondary) |
|---|---|---|---|
| P1. Complete transfer is rare at all six sites (upper bound below 0.20) | supported | **supported**: at most 2/72 at any site, every upper bound ≤ 0.10 | supported: at most 3/72, upper bounds ≤ 0.12 |
| P2. At the answer prefix the first token follows the donor in most pairs, above noise | supported or inconclusive | **inconclusive**: 45/72 (62%; exact 50–74%, donor-cluster 44–79%); 0.56 above noise (cluster 0.38 to 0.72) | inconclusive: 40/72 (56%; cluster 38–72%); 0.46 above noise |
| P3. The 0.6B one-token account explains the 4B answer prefix ((c) ≥ 0.80) | refuted | **refuted**: 27/72 (38%; upper bound 0.51) | refuted: 25/72 (35%; upper 0.49) |
| P3s. Everything after the write's own first token follows the unpatched model ((c′) ≥ 0.80) | refuted | **refuted**: 49/72 (68%; upper bound 0.79) | refuted: 50/72 (69%; upper 0.798) |
| P4. Operand writes act as operand substitution (lower bound ≥ 0.25 at each of four sites) | supported at all four | **supported at 2 of 4 sites**: block 20 first operand 28/63 (44%; lower 0.27), block 14 second operand 35/62 (56%; lower 0.39); inconclusive at block 9 (24/63, lower 0.21) and block 10 (21/63, lower 0.17) | supported at 2 of 4 sites (block 20: 31/64; block 14: 33/62; block 9: 21/64 and block 10: 23/64 inconclusive; on the 64 rather than 63 first-operand pairs, see the note below the outcome table) |
| P5. A write at the end of the first step leaves the answer unchanged (lower bound ≥ 0.80) | supported | **supported**: 71/72 (99%; lower 0.93) | supported: 72/72 |
| P6. At the answer prefix a same-leading-digit source gives the identical output in most pairs | supported | **inconclusive**: 41/72 (57%; lower bound 0.44); 0/72 joint successes | supported: 47/72 (65%; lower 0.51); 1/72 joint success |

Two verdicts differ from the development expectation in the primary run: P4, which holds at two of the four operand sites rather than all four, and P6, which is inconclusive. P2 is inconclusive too, which `PREDICTIONS.md` allowed ("supported or inconclusive"). P2 and P6 are inconclusive because the donor-cluster intervals are wide: the three pairs of a donor group share the donor, and outcomes that depend mainly on the donor move together. `PREDICTIONS.md` §3 estimated before the run that P2 would be inconclusive about four times in ten under development-like data. P4 holds at block 20 (first operand) and block 14 (second operand) and is inconclusive at blocks 9 and 10, where the substitution rate fell from 53% in development to 38% and 33%.

### Outcomes by site (box wording; conservative 95% range in brackets)

| Outcome | Block 23, answer prefix | Block 20, first operand | Block 9, first operand | Block 10, first operand | Block 14, second operand | Block 20, end of first step |
|---|---|---|---|---|---|---|
| (a) complete counterfactual | 2/72 [0–10] | 1/72 [0–7] | 1/72 [0–7] | 1/72 [0–7] | 1/72 [0–7] | 0/72 [0–5] |
| (b) first token follows donor | 45/72 [44–79] | 8/72 [5–21] | 8/72 [4–21] | 9/72 [6–22] | 9/72 [6–22] | 6/72 [3–17] |
| (c) equals first-digit text edit | 27/72 [24–51] | 3/72 [0–12] | 4/72 [1–14] | 5/72 [1–15] | 2/72 [0–10] | 5/72 [1–15] |
| (c′) equals own-first-token text edit | 49/72 [56–79] | 20/72 [14–43] | 21/72 [14–46] | 26/72 [19–53] | 9/72 [3–25] | 72/72 [95–100] |
| (d) operand substitution, informative pairs | – | 28/63 [27–62] | 24/63 [21–57] | 21/63 [17–51] | 35/62 [39–73] | – |
| (e) recipient answer unchanged | 6/72 [3–17] | 11/72 [4–29] | 13/72 [4–35] | 14/72 [6–35] | 6/72 [1–17] | 71/72 [93–100] |
| same-leading-digit source gives the identical output | 41/72 [44–69] | 4/72 [1–14] | 0/72 [0–5] | 2/72 [0–10] | 64/72 [79–96] | 71/72 [93–100] |
| answer leading-digit splice | 16/72 [11–35] | 1/72 [0–7] | 1/72 [0–7] | 2/72 [0–10] | 0/72 [0–5] | 0/72 [0–5] |
| intermediate leading-digit splice | 16/72 [13–34] | 0/72 [0–5] | 1/72 [0–7] | 0/72 [0–5] | 0/72 [0–5] | 0/72 [0–5] |
| donor answer copied | 6/72 [3–17] | 0/72 [0–5] | 0/72 [0–5] | 0/72 [0–5] | 0/72 [0–5] | 0/72 [0–5] |

The machine-wording table is in `results/machine/analysis.json`; its rates lie within a few pairs of these.

Further outcomes at block 23 (box wording):

- Without a write the first answer token already equals the donor's in 5 of 72 pairs, and under norm-matched noise in 5 of 72.
- Of the 45 pairs whose first token follows the donor, 16 (36%) later depart from the text edit that forces the same first token.
- The write's output tracks the donor's final answer, not its intermediate (a descriptive reading of the prespecified condition counts). A source with the donor's final answer but another intermediate gives the identical output in 32 of 72 pairs; a source with the donor's intermediate but another final answer does so in 3 of 72.
- The donor and the same-leading-digit source never both produce their own counterfactual (0 of 72), and the donor and both same-intermediate sources never all succeed (0 of 72).
- Unpatched and self-write outputs equal the recipient's baseline in 72 of 72 pairs at every site, in both wordings.
- The two complete counterfactuals at block 23 both have an addition as the recipient's second step.

In 9 of the 72 pairs the donor and the recipient share the first operand, so their vectors at the first-operand position are identical and the three first-operand writes leave those pairs unchanged. In the development wording those pairs drop out of outcome (d) by its definition and count as unchanged in (e). In the machine wording one of them counts as informative under the literal rule of `PROTOCOL.md` §8, because its unpatched recipient answer is wrong and so differs from the substituted program's answer: outcome (d) there is scored on 64 pairs rather than 63, the extra pair can only fail, which departs from the rule's parenthetical, and every verdict is the same with 63.

## What this means for the claim

The claim under test is that a single-position write moves the next token, not the intermediate. On held-out 4B pairs the second half holds: no site transfers the complete intermediate in more than 3% of pairs (upper bound 10%), a write at the end of the first step changes nothing in 71 of 72 pairs, and operand writes change the operand the model computes with. The first half is not established: the first answer token followed the donor in 45 of 72 pairs (62.5%; P2 inconclusive), 56 points above noise, and in this model the write also changes later tokens: the activation output equals the text edit of the donor's first digit in 38% of pairs and the text edit of its own first token in 68%. In Qwen3-4B the answer-prefix write therefore carries more than the next token. Its output depended on the donor's final answer rather than its intermediate (32 against 3 of 72 pairs; a reading that was not prespecified), although the donor's answer itself was the output in only 6 of 72 pairs.

## Notes and deviations

- The protocol and predictions were drafted at [local time] on 9 October, completed and revised after the drafting session was interrupted, and frozen at 14:26:43 UTC; no test output existed before the freeze. Their last revisions were saved at [local time] (`PROTOCOL.md`) and [local time] (`PREDICTIONS.md`), and the SHA-256 of both files matches `FREEZE.json`; the headers of the frozen files say "about [local time]", an approximate time that cannot be corrected inside the frozen files. The revisions made the development wording the primary run, defined (d) on informative pairs, compared later divergence with the text edit that forces the same first token, added the noise comparison to P2, and added chunked execution and the operating characteristics of the rules.
- The run used chunks of at most 25 minutes on the shared GPU (`results/<wording>/chunks.jsonl`); the journals resume by record key, and every record appears exactly once (per wording 432 baselines, 864 source baselines, 4,320 controls and 504 text controls).
- No change was made after the freeze; there is no `AMENDMENTS.md`.

## Files

- `PROTOCOL.md`, `PREDICTIONS.md`, `FREEZE.json`: the frozen design, the expectations and decision rules, and the SHA-256 of 104 frozen files with the freeze time.
- `data/`: the reserve, the two test files (box and machine wording), the reserve checks and the data manifest.
- `code/`: data preparation, reserve checks, the run script (`confirm.py`, `run_test.sh`), analysis, statistics, development reproduction, freeze and operating characteristics; `code/frozen/` holds the copied AI-02 pipeline.
- `results/box/`, `results/machine/`: raw journals (`baselines`, `source-baselines`, `controls`, `text-controls`), saved write vectors, gates, run records and `analysis.json`.
- `results/summary.json`: every number of the analysis by name, the verdicts of both runs and of the development data, run times and checks.
- `results/development-reference.json`, `results/operating-characteristics.json`, `results/dev-reproduction.json`: development values, verdict rates under development-like data, and the environment check.
- `verify.py`, `results/verification.json`: the independent recomputation (35 of 35 checks passed).

## Reproducibility

- **Environment.** The restored AI-01 environment: Python 3.12.14, torch 2.13.0 (CUDA 13.0), transformers 5.16.1, accelerate 1.14.0, NumPy 2.5.2, safetensors 0.8.0, tokenizers 0.23.1, triton 3.7.1; `kernels` 0.17.1 from `code/frozen/competence-v2/.packages`; FP8 kernel `kernels-community/finegrained-fp8` at revision `8c178950e8710e26a2210e4e909cd02dc16c8715`. One 6 GB NVIDIA GPU; peak allocation 5.59 GB.
- **Model.** `Qwen/Qwen3-4B-Instruct-2507-FP8` at revision `8591804019c8b22094c3b5b4454e0edc05dffc98`; `model.safetensors` 5,190,053,264 bytes, SHA-256 `b6154d74332140fd6dfbfbe70bbb3650dd6955861132bd59dda6789e6322b485` (`logs/weights-check.json`). The other eight files and their hashes are listed in `code/frozen/competence-v2/model-manifest-qwen4b.json`. The weights live in `models/qwen4b/` and can be fetched again with `logs/download-qwen4b.sh`.
- **Commands** (from this folder; `PY` is the AI-01 environment's python, `LOCK` the shared GPU lock file):
  - `$PY -B code/prepare_data.py` and `$PY -B code/check_reserve.py` (data and reserve checks, standard library);
  - `$PY -B code/analyze.py development` and `$PY -B code/operating_characteristics.py` (development reference values);
  - `flock $LOCK $PY -B code/reproduce_dev.py` (environment check on development pairs: 48 of 48 records and 18 of 18 vectors identical);
  - `$PY -B code/freeze.py`;
  - `GPU_LOCK=$LOCK PY=$PY bash code/run_test.sh` (both wordings, in chunks);
  - `$PY -B code/analyze.py box`, `$PY -B code/analyze.py machine`, `$PY -B code/analyze.py summary`;
  - `$PY -B verify.py`.
- **Run time.** 40 GPU-minutes for the box wording and 36 for the machine wording (`results/run-times.json`), plus about one minute for the development reproduction.
