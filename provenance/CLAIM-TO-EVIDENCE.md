# Claim-to-evidence map (manuscript of 9 October 2026, with Studies 3 and 4)

Every claim of `paper/manuscript.pdf`, the macros or table that carry its numbers (`paper/generated/`), and the saved
output and script they come from. Study 1 = `06-Study-History/original-studies/study1-value-transfer-and-readout`
(abbreviated S1), Study 2 = `.../study2-reasoning-interventions` (S2). "Recount" means
`paper/generated/reconciliation/numbers.json`, written by `analysis/reconcile_reports.py` from the journals; post hoc
results are in `analysis/posthoc/results/`. The map of the 2 October draft is kept in
`superseded-2026-10-09/05-Notes/CLAIM-TO-EVIDENCE.md`.

## Complete transfer is rare (Section 4.1, Table 2, abstract)

| Claim | Macros / table | Evidence | Script |
|---|---|---|---|
| Held-out: 0/128 at each frozen site; bound 2.3% per attempt, 4.6% per program (64 programs) | `\HoSuccess`, `\HoBranches`, `\HoBoundBranch`, `\HoBoundProgram`; `table-transfer.tex` | S2 `04-Results/final/qwen06-main-v1/records.jsonl` | `posthoc/transfer_rates.py` (per-program bound prespecified in S2) |
| Readout: 16/240 [3.9%, 10.6%], 7 of 30 program groups [9.9%, 42.3%]; 8 with one-digit donors; the 8 two-digit successes also under first-digit text | `\RoActSuccess`, `\RoActCI`, `\RoActUnits`, `\RoActUnitsCI`, `\RoOneDigitSuccess`, `\RoTwoDigitSuccess` | S1 `followup-readout/results/seed-*/interventions.jsonl` | `posthoc/transfer_rates.py`, `posthoc/first_digit_equivalence.py` |
| Qwen3-4B: best site 2/36 [0.7%, 18.7%] (2 of 12 donors [2.1%, 48.4%]); rank-8, attention, MLP 1/36 each; diagnostic (rank-8 projection at block 23) 0/72 per source, bound 4.1% (28.3% by donor value) | `\FbBestSuccess`, `\FbBestCI`, `\DgFullDisc`, `\DgDisc`, `\DgBoundPair`, `\DgBoundDonor` | S1 `hidden-value-transfer/competence-v3/results/{qwen4b-fewshot,eligible-sites}/controls.jsonl`, `subspace-v1/results/controls.jsonl`, `component-v1/results/*/controls.jsonl`, `subspace-v1/value-digit-qualified-sources/results/controls.jsonl` | `posthoc/transfer_rates.py` |
| Value as text: 151/240; 5/32 and 3/32 against 0/32; prompt baseline 36/36 | `\RoFullSuccess`, `\HcFullEarly`, `\HcFullLate`, `\HcActEarly`, `\FbPromptBaseline` | S1 readout journals; S2 `04-Results/final/qwen06-controls-v1/records.jsonl`; S1 `competence-v3/results/qwen4b-fewshot/baselines.jsonl` | recount; `posthoc/transfer_rates.py` |
| One of the two block-23 successes used a donor that answered its own problem incorrectly | (text) | S1 `hidden-value-transfer/paper/manuscript.md` (Study 1 draft, 2 Oct), from the same journals | read directly |

## Second-to-last block of Qwen3-0.6B sets the next token (Section 4.2, Figure 1)

| Claim | Macros | Evidence | Script |
|---|---|---|---|
| First token = donor's next token 240/240 | `\RoFirstTokCopy` | S1 readout journals | recount (`ai02.readout`) |
| Held-out: 119/119 parsed first numerals start with the donor digit, 1 complete numeral; recipient's answer in 91/128 | `\HoLeadDonor`, `\HoLeadParsed`, `\HoLeadComplete`, `\HoRecipient` | S2 main journal | recount (`ai04.main.*`) |
| First-digit text reproduces the output 233/238, 30/30, 30/32; 14 of the 16 successes also under the text edit | `\RoDigitExact`, `\HcDigitEarly`, `\HcDigitLate`, `\RoBothSuccess` | S1 readout journals; S2 controls journal | recount; `posthoc/first_digit_equivalence.py` |
| Same-leading-digit donors identical 117/120 and 8/8; late mismatches from one program | `\RoWidthIdentical`, `\DpPrograms`, `\DpOwnTarget` | S1 readout journals; S2 `04-Results/development/qwen06-donor-controls-v1/records.jsonl`, controls journal (program final-006) | recount |
| Margin +28.2 and +28.1 logits; complete answers 2 and 4 of 48 | `\HdAEarly`, `\HdALate`, `\HdCEarly`, `\HdCLate` | S2 `04-Results/development/qwen06-development-v2/records.jsonl` | recount |
| Intermediate already written: 119/119 and 232/232 distinct traces (post hoc); 40/40 earlier audit; 59/60 held-out programs | `\RoRecipientZ`, `\RoDonorZ`, `\VthreeTraces`, `\HoPriorZ`, `\HoTimingAvail` | S1 readout `baselines.jsonl`; S1 `results/v3/analysis/prefix-exposure-summary.json`; S2 `04-Results/analysis/main-secondary-v2.json` | `posthoc/trace_exposure.py`; read directly |

## Qwen3-4B answer prefix (Section 4.3, Tables 4–5)

| Claim | Macros | Evidence | Script |
|---|---|---|---|
| First digit follows the donor 25/36 (20 of 31 where digits differ); output = text edit 14/36; 11 diverge, 3 with the donor's second digit (post hoc) | `\FbFirstTok`, `\FbFirstTokDiffer`, `\FbIdentical`, `\FbDiverge`, `\FbSecondDonor` | S1 `eligible-sites/controls.jsonl` | `posthoc/first_digit_equivalence.py` |
| Same-leading-digit source identical 26/36; rank-8 32/36 | `\FbSameLead`, `\FbSubSameLead` | S1 eligible-sites and subspace journals | recount |
| Noise changed 11/36 at block 23, at most 1 elsewhere; donor 31/36 | `\FbNoiseChanged`, `\FbNoiseChangedOther`, `\FbActChanged` | same | `posthoc/first_digit_equivalence.py` |
| Diagnostic (rank-8 projection at block 23) splices 3/81, 7/81, 0 joint; text 12/81; first-step end unchanged 35/36 | `\DgCoarseDonor`, `\DgCoarseCoupled`, `\DgCoarseJoint`, `\DgCoarseText`, `\FbFseUnchanged` | S1 qualified-source and eligible-sites journals | recount |

## Operand substitution (Section 4.4, Table 3, Figure 2; post hoc)

| Claim | Macros | Evidence | Script |
|---|---|---|---|
| Donor at first-operand sites changed 36/36; 16–19/36 equal the program on the donor's operand; other sources 13–29/36; controls 0/405; block 14: 16/27 and 34/36; 419 of 435 differ beyond the first digit | `\FbOperandChanged`, `\OpDonorMin`, `\OpDonorMax`, `\OpOtherMin`, `\OpOtherMax`, `\OpControlMatches`, `\OpControlAttempts`, `\OpBDonor`, `\OpBSameZ`, `\OpBeyond`, `\OpMatches`; `table-operand.tex` | S1 `qwen4b-fewshot/controls.jsonl`, `eligible-sites/controls.jsonl` | `posthoc/operand_substitution.py` |

## Screen decomposition (Section 4.5, Figure 3; post hoc)

| Claim | Macros | Evidence | Script |
|---|---|---|---|
| 30.4 = 25.5 suppression + 4.9 gain at the top site; mean P(target) 1.45 × 10⁻⁵; top 3 of 144 at the first operand; ≥72% suppression at the 62 sites above 10 nats (minimum 72.75%, rounded down); max P 4.6 × 10⁻⁴ in blocks 0–11, peak 0.058 at block 23 | `\ScTopRatio`, `\ScTopSupp`, `\ScTopGain`, `\ScTopPTarget`, `\ScSites`, `\ScBigSites`, `\ScMinSuppShare`, `\ScMaxPEarly`, `\ScMaxP` | S1 `qwen4b-fewshot/screen.jsonl` (5,184 records) | `posthoc/screen_decomposition.py`; the 30.405-nat figure also in the recount |

## Tokenizer (Section 4.7; post hoc)

| Claim | Evidence | Script |
|---|---|---|
| Digits isolated; only the ten single digits among vocabulary entries with an ASCII digit; three tokenizer files byte-identical | S1 `models/Qwen3-0.6B/tokenizer.json`, `followup-readout/checkpoint-replication/model/tokenizer.json`, `hidden-value-transfer/competence-v2/models/qwen4b/tokenizer.json` | `posthoc/tokenizer_check.py` (asserted in `make_numbers.py`) |

## Held-out confirmation, Study 3 (Section 4.6, Table 4)

All from `studies/3-heldout-4b-confirmation/results/summary.json` (every number by name; macros `\Sth…`), read by
`paper/make_numbers.py`, which also requires `results/verification.json` to report all 35 checks passed. Prespecified
rules in that study's `PREDICTIONS.md`; protocol and code frozen in `FREEZE.json` before any test output.

| Claim | Keys in `summary.json` |
|---|---|
| 72 reserved pairs under 24 donors, never evaluated, no overlap with development | `reserve_checks` |
| Competence: donor 66/72, recipient 69/72, stated intermediate 72/72; both runs passed | `numbers` `box.competence.*`, `machine.competence.*` |
| P1–P6 verdicts, development wording, reserve wording, development pairs | `decisions.box`, `decisions.machine`, `decisions.development` |
| At most 2/72 complete transfers per site (upper ≤ 9.7%); 3/72 in the reserve wording; both block-23 successes under addition | `box.sites.*.complete_counterfactual.*`, `box.sites.block23_answer_prefix.complete_counterfactual_by_operation.*` |
| First token follows donor 45/72 [44.4%, 79.2%]; 56 points above noise [38, 72] | `box.sites.block23_answer_prefix.first_token_follows_donor.*`, `decisions.box.P2` |
| Output = first-digit text edit 27/72; = own-first-token text edit 49/72 | `box.sites.block23_answer_prefix.equals_first_digit_text_edit.*`, `...equals_own_first_token_text_edit.*` |
| Operand substitution 28/63, 24/63, 21/63, 35/62 with lower bounds | `box.sites.*.first_operand_substitution_informative.*`, `...second_operand_substitution_informative.*` |
| End-of-first-step unchanged 71/72; same-leading-digit identical 41/72 (reserve 47/72) | `box.sites.block20_first_step_end.recipient_answer_unchanged.*`, `{box,machine}.sites.block23_answer_prefix.same_leading_source_identical_output.*` |
| Not prespecified: same-answer source gives the donor's output 32/72, same-intermediate source 3/72 | `box.sites.block23_answer_prefix.conditions.{different_z_same_answer,same_z_different_answer}.identical_to_donor_write` |

## Multi-digit number tokens, Study 4 (Section 4.7)

Gate counts from `studies/4-multidigit-tokens/results/summary.json`, statuses and error profiles from
`results/candidates.json` (macros `\Sfo…`, Table 9 from `generated/table-study4.tex`); status, deviations and the
author's decision of 10 October in that study's `README.md` and `AMENDMENTS.md`.

| Claim | Keys |
|---|---|
| Three candidates in fixed order; all passed the tokenizer check | `candidates.order`, `tokenizer_check` |
| Phi-4-mini-instruct: instrumentation passed; donor 93/120, recipient 92/120, all three 72/120; each step and the stated intermediate 120/120; gate 90%/80% | `phi4mini_gate.*`, `candidates.candidates.phi4mini.gate.thresholds` |
| SmolLM3-3B: instrumentation passed; donor 63/120, recipient 72/120, all three 38/120; each step and the stated intermediate 120/120 | `smollm3_gate.*` |
| Errors most frequent with a multiplication second step (Phi-4-mini 26/54 against 7/52 and 3/53; SmolLM3 33/54 against 19/52 and 14/53) | `candidates.json`: `candidates.{phi4mini,smollm3}.error_profile` |
| Candidate 3 not tried (author's decision, 10 October); no screen, freeze or test | `candidates.json`: `candidates.olmo2-1b.status`, `outcome` |

## Appendix

| Claim | Evidence |
|---|---|
| Table 4 (six sites), Table 5 (three other writes) | recount `ai02.fresh.*`, `ai02.subspace.*`, `ai02.component.*`; `posthoc/first_digit_equivalence.py` |
| Table 6 (held-out outcomes), baselines 59, 61, 57 of 64; 7 paths at the 768-token cap | recount `ai04.main.*`, `ai04.controls`; S2 `02-Code/ai04/engine.py` (cap) |
| Table 7 (competence of four configurations); gate 90%/80% | recount `ai02.qualification.*`, `ai02.qwen17b.initial`; S1 `hidden-value-transfer/competence-v2/PROTOCOL.md`; S1 `hidden-value-transfer/STATUS.md` (Qwen3-1.7B truncation) |
| Study history | S1 `00-START-HERE.md`, `followup-readout/PROTOCOL.md`; S2 `00-START-HERE.md`, `followups/causal-carriers-v1/00-START-HERE.md` |

## Cited work

The related-work statements and the numbers quoted from Bhupatiraju and Nyaupane (76.9%, 88.2%) are checked against
their sources by `paper/review/check_references.py` (`QUOTES`); results in `paper/review/reference-check.json`.
