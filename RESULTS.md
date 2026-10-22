# Results

The findings of the manuscript ([`paper/manuscript.pdf`](paper/manuscript.pdf)) with their numbers. Every number is
generated from saved outputs by `paper/make_numbers.py`; brackets are exact (Clopper–Pearson) 95% intervals, and "≤"
is the one-sided 95% upper bound for a count of zero. Study 1 and Study 2 are the original study folders under
`06-Study-History/original-studies/`; Studies 3 to 6 are in `studies/` (see [`README.md`](README.md)). Analyses marked **post hoc** were written on
9 October 2026, after every outcome they analyse had been seen (`analysis/posthoc/`).

## Setting

- **Programs.** Two exact integer steps, `z = f(a, b)` (addition or subtraction) and `y = g(z) = z ∘ k` (addition,
  subtraction or multiplication), stated in words. A pair is a donor `d` and a recipient `r`; the counterfactual target
  is `y* = g_r(z_d)`, which differs from `y_r`, `y_d` and `z_d` in every pair.
- **Write.** The residual stream after one decoder block at one position is replaced by the donor's vector at the
  aligned position; the complete answer is then generated.
- **Experiments.** Readout (Study 1): Qwen3-0.6B, thinking mode, block 26 of 28, 30 program groups × 4 pairs ×
  2 seeds = 240 attempts. Held-out (Study 2): Qwen3-0.6B, blocks 20 and 26 chosen on 24 development programs and
  evaluated on 64 held-out programs × 2 continuations. Answer-only (Study 1): Qwen3-4B-Instruct-2507-FP8, greedy, 36
  development pairs under 12 donors, a 144-site screen and generation at six sites.

## 1. Complete transfer of the intermediate is rare

| Experiment | Write | Successes | 95% interval | Units with a success (programs, groups or donors) |
|---|---|---|---|---|
| Held-out, Qwen3-0.6B | block 20 or block 26, labelled boundary | 0/128 each | ≤ 2.3% | 0/64 programs, ≤ 4.6% (the prespecified bound) |
| Readout, Qwen3-0.6B | block 26 | 16/240 | [3.9%, 10.6%] | 7/30 groups [9.9%, 42.3%] |
| Answer-only, Qwen3-4B (development) | best of six sites (block 23, answer prefix) | 2/36 | [0.7%, 18.7%] | 2/12 donors [2.1%, 48.4%] |
| Answer-only, Qwen3-4B (development) | rank-8 projection, attention output, MLP output at block 23 | 1/36 each | [0.1%, 14.5%] | 1/12 donors [0.2%, 38.5%] |
| Diagnostic, Qwen3-4B, rank-8 projection at block 23 | 72 pairs separating the value from a leading-digit splice | 0/72 per source | ≤ 4.1% | 0/9 donor values, ≤ 28.3% |
| Held-out Qwen3-4B (Study 3) | six development sites, 72 reserved pairs | at most 2/72 per site | conservative upper bounds ≤ 9.7% | |

Text that states the value transfers it far more often: 151/240 (62.9%) in the readout experiment, 5/32 and 3/32 at the
two held-out boundaries (activation 0/32 at each), and 36/36 Qwen3-4B prompt baselines that state the donor's
intermediate. Of the 16 readout successes, 8 used one-digit donors (first digit = whole value) and 14 also occurred
under a text edit of the first digit alone (post hoc). The Qwen3-4B development rates come from the pairs on which
the sites were chosen; Study 3 (section 7) measured them again on 72 reserved pairs.

## 2. In Qwen3-0.6B's second-to-last block, the write sets the next token

- The first token after the write was the donor's next token in 240/240 readout attempts; in the held-out experiment
  119/119 parsed first numerals began with the donor's leading digit, 1 equalled the donor numeral, and the final
  answer was the recipient's own in 91/128 attempts.
- A text edit of the donor's first digit reproduced the activation output token for token in 233/238 readout attempts
  [95.2%, 99.3%] and in 30/30 and 30/32 held-out control paths (7 of the 30 are paths that reached the 768-token cap
  under both conditions; the 2 mismatches come from one program).
- Donors that share the leading digit but differ in value gave identical outputs in 117/120 readout pairs and 8/8
  donor-panel programs at both boundaries.
- This is close to built in: one block remains after the write, and the intermediate is already written in the trace
  (in 119/119 distinct recipient and 232/232 distinct donor readout traces, post hoc; in 59/60 held-out programs at the
  labelled boundary).

## 3. At the Qwen3-4B answer prefix (block 23 of 36), the write carries more than the next token

Development pairs (post hoc); the held-out test is in section 7.

- The first answer digit followed the donor in 25/36 pairs, but the output equalled the first-digit text edit in only
  14/36 [23.1%, 56.5%]; 11 pairs diverged after the first digit, 3 of them continuing with the donor's second digit.
- It rarely carried the value: a source differing by one but sharing the leading digit gave the donor's output in
  26/36 pairs (rank-8 projection: 32/36).
- Part of the effect is disruption: matched noise changed 11/36 answers at this site [16.3%, 48.1%], against 31/36
  for the donor write and at most 1/36 at the other five sites.
- Diagnostic (rank-8 projection at block 23): leading-digit splices 3/81 and 7/81 for the two source families, 0 pairs
  under both, versus 12/81 when the actual first token is forced as text.
- Held-out (Study 3): the output equalled the first-digit text edit in 27/72 pairs and the text edit of the write's own
  first token in 49/72, so the one-token account is refuted at this site (P3, P3s).

## 4. At operand positions, the write transfers the operand in about half the pairs (post hoc; positive control)

- At the three screen-selected first-operand sites (blocks 20, 9, 10) the donor write changed 36/36 answers; 16, 19 and
  19 of 36 [27.9%, 61.9%] to [35.5%, 69.6%] equalled the recipient's program run on the donor's first operand.
- The five other sources gave their own operand's program in 13 to 29 of 36 pairs; noise, unpatched and self-write
  controls in 0/405 attempts.
- Block 14 (second operand): 16/27 for the donor's second operand, 34/36 for the same-intermediate source.
- In 419 of the 435 matching answers more than the first digit differs from the recipient's answer: the model computed
  a new result from the substituted input.
- Held-out (Study 3, prespecified P4): supported at block 20, first operand (28/63) and block 14, second operand
  (35/62); inconclusive at blocks 9 and 10 (24/63, 21/63).

## 5. The likelihood screen rewards suppression of the recipient's answer (post hoc decomposition)

- Top site (block 20, first operand; top 3 of 144 sites are all at the first operand): mean change 30.4 nats =
  25.5 suppression of the recipient's answer + 4.9 gain of the target; mean P(target) after the write 1.45 × 10⁻⁵.
- Across the 62 sites above 10 nats, suppression supplies at least 72% (minimum 72.75%).
- Mean P(target) is at most 4.6 × 10⁻⁴ in blocks 0–11 and peaks at 0.058 at block 23 of the answer prefix.

## 6. Tokenizer

Qwen3's pre-tokenizer isolates every digit; the only vocabulary entries containing an ASCII digit are the ten single
digits (fullwidth digits also occur in the vocabulary but are never produced, because the pre-tokenizer isolates every
numeric character); the tokenizer files of Qwen3-0.6B, Qwen3-1.7B and Qwen3-4B-Instruct-2507-FP8 are byte-identical
(post hoc check of the shipped files). In Qwen3 the next token and the leading digit coincide; Study 4 was to separate
them (section 8).

## 7. Held-out confirmation in Qwen3-4B (Study 3)

`studies/3-heldout-4b-confirmation/`: the answer-only assay repeated without change on the 72-pair reserve (24 donors ×
3 recipients) generated with the development pairs on 1 October and never evaluated; protocol, predictions and code
frozen before any test output; primary run in the development wording, secondary in the reserve's own wording;
conservative 95% bounds (wider of exact and donor-cluster bootstrap). Both runs passed the competence gate (donor 66/72,
recipient 69/72, stated intermediate 72/72). Verdicts (development wording; reserve wording in brackets where different):

| Rule | Result | Verdict |
|---|---|---|
| P1 complete transfer rare at all six sites | at most 2/72 per site; upper bounds ≤ 9.7% (reserve: at most 3/72) | supported |
| P2 first token follows the donor in most pairs, above noise (block 23) | 45/72 [44.4%, 79.2%]; 56 points above noise | inconclusive |
| P3 output equals the first-digit text edit (block 23) | 27/72 [23.6%, 51.4%] | refuted |
| P3s output equals the text edit of the write's own first token | 49/72 [56.0%, 79.2%] | refuted |
| P4 operand substitution (blocks 20, 9, 10; 14) | 28/63, 24/63, 21/63; 35/62 | supported at 2 of 4 sites |
| P5 end-of-first-step write leaves the answer unchanged | 71/72 [92.5%, 100%] | supported |
| P6 same-leading-digit source gives the identical output | 41/72 [44.4%, 69.4%] | inconclusive (reserve: supported, 47/72) |

Not prespecified: at block 23 a source with the donor's final answer but another intermediate gave the donor's output in
32/72 pairs, a source with the donor's intermediate but another answer in 3/72: the output depends on the donor's
final answer, not its intermediate, although the donor's answer itself was the output in only 6/72. Both block-23 successes had addition as the recipient's second step.

## 8. Models with multi-digit number tokens (Studies 4 and 6)

`studies/4-multidigit-tokens/`: three candidates whose tokenizers write every number up to 999 as one token (all passed
the tokenizer check): Phi-4-mini-instruct, SmolLM3-3B, OLMo-2-0425-1B-Instruct, in fixed order. Phi-4-mini-instruct
(FP8 on loading) passed instrumentation and solved each step alone and the stated-intermediate prompt in 120/120
calibration pairs, but answered the complete problem for 93/120 donors (77.5%) and 92/120 recipients (76.7%), with
72/120 pairs correct on all three, against the 90%/80% gate. SmolLM3-3B (FP8 on loading) also passed instrumentation and
answered every step alone in 120/120, but the complete problem for only 63/120 donors (52.5%) and 72/120 recipients
(60.0%), with 38/120 pairs correct on all three. Both models erred most often where the second step is a multiplication
(distinct problems wrong with multiplication, addition and subtraction: Phi-4-mini 26/54, 7/52, 3/53; SmolLM3 33/54,
19/52, 14/53). On 10 October, after SmolLM3-3B failed, the author decided not to try the third candidate. No development
screen, prediction file, freeze or test inference exists.

**Study 6** (`studies/6-olmo2-7b-multidigit/`, 17 October 2026) ran Study 4's design unchanged with
OLMo-2-1124-7B-Instruct, a larger model whose tokenizer files are those of Study 4's third candidate, in bfloat16 on a
rented L4 (a new study, because a candidate chosen after two had failed changes the candidate list after outcomes were
seen). It passed the tokenizer check (every number 0–999 one token) and instrumentation (all 144 null writes 0.0), and
answered every step alone in 120/120 calibration pairs, but the complete problem for only 60/120 donors (50.0%) and
53/120 recipients (44.2%), with 22/120 pairs correct on all three; distinct problems wrong with multiplication,
addition and subtraction: 49/54, 21/52, 17/53. The study ended at the gate. Following Baeumel et al. (2025), its
protocol had added a third outcome, one digit changing inside the answer token; it was never reached. Three models with
multi-digit number tokens have now failed the gate, and the tokenizer question remains open.

## 9. An added direction and span writes in Qwen3-4B (Study 5)

`studies/5-direction-and-span/` (17 October 2026, Modal L4 GPUs): the model, prompts and code of Study 3; two
interventions swept over all 36 blocks and scored by the complete generated answer. (1) Gurnee et al.'s added
mean-difference direction: the class mean of the residual vector over 24 fitting problems per intermediate value
(1,488 problems), the donor-minus-recipient difference added at one position (α = 1; α = 4 on development only). (2)
Span writes: the donor's vectors over the 6 tokens after the second operand, or over the 15 non-operand tokens of the
first-step sentence. Sites selected on the 36 development pairs; tested on Study 3's 72 held-out pairs; protocol,
predictions, code and development records frozen before the test. The first cloud stage reproduced 27 saved Study 3
records token for token and 18 donor vectors bit for bit.

| Rule | Selected site | Held-out result (conservative 95%) | Verdict |
|---|---|---|---|
| R1 direction, end of first step | block 1, α = 1 | 0/72 [0, 5.0%] | does not carry |
| R1 direction, answer prefix | block 27, α = 4 | 4/72 [0, 13.6%] (noise 1/72) | does not carry |
| R2 span after the second operand | block 2 | 0/72 [0, 5.0%] | does not carry |
| R2 span of the whole first step | block 0 | 0/72 [0, 5.0%] | does not carry |
| R4 every block | | at most 10/72 (answer prefix, blocks 25–29: 7–10; upper bounds to 25.0%) | inconclusive |

Overall verdict by the frozen rules: **inconclusive** (R1 and R2 not supported; R4 inconclusive). Descriptive: the span
after the operands left ≥ 68/72 answers unchanged at every block; the whole-sentence span gave the donor's first
operation applied to the recipient's operands in 66–69/72 pairs at blocks 0–11 and nothing from block 19 on; the
direction left ≥ 70/72 answers unchanged at the end of the first step at every block; at the answer prefix (blocks
25–30) it changed 44–56/72 answers, most often to the recipient's second step applied to an intermediate with the
donor's leading digit (17–24/72; 12/24 multiplication recipients, post hoc), and its teacher-forced score ranked the
counterfactual first in 43/72 pairs at the selected site against 4 generated. Four of the five descriptive
expectations stated before the test held in full. Cost $4.47 (Modal billing report).

## Corrections to the 2 October draft

The 2 October merged draft (`provenance/superseded-2026-10-09/01-Papers/manuscript.md`) said that a single-position
write "moves exactly one token, never the intermediate" and that "everything measured is consistent with this
reading"; the Qwen3-4B data contradict the universal form (14/36 above), as the original Study 1 draft had reported.
Its Table 1 also gave matched-noise "unchanged" counts of 35/36 and 36/36 at blocks 14 and 23; the recount gives 36/36
and 25/36. Details are in [`paper/CHANGELOG.md`](paper/CHANGELOG.md).
