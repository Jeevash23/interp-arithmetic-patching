# Compositional value-transfer experiment — measured results

Qwen3-0.6B · local 6 GB NVIDIA GPU · 9 September 2026 · exploratory development study

The tested English whole-vector intervention does **not** justify a claim of reusable complete intermediate values. Its apparent pilot success is closely explained by moving the next digit at a late output position.

- The selected site fell from 6/20 discovery successes to 8/100 on full development, with only 2/80 successes in the additional pairs. It failed both declared progression requirements: at least 20% counterfactuals and at least a ten-point margin over the matched controls.
- The activation emitted the donor's first digit in 97/97 runnable cases, versus 31/99 for unpatched replay. A first-digit-only text intervention reproduced all 100 success indicators and 93/97 complete token sequences on jointly runnable cases. Four continuations differed, so an exact reduction to text alone is not established.
- All eight counterfactual successes used single-digit donor values: 8/25 versus 0/75 for two-digit values. No two-digit donor value was printed completely by this intervention. In contrast, inserting the completed value as text produced 58/100 counterfactuals, including 48/75 two-digit cases. These value-length strata are exploratory and confounded by other task differences.
- With the recipient's completed intermediate line fixed to its original value, activation and text-only continuations agreed on all 20 cases and produced no counterfactuals. With donor z fixed instead, both produced 8/20 counterfactuals; one final answer differed. A pre-final donor state transported the first digit of its **final answer** in 96/96 runnable cases, showing that this behavior is not restricted to intermediate-value roles.

The source captured **after** the donor's final numeral caused immediate EOS in 96/96 runnable cases. Before-number digit copying and after-number termination together favor a late next-token readout account. The after-final control is therefore a termination control, not a clean negative test of numeric information at a matched before-number site.

This supports a narrow account of late numeric readout and sometimes recomputing a final answer from altered printed text. It does not identify a unique literal z register, selective complete-value transfer, a hidden arithmetic algorithm, or a multilingual computation barrier. Full-expression transport remains observationally equivalent to z transfer in the task; source-control invariance alone cannot distinguish them.

Stop expansion of this intervention family. Keep all 100 English validation pairs reserved; no held-out results are claimed. The early equation-site search was too disruptive to supply a better candidate, and the present evidence does not specifically motivate fitting a distributed alignment. Do not start a model/language grid or cross-language activation interchange from this failed English gate. The Spanish/Tamil work is a separately labelled exploratory behavioral supplement on draft translations.

This is a coherent negative development result with an explanatory control, not a submission-ready paper or evidence that reusable internal values do not exist elsewhere.

The behavioral supplement is complete: Spanish full donor/recipient scores are 18/20 and 20/20. Tamil remains at 3/20 and 0/20 even after the separately recorded 1024-token budget diagnostic; supplied-counterfactual competence remains 2/20. This is insufficient for interpreting a Tamil activation test on these unverified drafts. No French run or cross-language activation matrix was performed.


## Behavioral competence

All counts below use the complete 100-pair English development set. The first 20 pairs supplied the initial instrumentation and discovery sample. Full-task failures remain included; the prespecified triple-competence subset contains 96 pairs.

| Baseline | Correct | Parser failures | Truncations |
|---|---|---|---|
| donor full | 97/100 | 3 | 2 |
| recipient full | 99/100 | 1 | 0 |
| donor first | 100/100 | 0 | 0 |
| recipient first | 100/100 | 0 | 0 |
| donor remaining | 100/100 | 0 | 0 |
| recipient remaining | 100/100 | 0 | 0 |
| textual counterfactual | 100/100 | 0 | 0 |

The earlier non-thinking configurations achieved only 6/20 donor and 6/20 recipient full-task accuracy (v1), and 2/20 and 5/20 (v2). Their failures and source snapshots remain saved. These are configuration comparisons, not an isolated effect of thinking mode.

## Bounded intervention screen

The complete 24-site search used 20 pairs (four donor-value clusters). The maximum was 6/20 at block 26 before the labelled Intermediate numeral. Both original arithmetic answers were already present in all 40 donor/recipient thinking traces.

![Complete screening counts](results/v3/analysis/screen.png)

[Every screening count and failure](results/v3/analysis/screen.csv). Equation timings were available for 16/20 pairs; unavailable cases remain unsuccessful in the complete denominator. An early equation location is not proof of first hidden computation.

## Development controls at the selected site

| Condition | Original counterfactual | Unchanged answer | Copied donor answer | Copied donor z | Invalid incl. unavailable | Unavailable | Truncated |
|---|---|---|---|---|---|---|---|
| different z same answer | 0/100 | 60 | 1 | 3 | 3 | 1 | 0 |
| donor | 8/100 | 63 | 1 | 0 | 6 | 3 | 0 |
| donor distractor | 8/100 | 63 | 1 | 0 | 5 | 2 | 0 |
| donor final position | 0/100 | 0 | 0 | 0 | 100 | 4 | 0 |
| donor pre final position | 0/100 | 56 | 1 | 3 | 6 | 4 | 0 |
| first operand position | 0/100 | 63 | 0 | 0 | 7 | 1 | 0 |
| first token position | 0/100 | 73 | 0 | 0 | 27 | 1 | 0 |
| isotropic delta | 0/100 | 97 | 0 | 0 | 3 | 3 | 0 |
| mismatched z | 0/100 | 64 | 1 | 1 | 3 | 2 | 0 |
| random source | 0/100 | 62 | 0 | 0 | 4 | 3 | 0 |
| recipient distractor | 9/100 | 62 | 0 | 0 | 9 | 7 | 2 |
| same example | 0/100 | 99 | 0 | 0 | 1 | 1 | 0 |
| same z decomposition | 8/100 | 65 | 1 | 0 | 4 | 1 | 0 |
| same z different answer | 8/100 | 65 | 1 | 0 | 4 | 1 | 0 |
| unpatched | 0/100 | 99 | 0 | 0 | 1 | 1 | 0 |

![Development controls](results/v3/analysis/controls.png)

For crossed/random sources, [the full table](results/v3/analysis/controls.csv) also scores the counterfactual using the actual source value. Donor-answer-as-intermediate predictions, all overlapping alternatives, distractor predictions and generated parser failures are separate fields. Unavailable positions and parser failures are not conflated in that table.

## Printed-number diagnostics

![Value and digit comparators](results/v3/analysis/value-vs-digit.png)

The original text-prefix assay replaces the intermediate numeral and allows fresh continuation immediately. New digits can extend that numeral. The line-clamp assay includes the original line ending, then generates the final answer freshly, with or without the same pre-z activation write.

| Completed-line clamp condition | Counterfactual | Unchanged | Invalid | Truncated |
|---|---|---|---|---|
| text recipient z clamped | 0/20 | 20 | 0 | 0 |
| text donor z clamped | 8/20 | 8 | 0 | 0 |
| activation recipient z clamped | 0/20 | 20 | 0 | 0 |
| activation donor z clamped | 8/20 | 9 | 0 | 0 |

[Paired text/activation comparison](results/v3/analysis/text-activation-comparison.json), [all numeral-prefix outcomes](results/v3/development/textual-scratchpad-summary.json), and [value-length/operation diagnostics](results/v3/analysis/mechanism-diagnostics.csv). These are exploratory controlled continuations, not natural direct/indirect-effect estimates.

## Spanish and Tamil competence supplement

The same 20 underlying development pairs were used in all languages. English outputs were reused. Spanish/Tamil prompts are [redacted]-reviewed drafts without independent fluent verification; output labels and numeric answers are shared, and thinking language is unconstrained.

| Baseline | English | Spanish | Tamil |
|---|---|---|---|
| donor full | 20/20 | 18/20 | 2/20 |
| recipient full | 20/20 | 20/20 | 0/20 |
| donor first | 20/20 | 20/20 | 5/20 |
| recipient first | 20/20 | 20/20 | 5/20 |
| donor remaining | 20/20 | 20/20 | 3/20 |
| recipient remaining | 20/20 | 20/20 | 2/20 |
| textual counterfactual | 20/20 | 20/20 | 2/20 |

![Matched language competence](results/v3/analysis/multilingual-baselines.png)

[Language failures, truncations and token lengths](results/v3/analysis/multilingual-baselines.csv). This small draft-translation comparison does not identify a language-family effect or a cross-language mechanism barrier.

A separate 1024-token diagnostic extended all 70 original truncations, preserving their generated prefixes and leaving EOS-completed runs unchanged. Native and resumed/batched continuation agreed on a development example used for instrumentation validation. English and Spanish had no original truncations, so their outcomes were reused. The Tamil comparison is:

| Tamil baseline | Correct at 512 | Correct at 1024 | Truncated at 512 | Truncated at 1024 |
|---|---|---|---|---|
| donor full | 2/20 | 3/20 | 14 | 6 |
| recipient full | 0/20 | 0/20 | 17 | 10 |
| donor first | 5/20 | 7/20 | 7 | 2 |
| recipient first | 5/20 | 5/20 | 6 | 0 |
| donor remaining | 3/20 | 5/20 | 10 | 5 |
| recipient remaining | 2/20 | 2/20 | 7 | 3 |
| textual counterfactual | 2/20 | 2/20 | 9 | 5 |

[Paired budget diagnostic](results/v3/analysis/language-budget-comparison.csv). This is an adapted behavioral supplement, not a new independent sample or a cross-language activation experiment.

## Uncertainty, validation and audit

The full development set reuses 20 donor values, 20 recipient values, 12 arithmetic structures and two fixed templates. [Paired analysis](results/v3/analysis/paired-analysis.json) retains complete and prespecified competence-filtered samples. Separate cluster bootstraps and their envelope are sensitivity descriptions; they are not exact multiway or selection-adjusted confidence intervals.

[Pilot-informed sample-size simulations](results/v3/analysis/sample-size-planning.json) evaluate a ten-percentage-point target under explicit dependence assumptions. These are statistical design simulations, not measured model outcomes or a finalized confirmatory sample size.

| Set | Planned pairs | Evaluated pairs | Status |
|---|---|---|---|
| English discovery | 20 | 20 | exploratory |
| English full development | 100 | 100 | includes discovery |
| Reserved English validation | 100 | 0 | not evaluated; see decision above |
| Spanish/Tamil competence | 20 | 20 | matched development pairs; no activation transfer claims |

The declared numerical English progression gate is **not passed**. [Exact gate calculation](results/v3/analysis/development-gate.json). Interpretation requires the controls in addition to this numerical threshold.

[Artifact audit](results/v3/analysis/artifact-audit.json) passed. It checks split/model hashes, unique raw records, parser and causal scoring, token bookkeeping, identity continuations and stored activation hashes. [Measurement boundaries](docs/MEASUREMENT-BOUNDARIES.md) specify what each check establishes.

## Files for continuing the research

- [Living manuscript](paper/manuscript.md) and [prioritized next experiments](docs/NEXT-STEPS.md).
- [Protocol](docs/PROTOCOL.md), amendments 001–009, [research log](docs/RESEARCH-LOG.md).
- [Primary-source comparison](docs/RELATED-WORK.md), [environment](docs/ENVIRONMENT.md), [reproduction instructions](docs/REPRODUCIBILITY.md).
- Deterministic datasets under `data/`; complete outputs, failures and vectors under `results/`; commands/errors under `logs/`.
