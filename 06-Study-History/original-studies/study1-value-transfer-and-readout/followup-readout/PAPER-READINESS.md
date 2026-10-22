# Paper readiness after the follow-up

**There is enough evidence for a substantive paper draft. A broader multilingual-value-transfer or model-general claim is not supported.** The completed draft is organized around an independently tested boundary case: a whole-vector intervention can appear to transfer an arithmetic value while its output is almost entirely reproduced by a first-digit text intervention.

| Question | Evidence and current decision |
|---|---|
| Is there a finding worth drafting? | Yes. The fixed follow-up yields 233/238 identical digit-control sequences and 117/120 identical short/long-source continuations. Every apparent two-digit success is also produced by a one-digit source. |
| Is it only a selected pilot effect? | No. The original exploratory selection is followed by new programs, a paired-source design, a new wording and two fixed seeds, with 30 underlying program groups. |
| Does it establish complete-value reuse? | No. The original progression gate failed, and the new controls favor a readout account with residual effects in a few continuations. |
| Are the observations numerically audited? | Yes for the 0.6B experiment: 21 follow-up audit checks passed, in addition to the original 64. The main run preserves all 720 baseline and 2,220 intervention records. |
| Does it generalize to another model? | Not established. The separate 1.7B bfloat16 attempt failed its declared replay/cache gate. Its one replay mismatch was whitespace-only, and its final answer was unchanged. This is an execution-compatibility limit, not a failed arithmetic result. |
| Are language-family claims justified? | No. Spanish/Tamil results are exploratory behavioral checks on drafts without independent fluent review. |
| Is novelty established? | No. The updated primary-source comparison identifies close work on readout, variable binding, arithmetic abstraction and intermediate localization. The proposed contribution is the controlled boundary case and combined audit. |
| Is the document ready to review? | Yes: the editable manuscript and rendered PDF include actual results, failures, controls, figures and limitations. |
| Is a TMLR/ARR submission automatically justified? | No. A venue decision requires assessing whether the bounded single-checkpoint contribution is sufficient; a broader claim requires further evidence. No submission or public release has occurred. |

The most useful next experiment is a separately declared second-checkpoint study with execution fidelity comparable to the primary run. A float32 setup with carefully validated CPU/GPU placement is one possible development direction, subject to measured memory and speed. Another precision configuration needs its own pre-outcome validation criteria and fresh evaluation; the failed study's tolerances must not be retrospectively changed. An additional language grid would not resolve the current model-comparability gap.

Independent fluent language review is needed only if the multilingual research question is pursued further. The present paper can remain explicitly about the validated English readout boundary. Author/contribution review, an appropriate data/code release and venue formatting are still necessary before submission.
