# Trajectory and stage-quality clarification

21 September 2026, after the complete main evaluation. This is descriptive analysis; primary selection, eligibility, outcomes and inference are unchanged.

The audit found three main donor and one recipient baselines with correct final answers but incorrect intermediate labels. An observed numeral differing from the oracle therefore need not have been changed by an intervention. The earlier trajectory_audit.py outputs compare first numeral to the oracle and are preserved as that diagnostic. Reporting uses secondary_analysis.py, which compares to the actual baseline numeral saved in timing.observed_text, retains oracle mismatch separately, and requires an actual baseline change before counting the later-oracle/correct-final signature. For first_z, the cut itself is defined by the oracle literal, so these definitions coincide. They can differ at label_z.

A donor-leading-digit match is an observed match, not proof that the intervention changed that digit: an erroneous baseline label can already start with it. First-number change and exact baseline-sequence agreement are therefore separately retained.

A post-freeze descriptive sensitivity subset requires correct donor and recipient intermediate labels plus all three correct final baselines (48 main programs; 96 branches per policy). This does not replace the frozen final-only competence definition or constitute new independent evidence. A stricter all-six-labelled-stage subset has 46 programs and is recorded separately.

An additional arithmetic diagnostic checks whether the final answer equals the recipient remaining operation applied to the changed first number. This is numerical consistency with the changed numeral, not proof of a unique computation or internal representation. It is not applicable to the lookup task. None of these diagnostics alters the primary counterfactual target.
