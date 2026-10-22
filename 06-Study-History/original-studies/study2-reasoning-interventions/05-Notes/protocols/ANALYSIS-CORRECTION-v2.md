# Secondary semantic-agreement correction

21 September 2026, discovered by code inspection after the main run was frozen and before held-out control outcomes. The frozen `final_analysis.py` compares parsed final fields directly. Two null fields therefore compare equal, and equal numbers without native completion also compare equal. That is a literal field-equality diagnostic, not agreement on two valid completed semantic answers.

Preserve the frozen file and its `final-v1` analysis output. `final_analysis_v2.py` adds a strict agreement function requiring two available, naturally completed, non-null parsed answers with the same integer. It retains raw parsed-field equality and the joint-valid-completion count as separate diagnostics. Exact token-sequence agreement continues to include matching truncated outputs and is labelled separately.

The primary counterfactual-success definition, selector identities, group means, stratified bootstrap, control outcome counts and scientific run data are unchanged. The authoritative reporting output is `04-Results/analysis/final-v2/results.json`, and its source hash identifies the correction. Verify v1/v2 primary results are identical. This corrects reporting semantics; it does not exclude failures from the all-intended primary denominator or tune a statistical comparison.

Before producing the complete secondary analysis, the same natural-completion requirement was applied to its actual-baseline-answer and donor-original-answer agreement fields. Token-sequence equality remains a separate field and may include identical capped responses. This consistency correction changes no generated observation or target-success grade.
