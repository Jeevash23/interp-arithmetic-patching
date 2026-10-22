# Exploratory check of causally eligible whole-vector sites

Declared 2 October 2026 after the full development likelihood screen, while the original selected-site controls were still completing. This is a development revision, not a new preregistration or confirmation. Preserve and report the original ratio-selected results.

The original ratio criterion selected only first-operand positions. Under causal attention those positions have not seen the first operation and second operand, so they cannot contain the current task's already-computed intermediate. Moreover, a ratio can improve by suppressing the normal answer. Neither fact establishes that another site carries a transferable value.

Use one bounded follow-up: choose exactly one block for each of `b_end`, `first_stage_end`, and `prompt_end`, maximizing the mean probability of the complete canonical counterfactual including EOS, using the already completed screen. Break ties by lower block number. These positions can see the complete first-stage expression. This deliberately excludes `a_end`. Selection is adaptive to the same development data and must be described that way.

The resulting sites are block 14 at `b_end`, block 20 at `first_stage_end`, and block 23 at `prompt_end` (zero based). The existing score records yield mean complete counterfactual probabilities of approximately 0.02778, 0.0001325, and 0.05783 respectively. These probabilities are not measured fresh-generation success rates.

Run all ten original fresh-generation control conditions on all 36 pairs at all three sites. Reuse unchanged behavioral baselines and source baselines with explicit provenance. Keep the original model, examples, parser, greedy generation, full-strength write, source variants, metrics and progression heuristics. No additional strength, site, or prompt search is authorized by this follow-up. Include the actual donor first-output-token text comparison by reusing the already measured text-only generations; those have no activation site.

Audit every vector hash, write location, unpatched/self output, complete grid, and the implication that a complete sequence probability above one half must yield that greedy sequence. Report joint same-z and matched-leading-digit contrasts, not just marginal rates. Do not interpret failure of this bounded family as absence of intermediate computation throughout the model.

After this check, assess whether a targeted distributed subspace is scientifically worth testing. Any such method needs its own finite protocol, development-only fitting, matched controls, and an unchanged held-out set. This file does not authorize an unbounded search for a positive result.
