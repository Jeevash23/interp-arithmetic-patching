# Claim-to-evidence register

| Claim or question | Finding/status | Evidence and boundary |
| --- | --- | --- |
| Primary held-out policy comparison | Across 64 held-out programs, the replay-selected site (label_z-L20) produced 0/128 complete counterfactual answers; the continuation-selected site (label_z-L26) produced 0/128. Their paired difference was 0.00 percentage points. | `04-Results/analysis/final-v2/results.json`; 64 program groups; paired, template-stratified bootstrap; two streams |
| Complete transfer follows a large immediate digit effect | Not established by A alone | Development A/B/C table and `04-Results/figures/01-assay-comparison.png`; logits are not calibrated probabilities |
| Activation outputs can be reproduced by a leading-digit text edit | first_z-L26: 30/30 exact activation/first-digit sequence matches; label_z-L26: 30/32 exact activation/first-digit sequence matches | `digit_pairs` in primary analysis; jointly runnable pairs; exact sequence and final-answer endpoints separately |
| A complete donor value was transferred whenever its first digit appeared | Unsupported | `04-Results/analysis/secondary-v1/secondary.json`; complete numeral, leading digit and complete target counted separately |
| Visible recomputation after an effect occurs | Supported for the inspected pilot-000 early trace | `pilot/qwen06-pilot-v1/records.jsonl`, key `pilot-000/first_z-L26/activation/s0`: 51 -> 46 -> 91 -> 86; a readable trajectory, not private-state identification |
| Every return to the recipient answer proves correction | Unsupported | Operational changed/later-original/recipient-final signature is not sufficient alone; unchanged final can reflect ignored perturbation |
| The labelled intermediate boundary is early reasoning | False for the declared boundary | `*-boundaries.json` and secondary timing table; labelled value follows thinking |
| Norm-matched random control originally passed every check | False, failure preserved | `pilot-integrity-v1.json`: one vector, two stream records, 0.0001300014 > 0.0001; corrected operator separately qualified and rerun |
| Corrected random controls change the pilot conclusion | No observed output change | All 32 rerun sequences exactly agree; qualified `pilot/qwen06-random-repair-v2` |
| Numerical consistency is assumed from earlier studies | False | AI-04's 116-check Qwen audit and independent output/hash checks; bounded audit coverage only |
| Second model's initial formatting failure proves poor arithmetic | Unsupported | Correct visible arithmetic can lack labelled sites; `audits/deepseek-cpu-v1` failed 2/30 required-site checks |
| Second checkpoint and lookup task validate broad generality | Not claimed | Separate bounded competence/instrumentation gates and panels in report; no pooling with main estimate |
| Complete-expression versus computed-value transfer is identified | Unsupported | Both produce g_r(z_d) in this generator; collisions recorded |
| Replay is universally invalid or continuation selection is universally superior | Unsupported | B deliberately clamps original future text; primary contrast is limited to frozen policies and declared task |
| Efficiency advantage, full circuit identity, priority or publication readiness | Unsupported | Unequal selection budgets, few sites, small models, targeted literature review and no independent external review |

The report gives competent subsets alongside all-intended denominators. Saved raw matches retain alternative-hypothesis collisions; the primary categorical order does not eliminate them. Zero-event bootstrap intervals must be read with the separate one-sided group bound, and all secondary uncertainty is descriptive.
