# Reconciliation of recounted numbers against the two original reports

50 of 50 checks pass. Each row recounts a figure from the journals in the original study folders (Study 1: `study1-value-transfer-and-readout`, Study 2: `study2-reasoning-interventions`, both under `06-Study-History/original-studies/`) and compares it with the number stated in the Study 1 draft (2 October 2026) or the Study 2 report (21 September 2026).

| Claim | Recount | Stated | Result | Journal |
|---|---|---|---|---|
| AI-02 selected configuration donor full 33/36, recipient full 36/36 | ((33, 36), (36, 36)) | ((33, 36), (36, 36)) | pass | competence-v3 baselines.jsonl |
| AI-02 calculator-only 4B donor 30/36, recipient 24/36 | ((30, 36), (24, 36)) | ((30, 36), (24, 36)) | pass | competence-v2 qwen4b baselines.jsonl |
| AI-02 Qwen2.5-1.5B donor 6/36, recipient 4/36 | ((6, 36), (4, 36)) | ((6, 36), (4, 36)) | pass | competence-v2 qwen25 baselines.jsonl |
| AI-02 Qwen3-1.7B initial donor two-step 0/36 | (0, 36) | (0, 36) | pass | hidden-value-transfer/results/baselines.jsonl |
| AI-02 screen has 5,184 records | 5184 | 5184 | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/competence-v3/results/qwen4b-fewshot/screen.jsonl |
| AI-02 ratio-selected sites are blocks 20, 9, 10 at the first operand | [(20, 'a_end'), (9, 'a_end'), (10, 'a_end')] | [(20, 'a_end'), (9, 'a_end'), (10, 'a_end')] | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/competence-v3/results/qwen4b-fewshot/screen.jsonl |
| AI-02 ratio-selected mean changes 30.405, 30.159, 30.159 nats | [30.405, 30.159, 30.159] | [30.405, 30.159, 30.159] | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/competence-v3/results/qwen4b-fewshot/screen.jsonl |
| AI-02 eligible-site revision chooses blocks 14, 20, 23 | [14, 20, 23] | [14, 20, 23] | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/competence-v3/results/qwen4b-fewshot/screen.jsonl |
| AI-02 ratio sites: 0/36 complete counterfactuals at blocks 20, 9, 10 | [0, 0, 0] | [0, 0, 0] | pass | competence-v3 controls.jsonl |
| AI-02 ratio sites: first-operand-only outputs 16, 19, 19 | [16, 19, 19] | [16, 19, 19] | pass | competence-v3 controls.jsonl |
| AI-02 ratio sites: first-digit text control 1/36 at each site | [1, 1, 1] | [1, 1, 1] | pass | competence-v3 controls.jsonl |
| AI-02 noise unchanged 35, 35, 36 at ratio sites | [35, 35, 36] | [35, 35, 36] | pass | competence-v3 controls.jsonl |
| AI-02 revised sites: complete counterfactuals 1, 0, 2 | [1, 0, 2] | [1, 0, 2] | pass | eligible-sites controls.jsonl |
| AI-02 revised sites: same-leading-digit sources identical on 33, 35, 26 pairs | [33, 35, 26] | [33, 35, 26] | pass | eligible-sites controls.jsonl |
| AI-02 revised sites: unchanged recipient 3, 35, 5 | [3, 35, 5] | [3, 35, 5] | pass | eligible-sites controls.jsonl |
| AI-02 block 14 different-z/same-answer source counterfactual 4/36 | 4 | 4 | pass | eligible-sites controls.jsonl |
| AI-02 joint original-and-both-same-z successes are zero at all six whole-vector sites | [0, 0, 0, 0, 0, 0] | [0, 0, 0, 0, 0, 0] | pass | competence-v3 and eligible-sites controls.jsonl |
| AI-02 components: 1/36 original counterfactual each | [1, 1] | [1, 1] | pass | component-v1 controls.jsonl |
| AI-02 components: unchanged 10 and 15 | [10, 15] | [10, 15] | pass | component-v1 controls.jsonl |
| AI-02 components: successes only under multiplication | [{'mul': 1}, {'mul': 1}] | [{'mul': 1}, {'mul': 1}] | pass | component-v1 controls.jsonl |
| AI-02 subspace: 1/36 original counterfactual, 23/36 unchanged | (1, 23) | (1, 23) | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/subspace-v1/results/controls.jsonl |
| AI-02 subspace: noise 0 (35 unchanged), random subspace 0 (34 unchanged), shuffled labels 0 (6 unchanged) | [(0, 35), (0, 34), (0, 6)] | [(0, 35), (0, 34), (0, 6)] | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/subspace-v1/results/controls.jsonl |
| AI-02 subspace: same-z sources 2/36 and 1/36 with one joint success | (2, 1, 1) | (2, 1, 1) | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/subspace-v1/results/controls.jsonl |
| AI-02 subspace: same-leading-digit sources identical on 32/36 | 32 | 32 | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/subspace-v1/results/controls.jsonl |
| AI-02 subspace: success only under multiplication | {'mul': 1} | {'mul': 1} | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/subspace-v1/results/controls.jsonl |
| AI-02 subspace fitting set: first step 756/756, full problem 677/756 | ((756, 756), (677, 756)) | ((756, 756), (677, 756)) | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/subspace-v1/results/training-baselines.jsonl |
| AI-02 qualified sources: 0/72 full-value successes for both sources | (0, 0) | (0, 0) | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/subspace-v1/value-digit-qualified-sources/results/controls.jsonl |
| AI-02 qualified sources: coarse 3/81 and 7/81; text 12/81 | (3, 7, 12) | (3, 7, 12) | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/subspace-v1/value-digit-qualified-sources/results/controls.jsonl |
| AI-02 qualified sources: no joint full or coarse success across sources | (0, 0) | (0, 0) | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/subspace-v1/value-digit-qualified-sources/results/controls.jsonl |
| AI-02 qualified sources: 27 cells, none all-full, one all-coarse, 19 identical | (27, 0, 1, 19) | (27, 0, 1, 19) | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/subspace-v1/value-digit-qualified-sources/results/controls.jsonl |
| AI-02 qualified sources: primary coarse matches all under addition | {'add': 3} | {'add': 3} | pass | 06-Study-History/original-studies/study1-value-transfer-and-readout/hidden-value-transfer/subspace-v1/value-digit-qualified-sources/results/controls.jsonl |
| Readout: activation 16/240 versus full-value text 151/240 | (16, 151) | (16, 151) | pass | followup-readout interventions.jsonl |
| Readout: first-digit text reproduces activation output 233/238 | (233, 238) | (233, 238) | pass | followup-readout interventions.jsonl |
| Readout: first emitted token matches donor 240/240 | (240, 240) | (240, 240) | pass | followup-readout interventions.jsonl |
| Readout: paired short and long donors identical 117/120 | (117, 120) | (117, 120) | pass | followup-readout interventions.jsonl |
| AI-04 held-out baselines donor 59/64, recipient 61/64, supplied 57/64 | [59, 61, 57] | [59, 61, 57] | pass | qwen06-main-v1 records.jsonl |
| AI-04 held-out natural EOS 60, 63, 61 | [60, 63, 61] | [60, 63, 61] | pass | qwen06-main-v1 records.jsonl |
| AI-04 held-out correct intermediate labels 56, 60, 58 | [56, 60, 58] | [56, 60, 58] | pass | qwen06-main-v1 records.jsonl |
| AI-04 held-out: 0/128 at label_z-L20 and label_z-L26 | [0, 0] | [0, 0] | pass | qwen06-main-v1 records.jsonl |
| AI-04 held-out: recipient answers 113/128 and 91/128; other numbers 4 and 26 | [(113, 4), (91, 26)] | [(113, 4), (91, 26)] | pass | qwen06-main-v1 records.jsonl |
| AI-04 held-out: 8 unavailable, 2 exhausted, 1 invalid per site | [(8, 2, 1), (8, 2, 1)] | [(8, 2, 1), (8, 2, 1)] | pass | qwen06-main-v1 records.jsonl |
| AI-04 held-out: competent subset 0/104 at both sites | [(0, 104), (0, 104)] | [(0, 104), (0, 104)] | pass | qwen06-main-v1 records.jsonl |
| AI-04 controls: full-value text 5/32 early and 3/32 late; activation 0/32 at both | [5, 3, 0, 0] | [5, 3, 0, 0] | pass | qwen06-controls-v1 records.jsonl |
| AI-04 controls: full-value text exhausts 11/32 early; activation 7/32 | [11, 7] | [11, 7] | pass | qwen06-controls-v1 records.jsonl |
| AI-04 controls: random directions give 0 targets and 29 or 30 recipient answers | [(0, 29), (0, 30)] | [(0, 29), (0, 30)] | pass | qwen06-controls-v1 records.jsonl |
| AI-04 controls: first-digit text reproduces activation 30/30 early and 30/32 late | [(30, 30), (30, 32)] | [(30, 30), (30, 32)] | pass | qwen06-controls-v1 records.jsonl |
| AI-04 development C counts 0, 2, 2, 0, 0, 4 | [0, 2, 2, 0, 0, 4] | [0, 2, 2, 0, 0, 4] | pass | qwen06-development-v2 records.jsonl |
| AI-04 development mean A changes 0.0442 ... 28.0946 | [0.0442, 23.5377, 28.2053, 0.1344, 12.677, 28.0946] | [0.0442, 23.5377, 28.2053, 0.1344, 12.677, 28.0946] | pass | qwen06-development-v2 records.jsonl |
| AI-04 donor semantics: same-leading wide sources produce none of their own targets and reproduce 8/8 original sequences | [(0, 8), (0, 8)] | [(0, 8), (0, 8)] | pass | qwen06-donor-controls-v1 records.jsonl |
| AI-04 donor semantics: different-z/same-answer sources reproduce 3/8 early and 4/8 late sequences | [3, 4] | [3, 4] | pass | qwen06-donor-controls-v1 records.jsonl |
