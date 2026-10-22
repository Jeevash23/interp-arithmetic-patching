# Reproducing the single learned-subspace branch

This exploratory development branch follows the completed qualification and whole-vector studies in `../competence-v3`. It uses their pinned Qwen3-4B-Instruct-2507-FP8 checkpoint, isolated FP8 kernel and unchanged shared Python environment. No installation, new checkpoint, model-weight training or external service is needed.

From this directory use `../../.venv/bin/python`. Model loading needs host GPU access. Run `study_subspace.py --prepare-only` to reproduce the 756 fitting tasks. Their final answers are balanced within every intermediate-value class, and exact complete-task overlaps with the evaluation pairs and their source variants are excluded. Repeated numeric values are intentional.

Run `study_subspace.py` for fitting-vector capture, behavioral baselines, closed-form basis fitting and fixed causal controls. It requires completion of the eligible whole-vector test and refuses to run if that simpler method passed its progression criteria. It verifies or creates `source-freeze.json` before inference. Source and fitting data must match the freeze on resumption.

The fitting journal has 1,512 paired records: full-task and first-step-only outputs for each fitting task. Identical prompts may reuse deterministic outputs and record their source. The block-23 answer-prefix vector is captured from the full two-stage prompt before any answer generation, not from the first-step-only prompt. All 756 examples enter the class means irrespective of correctness.

`results/learned-freeze.json` records the rank-eight objective, singular values, seeds, algebra checks and artifact hashes before causal outcomes. The label-shuffled control permutes labels across individual examples. No basis is retuned after outcomes.

The evaluation journal has 504 records across 36 pairs and 14 conditions: 360 fresh generations and 144 reused comparisons. Unpatched generations, both textual first-token controls and the whole-vector write at the same site retain their original output IDs, text, parser result and provenance. Reused conditions are not additional independent observations. `source_hash` identifies the saved replacement target; for a projection this differs from the raw donor vector.

Run `analyze_subspace.py` after completion. It reports full-sample and original triple-competent counts, source-specific counterfactuals and competence, joint source invariance, competing predictions, invalid outputs, paired donor-group bootstrap summaries, and the fixed progression rules. These are development analyses; intervals do not adjust for site selection.

Run `audit_subspace.py` to recapture evaluation source vectors independently and reconstruct all 360 replacement targets, including seeded noise and random-subspace controls. It checks all 756 fitting-vector hashes and all 144 reused outputs, then writes the evaluation-source vector file and exact-prompt ledger. Float32 displacement norms and actual bfloat16 write norms are retained separately; rounding need not preserve norms exactly.

The original validation and hidden-study reserve are not loaded by these commands. The [protocol](PROTOCOL.md) fixes the single-method stopping rule: a later rank/site/strength search cannot silently replace this result.
