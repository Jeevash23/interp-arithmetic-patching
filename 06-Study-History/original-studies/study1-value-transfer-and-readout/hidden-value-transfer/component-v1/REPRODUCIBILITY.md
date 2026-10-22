# Two fixed component interventions

Use `../../.venv/bin/python study_components.py` from this directory. Host GPU access is needed. The script uses the existing pinned 4B FP8 model and three-example context, without installing packages or training weights.

The two tested tensors are block 23's attention output projection (`self_attn.o_proj`) and feed-forward down projection (`mlp.down_proj`), at the last assistant-prefix token. `model-forward-inspection.json` stores the installed Qwen3 forward definitions and source hash confirming the order of projections and residual additions. The intervention replaces a contribution before the residual addition, retaining the other recipient terms. It is reapplied at the same prompt node during every no-cache generation step.

`freeze.json` covers the protocol, source, exact original development data, imported implementations and model-forward inspection. The common runtime and exact examples are recorded in `results/runtime`. Each component has its own instrumentation, controls, vector file, origin ledger, audit and analysis. The original 216 behavioral and 432 source-baseline records are inherited with hashes, not rerun or counted as fresh evidence.

Before either component's 360-control journal, the script repeats native/custom generation, read/identity/captured-self checks at three blocks and four roles, captured-self generation and a positive write. It stops that component if instrumentation fails. All original scientific thresholds are unchanged.

`analyze_components.py --candidate attention` or `--candidate mlp` reruns the original progression analysis for that component. Its scientific calculations match competence-v2. The former residual-screen heatmap is intentionally disabled: no component likelihood grid was conducted. The selected block comes from earlier adaptive residual-state discovery, and must not be described as independent component localization.

The audit verifies all 288 stored vector hashes and 72 identity outputs per completed component, as well as write locality. `actual-source-ledger.jsonl` distinguishes real donor activations, recipient self states, and conditions without an activation donor; the raw inherited schema's default donor reference does not establish an activation origin for noise or text conditions. The actual-donor-first-token text comparison reuses the original site-independent outputs with explicit provenance.

These two fixed checks do not authorize a further head, neuron, block, strength or position search. Consult each component's `completed.json`, `stopped.json` if present, and `analysis/summary.json` before describing it as complete or successful.
