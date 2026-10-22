# Compositional value transfer during generation

Local Qwen3-0.6B research begun 9 September 2026. The experiment tests whether an activation transfers donor intermediate value z while preserving the recipient's remaining operation.

The completed English development experiment found a late **next-token readout effect**: 8/100 counterfactuals, no complete two-digit donor values, and 93/97 continuations reproduced by inserting only the donor's first digit as text. The candidate failed its declared progression gate. Validation remains reserved. Spanish/Tamil work is a separately labelled behavioral supplement using draft translations.

Start with:

- [Measured results and figures](RESULTS.md).
- [Research decision and claim boundaries](docs/DECISION.md).
- [Living manuscript](paper/manuscript.md) and [prioritized next experiments](docs/NEXT-STEPS.md).
- [Reproduction and audit instructions](docs/REPRODUCIBILITY.md).
- [Protocol](docs/PROTOCOL.md), numbered amendments and [research log](docs/RESEARCH-LOG.md).
- [Literature comparison](docs/RELATED-WORK.md), [environment](docs/ENVIRONMENT.md) and [measurement definitions](docs/MEASUREMENT-BOUNDARIES.md).

The `.venv` symlink reuses `../<environment-project>/.venv` without changing its packages. Model files are revision-pinned and hashed. Executed instrumentation uses native Hugging Face Qwen3 with PyTorch block-output hooks; inspected TransformerLens support was not the execution backend. All inference is local and uses no paid cloud service.

Code is under `src/` and `scripts/`; deterministic data and manifests under `data/`; complete outputs, failed configurations, saved vectors, source snapshots and analyses under `results/`; commands and errors under `logs/`. V1/v2 failures are retained. Existing journals resume completed rows and do not constitute independent reruns. Do not change a scientific configuration and resume into the same journal.

This project is an exploratory research result, not a submission-ready paper or a claim that reusable hidden values do not exist.
