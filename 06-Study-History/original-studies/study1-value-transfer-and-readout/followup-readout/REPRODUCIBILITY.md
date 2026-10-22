# Reproducing the follow-up

Run commands from the parent project `<study1-alias>`. The `.venv` still reuses project 04's installed environment. Original model hashes and environment inventory remain unchanged. No clean-environment installation has been verified.

The original pre-outcome freeze is preserved in `source-before-budget-fix/freeze.json`. The active `freeze.json` additionally records the single exhausted-budget implementation amendment, made during baseline generation and before intervention outcomes. Dataset, sampling, layer, parser, scientific criteria and statistical analysis did not change. Both versions of the runner and the interruption log remain available. The latest execution log is `logs/execution-attempt-2.log`.

The main run resumes completed records rather than redoing them:

```bash
.venv/bin/python followup-readout/run_replication.py
.venv/bin/python followup-readout/analyze_replication.py
.venv/bin/python followup-readout/draw_design.py
.venv/bin/python followup-readout/build_paper.py
```

The dataset and freeze creator intentionally refuses to overwrite a freeze. To perform an independent rerun, copy the project with a fresh results branch and explicitly document the new run. Do not change a frozen source or configuration and resume into the current journal. Reusing saved recipient prompts across donor widths is part of the paired design; it is not an additional independent observation.

The second checkpoint has its own protocol, revision/file manifest and freeze. It must run after the first GPU process ends:

```bash
.venv/bin/python followup-readout/checkpoint-replication/run_checkpoint.py
.venv/bin/python followup-readout/checkpoint-replication/analyze_checkpoint.py
```

A failed second-checkpoint instrumentation gate closes that branch before a mechanism claim. Lower precision is a different execution condition, even when the model architecture belongs to the same family. Its tolerances and identity requirements must not be relaxed after seeing failures.

The PDF uses the separately bundled document runtime, leaving research packages untouched:

```bash
<runtime>/dependencies/python/bin/python3 followup-readout/build_pdf.py
pdftoppm -r 120 -png followup-readout/output/pdf/intermediate-value-readout-paper.pdf followup-readout/tmp/pdfs/final
```

Native output equivalence, cached/fixed-node comparisons, source hashes, exact identities, parser/score reconstruction, complete record counts and token bookkeeping are checked in the analysis audit. Figures and rendered PDF pages are inspected separately. Neither code/audit hashes nor seeded determinism establish scientific novelty or universal model behavior.
