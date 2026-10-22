# What single-position activation patches transfer in Qwen3 arithmetic

Code and results for the preprint *What single-position activation patches transfer in Qwen3 arithmetic: operands and output digits, not intermediates*, by Richard Jeevash (link to be added when it is posted).

This repository holds the code, frozen inputs, study records and saved outputs behind it; the saved outputs and the
frozen protocols are attached to its v1.0 release (see [Release files](#release-files)). Every number, generated
table and plotted value in the manuscript can be recomputed from the saved outputs on a CPU in under a minute,
without a model or network.

## Release files

The saved outputs and the frozen protocols are not in this repository's files; they are attached to its v1.0 release:

| File | Size | Contents |
|---|---|---|
| `per-run-outputs.tar.gz` | 7.0 MB | The saved outputs: the 6 compressed archives in `data/saved-outputs/`, which `make unpack` places in the study folders and checks file by file |
| `frozen-protocols.tar.gz` | 96 kB | The studies' protocols, predictions, amendments, approvals and freeze records (62 files), at their places in the folder, where the checks read them |
| `SHA256SUMS` | | Checksums of the two archives |

Download the three files into an empty folder, check them there, and unpack them into this one:

```sh
sha256sum -c SHA256SUMS
tar -xzf per-run-outputs.tar.gz -C /path/to/interp-arithmetic-patching
tar -xzf frozen-protocols.tar.gz -C /path/to/interp-arithmetic-patching
```

After unpacking, `sha256sum -c SHA256SUMS` in this folder checks every file that its `SHA256SUMS` lists, the
unpacked attachments included, and the checks below run as described.

## What the paper asks

Activation patching asks whether a hidden state carries a value: copy the state from a donor run into a recipient
run and see whether the recipient now behaves as if it held the donor's value. The paper applies that test to
two-step arithmetic, `z = f(a, b)` then `y = g(z)`, in Qwen3-0.6B (which reasons in a written thinking trace) and
Qwen3-4B-Instruct-2507-FP8 (which answers directly), with one residual-stream vector replaced at one position and
the complete generated answer scored. What a write transfers depends on where it is made: the intermediate rarely
transfers; late in Qwen3-0.6B a write sets the next token; at the Qwen3-4B answer prefix it carries more than the
next token but rarely the intermediate; at operand positions it transfers the operand in about half of the pairs.

## Studies

| Manuscript | Folder | What it did |
|---|---|---|
| Study 1 | `06-Study-History/original-studies/study1-value-transfer-and-readout` | Readout experiment in Qwen3-0.6B (block 26, 240 attempts); answer-only experiments in Qwen3-4B (144-site screen, generation at six sites, rank-8 projection, attention and MLP writes, 81-pair diagnostic) |
| Study 2 | `06-Study-History/original-studies/study2-reasoning-interventions` | Held-out experiment in Qwen3-0.6B: sites frozen on 24 development programs, evaluated on 64 held-out programs, control panel of 16 programs |
| Study 3 | `studies/3-heldout-4b-confirmation` | Held-out confirmation of the Qwen3-4B results on 72 reserved pairs, protocol and decision rules frozen before any test output |
| Study 4 | `studies/4-multidigit-tokens` | The answer-only assay in models whose tokeniser writes numbers up to 999 as single tokens; ended at the competence gate (two candidates failed it) |
| Study 5 | `studies/5-direction-and-span` | An added mean-difference direction at every block and four positions, and writes over the first-step sentence without its operand tokens, in Qwen3-4B on Study 3's held-out pairs; cloud GPUs |
| Study 6 | `studies/6-olmo2-7b-multidigit` | Study 4's design with OLMo-2-1124-7B-Instruct; ended at the competence gate; cloud GPU |

Studies 1 and 2 are complete original study trees: their code, protocols, amendments, freeze records, inputs and
saved outputs, without the parts listed under "What is not included". Studies 3 to 6 each keep their protocol,
predictions, freeze records, code and results. The freezes are local hash records written by the study code, not
public preregistrations.

## What is in this folder

| Path | Contents |
|---|---|
| `analysis/reconcile_reports.py` | Recounts 50 figures stated in the two original study reports from the saved journals (`paper/generated/reconciliation/`) |
| `analysis/posthoc/` | The post hoc analyses (first-digit equivalence, operand substitution, screen decomposition, trace exposure, tokeniser check, transfer rates) and their results |
| `paper/make_numbers.py`, `paper/make_figures.py` | Every macro and generated table of the manuscript, and its four figures, from the saved outputs; `paper/generated/` holds the values the manuscript was built with |
| `studies/`, `06-Study-History/original-studies/` | The six studies (above); after `make unpack`, also their saved outputs |
| `provenance/CLAIM-TO-EVIDENCE.md` | Each claim of the manuscript of 9 October with the macros, saved outputs and scripts behind it (Studies 5 and 6 were added later; their numbers come from their `results/summary.json`) |
| `RESULTS.md` | The findings with their numbers |
| `data/saved-outputs/` | Saved outputs as compressed archives (see its README) (attached to the release: `per-run-outputs.tar.gz`) |
| `tools/`, `models/` | Unpacking, freeze checking, the replay of the studies' analyses, and model download helpers |
| `pyproject.toml`, `uv.lock` | The exact environment of the runs (see `ENVIRONMENT.md`) |
| `RELEASE-CHANGES.md`, `release-changes.json` | Every file that differs from the original project, with the original's SHA-256, and every frozen file not shipped |

## Install

```sh
make setup        # uv sync --frozen --extra experiment --group dev: creates .venv from uv.lock
```

This downloads about 3.0 GB of packages (most of it PyTorch's CUDA libraries); the checks below need only NumPy and
Matplotlib, and `ENVIRONMENT.md` gives a small CPU-only route. Pass `PYTHON=/path/to/python` to any `make` target to
use another environment.

## Quick check (CPU, under a minute, no model, no network)

First unpack the two release attachments into this folder (see [Release files](#release-files)): the checks read
the saved outputs and the freeze records they contain.

```sh
make quick-check
```

It unpacks the saved outputs and compressed inputs (checking every SHA-256), checks every freeze record against the
files it names, recounts the 50 figures of the original reports, reruns five post hoc analyses and compares them
with their saved results, recomputes every macro and generated table of the manuscript (`paper/make_numbers.py
--check`), and reruns the saved-output analyses of Studies 3, 5 and 6 in a temporary copy, comparing each output
with the shipped file byte for byte. Expected end of the output:

```text
total: 34 changed for the release (documented), 7 frozen copy matches, 382 not shipped (documented), 6 superseded by a recorded amendment, 367 unchanged
50/50 checks pass; 0 saved output(s) differ from the recount
ok  287 macros and 10 tables match the saved outputs
10 analyses rerun from the saved records: every output identical to the shipped file
```

"Frozen copy matches" are Study 2's early runs: the study changed its code after them and kept every version it ran in
a `source/` folder beside the run, which the freeze check compares instead.

## More from the saved outputs

```sh
make numbers          # rewrite paper/generated/
make figures          # Figures 1-4 into paper/figures/
make tokenisers       # download the three Qwen3 tokenisers (about 50 MB) ...
make tokeniser-check  # ... and rerun the post hoc tokeniser check
```

Study 3's independent verifier (`studies/3-heldout-4b-confirmation/verify.py`) hashes the model files and the saved
write vectors (377 MB per wording), which are not included; its saved result is `results/verification.json` (35 of
35 checks passed), which `make quick-check` requires. Study 5's verifier is rerun by the quick check.

## Model reruns

Nothing in the manuscript needs a model to run again. Every model stage refuses to overwrite its outputs, so a rerun
starts from a copy of the study folder with its results removed, following the study's own instructions (its
`README.md`, `REPRODUCIBILITY.md` or `docs/REPRODUCIBILITY.md`). Studies 1 to 4 ran on one 6 GB NVIDIA GPU: Study 3's test took 76 GPU-minutes for both wordings, with at most 5.6 GB of GPU memory. Studies 5 and
6 ran on Modal with one NVIDIA L4 per worker: about 3.5 L4-hours of generation for Study 5 and 143 seconds for Study 6's gate.

```sh
make models           # Qwen3-0.6B and Qwen3-4B-Instruct-2507-FP8 at the pinned revisions (about 6.7 GB)
make link-models      # put them where the study code reads them
```

`models/README.md` lists every checkpoint. The Qwen3-4B pipeline also needs the `kernels` package and the FP8 kernel
in the frozen code's `.packages/` and `.cache/` folders; `ENVIRONMENT.md` shows how to prepare them.

### Studies 5 and 6 on Modal (optional, your own account)

From the study folder, with the Modal CLI logged in to your own account: `modal run --detach
code/modal_app.py::<stage>` for each stage that the study's `README.md` lists (`run_stage.sh` retries a stage that
failed to connect), then `modal run code/modal_app.py::pull --stage <stage>` to copy the records back. The image is
built from this folder's `pyproject.toml` and `uv.lock`, the weights are downloaded inside it, and the records go to
a Modal Volume created in your workspace. `watch_billing.sh` stops the study's apps at the cost limits of Study 5's
protocol. Study 5 cost $4.47 and Study 6 $0.07 on Modal's billing report.

## Tables and figures

| Manuscript | Source |
|---|---|
| All numbers in the text | macros in `paper/generated/numbers.tex`, written by `paper/make_numbers.py` |
| Table 2 (complete transfer) | `paper/generated/table-transfer.tex` |
| Table 3 (operand substitution) | `paper/generated/table-operand.tex` |
| Table 4 (Study 3) | `paper/generated/table-study3.tex` |
| Table 5 (Study 5) | `paper/generated/table-study5.tex` |
| Table 6 (Qwen3-4B sites) | `paper/generated/table-fourb-sites.tex` |
| Table 7 (Qwen3-4B variants) | `paper/generated/table-fourb-variants.tex` |
| Table 8 (held-out experiment) | `paper/generated/table-heldout.tex` |
| Table 9 (competence) | `paper/generated/table-competence.tex` |
| Table 10 (Study 4) | `paper/generated/table-study4.tex` |
| Table 11 (Study 5 sites) | `paper/generated/table-study5-sites.tex` |
| Table 1 (settings) | written in the manuscript's source |
| Figures 1-4 | `paper/make_figures.py` → `paper/figures/fig-one-token.pdf`, `fig-operand.pdf`, `fig-screen.pdf`, `fig-study5.pdf` |

The values come from the recount (`paper/generated/reconciliation/numbers.json`), the post hoc results
(`analysis/posthoc/results/`) and the studies' saved summaries (`studies/*/results/summary.json`); the post hoc
analyses are labelled post hoc in the manuscript.

## What is not included

- Model weights, tokeniser files and the FP8 kernel package (download them; see `models/README.md` and
  `ENVIRONMENT.md`).
- The saved write vectors and other saved tensors (Study 3's `control-vectors.pt`, 377 MB per wording; Study 1's
  `.pt` files, 767 MB), logs, rendered figures, the original studies' written reports and drafts, review notes,
  briefs, the reference papers the original studies collected, code snapshots the studies kept as history, Study 2's
  follow-up `followups/causal-carriers-v1` (which contributes no result to the manuscript), and the original
  project's provenance history and manuscript source. Frozen files not shipped are listed with their SHA-256 in
  `RELEASE-CHANGES.md`.

## Changes from the original project

Files are copied byte for byte, except those listed in `RELEASE-CHANGES.md` with the original's SHA-256: the two
original study folders are renamed (`study1-value-transfer-and-readout`, `study2-reasoning-interventions`) in paths
and text; records had personal paths, other project folders' names and the names of software tools in working records replaced by placeholders (the assistants used are named under "Use of AI assistants" below); the
cloud apps of Studies 5 and 6 and Study 3's freeze script read the environment files from the release root; and
Study 5's verifier accepts the documented changes when it checks freeze records.

## Licence

MIT (`LICENSE`), covering the code and the studies' records and saved outputs. The model checkpoints are distributed under their own licences.

## Study codes

The records and code name the studies by the codes they had while the work was done. Each code that occurs here,
and where that study is released:

| Code | Released in |
|---|---|
| AI-01 | `interp-answer-repair` |
| AI-02 | this repository |
| AI-02-04 | this repository |
| AI-03 | `interp-donor-pooling` |
| AI-04 | this repository |

## Use of AI assistants

As the preprint states:

> Language-model assistants (Claude, Anthropic; Codex, OpenAI) were used throughout this work, under the author's
> direction, to help design the studies, to write and run the code for the experiments and analyses, to draft and
> revise the text and to check the references against their online records. All reported results come from the
> saved study outputs, and the numbers in the text were generated from those outputs or checked against them.

In the working records shipped here, the names of software tools were replaced by placeholders such as
`[redacted]`; the assistants used are the ones named above.

## Citation

Please cite the preprint (the link will be added here when it is posted) and, for the code and results, this
repository; [CITATION.cff](CITATION.cff) has the details.

```bibtex
@software{interp_arithmetic_patching_2026,
  author  = {Jeevash, Richard},
  title   = {{What single-position activation patches transfer in Qwen3 arithmetic: code and results}},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/Jeevash23/interp-arithmetic-patching}
}
```
