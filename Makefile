# What single-position activation patches transfer in Qwen3 arithmetic: code release.
# `make help` lists the targets. PYTHON defaults to the environment that `make setup` (uv sync) creates.
PYTHON ?= .venv/bin/python
PY := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
export PYTHONDONTWRITEBYTECODE := 1
export OPENBLAS_NUM_THREADS ?= 1
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
.NOTPARALLEL:

S1 := 06-Study-History/original-studies/study1-value-transfer-and-readout
POSTHOC := first_digit_equivalence operand_substitution screen_decomposition trace_exposure transfer_rates
STUDIES := 3-heldout-4b-confirmation 4-multidigit-tokens 5-direction-and-span 6-olmo2-7b-multidigit

.PHONY: help setup python-check unpack check-freezes quick-check numbers figures tokenisers tokeniser-check \
        models link-models

help:
	@echo "make setup            create .venv with the pinned environment (uv sync; downloads packages, see ENVIRONMENT.md)"
	@echo "make quick-check      CPU only, under a minute, no model and no network: freeze records, the 50 recounts,"
	@echo "                      five post hoc analyses, every number and table, and the studies' own analyses rerun"
	@echo "make numbers          rewrite paper/generated/ from the saved outputs"
	@echo "make figures          draw Figures 1-4 into paper/figures/"
	@echo "make tokenisers       download the three Qwen3 tokenisers (about 50 MB) for make tokeniser-check"
	@echo "make tokeniser-check  the post hoc tokeniser check (needs make tokenisers)"
	@echo "make models           download Qwen3-0.6B and Qwen3-4B-Instruct-2507-FP8 (about 6.7 GB), for model reruns"
	@echo "make link-models      put the downloaded checkpoints where the study code reads them"

setup:
	uv sync --frozen --extra experiment --group dev

python-check:
	@$(PY) -c "import numpy, matplotlib" 2>/dev/null || { echo "No usable Python at $(PYTHON): run make setup, or set PYTHON=..."; exit 1; }

unpack: python-check
	$(PY) tools/unpack.py

check-freezes: unpack
	$(PY) tools/check_freezes.py

quick-check: unpack
	$(PY) tools/check_freezes.py
	$(PY) analysis/reconcile_reports.py --check
	for s in $(POSTHOC); do $(PY) analysis/posthoc/$$s.py --check || exit 1; done
	$(PY) paper/make_numbers.py --check
	$(PY) tools/replay_analyses.py

numbers: unpack
	$(PY) paper/make_numbers.py

figures: unpack
	$(PY) paper/make_figures.py

# The tokeniser check compares the tokeniser files of the three Qwen3 models at the paths the original studies
# kept them; the files are copied there (links would change the paths the check records).
tokenisers: python-check
	$(PY) tools/get_models.py --no-weights qwen3-0.6b qwen3-1.7b qwen3-4b-fp8
	cp -f models/qwen3-0.6b/tokenizer.json $(S1)/models/Qwen3-0.6B/tokenizer.json
	mkdir -p $(S1)/followup-readout/checkpoint-replication/model
	cp -f models/qwen3-1.7b/tokenizer.json $(S1)/followup-readout/checkpoint-replication/model/tokenizer.json
	cp -f models/qwen3-4b-fp8/tokenizer.json $(S1)/hidden-value-transfer/competence-v2/models/qwen4b/tokenizer.json

tokeniser-check: unpack
	@[ -f $(S1)/followup-readout/checkpoint-replication/model/tokenizer.json ] || { echo "run make tokenisers first"; exit 1; }
	$(PY) analysis/posthoc/tokenizer_check.py --check

models: python-check
	$(PY) tools/get_models.py qwen3-0.6b qwen3-4b-fp8

# Study 1's readout experiment and Study 2 read Qwen3-0.6B from Study 1's models/Qwen3-0.6B; the Qwen3-4B
# pipeline reads its checkpoint through code/frozen/competence-v2/models (a link to the study's models/ folder in
# Studies 3 to 6; a folder in Study 1). Studies 4 and 6 download their candidates themselves (see their README).
link-models:
	@[ -f models/qwen3-0.6b/model.safetensors ] && [ -f models/qwen3-4b-fp8/model.safetensors ] || { echo "run make models first"; exit 1; }
	for f in models/qwen3-0.6b/*; do ln -sfn "$(CURDIR)/$$f" $(S1)/models/Qwen3-0.6B/; done
	mkdir -p studies/3-heldout-4b-confirmation/models/qwen4b
	for f in models/qwen3-4b-fp8/*; do \
	  ln -sfn "$(CURDIR)/$$f" $(S1)/hidden-value-transfer/competence-v2/models/qwen4b/; \
	  ln -sfn "$(CURDIR)/$$f" studies/3-heldout-4b-confirmation/models/qwen4b/; done
	for s in $(STUDIES); do [ -e studies/$$s/code/frozen/competence-v2/models ] || \
	  ln -s ../../../models studies/$$s/code/frozen/competence-v2/models; done
