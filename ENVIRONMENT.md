# Environment

Studies 1 to 3 ran on one 6 GB NVIDIA GPU in a single Python environment, and the cloud images of Studies 5 and 6 were
built from the same lock. `uv.lock` records it exactly (180 locked packages, including this release as a virtual
project); `pyproject.toml` declares it as the original project did: the `experiment` extra is the full research
environment and the `dev` group adds pytest.

| Package | Version |
|---|---|
| Python | 3.12.14 (`uv` installs it if it is missing) |
| torch | 2.13.0 (CUDA 13 build from PyPI) |
| transformers | 5.16.1 |
| numpy | 2.5.2 |
| matplotlib | 3.11.1 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.29.0 |

The checks in this release need only NumPy and Matplotlib: the recount, the post hoc analyses and the number script
use the Python standard library, Study 3's analysis uses NumPy for its bootstrap, and the figure script uses
Matplotlib. The model code imports torch and transformers.

## A fresh install

```sh
uv sync --frozen --extra experiment --group dev        # what make setup runs
```

On Linux x86-64 this downloads 176 packages, about 3.0 GB of wheels. The largest are torch (527 MB) and the
NVIDIA CUDA libraries it depends on (about 2.0 GB together), then triton (198 MB). The sizes come from `uv.lock`;
installing creates `.venv/` in this folder.

## A smaller CPU-only environment

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv numpy==2.5.2 matplotlib==3.11.1
```

This is enough for `make quick-check`, `make numbers` and `make figures`. It is not the locked environment; for
model reruns use `make setup`.

## The FP8 kernel of the Qwen3-4B experiments

Study 1's answer-only experiments and Studies 3 and 5 ran Qwen3-4B-Instruct-2507-FP8 with the `finegrained-fp8`
kernel of `kernels-community`, loaded through the `kernels` package (version 0.17.1) at kernel revision
`8c178950e8710e26a2210e4e909cd02dc16c8715`. The frozen pipeline (`code/frozen/competence-v2/study.py` in each of
those studies, and `hidden-value-transfer/competence-v2/study.py` in Study 1) imports `kernels` from a
`.packages/` folder next to it and reads the kernel from a Hugging Face cache in `.cache/huggingface/` there, with
the Hub set to offline mode. Neither folder is included (a third-party package and a download cache). To prepare
them for a rerun, run this in the folder that holds `study.py`, with `PY` set to the release environment's
Python (for example the `.venv/bin/python` that `make setup` creates):

```sh
uv pip install --python "$PY" --target .packages kernels==0.17.1
PYTHONPATH=.packages HF_HUB_CACHE=.cache/huggingface "$PY" -c \
  "import kernels; kernels.get_kernel('kernels-community/finegrained-fp8', revision='8c178950e8710e26a2210e4e909cd02dc16c8715')"
```

## Environments recorded by the original studies

Kept as records, not needed for the checks:

- Study 1: `configs/environment-source.uv.lock` and `docs/ENVIRONMENT.md` (the September development stage).
- Study 2: `uv.lock`, `pyproject.toml` and `06-Study-History/environment/requirements.lock.txt` (torch, transformers,
  NumPy, Matplotlib and safetensors at the versions above).
- Studies 3 and 5: the `environment` section of `FREEZE.json` (package versions and the digest of the lock).
- Studies 5 and 6: `code/modal_app.py` builds the cloud image from this folder's `pyproject.toml` and `uv.lock`.
