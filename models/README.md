# Model checkpoints

No weights or tokeniser files are included, except the two `config.json` files from which `paper/make_numbers.py`
reads the layer counts (Study 1's `models/Qwen3-0.6B/config.json` and
`hidden-value-transfer/competence-v2/models/qwen4b/config.json`). `tools/get_models.py` downloads a checkpoint
into `models/<folder>/` at the revision the studies used and stops if any file's SHA-256 differs from the record:

| Name and folder | Hugging Face repository | Revision | Size | Used by |
|---|---|---|---|---|
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | 1.52 GB | Study 1 readout experiment, Study 2; the tokeniser check |
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 4.08 GB | Study 1 checkpoint replication; the tokeniser check |
| `qwen3-4b-fp8` | `Qwen/Qwen3-4B-Instruct-2507-FP8` | `8591804019c8b22094c3b5b4454e0edc05dffc98` | 5.21 GB | Study 1 answer-only experiments, Studies 3 and 5; the tokeniser check |
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 3.10 GB | Study 1 qualification attempt (failed) |
| `phi-4-mini-instruct` | `microsoft/Phi-4-mini-instruct` | `cfbefacb99257ffa30c83adab238a50856ac3083` | 7.69 GB | Study 4 competence gate |
| `smollm3-3b` | `HuggingFaceTB/SmolLM3-3B` | `a07cc9a04f16550a088caea529712d1d335b0ac1` | 6.17 GB | Study 4 competence gate |
| `olmo-2-1b-instruct` | `allenai/OLMo-2-0425-1B-Instruct` | `48d788eca847d4d7548f375ad03d3c9312f6139e` | 0.01 GB | Study 4 tokeniser check (candidate 3, not run); tokeniser files only |
| `olmo-2-7b-instruct` | `allenai/OLMo-2-1124-7B-Instruct` | `470b1fba1ae01581f270116362ee4aa1b97f4c84` | 14.61 GB | Study 6 competence gate (downloaded inside its cloud image) |

`models.json` lists every file with its SHA-256 as the studies recorded it (Study 1's model manifests, Study 4's and
Study 6's weight checks).

```sh
python tools/get_models.py --list                     # the table above
python tools/get_models.py qwen3-0.6b qwen3-4b-fp8    # what Studies 1 to 3 and 5 need (make models)
python tools/get_models.py --no-weights qwen3-1.7b    # configuration and tokeniser files only
python tools/get_models.py --check                    # check what is here, without network access
```

`make link-models` then puts the files where the study code reads them: Study 1's `models/Qwen3-0.6B/` (also
read by Study 2's code), Study 1's `hidden-value-transfer/competence-v2/models/qwen4b/`, and Study 3's
`models/qwen4b/`, reached from each study's `code/frozen/competence-v2/models`. Studies 4 and 6 fetched their
candidates with their own code (`code/download.sh`, `code/hub_files.py`; Study 6 inside its cloud image).

The checkpoints are distributed under their own licences: Apache 2.0 for Qwen3, Qwen2.5, SmolLM3 and OLMo 2, MIT
for Phi-4-mini. The two `config.json` files included here are Qwen3's (Apache 2.0).
