# Inspected local environment

Measured on 9 September 2026; machine output in `results/environment.json`, distribution inventory in `results/environment-packages.txt`. The current physical directory resolves through Desktop to Documents/[redacted]/Research; both names refer to the same project. The GPU initially appeared unavailable only inside the restricted sandbox. Host-level commands and a CUDA matrix multiplication passed.

| Item | Inspected value |
|---|---|
| Python | 3.12.14, shared project 04 environment |
| Operating system | Linux, kernel 7.2.3-1-[distro] |
| GPU | [6 GB NVIDIA GPU] |
| Driver | 610.57.04 |
| GPU memory at host inspection | 6141 MiB total, 5799 MiB free |
| GPU memory at recorded preparation | 6141 MiB total, 5796 MiB free |
| CUDA allocator free/total at preparation | 5,985,861,632 / 6,081,347,584 bytes; allocator/driver accounting differs from nvidia-smi |
| System RAM | 16,339,464,192 bytes total; 10,935,205,888 available at preparation |
| PyTorch / bundled CUDA | 2.13.0 / 13.0 |
| Transformers | 5.16.1 |
| TransformerLens | 3.8.0 |
| TransformerBridge | Bundled in TransformerLens; no separate installed distribution |
| NumPy / SciPy | 2.5.2 / 1.18.1 |
| Hugging Face Hub / safetensors | 1.29.0 / 0.8.0 |
| Accelerate / Matplotlib | 1.14.0 / 3.11.1 |
| Parameters actually loaded | 596,049,920 |
| Initial instrumentation peak GPU allocation | 2,463,417,344 bytes, v1 measured run |

## Checkpoint and hooks

Official public checkpoint `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`. Model/config/tokenizer file hashes are in `data/model-manifest.json`. No quantization. Eager attention, float32, TF32 disabled, model in evaluation mode, all parameters frozen. Deterministic algorithms enabled, CUBLAS workspace `:4096:8`.

Inspected installed `transformer_lens/model_bridge/supported_architectures/qwen3.py`: Qwen3ArchitectureAdapter maps dense Qwen3 blocks, gated MLP, RMSNorm, Q/K normalization, rotary embeddings and grouped-query attention. Its component mapping points to `model.layers`, matching the native checkpoint. This is source-level architecture support evidence; TransformerBridge was not instantiated for this experiment and is not claimed runtime-validated.

Executed backend: native Transformers Qwen3 plus PyTorch forward hooks on `model.model.layers[layer]`. The hook receives the block output Tensor `[1, prefix_length, 1024]`. Whole-vector edits change one position only. This equivalent native instrumentation avoids a weight conversion and is tested against native `model.generate` and raw forwards.

## Environment reuse and reproducibility

No new environment installation was necessary. `.venv` is a symlink to project 04; do not delete or modify that source environment. The predecessor's `uv.lock` is retained as an external dependency and an exact distribution inventory is saved here. Initial `run.py` invocations record code/config/data-manifest hashes; later batch stages have command logs, explicit plans, raw-record provenance and source snapshots. A per-invocation hash manifest was not implemented for every later utility. Older scientific configurations have archived source snapshots. Model weights are separately cached inside this project.

Four-item batch validation measured peak GPU allocation of 3,091,499,520 bytes. This is a measured validation peak, not a claimed maximum over every subsequent development run. Batched writes use left padding, masks and original unpadded position IDs; serial/native equivalence was checked before scientific batch use.

The initial `pip freeze` command status is retained even if pip is absent; the importlib distribution inventory is the successful package inventory and does not require package installation. Matplotlib uses a project-local writable cache after the sandbox rejected its default home configuration directory; no figures depend on the temporary default-cache attempts.
