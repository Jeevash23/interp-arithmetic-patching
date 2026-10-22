# Local preparation record

1 October 2026. No candidate-model arithmetic outputs were inspected during these setup changes.

The local machine reports a 6 GB NVIDIA GPU, CUDA capability 8.9, 6141 MiB total GPU memory and initially 5852 MiB free. The model occupies about 5.2 GB on disk; the downloader checks remaining storage and keeps at least 1 GiB free. Existing models and journals are preserved.

Installed kernels 0.17.1 without dependencies into this experiment's .packages folder. The shared environment was not modified. The installed Transformers FP8 loader accepts capability 8.9. Its actual v4 Triton backend passed a deterministic matrix multiplication on this GPU: all finite outputs, zero repeated-output discrepancy and 0.02614 relative RMS difference against a bfloat16 multiplication with dequantized weights. That last difference includes FP8 activation rounding; this is a hardware smoke test, not a statement of model equivalence.

Two initial kernel discovery attempts failed before model download or inference. The Hub has migrated kernels from model repositories to a distinct kernel repository type, so the model-repository commit did not resolve. The kernel repository main branch then exposed an older API. The published v4 branch resolves to 8c178950e8710e26a2210e4e909cd02dc16c8715 and supplies all three entry points required by Transformers 5.16.1. Failure logs are preserved. Scientific inference pins that v4 implementation and binds its documented functions to the same Transformers FP8 integration used for native and instrumented execution.

The scientific protocol was clarified before candidate inference to use sequential complete-answer likelihood scoring. A single-forward-pass score is a numerical diagnostic only. Multi-token answer generation remains autoregressive; no scratchpad or intermediate value is inserted into activation runs.

The initial single-stream HTTP model download made slow progress, and an attempted switch to the installed Xet backend stalled without writing model bytes. Only the task-owned transfer processes were stopped. A bounded four-worker HTTP range downloader resumed the existing verified-source prefix; each response is checked for the exact requested Content-Range and length, and the completed file must match the provider's SHA256 before loading. Transfer logs and resume state are retained. This changes transport only, not checkpoint bytes or scientific settings.

The official 4B tokenizer's native chat template and all 180 donor, recipient and constructed-source prompt role mappings were checked before model inference. The prefix ends at the assistant response boundary and supplies neither an intermediate nor an answer.

After adding the same-leading-digit source, all 216 full-problem role mappings and arithmetic/source invariants passed (`data-checks.json`). The complete weight file matched the provider's SHA256 before the first load attempt.

The first baseline launch stopped before model loading: offline kernel resolution required a complete repository snapshot, while the earlier online smoke test had downloaded only the CUDA variant. The original failure log is retained as `logs/baseline-qwen4b-kernel-cache-incomplete.log`. Completing the same pinned kernel snapshot supplies the missing documentation and alternative-backend source files; it does not change the CUDA implementation, model, protocol, or analysis freeze. No arithmetic output was produced by this failed launch.
