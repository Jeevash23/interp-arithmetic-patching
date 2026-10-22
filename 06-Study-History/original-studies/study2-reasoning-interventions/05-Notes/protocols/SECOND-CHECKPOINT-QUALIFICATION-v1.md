# Second-checkpoint qualification v1

Declared before this checkpoint's outputs. DeepSeek-R1-Distill-Qwen-1.5B is an SFT-distilled Qwen2.5-Math derivative, not full R1, an RL-only model, or an independent broad lineage. Public checkpoint revision/hash is fixed by checkpoint-fetch record. First profile float32 CPU inference (4 threads, eager, singleton) with native thinking prompt, temperature .6/top-p .95/top-k 20; 96 generated tokens for cost only. This partial response is not a competence or mechanism test.

If profiling permits, extend under a new named configuration to 768 total tokens and qualify native/custom, no-op, identity and historical-cache/recomputation with the same 0.0005 full-vocabulary gate. CPU >0.5 seconds/token makes a 24-group broad continuation replication impractical for this local session (>several hours); retain that measured boundary and consider a separately qualified mixed CPU/GPU float32 path. Do not substitute bfloat16 or loosen tolerances implicitly.
