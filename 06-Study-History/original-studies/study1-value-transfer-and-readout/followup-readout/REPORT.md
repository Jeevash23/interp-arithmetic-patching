# Follow-up findings and paper status

The fixed-site replication meets all three declared descriptive criteria for a dominant next-digit account.

The local follow-up is complete. The expanded manuscript includes the original development study and the separately frozen replication. It remains a review draft; a second checkpoint and author review of the contribution are still needed before claiming broader publication readiness.

- Activation counterfactuals: 16/240; complete-value text: 151/240.
- Full-sequence agreement with first-digit text: 233/238 jointly runnable cases.
- Short/long donors with the same leading digit: 117/120 identical continuations on the same recipient.
- Artifact audit: 21 checks passed. Original validation pairs remain reserved.

![Replication figure](results/analysis/replication.png)

| Intervention | All pairs | One-digit donors | Two-digit donors | Available |
|---|---|---|---|---|
| activation | 16/240 | 8/120 | 8/120 | 240 |
| unpatched | 0/240 | 0/120 | 0/120 | 240 |
| same example | 0/240 | 0/120 | 0/120 | 240 |
| isotropic delta | 0/240 | 0/120 | 0/120 | 240 |
| text first digit | 14/240 | 6/120 | 8/120 | 238 |
| text full value | 151/240 | 48/120 | 103/120 | 238 |
| text recipient clamp | 0/240 | 0/120 | 0/120 | 238 |
| activation recipient clamp | 0/240 | 0/120 | 0/120 | 238 |
| activation donor clamp | 151/240 | 48/120 | 103/120 | 238 |

| Prespecified descriptive criterion | Met |
|---|---|
| exact digit at least 90pct | yes |
| source copy at least 95pct | yes |
| two digit full text advantage at least 10pp | yes |

Read the [manuscript](paper/manuscript.md), [paper-readiness assessment](PAPER-READINESS.md), [fixed protocol](PROTOCOL.md), [complete analysis](results/analysis/summary.json), [all differing continuations](results/analysis/nonidentical-continuations.json), [literature update](LITERATURE-UPDATE.md), [reproduction instructions](REPRODUCIBILITY.md), and [research log](RESEARCH-LOG.md). The paper PDF is in output/pdf/.

## 9.1. Bounded second-checkpoint attempt

The official Qwen3-1.7B checkpoint was downloaded with a pinned revision and file hashes. Its 28-layer architecture permits testing the same block index. A separate pre-outcome protocol selected the first 15 numeric groups (60 pairs) and the first sampling seed. To fit local memory, execution used bfloat16 with native eager attention and serial generation. Model size and precision are consequently confounded.

Four predetermined feasibility cases test native generation, replay identity, same-example writes and cached/fixed-node continuation. Exact output-token equality is required; the declared maximum cache-logit difference is 0.125 for this precision. The result is:

| Case | Native donor / recipient | Replay / same example | Cache exact tokens | Maximum logit difference | Overall |
|---|---|---|---|---|---|
| 00 box w1 | pass / pass | fail / fail | pass | 0.87500 | fail |
| 00 register w1 | pass / pass | pass / pass | pass | 1.50000 | fail |
| 01 box w2 | pass / pass | pass / pass | pass | 1.00000 | fail |
| 01 register w2 | pass / pass | pass / pass | pass | 0.93750 | fail |

The checkpoint branch stopped because its prespecified instrumentation gate failed. Only 12 baseline records were evaluated. No comparable mechanism-replication result is claimed for this checkpoint. Missing timing, if present, concerns trajectory availability; numerical replay failure concerns execution comparability. Neither establishes absence of an intermediate representation. Full failure records and native/patched outputs are preserved.

Reconstruction of the failed identity case confirmed a whitespace-only change: both outputs printed intermediate 86 and final 88. All four patched cached/recomputed output sequences also agreed, despite maximum logit discrepancies of 0.875-1.5. This is a conservative rejection under the declared execution-equivalence requirements, not an observed arithmetic failure or proof that bfloat16 cannot support such studies. The tolerance was not relaxed after seeing the results.

