# Fixed value-projection result and targeted diagnostics

2 October 2026. These are development results, not held-out validation. The fixed model is Qwen3-4B-Instruct-2507-FP8 with the qualified three-example context. No model weights were trained.

## Learned intervention

The rank-eight class-mean projection was fitted on 756 value/final-answer/first-operation combinations. All isolated first steps are correct; 677/756 full tasks are correct. Every fitting record is retained. The basis explains 71.6% of between-value mean variance, a geometric quantity rather than a causal-mechanism percentage.

On the original 36 pairs, the learned donor projection produces 1/36 complete counterfactuals, including 1/33 on the original triple-competent subset. Same-z/new-decomposition and same-z/new-answer sources produce 2/36 and 1/36, with one joint success. Changed-z/same-answer and matched-leading-digit/new-value sources each produce zero respective counterfactuals. Noise, random-subspace and shuffled-label controls each give zero; each first-digit text control gives one, on a different pair from the projection. No progression criterion set passes.

The independent [audit](AUDIT-NOTE.md) exactly reconstructs all 360 intervention targets and checks all 144 reused outputs and 756 fitting vectors. The original learned artifact remains unchanged. The [full control figure](results/analysis/subspace-controls.png), [summary](results/analysis/summary.json) and raw journals retain every result.

## Investigation of the isolated success

The projection produces 112 for source z=56 and recipient z=76 followed by multiplication by two, including two alternative sources with the same value. But changing the source to z=57 still produces 112 instead of 114. This is compatible with several partial-information mechanisms and does not establish full-value transfer.

The first [factorial diagnostic](value-digit-diagnostic/PROTOCOL.md) preserves nine donor values, three recipient values and three recipient operations. Its donors score 63/81 paired records (7/9 unique) and fail the competence gate. A single [no-borrow calibration](value-digit-no-borrow/PROTOCOL.md) scores 54/81 (6/9 unique) on its original donors and also stops before interventions. All recipients and explicit counterfactuals pass in both grids. The two failed grids are preserved and are not intervention failures.

Both predefined auxiliary source families in the no-borrow version score 81/81 on full problems, isolated steps and explicit counterfactuals. A separately declared [qualified-source contrast](value-digit-qualified-sources/PROTOCOL.md) therefore uses those already measured sources, with exact inherited baseline provenance and no new learned parameters. One source performs `(z+1)-1`, then +11; the other performs `(z-1)+1`, then +4. Their same-z comparison jointly changes decomposition and final answer; it does not isolate those changes separately.

| Outcome | Primary qualified source | Coupled same-value source |
|---|---:|---:|
| Complete-value target, all pairs | 1/81 | 1/81 |
| Coarse leading-digit target, all pairs | 3/81 | 7/81 |
| Complete-value target on discriminating pairs | 0/72 | 0/72 |
| Coarse target on discriminating pairs | 2/72 | 6/72 |
| Unchanged recipient answer | 48/81 | 51/81 |

Joint success across both sources is zero for both targets. The first-token text control matches the coarse target on 10/72 discriminating pairs, compared with 2/72 for the primary projection. Noise and random-subspace controls match neither target. The [complete summary](value-digit-qualified-sources/results/summary.json) reports operations, source agreement and donor-unit triplets. Its 405 activation targets reconstruct exactly; 81 self outputs match baseline.

The qualified grid has a further [structural ambiguity](value-digit-qualified-sources/INTERPRETATION-ADDENDUM.md): source operand and intermediate leading digits coincide. Even a stronger coarse pattern could not distinguish those origins. The observed weak matches are therefore not a partial-computation discovery. The isolated original success did not become a reliable pattern under this fixed-intervention challenge.

The original validation, reserve and language interventions remain untouched. No rank, strength or layer was retuned after these outcomes. Native-component checks are a separate untrained branch in [component-v1](../component-v1/PROTOCOL.md).
