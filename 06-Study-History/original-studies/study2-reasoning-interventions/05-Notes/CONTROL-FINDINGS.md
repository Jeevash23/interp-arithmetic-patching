# Held-out controls: completed

The fixed block-26 panel evaluates the first 16 final programs at first_z and label_z, two streams, with activation, first-digit text, full-value text and three seeded norm-matched random controls. All 384 planned intervention rows and 48 reused baselines are retained. Integrity verification passes for all 420 actual outputs, with 12 unavailable attempts retained separately.

First-digit text exactly matches activation in 30/30 runnable early branches and 30/32 late branches. The early matches include seven identically capped responses: strict completed-answer agreement is 23/30, not 30/30. All 32 late pairs finish validly; 30 agree. Both late mismatches belong to final-006, so 15/16 late programs match both streams. Exact suffixes are saved in `04-Results/analysis/control-mismatch-audit-v1.json`.

Neither activation nor digit text yields a complete counterfactual. Full-value text does so in 5/32 early branches across four programs and 3/32 late branches across three programs. It exhausts the cap in 11/32 early branches, compared with 7/32 activation branches. Each random direction gives zero targets and zero capped branches. There are 16 independent program groups, not 384 independent experiments.

The final-006 stream-0 mismatch is informative: both activation and digit text print Intermediate 128, but their finals are 124 and 214 respectively. The intended donor-value target is 130. Activation has an observable residual effect beyond identical first-digit insertion on this path, without establishing complete donor-value transfer. Stream 1 also differs (1018 -> 1014 versus 100 -> 214). This prevents universal first-digit equivalence claims.

Reporting uses corrected `final-v2/results.json`. The frozen `final-v1` analysis remains available and its primary results must agree exactly. The correction excludes null or unfinished parsed fields from semantic agreement, while retaining raw equality and exact sequence equality separately.
