# Completed development comparison

All 24 programs, 72 baselines and 576 intended intervention rows completed. Baselines: donor 22/24; recipient 24/24; supplied 24/24. Integrity checks passed, including all 144 same-stream unchanged records exactly matching their baseline sequences (some identical executions are explicitly reused across layers).

Counterfactual success counts by site, each 48 branches from 24 programs: first_z L13 0; L20 2; L26 2; label_z L13 0; L20 0; L26 4. Every success belongs to development-007 or development-010. These are sparse successes, not eight independent discoveries. All source/candidate boundaries were available.

The replay selector chose label_z-L20 (mean B change 0.426954 nats; raw mean target-minus-recipient margin -22.3556; 0/48 C successes). The continuation selector chose label_z-L26 (4/48 C successes across two groups; mean B change 0.132597 nats). No activation replay score at any development site preferred the counterfactual over the recipient, so selecting its largest relative increase is not a claim that B predicted a high absolute success probability.

Both selected sites occur after thinking. The primary final comparison is consequently a late-output comparison; the first-mention layer-26 panel is a separately frozen diagnostic of continuing reasoning. Selection was frozen at 2026-09-21T04:00:00.817228+00:00, before final model outcomes. The full ranking, rules and development journal hash are in protocols/frozen-selection-v1.json.

Leading digits were copied in every early L20/L26 and late L26 activation branch, while complete donor first numerals appeared in only 10/48, 9/48 and 6/48, respectively. First_z-L13 and label_z-L13 changed no first numeral. Explicit error words occurred even in unchanged trajectories, so words alone do not identify intervention recovery. The broader control panels remain necessary for interpretation.
