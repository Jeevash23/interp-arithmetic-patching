# Lookup-composition qualification: completed boundary

All 24 planned baseline records were collected for eight program groups under the fixed 768-token total assistant budget. Correct naturally completed final answers: donor 5/8, recipient 2/8, supplied intermediate 5/8. Each family needed at least 7/8. The gate failed, and no conditional mechanism intervention was run.

The 12 failed rows all exhausted the budget with unclosed thinking: three donor, six recipient and three supplied. All 12 naturally completed rows parsed and had correct final answers. This is a task/prompt/budget qualification boundary, not evidence that no representation transfers or that a longer-budget setup would fail.

All 24 outputs passed the saved-record integrity verifier. `completion.json` correctly records complete=true, status=competence_boundary, mechanism_run=false. The collector's intentional nonzero exit was caught by the generic coordinator as a failure; its exact status was preserved in `sequence-status-boundary-exit-v1.json` and reconciled in `sequence-status.json`. This bookkeeping reconciliation did not change any journal or gate.

The arithmetic main result is unchanged. The generated symbolic data and frozen prompt/configuration remain available for a future separately qualified study. No outcome-driven prompt or budget tuning was performed on these groups.
