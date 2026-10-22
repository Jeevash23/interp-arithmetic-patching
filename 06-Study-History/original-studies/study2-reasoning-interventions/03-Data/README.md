# Data and split records

Arithmetic programs are synthetic, exactly solved by `02-Code/ai04/tasks.py`. Files in `generated/` and `splits/` were created before model outcomes and remain unchanged. Their `status` fields describe creation-time state, not the current completion of experiments. Run journals and the project guide record actual evaluation status.

Four audit, eight pilot and 24 development groups have separate IDs/seeds. Eighty final groups were generated; the scientific protocol evaluates indices 0-63, reuses indices 0-15 for diagnostics, and leaves 64-79 without model outputs. Templates and ranges shift together. Duplicate-prompt and exact-solver checks are in the scientific tests. These are independent random program constructions conditional on fixed operator/template structure, not independent models or independent branches.

`generated/symbolic.json` contains eight exploratory two-map state-composition programs, with separate seed and manifest. The same numeric output grammar is used, but the numbers are state labels.

`raw/models/` holds the publicly fetched, pinned second checkpoint and its original license/card. The primary checkpoint is read only from the adjacent AI-02 cache, without copying or changing that earlier study. Model file hashes and revisions are in `06-Study-History/environment`. No external participant data was collected, and AI-02's original reserved validation was not read.
