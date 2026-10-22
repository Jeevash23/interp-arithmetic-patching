# Saved outputs

The studies' results, packed as deterministic tar archives compressed with xz. Member paths are relative to the release root, so `make unpack` (or `python tools/unpack.py`) puts every file where the study code reads it and checks each SHA-256 against `data/PACKED-FILES.json`. Text records inside the archives had personal strings replaced, as listed in `RELEASE-CHANGES.md`.

| Archive | Files | Packed (MB) | Unpacked (MB) | Contents |
|---|---|---|---|---|
| `study1-value-transfer-and-readout.tar.xz` | 224 | 1.7 | 113.6 | Study 1: every saved journal, analysis and run record of the readout and answer-only experiments and of the September development stage (saved write vectors not included) |
| `study2-reasoning-interventions.tar.xz` | 2140 | 3.9 | 54.5 | Study 2: run records, assays, activation summaries, audits and analyses |
| `study3-heldout-confirmation.tar.xz` | 34 | 0.4 | 44.0 | Study 3: journals of both wordings, analyses, gates, run records (saved write vectors not included) |
| `study4-multidigit-tokens.tar.xz` | 22 | 0.0 | 3.9 | Study 4: competence-gate journals and summaries of both candidates |
| `study5-direction-and-span.tar.xz` | 174 | 1.0 | 1.7 | Study 5: every record pulled back from the cloud run, analyses and verification |
| `study6-olmo2-7b.tar.xz` | 17 | 0.0 | 1.9 | Study 6: competence-gate journal and summaries |

Total packed size: 7.0 MB. Rendered figures are not included; `make figures` draws them from these outputs.
