# Leaderboard, uploads and decisions: index

Summary: what was uploaded, how the public score was read, how it relates to cross-validation (CV),
which decisions govern modelling and validation, and what a submission must satisfy so the
organisers can re-run it. Almost everything here is an **earlier measurement** (from earlier
research); the only numbers **measured in this repo** are the leave-one-month-out (LOMO) and
October-like scores in [RESULTS.md](../../../RESULTS.md), re-derivable with
`python -m ringtrace.build recipes/champion.json` and `python -m ringtrace.octlike`.
No public score was produced from this repository: it makes no uploads itself.

## Files

| File | Contents |
|---|---|
| [uploads.md](uploads.md) | Every upload in order: model, public score, forecast and read-out rule, lesson |
| [regime.md](regime.md) | How the all-legit upload read the test labels, the regime table, the decision playbook |
| [cv-vs-public.md](cv-vs-public.md) | CV against public gaps; public-slice noise; why near-identical files differ by chance |
| [decisions.md](decisions.md) | Canonical folds, device counter, upload rule and its evolution, merge policy, design choices |
| [reproducibility.md](reproducibility.md) | What the organisers re-run, what a submission must satisfy, evidence of byte-identical rebuilds |

## The headline facts

- Task: one of 12 labels per October transaction; metric macro-F1; all-legit scores about 0.08.
  Public = 30% of test (about 78,175 rows), private = 70% (test total 260,582 rows).
- Each team's **best public upload is what gets scored on private**; there is no final-selection
  choice.
- Upload 1 (all-legit) scored 0.0816, which means the ordinary-looking fraud rows count in the test
  labels ("regime A"); the ceiling on cleaned labels is about 0.61.
- The XGBoost champion scored **0.5470** public against LOMO 0.5126 (given labels) / 0.5662
  (cleaned labels), inside its forecast of about 0.54 +/- 0.02-0.04.
- Three later near-identical blends scored 0.5487, 0.5488 and 0.5488. Their LOMO gains were
  +0.0003 or less on cleaned labels, so the public gap is noise. Best public: 0.5488.
- A public gain under about 0.005 (row split) to 0.008 (event split) between near-identical files
  is zero; one right row in a rare class moves the public score by up to 0.0013.
