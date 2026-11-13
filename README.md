# ringtrace

A pipeline for the Hack4Hack "UPI Fraud Rings" datathon: label every October
transaction with one of 12 classes (legit + 11 fraud types), scored by macro-F1.

## One command

On a machine with the organisers' files in `~/data` (or `RINGTRACE_DATA`), a CUDA GPU and
Python 3.13:

```bash
uv venv --python 3.13 ~/venv && VIRTUAL_ENV=~/venv uv pip install -e ".[dev]"
python -m ringtrace.build recipes/champion.json
```

This builds every feature family (cached by content hash under `~/cache/ringtrace`), freezes the
cleaned labels used for scoring, runs the leave-one-month-out cascade, decodes, applies the
template rule, scores every stage on the given and cleaned labels, and writes and validates
`~/outputs/submission.csv` against `sample_submission.csv`.

## Layout

| module | role |
|---|---|
| `ringtrace.tables` | typed CSV cache and the stacked train+test "ledger" |
| `ringtrace.context` | relational window features (sender, receiver, phone, pair, money flow) |
| `ringtrace.evidence.*` | label-free generator evidence: row marks, drains, bot bursts, money flow, collect requests, merchant rings, repeat payers, the device-number counter |
| `ringtrace.store` | builds and caches feature families |
| `ringtrace.silver` / `ringtrace.teacher` | cleaned labels for scoring, from an independent teacher |
| `ringtrace.cascade` | P(fraud) x P(type given fraud), XGBoost on the GPU |
| `ringtrace.decode` | label-free fixed-point rule for macro-F1, per month |
| `ringtrace.rules` | counter-anchored look-alike template rule |
| `ringtrace.run` | one recipe end to end on month folds |

Recipes are JSON files in `recipes/`. Results of every run are in `RESULTS.md`, progress in
`PROGRESS.md`, design choices in `docs/decisions.md` and sources in `docs/research.md`.

## Rules we keep

Train rows may be inspected; test rows are used only through code and aggregates. No external
data, no leaderboard uploads from here, no data or model files in git.

## Working with the compute box

`scripts/ship.sh` copies the committed tree to `~/work/ringtrace`; `scripts/launch.sh <name>
<command>` starts a job in tmux from its own copy of that tree, logging to `~/logs/<name>.log`;
`scripts/box.sh` / `scripts/box_in.sh` run short commands there. Thread budgets are explicit:
`RINGTRACE_THREADS` sets Polars, BLAS/OpenMP and XGBoost threads (default 16).

Other entry points: `python -m ringtrace.run <recipe> [--no-test] [--without fam,...]
[--rules ...]` (scores only), `python -m ringtrace.rescore <run> [--rules ...] [--scale x]`
(re-decode saved probabilities), `python -m ringtrace.teacher` (freeze the cleaned labels),
`bash scripts/ablate.sh <recipe> <parallel> <threads> <family...>` (leave-one-family-out).
