# ringtrace

A pipeline for the "UPI Fraud Rings" datathon: label every October transaction with
one of 12 classes (legit + 11 fraud types), scored by macro-F1. Train is July-September, test is
October.

The submission is an **equal-weight blend of five cascade models**. Each member is a cascade
(P(fraud) x P(type | fraud)) over the same 230 label-free features (relational windows plus seven
evidence families and the device-number counter), averaged over three seeds (0, 1, 2) on
leave-one-month-out folds: out-of-fold probabilities on train rows, the mean of the three fold
models on October.

| # | member | model | training labels | recipe |
|---|---|---|---|---|
| 1 | `champion` | XGBoost cascade | given | `recipes/champion.json` |
| 2 | `lgbm` | LightGBM cascade | given | `recipes/lgbm.json` |
| 3 | `catboost` | CatBoost cascade | given | `recipes/catboost.json` |
| 4 | `clean-event` | XGBoost cascade | event-cleaned | `recipes/clean-event.json` |
| 5 | `clean-l2` | XGBoost cascade | event-cleaned + noise-floor flips | `recipes/clean-l2.json` |

The blend (`recipes/blend.json`, `ringtrace.blend`) is the arithmetic mean of the five members'
probabilities with weights 1/5. It is decoded per month, October on its own: the label-free
fixed point for expected macro-F1 at fraud scale 0.875, then the counter-anchored template rule,
the same decoder and rule as the champion recipe.

## One command

On a machine with the organisers' files in `~/data` (or `RINGTRACE_DATA`), a CUDA GPU and
Python 3.13, from a clean checkout and an empty cache:

```bash
uv venv --python 3.13 ~/venv && VIRTUAL_ENV=~/venv uv pip install -e ".[dev]"
RINGTRACE_THREADS=4 bash scripts/build_submission.sh recipes/blend.json --final
```

This builds every feature family (cached by content hash under `~/cache/ringtrace`), freezes the
cleaned labels used for scoring, runs the five member recipes
(`python -m ringtrace.run <recipe> --name <run>`, each retried if the shared GPU runs out of
memory), then `python -m ringtrace.blend recipes/blend.json --final`: it checks every member's
shapes and row alignment, averages, decodes, scores the blend and each member on the given,
cleaned and gen-only labels per held-out month, and writes the submission in the sample's id
order, validated against `sample_submission.csv`, to `~/outputs/runs/blend/submission.csv` (and,
with `--final`, `~/outputs/submission.csv`). `summary.json` next to it holds the scores, the
sha256, the October calls per class and the rows changed against each member's own decode.

A single member: `python -m ringtrace.build recipes/champion.json` (features, cleaned labels,
run, validated submission).

## Reproducibility

- Seeds: cascade seeds 0, 1, 2 for every member; every other random step is seeded; the decoder
  and rules are deterministic. Rebuilding member 1 from a clean checkout and an empty cache gives
  a byte-identical submission (see `RESULTS.md`).
- Libraries (box): Python 3.13.18, numpy 2.5.4, polars 2.1.0, xgboost 3.4.1, lightgbm 4.7.0,
  catboost 1.2.10, scikit-learn 1.9.1, pyarrow and scipy as installed with them.
- Hardware: one NVIDIA RTX 5060 Ti 16 GB and a shared AMD EPYC CPU; `RINGTRACE_THREADS` sets the
  CPU thread budget (Polars, BLAS/OpenMP and the boosters).
- Wall times: member 1 about 11-13 min with the GPU shared by other jobs (`RESULTS.md`); the
  other members and the full build are recorded there as they are run.

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
| `ringtrace.blend` | weighted mean of finished runs, decoded, scored and validated |

Recipes are JSON files in `recipes/`. Results of every run are in `RESULTS.md`, progress in
`PROGRESS.md`, design choices in `docs/decisions.md` and sources in `docs/research.md`.

## Rules we keep

Train rows may be inspected; test rows are used only through code and aggregates. No external
data, no leaderboard uploads from here, no data or model files in git.

## Working with the compute box

`scripts/ship.sh` copies the committed tree to `~/work/$RINGTRACE_BOX_DIR`; `scripts/launch.sh
<name> <command>` starts a job in tmux from its own copy of that tree, logging to
`~/logs/<name>.log`; `scripts/box.sh` / `scripts/box_in.sh` run short commands there;
`scripts/retry_gpu.sh <tries> <command>` reruns a job after a GPU out-of-memory failure.

Other entry points: `python -m ringtrace.run <recipe> [--no-test] [--name <run>] [--without
fam,...] [--rules ...]` (one recipe), `python -m ringtrace.rescore <run> [--rules ...] [--scale x]`
(re-decode saved probabilities), `python -m ringtrace.teacher` (freeze the cleaned labels),
`bash scripts/ablate.sh <recipe> <parallel> <threads> <family...>` (leave-one-family-out).
