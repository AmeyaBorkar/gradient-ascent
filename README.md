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

## Reproduce the uploaded files

The uploads are final recipes over saved member runs. Each command re-blends the members' arrays
(`$RINGTRACE_OUT/runs/<run>/oof.npy` and `test.npy`), decodes, applies the recipe's rules,
validates the file against `sample_submission.csv` and prints its sha256:

```bash
python -m ringtrace.final build recipes/bait-edge.json --out <dir>    # sha256 b571af3e...2f9dfc (best public, 0.5506)
python -m ringtrace.final build recipes/research-combo-2.json --out <dir>  # sha256 214926a6...986344 (public 0.5506, tied)
python -m ringtrace.final build recipes/counter-news.json --out <dir>  # sha256 b8c3183a...967799 (rules on the refined counter)
python -m ringtrace.final build recipes/bait-combo.json --out <dir>   # sha256 3b25869e...e1974
python -m ringtrace.final build recipes/final-half.json --out <dir>   # sha256 8a461426...fb926eb
```

`final-half` is the five tree cascades for the type split, with the fraud mass taken half from
TabM and a tenth from each tree, decoded at fraud scale 0.875 with the anchored template rule and
`blank_merchant_qr`. `bait-combo` adds the two operator-role bait rules
(`docs/findings/bait-operator-roles.md`), and `bait-edge` the counter's mule/bot edge
(`mule_edge`). `research-combo-2` adds three October base-rate rules to `bait-edge` (`far_segment`,
`legit_topup` for qr_swap, `just_under_pay`; 4 October rows) and builds from the same six member runs.

`counter-news` keeps those six member runs, which were trained on the champion device counter
(`ringtrace.evidence.phones`). Its rules read the refined counter instead
(`"rule_counter": "refined"`, `ringtrace.evidence.refined`): the same blocks and HMM, plus each
phone's allocation geometry, meaning the run of unused numbers below it and its row count. A last
base-rate rule (`counter_news`) then moves a row whose call still follows the champion segment
when the refined counter puts it one class away with posterior at least 0.7. Against
`research-combo-2` this changes 4 October rows: sim_swap → account_takeover 3, velocity_bot →
mule_account 1. The evidence is in `docs/findings/counter-news.md`.

From a clean checkout and an empty cache, one command builds the features and cleaned labels, runs
the six member recipes behind the recipe's runs (`python -m ringtrace.run recipes/<member>.json
--name <run>`, retried if the GPU runs out of memory) and builds the file:

```bash
RINGTRACE_THREADS=8 bash scripts/build_final.sh recipes/bait-edge.json <dir> [--against <uploaded csv>]
```

Measured on the box (one RTX 5060 Ti, torch 2.14.1+cu130, code at `c7eba12`): 93 minutes, and the
file is byte-identical to the uploaded `bait-edge.csv` (sha256 `b571af3e...2f9dfc`, 0 rows
different). TabM is the one member whose arrays depend on the software and hardware it ran on:
two fits on the box's two GPUs are bit-identical, while the uploaded run's arrays (fitted earlier
in another environment) differ slightly from the rebuild's (oof sha256 `5bef8fbc...` against
`a78a87a1...`); no October call of the final file moves. CatBoost on the GPU is not
bit-reproducible either (decision 14); it also moved no call here.

## Results

Leave-one-month-out macro-F1 (train on two months, score the third), after the fixed point and
the template rule; mean of July / August / September. Details per month in `RESULTS.md`.

| model | given | cleaned | gen-only | wall |
|---|---|---|---|---|
| 1 `champion` (run `blend-champion-m1`) | 0.5122 | 0.5665 | 0.9010 | 10.7 min |
| 2 `lgbm` | 0.5122 | 0.5667 | 0.9011 | 14.9 min (CPU, 10 threads) |
| 3 `catboost` | 0.5120 | 0.5662 | 0.9008 | 3.6 min |
| 4 `clean-event` | 0.5123 | 0.5666 | 0.9012 | 12.4 min |
| 5 `clean-l2` | 0.5122 | 0.5666 | 0.9012 | 12.0 min |
| **blend** | **0.5123** | **0.5667** | **0.9013** | 22 s |

The submission (`~/outputs/runs/blend/submission.csv`) has 260,582 rows in the sample's id order
and sha256 `c3af1418ffab08f7843b1813359ad7353ee0ac6676f4293609d3cf530a696c05`.

## Verifying the submission

```bash
sha256sum ~/outputs/runs/blend/submission.csv          # or ~/outputs/submission.csv after --final
python -m ringtrace.blend recipes/blend.json           # re-blend the saved member arrays
```

Re-blending the five saved runs is deterministic and reproduces the sha256 above byte for byte;
`summary.json` records it with the scores. A full rebuild reproduces members 1, 2, 4 and 5 exactly
(member 1 from a clean checkout and an empty cache gives the champion's sha256
`a36833d5...`), but CatBoost on the GPU is not bit-reproducible (decision 14: a few rows per
month can move), so a rebuilt member 3, and through it the blend, can differ in a handful of
October rows. Compare such a rebuild through the scores and October calls in `summary.json`.

## Reproducibility

- Seeds: cascade seeds 0, 1, 2 for every member, averaged; every other random step is seeded;
  the decoder, the template rule and the blend are deterministic. LightGBM runs with
  `deterministic` and row-wise histograms; XGBoost on the GPU reproduced byte for byte here.
- Libraries (box): Python 3.13.18, numpy 2.5.4, polars 2.1.0, xgboost 3.4.1, lightgbm 4.7.0,
  catboost 1.2.10, scikit-learn 1.9.1, pyarrow and scipy as installed with them.
- Hardware: one NVIDIA RTX 5060 Ti 16 GB (XGBoost and CatBoost) and a shared AMD EPYC CPU
  (LightGBM, features, decoding). `RINGTRACE_THREADS` sets the CPU thread budget (Polars,
  BLAS/OpenMP and the boosters).
- Wall times: as in the table above, with the GPU shared by other jobs; building every feature
  family into an empty cache adds about 40 s (all nine families; the cleaned labels are built once, a few minutes). The one command runs the members one after
  another: about an hour in total.

## Layout

| module | role |
|---|---|
| `ringtrace.tables` | typed CSV cache and the stacked train+test "ledger" |
| `ringtrace.context` | relational window features (sender, receiver, phone, pair, money flow) |
| `ringtrace.evidence.*` | label-free generator evidence: row marks, drains, bot bursts, money flow, collect requests, merchant rings, repeat payers, the device-number counter |
| `ringtrace.store` | builds and caches feature families |
| `ringtrace.silver` / `ringtrace.teacher` | cleaned labels for scoring, from an independent teacher |
| `ringtrace.cascade` | P(fraud) x P(type given fraud): XGBoost on the GPU, LightGBM, or CatBoost stages |
| `ringtrace.eventclean` | event-cleaned training labels (members 4 and 5), rebuilt inside each fold |
| `ringtrace.decode` | label-free fixed-point rule for macro-F1, per month |
| `ringtrace.evidence.refined` | the refined device counter (allocation geometry), read by rules only |
| `ringtrace.rules` | counter-anchored look-alike template rule, operator-role and base-rate rules |
| `ringtrace.counternews` | the `counter_news` rule: champion against refined counter |
| `ringtrace.final` | final recipes: mix, decode, rules, scoring harness and the October file |
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
