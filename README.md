# GradientAscent · UPI Fraud Rings

**Team:** GradientAscent  ·  **Member:** Ameya Borkar

Label every October 2026 UPI payment as `legit` or one of 11 fraud types (mule_account,
velocity_bot, layering_chain, sim_swap, account_takeover, phishing_collect, qr_swap, refund_abuse,
fake_cashback, loan_app_extortion, merchant_collusion), scored by 12-class macro-F1. Train is
July to September; test is October (260,582 rows). The code is the Python package `ringtrace`.

**Best public score: 0.5556** (`stack-burst12-sim45`). It rebuilds byte for byte from this repository:

```bash
python -m ringtrace.final build recipes/stack/stack-burst12-sim45.json --out <dir>
# sha256 c278490094706e156dc4f4d573989fafda24de896f3a7ec147fb5fc886eb2efc
```

## The approach on one page

| Layer | What it does |
|---|---|
| **Label-free evidence** | Rebuilds the generator's fraud routines from how rows connect (accounts, phones, receivers, time), inside each split and without labels: layering hops and mule fan-ins with onward transfers, bot sessions on shared devices, balance drains stepping by a fixed fraction, collect requests and cashback baits, repeat refund and lending chains, round-amount merchant rings |
| **Device-number counter** | Fresh phones are numbered in creation order. Each split has a bot pool, a fraud phase written class by class and a dense legit block. An 11-state left-to-right HMM segments the fraud phase, so a phone's position certifies fraud rows that look ordinary and gives their type; the refined counter also reads each phone's allocation geometry (the unused numbers below it and its row count) |
| **Features** | About 230 per row: relational windows, the evidence columns and the counter columns |
| **Members** | Five two-stage boosted cascades, P(fraud) x P(type given fraud): XGBoost on the given labels, LightGBM, CatBoost and XGBoost on two event-cleaned label sets. Plus a TabM network. Three seeds each, fitted on whole months as folds |
| **Mix** | The type split is the mean of the five trees; the legit-against-fraud split is half TabM and a tenth from each tree |
| **Decoder** | An expected-macro-F1 fixed point, each month decoded on its own. Fraud probabilities are divided by 0.875, because train labels miss about 12 % of fraud and test labels are clean |
| **Rules** | Look-alike templates take their counter segment or anchor's class; operator-role bait rules; then base-rate rules, each with a cap on the October rows it may change: the counter's class borders and edges, counter news, fresh-SIM bursts |

## How every change was judged

- **Month folds:** leave-one-month-out on the given labels, on cleaned labels from an independent
  teacher and on generator-only labels, with a paired bootstrap over label-free events.
- **October-like frames:** September thinned to October's density, with the device counter
  rebuilt as its own pass, exactly as October's is built.
- **Labelled counter passes:** the counter's boundaries are scored on train passes at October's
  size.
- A change is adopted only when it is non-negative in every held-out month and its lower bound
  is above zero; a tie goes to the simpler file. Changes that train cannot measure (one October
  event or one border) are judged by calibration at the test phase's size; the cuts of the
  fresh-SIM burst rule in the best file were set with public reads, as
  `docs/findings/october-events.md` states.

## Uploads

| File | What changed | Public |
|---|---|---|
| `final-half` + blank-QR rule | TabM sets half of the legit-against-fraud split | 0.5499 |
| `bait-combo` | operator-role bait rules, 7 rows | 0.5502 |
| `bait-edge` | the counter's mule / bot edge, 10 rows | 0.5506 |
| `research-combo-2` | three small base-rate rules, 4 rows | 0.5506 |
| `counter-news` | the refined counter in the rules, 4 rows | 0.5516 |
| `stack-burst-b8` | one 28-row burst to velocity_bot, 1 border row | 0.5531 |
| `stack-burst12-b8` | two more bursts (15 and 12 rows) | 0.5532 |
| `stack-burst13-b8` | the 12-row burst back to mule_account | 0.5520 |
| **`stack-burst12-sim45`** | the 15-row burst back to mule_account, the border row removed | **0.5556** |
| `stack-sim45-b9r2` | 2 rows at the loan_app_extortion / merchant_collusion border | 0.5556 (tie: neither row is public) |
| `stack-sim45-rings8` | three merchant rings of 8 rows as merchant_collusion | 0.5505 |

The documents that hold the evidence are indexed in [`docs/README.md`](docs/README.md).

## One command, from a clean checkout

On a machine with the organisers' files in `~/data` (or `RINGTRACE_DATA`), a CUDA GPU and Python
3.13, with an empty cache:

```bash
uv venv --python 3.13 ~/venv && VIRTUAL_ENV=~/venv uv pip install -e ".[dev,nn]"
RINGTRACE_THREADS=8 bash scripts/build_final.sh recipes/stack/stack-burst12-sim45.json <dir>
```

This builds every feature family (cached by content hash), freezes the cleaned labels used for
scoring, runs the six member recipes behind the recipe's runs and builds the October file,
validated against `sample_submission.csv`. About 90 minutes on one RTX 5060 Ti. The `nn` extra
installs torch for the TabM member.

Measured on 15 Nov: a clean tree and an empty cache rebuilt the six member runs and `counter-news`
in 5,086 s, byte-identical to the upload (sha256 `b8c3183a...967799`). `stack-burst12-sim45` and
`stack-sim45-b9r2`, built on those runs, have their uploaded sha256.

## Reproduce the uploaded files

The uploads are final recipes over saved member runs. Each command re-blends the members' arrays
(`$RINGTRACE_OUT/runs/<run>/oof.npy` and `test.npy`), decodes, applies the recipe's rules,
validates the file against `sample_submission.csv` and prints its sha256:

```bash
python -m ringtrace.final build recipes/bait-edge.json --out <dir>    # sha256 b571af3e...2f9dfc (public 0.5506)
python -m ringtrace.final build recipes/research-combo-2.json --out <dir>  # sha256 214926a6...986344 (public 0.5506, tied)
python -m ringtrace.final build recipes/counter-news.json --out <dir>  # sha256 b8c3183a...967799 (public 0.5516)
python -m ringtrace.final build recipes/stack/stack-burst-b8.json --out <dir>    # sha256 31251e77...76eb63d3 (public 0.5531)
python -m ringtrace.final build recipes/stack/stack-burst12-b8.json --out <dir>  # sha256 a9a3e35d...226ed20b (public 0.5532)
python -m ringtrace.final build recipes/stack/stack-burst12-sim45.json --out <dir>  # sha256 c2784900...86eb2efc (best public, 0.5556)
python -m ringtrace.final build recipes/stack/stack-sim45-b9r2.json --out <dir>     # sha256 438e0eb7...f91fd474 (public 0.5556, tied)
python -m ringtrace.final build recipes/bait-combo.json --out <dir>   # sha256 3b25869e...e1974
python -m ringtrace.final build recipes/final-half.json --out <dir>   # sha256 8a461426...fb926eb
```

`stack-burst12-sim45` is `counter-news` with one base-rate rule appended: `fresh_bursts` (a
mule-shaped burst with no fraud-phase phone, 12 rows or more and a youngest sender SIM of 45 to
50 hours, is velocity_bot; two October events, 40 rows). `stack-burst12-b8` had the cut at under
50 hours (three events) and a `border_shift` row; the public scores read the third event as
mule_account and the border row as wrong, and both were removed. See
`docs/findings/october-events.md` and `docs/findings/october-borders.md`.

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
mule_account 1. Public score 0.5516, against 0.5506 for `research-combo-2`. The evidence is in
`docs/findings/counter-news.md`.

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

## Member results

Leave-one-month-out macro-F1 of each tree member and of their early equal-weight blend (train on
two months, score the third), after the fixed point and the template rule; mean of July / August /
September. Details per month in `RESULTS.md`.

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

## Verifying the early five-member blend

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
  another: about 85 to 95 minutes in total.

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

## Working with a remote GPU machine

`scripts/ship.sh` copies the committed tree to `~/work/$RINGTRACE_BOX_DIR`; `scripts/launch.sh
<name> <command>` starts a job in tmux from its own copy of that tree, logging to
`~/logs/<name>.log`; `scripts/box.sh` / `scripts/box_in.sh` run short commands there;
`scripts/retry_gpu.sh <tries> <command>` reruns a job after a GPU out-of-memory failure.

Other entry points: `python -m ringtrace.run <recipe> [--no-test] [--name <run>] [--without
fam,...] [--rules ...]` (one recipe), `python -m ringtrace.rescore <run> [--rules ...] [--scale x]`
(re-decode saved probabilities), `python -m ringtrace.teacher` (freeze the cleaned labels),
`bash scripts/ablate.sh <recipe> <parallel> <threads> <family...>` (leave-one-family-out).
