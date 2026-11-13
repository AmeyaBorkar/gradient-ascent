# Onboarding

**Summary.** GradientAscent (author: Ameya Borkar) labels every October 2026 UPI transaction with
one of 12 classes (legit + 11 fraud types), scored by macro-F1, using the Python package
`ringtrace`. All five blend members and the equal-weight blend are rebuilt here: the blend scores
leave-one-month-out mean 0.5123 given / 0.5667 cleaned (champion 0.5122 / 0.5665), and its
submission has sha256 `c3af1418ffab08f7843b1813359ad7353ee0ac6676f4293609d3cf530a696c05`. Rules for working here are in [`../AGENTS.md`](../AGENTS.md).

Status labels used below: **measured in this repo** = a run recorded in [`../RESULTS.md`](../RESULTS.md);
**earlier measurement** = carried over from earlier research, not yet re-derived here.

## 1. Current state

**Goal** ([`../PROGRESS.md`](../PROGRESS.md)): rebuild the latest submission end to end in this
codebase. That submission is an equal-weight blend of five cascade models, each a mean of three seeds
(0, 1, 2) over the leave-one-month-out folds, with October as the fold mean. The blend is decoded by
the fixed point at fraud scale 0.875 per month, then the counter-anchored template rule.

| # | member | status |
|---|---|---|
| 1 | XGBoost cascade, given labels (`recipes/champion.json`) | **done** (runs 4, 9, 11), 0.5122 / 0.5665 |
| 2 | LightGBM cascade, given labels (`recipes/lgbm.json`) | **done**, 0.5122 / 0.5667 |
| 3 | CatBoost cascade, given labels (`recipes/catboost.json`) | **done**, 0.5120 / 0.5662 |
| 4 | XGBoost cascade, event-cleaned labels (`recipes/clean-event.json`) | **done**, 0.5123 / 0.5666 |
| 5 | XGBoost cascade, event-cleaned labels with noise-floor flips (`recipes/clean-l2.json`) | **done**, 0.5122 / 0.5666 |
| - | the blend (`recipes/blend.json`), one-command build, README | **done**, 0.5123 / 0.5667 |

Scores are leave-one-month-out given / cleaned means, measured in this repo (`RESULTS.md`).

**What exists (measured in this repo).** Full pipeline: typed ledger, relational context features,
eight label-free evidence families (marks, bursts, drains, flows, requests, merchant, repeats,
device counter), cleaned ("silver") labels for scoring, the two-stage cascade, the fixed-point
decoder, the template rule, LightGBM and CatBoost stages, event-cleaned training labels, the
blend, leave-one-month-out evaluation, the native October-like validator and its gate, refit on all
train rows, and a paired event bootstrap. Findings are in `docs/findings/`; 19 test modules under
`tests/`.

**Headline numbers (measured in this repo, `RESULTS.md`).** Mean over the three held-out months,
macro-F1:

| run | given | cleaned |
|---|---|---|
| context-only (run 1) | 0.4671 | n/a at the time (0.5192 rescored, run 5) |
| champion (runs 4, 9, 11) | **0.5122** | **0.5665** |
| champion, fixed point only (no rule) | 0.5064 | 0.5601 |
| champion without counter features (run 12) | 0.4846 | 0.5388 |
| champion without money-flow features (run 13) | 0.5116 | 0.5661 |
| equal-weight five-member blend | **0.5123** | **0.5667** |

The champion's submission (260,582 rows, valid against the sample file) has sha256
`a36833d5fb8c497ff3b6e7ea116226796ec8892ad661c17afa87bf92fb8e380e`, reproduced from a clean
checkout in run 11. The blend's submission has sha256
`c3af1418ffab08f7843b1813359ad7353ee0ac6676f4293609d3cf530a696c05`. On the native October-like
validator the champion's mean over 9 cells is 0.5155 given / 0.5731 cleaned / 0.8980 gen-only. The device counter is worth about +0.028 on both label sets.

**What is under way.** The October-like gate of each member and of the blend against the champion,
a test of refitting October on all train rows, and the blind-spot rule (section 5). Check the
compute box before assuming it is idle.

**Earlier measurements to re-derive** (not in this repo): the leaderboard behaviour of earlier
uploads and most of the ensemble and rule experiments; see the knowledge base below.

## 2. Map of the code

| path | what it is |
|---|---|
| `src/ringtrace/paths.py`, `threads.py`, `classes.py` | data/cache/output paths from environment; one thread budget for every library; the 12 labels in fixed order |
| `src/ringtrace/tables.py` | typed CSV cache; the stacked train+test "ledger" with integer keys |
| `src/ringtrace/windows.py` | fast per-entity time-window counts and sums (one `searchsorted` over a composite key) |
| `src/ringtrace/context.py` | relational features: what the same sender, receiver, phone and pair did around a row |
| `src/ringtrace/evidence/` | label-free generator evidence, one module per shape: `marks` (single-row certificates), `bursts` (bot sessions), `drains` (balance-step sweeps), `flows` (mule/layering chains and fan-ins), `requests` (collect requests), `merchant` (merchant rings), `repeats` (repeat payers), `phones` (device counter + HMM), `anchors` (row anchors the counter reads) |
| `src/ringtrace/store.py` | builds and caches each feature family, keyed by content hash of its source |
| `src/ringtrace/teacher.py`, `silver.py` | cleaned labels for scoring: an independent teacher plus noise inversion, frozen on first build |
| `src/ringtrace/cascade.py` | P(fraud) x P(type given fraud); XGBoost, LightGBM or CatBoost stages; leave-one-month-out |
| `src/ringtrace/eventclean.py` | event-cleaned training labels (event majority; `l2` noise-floor flips), rebuilt inside each fold |
| `src/ringtrace/blend.py` | fixed-weight blend of saved member runs, decoded per month, and its submission |
| `src/ringtrace/refit.py` | learning curve, and October from one fit on all train rows |
| `src/ringtrace/decode.py` | label-free fixed-point decoder for macro-F1, per month |
| `src/ringtrace/rules.py` | counter-anchored look-alike template rule applied after decoding |
| `src/ringtrace/scoring.py` | macro-F1 and its per-class parts |
| `src/ringtrace/run.py` | one recipe on month folds: features, cascade, decode, rules, scores |
| `src/ringtrace/build.py` | the one command: build, freeze labels, run, validate, write submission + sha256 |
| `src/ringtrace/rescore.py` | re-decode saved probabilities without refitting |
| `src/ringtrace/compare.py` | paired event bootstrap between two finished runs |
| `src/ringtrace/octlike.py` | native October-like validator: thinned September frames (`build`), fits (`fit`), paired gate (`score`) |
| `recipes/*.json` | JSON recipes: families, dropped columns, cascade seeds and booster, decoder scale, rules. `champion` is the XGBoost champion; `lgbm`, `catboost`, `clean-event`, `clean-l2` are members 2-5; `blend` is the submission; `champion-cleantrain` trains on cleaned labels; `context-only`, `context-hygiene`, `evidence`, `evidence-nophones` are earlier stages and ablations |
| `scripts/` | compute-box helpers: `ship.sh`, `sync_to.sh`, `launch.sh`, `box.sh`, `box_in.sh`, `ablate.sh`, `retry_gpu.sh`; `build_submission.sh` (the one command); `refit_october.sh`, `refit_tables.py`; `catboost_repro.py` |
| `tests/` | one test module per component, on synthetic data |
| `analysis/` | `findings.py` (the aggregates behind `docs/findings/`) and its helpers in `core.py` |
| `docs/` | `decisions.md`, `research.md`, this file, `knowledge/`, `findings/` |

## 3. Reproducing every result in `RESULTS.md`

Prerequisites: install as in [`../AGENTS.md`](../AGENTS.md), put the organisers' files under
`RINGTRACE_DATA`, a CUDA GPU, and set `RINGTRACE_CACHE` / `RINGTRACE_OUT`. Run from the repository
root on the commit recorded in the table. Runs are named after their recipe; `rescore` and
`compare` read saved runs from `RINGTRACE_OUT/runs/<name>/`.

| result | command |
|---|---|
| runs 1, 2, 4, 10 (stage ladder: context-only, hygiene, evidence, no counter) | `python -m ringtrace.run recipes/<recipe>.json` for `context-only`, `context-hygiene`, `evidence`, `evidence-nophones` |
| cleaned labels (silver; 3,629 rows changed) | `python -m ringtrace.teacher` (also done by `build`) |
| run 9: the champion and its submission | `python -m ringtrace.build recipes/champion.json` |
| run 11: clean-checkout check | `git archive HEAD` into an empty directory, empty `RINGTRACE_CACHE`, then the run 9 command; compare the sha256 |
| runs 5-8: rescores (cleaned labels, plain template rule, no rule, scale 1.0) | `python -m ringtrace.rescore <run> [--rules <list>] [--scale <x>]` on the saved run |
| runs 12-14: leave-one-family-out | `python -m ringtrace.run recipes/champion.json --no-test --without <family>`, or on the box `bash scripts/ablate.sh recipes/champion.json 3 6 <family...>` |
| paired bootstrap table | `python -m ringtrace.compare <base-run> <challenger-run> [--target cleaned] [--draws 2000]` |
| members 2-5 | `python -m ringtrace.run recipes/<recipe>.json` for `lgbm`, `catboost`, `clean-event`, `clean-l2` |
| the blend and its submission | `python -m ringtrace.blend recipes/blend.json` on the saved members, or from scratch `bash scripts/build_submission.sh recipes/blend.json` |
| findings | `python analysis/findings.py`; refit: `python -m ringtrace.refit curve <recipe>` or `october <recipe>`, tables `python scripts/refit_tables.py` |
| October-like tables | `python -m ringtrace.octlike build`, then `fit <recipe>:0,1,2`, then `score <base> <challenger>` |

Wall times in `RESULTS.md` are for one GPU shared by up to three jobs (about 6 minutes alone for the
champion's folds). The October-like frames are built in this repository (`octlike build`).

## 4. Knowledge base

Curated background from earlier research (each with claims labelled measured/earlier):

- [Data](knowledge/data/index.md): what the generated ledger looks like and what is known about it.
- [Modeling](knowledge/modeling/index.md): features, models, decoders, blends, cleaned labels.
- [Experiments](knowledge/experiments/index.md): what was tried, with scores, including what failed.
- [Leaderboard](knowledge/leaderboard/index.md): upload history and how leaderboard scores related
  to local ones.

If a link is missing, that folder is not written yet.

## 5. Next steps (under way)

1. **Gate the members and the blend** on the native October-like validator: cells s13-15 x r0-2
   for `lgbm`, `catboost`, `clean-event`, `clean-l2` and the champion; score each member and the
   equal five-way blend against the champion.
2. **Refit test**: October from one fit on all train rows against the fold mean
   (`docs/findings/more-data.md`).
3. **Blind-spot rule**: a label-free rule that calls fraud on rows with generator evidence that the
   champion calls legit (`docs/findings/validation-blind-spot.md`); adopted only if it never loses
   on leave-one-month-out (all three targets, per month) or on the October-like cells.
4. **Finish the ablations** left over in run 14 (drains, merchant, requests, marks, bursts, repeats,
   context).
