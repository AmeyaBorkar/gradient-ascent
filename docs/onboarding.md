# Onboarding

**Summary.** GradientAscent (author: Ameya Borkar) labels every October 2026 UPI transaction with
one of 12 classes (legit + 11 fraud types), scored by macro-F1, using the Python package
`ringtrace`. The code in this repository already reproduces the first of five blend members (the
XGBoost champion, leave-one-month-out mean 0.5122 given / 0.5665 cleaned). The other four members and
the blend are still to be rebuilt. Rules for working here are in [`../AGENTS.md`](../AGENTS.md).

Status labels used below: **measured in this repo** = a run recorded in [`../RESULTS.md`](../RESULTS.md);
**earlier measurement** = carried over from earlier research, not yet re-derived here.

## 1. Current state

**Goal** ([`../PROGRESS.md`](../PROGRESS.md)): rebuild the latest submission end to end in this
codebase. That submission is an equal-weight blend of five cascade models, each a mean of three seeds
(0, 1, 2) over the leave-one-month-out folds, with October as the fold mean. The blend is decoded by
the fixed point at fraud scale 0.875 per month, then the counter-anchored template rule.

| # | member | status |
|---|---|---|
| 1 | XGBoost cascade, given labels (`recipes/champion.json`) | **done**, measured in this repo (runs 4, 9, 11) |
| 2 | LightGBM cascade, given labels | not started (LightGBM is not yet a dependency) |
| 3 | CatBoost cascade, given labels | not started (CatBoost is not yet a dependency) |
| 4 | XGBoost cascade, event-cleaned labels | not started |
| 5 | XGBoost cascade, event-cleaned labels with noise-floor flips | not started |
| - | the blend, its one-command build, README with seeds/versions/hardware | not started |

**What exists (measured in this repo).** Full pipeline: typed ledger, relational context features,
eight label-free evidence families (marks, bursts, drains, flows, requests, merchant, repeats,
device counter), cleaned ("silver") labels for scoring, the two-stage cascade, the fixed-point
decoder, the template rule, leave-one-month-out and October-like evaluation, and a paired event
bootstrap. 13 test modules under `tests/`.

**Headline numbers (measured in this repo, `RESULTS.md`).** Mean over the three held-out months,
macro-F1:

| run | given | cleaned |
|---|---|---|
| context-only (run 1) | 0.4671 | n/a at the time (0.5192 rescored, run 5) |
| champion (runs 4, 9, 11) | **0.5122** | **0.5665** |
| champion, fixed point only (no rule) | 0.5064 | 0.5601 |
| champion without counter features (run 12) | 0.4846 | 0.5388 |
| champion without money-flow features (run 13) | 0.5116 | 0.5661 |

The champion's submission (260,582 rows, valid against the sample file) has sha256
`a36833d5fb8c497ff3b6e7ea116226796ec8892ad661c17afa87bf92fb8e380e`, reproduced from a clean
checkout in run 11. October-like September mean over 9 cells: 0.5093 (champion), 0.5091 (trained on
cleaned labels), against 0.5097 at full density, so the champion's score does not depend on October
being sparser. The device counter is worth about +0.028 on both label sets.

**What is running.** Nothing is recorded as running in the repository. The last queue (leave-one-family-out
ablations) was stopped at wrap-up; the ablations of drains, merchant, requests, marks, bursts, repeats
and context were not finished (run 14). Check the compute box before assuming it is idle.

**Earlier measurements to re-derive** (not in this repo): the blend's own scores, anything about
LightGBM/CatBoost members, noise-floor flips, and the leaderboard behaviour of earlier uploads. See
the knowledge base below.

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
| `src/ringtrace/cascade.py` | P(fraud) x P(type given fraud), XGBoost on GPU; leave-one-month-out |
| `src/ringtrace/decode.py` | label-free fixed-point decoder for macro-F1, per month |
| `src/ringtrace/rules.py` | counter-anchored look-alike template rule applied after decoding |
| `src/ringtrace/scoring.py` | macro-F1 and its per-class parts |
| `src/ringtrace/run.py` | one recipe on month folds: features, cascade, decode, rules, scores |
| `src/ringtrace/build.py` | the one command: build, freeze labels, run, validate, write submission + sha256 |
| `src/ringtrace/rescore.py` | re-decode saved probabilities without refitting |
| `src/ringtrace/compare.py` | paired event bootstrap between two finished runs |
| `src/ringtrace/octlike.py` | October-like evaluation on thinned September frames |
| `recipes/*.json` | JSON recipes: families, dropped columns, cascade seeds and booster, decoder scale, rules. `champion` is the XGBoost champion; `champion-cleantrain` trains on cleaned labels; `context-only`, `context-hygiene`, `evidence`, `evidence-nophones` are earlier stages and ablations |
| `scripts/` | compute-box helpers: `ship.sh`, `sync_to.sh`, `launch.sh`, `box.sh`, `box_in.sh`, `ablate.sh`; `octlike_table.py` summarises October-like output |
| `tests/` | one test module per component (build, bursts, cascade, compare, core, drains, flows, marks, merchant, phones, repeats, requests, rules) |
| `analysis/` | does not exist yet; scratch analyses that produce a finding should go here with a short note in `docs/findings/` |
| `docs/` | `decisions.md`, `research.md`, this file, `knowledge/`, and `findings/` once the first finding is written |

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
| October-like tables | `python -m ringtrace.octlike export-counter` once, then `python -m ringtrace.octlike run <frame-dir> champion:0,1,2 champion-cleantrain:0,1,2` per thinned frame; summarise with `scripts/octlike_table.py` |

Wall times in `RESULTS.md` are for one GPU shared by up to three jobs (about 6 minutes alone for the
champion's folds). The thinned September frames used by October-like mode are built outside this
repository (September thinned to October's density, fraud events kept whole); rebuilding that step
is an open item below.

## 4. Knowledge base

Curated background from earlier research (each with claims labelled measured/earlier):

- [Data](knowledge/data/index.md): what the generated ledger looks like and what is known about it.
- [Modeling](knowledge/modeling/index.md): features, models, decoders, blends, cleaned labels.
- [Experiments](knowledge/experiments/index.md): what was tried, with scores, including what failed.
- [Leaderboard](knowledge/leaderboard/index.md): upload history and how leaderboard scores related
  to local ones.

These folders are being written by other sessions; if a link is missing, the folder is not merged yet.

## 5. Next steps

1. **Member 2, LightGBM cascade** on the given labels: add the dependency, a recipe, run on the three
   month folds with seeds 0, 1, 2; record in `RESULTS.md`.
2. **Member 3, CatBoost cascade**, same protocol.
3. **Members 4 and 5**: event-cleaned labels (event majority, else teacher type) used as training
   targets; then the noise-floor-flip variant. Check each against the validation bar, including the
   October-like frames (the cleaned-label champion tied the given-label one there: 0.5091 vs 0.5093).
4. **The blend**: equal-weight mean of the five members' probabilities, fixed-point decoder at
   fraud scale 0.875 per month, then the template rule; one command; verify the submission's
   sha256 from a clean checkout.
5. **Finish the ablations** left over in run 14 (drains, merchant, requests, marks, bursts, repeats,
   context).
6. **Bring the thinned-frame builder into the repository** so October-like evaluation is fully
   reproducible here.
7. **Write findings** into `docs/findings/` as they are made, and update the README with commands,
   seeds, versions and hardware once the blend builds.
