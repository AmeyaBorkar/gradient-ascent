# Progress

## Close look at the best file (14 Nov, under way)

Audit of every number behind the best public file (`blend-tabmF-bq`, 0.5499354), rebuilt from
the saved arrays (`python analysis/closelook.py prep`: 0 October rows differ from the uploaded
file). Goal: a lever worth +0.0025 macro-F1 on October, proven offline first.

- [ ] Ledger of every scored run and candidate (`docs/findings/ledger.md`).
- [x] Per-class anatomy (given / cleaned / A-clean, per month). Under cleaned labels every class
      but the look-alike ones runs at precision 0.89-0.99 with recall 0.23-0.48: the calls stop
      where the generator rows stop.
- [x] Per-class volume scaling (admission multiplier 0.3-2.0 per class): call volume is rigid
      (0-20 rows per month for most settings). Only qr_swap at 0.5 never loses on LOMO
      (+0.0001 mean, 4 October rows), far below +0.0005.
- [x] Look-alike pairs: every template already gives its unheld pool to the class that maximises
      the pair's F1 sum; within-pool AUC 0.44-0.64; nested top-score splits lose or pick nothing.
- [x] October aggregates: volume departures (velocity_bot, merchant_collusion down, mule up) are
      event-count changes; every label-free anchor is called fraud in October at the train rate.
- [ ] Gate the qr_swap volume lever on the full bar; write `docs/findings/closelook.md`.

Current best idea: none above +0.0005 so far; the measured levers are ties.

## Goal
Rebuild the latest submission end to end in this codebase alone, from ideas only. The submission
is an equal-weight blend of five cascade models, decoded per month:

1. the XGBoost cascade on the given labels (`recipes/champion.json`);
2. a LightGBM cascade on the given labels;
3. a CatBoost cascade on the given labels;
4. the XGBoost cascade on event-cleaned labels (`clean`);
5. the XGBoost cascade on event-cleaned labels with noise-floor flips (`l2`).

Every member is a mean of three seeds (0, 1, 2) over the leave-one-month-out folds; October is the
fold mean. The blend is decoded with the fixed point at fraud scale 0.875 per month, then the
counter-anchored template rule.

## Status
- [x] Import of the standalone pipeline (features, evidence modules, cascade, decoder, template
      rule); `recipes/champion.json` reproduces member 1.
- [x] Member 2: LightGBM cascade (`recipes/lgbm.json`; results in RESULTS.md).
- [x] Member 3: CatBoost cascade (`recipes/catboost.json`; results in RESULTS.md).
- [x] Members 4 and 5: event-cleaned labels and their XGBoost cascades (`recipes/clean-event.json`,
      `recipes/clean-l2.json`).
- [x] The blend (`ringtrace.blend`, `recipes/blend.json`): leave-one-month-out given 0.5123 /
      cleaned 0.5667 / gen-only 0.9013; one command (`scripts/build_submission.sh`); submission
      sha256 `c3af1418ffab08f7843b1813359ad7353ee0ac6676f4293609d3cf530a696c05`. The champion
      rerun gives the champion's sha256; the tree is `ruff format`-clean.
- [x] Findings in `docs/findings/`: headroom per class, member disagreement, public-score noise,
      the validation blind spot, and more data (learning curve, fold mean against refit).
- [x] Native October-like validator (`python -m ringtrace.octlike build|fit|score`): September
      thinned by protected units, fitted on July + August, decoded like a test month, with a
      paired-bootstrap gate (`docs/decisions.md` item 13).
- [x] Knowledge base in `docs/knowledge/` (data, modeling, experiments, leaderboard) and
      `docs/onboarding.md`.

## Next steps (under way)
- [ ] The October-like gate of each member and of the equal five-way blend against the champion
      (cells s13-15 x r0-2).
- [ ] The refit test: October from one fit on all train rows against the fold mean.
- [ ] The blind-spot rule: a label-free rule that calls generator-made rows the champion calls
      legit, adopted only if it never loses on leave-one-month-out and the October-like cells.
- [ ] Finish the leave-one-family-out ablations left over in run 14.
