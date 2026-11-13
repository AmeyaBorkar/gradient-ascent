# Progress

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

## Wide look for a larger lever (14 Nov)
- [x] `python analysis/farlook.py all` (21 s): headroom, pair typing, generator checks and the
      counter's segments at October's size; finding `docs/findings/farlook.md`.
- No lever worth +0.0025 was found, and no candidate file was built.
  - The cleaned headroom is the typing of events inside the six look-alike templates: +0.032 to
    +0.034 a month with perfect typing. Nothing label-free reaches it (AUC 0.49 to 0.64 inside a
    pool; shared accounts do not keep a pair's class), and the template classes are already the
    best split of every pool.
  - Every other error adds up to 0.0043. Finding ordinary-looking fraud cannot pay without its
    type: the largest class share among those rows is 19 %, below every class's break-even.
- **Current best idea:** the counter's segments are fitted on one sixth of train's phones in
  October. Thinning train's block to that size loses 8.6 ± 5.1 unanchored rows of 171 (agreement
  0.897 against 0.942). Recovering it is worth about +0.0005, at most +0.002. The experiment (a
  cut-position prior and per-segment occupancy in `phones.decode_phase`, two gates) is specified
  in the finding and has not been run.

## The counter's segments at October's size (14 Nov)
- [x] Reproduced the thinned baseline: 0.897 ± 0.028 on unanchored fraud rows, 0.800 in the last
      four segments.
- [x] A train-taught prior in `phones.decode_phase`, off by default. It adds a normal prior on each
      cut's place on the number line and a per-segment geometric gap emission. The `phones_prior`
      family holds the test split decoded with it, and the `counter` rule can read it (`source`,
      `unanchored`, `below`). Tests on synthetic data.
- [x] First gate passes. Spread 0.01 / gap weight 1.0 was chosen on seeds 0-14 by the worse of the
      plain and the October-shifted simulation. Agreement rises +0.036 (z 3.7) on seeds 15-29 and
      +0.030 (z 5.1) on all 30. The last four segments gain +0.09 and the full block does not fall.
- [x] Second gate fails. The counter rule after `anchored_templates` loses in September (-0.00003
      to -0.00004 given) on every variant that touches train. In train, when the model's fraud type
      and the segment disagree, the model is right on 17 of 19 rows.
- [x] Candidates built and rejected: `counter-final*.csv`, `counter-tabmF*.csv`.
  - October moves are 5 to 10 rows, all velocity_bot -> mule_account, a pattern the model wins 6
    of 6 times on train.
  - The prior moves only 4 October segments.
- **Current best idea:** none from this lever. It is worth about 0 on October, because the model
  already follows the segment (545 of 564 fraud-block rows) and wins where they disagree. The best
  file stays `blend-tabmF-bq`.

## Next steps (under way)
- [ ] The October-like gate of each member and of the equal five-way blend against the champion
      (cells s13-15 x r0-2).
- [ ] The refit test: October from one fit on all train rows against the fold mean.
- [ ] The blind-spot rule: a label-free rule that calls generator-made rows the champion calls
      legit, adopted only if it never loses on leave-one-month-out and the October-like cells.
- [ ] Finish the leave-one-family-out ablations left over in run 14.
