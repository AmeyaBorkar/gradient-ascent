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

## Next steps (under way)
- [ ] The October-like gate of each member and of the equal five-way blend against the champion
      (cells s13-15 x r0-2).
- [ ] The refit test: October from one fit on all train rows against the fold mean.
- [ ] The blind-spot rule: a label-free rule that calls generator-made rows the champion calls
      legit, adopted only if it never loses on leave-one-month-out and the October-like cells.
- [ ] Finish the leave-one-family-out ablations left over in run 14.

## Clean-label decoding (14 Nov)
- [x] Noise process by class, template and observables: unreported 12 % and wrong type 3 %,
      uniform; no group the model could learn to suppress.
- [x] Counter proxy target (A-clean with the label-free counter imposed, 200 rows).
- [x] Clean-posterior decoder (`ringtrace.decode.clean_posterior`, `fixed_point(clean=...)`):
      every variant moves 0-2 October rows; best passing variant is a one-row tie
      (`cleandec-clean-uk-flip.csv`, not worth an upload).
- [x] Public check: the template-4 switch's public cost (-0.0078) rejects template-pure clean
      labels and fits the train partner share.
- Current best idea from this line: none; the gap to higher public scores is not in decoding.
