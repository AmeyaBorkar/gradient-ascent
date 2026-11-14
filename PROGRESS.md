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

## Density: the device counter at October's density (in progress, 14 Nov)
- October's counter is computed on October alone (its own pass, half density). The October-like
  frames held September's counter at its value on the shared three-month, full-density pass.
- `ringtrace.density` rebuilds a held-out month's counter as its own pass (renumbered phones,
  own gaps kept, dropped units' phones removed): block assignment unchanged (100 %), segment
  agreement with the held counter 0.92 to 0.96 on the thinned frames, mean segment confidence
  0.971 to 0.975 (held 0.993 to 0.997; October's own 0.975).
- First cells (seed 13): the held counter overstates a tree member by about 0.004 macro-F1.
- Every box job runs CPU-only: `scripts/density_queue.sh` hides every GPU
  (`CUDA_VISIBLE_DEVICES` empty) and sets `RINGTRACE_DEVICE=cpu`, which forces XGBoost `device`
  cpu and CatBoost `task_type` CPU whatever the recipe says. With no GPU visible, the CUDA build
  of XGBoost 3.4.1 fails in `xgb.train`'s per-round evaluation of a multiclass model; CPU fits now
  run the same boosting loop without that evaluation (identical model bytes, tested).
- The box container has a CPU quota of 46 cores shared by every job, so the queue was ordered by
  value: September cells of the five trees (seeds 13 to 15), month folds and cells of the three
  XGBoost members with and without `ctr_pos`, then the champion on a half-density counter (draw
  0, then draw 1), then October-mix frames and the LightGBM / CatBoost `ctr_pos` variants.
  TabM cells and month folds with the standalone counter run on Colab.
