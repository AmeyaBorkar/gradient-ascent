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

## Density: the device counter at October's density (14 Nov)
- October's counter is computed on October alone (its own pass, half density). The October-like
  frames held September's counter at its value on the shared three-month, full-density pass.
- `ringtrace.density` rebuilds a held-out month's counter as its own pass (renumbered phones,
  own gaps kept, dropped units' phones removed): block assignment unchanged (100 %), segment
  agreement with the held counter 0.92 to 0.96 on the thinned frames, mean segment confidence
  0.971 to 0.975 (held 0.993 to 0.997; October's own 0.975).
- `--train-counter half<d>` trains on every train month split into two halves by protected units,
  each half its own pass (October's density). Month folds also predict every held-out month with
  its standalone counter (`oof_alone.npy`) and at October's density (`oof_half0.npy`).
- `RINGTRACE_DEVICE=cpu|cuda` forces every XGBoost and CatBoost booster onto one device; cells and
  month folds record it. The September cells of the XGBoost members and CatBoost are GPU fits on
  one RTX 4060 (they reproduce the earlier GPU cells under the held counter to four decimals);
  LightGBM cells s14 and s15 are CPU fits (s13 is the earlier CPU fit); TabM cells came from an L4.
- Jobs: `scripts/density_queue.sh <queue> 2 <logs> cuda`, gates `scripts/density_gate.sh <step>`.
- Results and verdicts: RESULTS.md, "Density: the counter at October's density";
  `docs/findings/density-counter.md`.
- Verdicts (14 Nov): the held counter overstated the best recipe by 0.0035 given per cell;
  (a) dropping `ctr_pos` rejected; (b) the three XGBoost members trained at October's density pass
  the October-density cells and month folds and fail the production-counter month folds
  (candidate `candidates/density-half3.csv`, 11 October rows).
- Not run for time: the no-position variants of clean-event, clean-l2, LightGBM and CatBoost and
  the champion's no-position month folds; LightGBM and CatBoost at October's density; the second
  draw beyond the champion's cells; the October-mix frames.
- [ ] TabM at October's density: one like-for-like frame gains +0.0044 given. Queued on the box:
      TabM half0 cells s14 and s15, TabM refitted on the same GPU for s14 and s15, month folds
      `density-tabm-half0` and `density-tabm`; then `bash scripts/density_gate.sh tabm` and
      `tabm-lomo`, and build `recipes/density/final-half-chalf3t.json`.
