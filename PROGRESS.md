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
- [x] Members 4 and 5: event-cleaned labels and their XGBoost cascades (`recipes/clean-event.json`, `recipes/clean-l2.json`).
- [x] The blend (`ringtrace.blend`, `recipes/blend.json`): given 0.5123 / cleaned 0.5667, submission sha256 `c3af1418...`; the one-command build (`scripts/build_submission.sh`); the README. Member 1 rerun from this branch gives the champion's sha256; the four member branches are merged and the repository is `ruff format`-clean.
