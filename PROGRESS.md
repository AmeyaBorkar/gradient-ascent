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
- [ ] Member 2: LightGBM cascade.
- [ ] Member 3: CatBoost cascade.
- [ ] Members 4 and 5: event-cleaned labels and their XGBoost cascades.
- [~] The blend (`ringtrace.blend`, `recipes/blend.json`), the one-command build (`scripts/build_submission.sh`) and the README drafted; member 1 rerun from this branch (`blend-champion-m1`, same sha256). Blend run waits for members 2-5.
