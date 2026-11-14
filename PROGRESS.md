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

## The 0.04-cut cascade member (14 Nov, from 13:00 IST)
- What the champion cascade trains on now (`ringtrace.cascade.training_rows`): stage one keeps
  every fraud-labelled row and the legit-labelled rows whose cross-fitted screening score is
  below 0.70 (legit rows at 0.70 or more are left out as suspected unreported fraud); its target
  is 1 only on fraud-labelled rows scoring at least 0.10. Stage two trains on those fraud rows
  scoring at least 0.10. So the legit-side drop at 0.70 is already in the champion, and a 0.04
  type cut adds rows to stage two (about 270 to 430 per fit) rather than removing any.
- Built: `ringtrace.member` (seed-by-seed fits; one set of fits serves several seed subsets,
  byte-equal to `fit_predict`), the `type` blend role in `ringtrace.final.mix`, recipes
  `member-x5` (0.04 cut, seeds 0-4) and `member-cut04` (0.04 cut, seeds 0-2), five blend
  candidates in `recipes/cand/member-*.json` and the gate `analysis/member_gate.py`.
- Running on the box GPU: leave-one-month-out seeds 0-4 and October-like frames s13-15 with seeds
  0-14; a champion control on frame s14 checks that the cells reproduce.
