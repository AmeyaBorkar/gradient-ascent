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
- [x] Signal audits (`python -m ringtrace.audit`, `docs/decisions.md` item 16): ordinary-looking
      fraud carries no signal in train (AUC 0.490) or October (0.490); October's reachable ceiling is
      about 0.556; the blend's expected public score from its call counts is 0.5475 (public 0.5474)
      (`docs/findings/ordinary-rows-audit.md`).

## Next steps (under way)
- [ ] The October-like gate of each member and of the equal five-way blend against the champion
      (cells s13-15 x r0-2).
- [ ] The refit test: October from one fit on all train rows against the fold mean.
- [ ] The blind-spot rule: a label-free rule that calls generator-made rows the champion calls
      legit, adopted only if it never loses on leave-one-month-out and the October-like cells.
- [ ] Finish the leave-one-family-out ablations left over in run 14.

## Look-alike pairs (14 Nov)
- [x] Who writes the partner label: the routine (device-counter zones agree with the label on
      every template event with a fraud-block phone). Clean October labels keep the partner
      types; retyping by template is rejected.
- [x] Current best idea: `pairs-tmplA-flip`, a takeover flip inside the own-phone drain.
      Leave-one-month-out given +0.0007 / +0.0002 / +0.0007, cleaned +0.0007 / +0.0002 / +0.0006;
      6 October rows. Small; fold into a combined upload.
- [x] phishing_collect / fake_cashback (col7, col1) and loan_app_extortion in col1: at chance.

## Unanswered baits and operator roles (14 Nov)
- [x] The best file calls every unanswered Rs 1 / 2 / 5 bait; the only bait rows left are lone
      unanswered Rs 10s (31 cleaned phishing rows called legit in train, 142 legit calls in
      October).
- [x] Label-free operator roles (QR and just-under collect receivers, bait senders, answer
      receivers, bot-pool senders, round merchant payers) in another month: 61 % of recognised
      bait senders against 1.2 % of legit Rs 10 senders. The label-history control fails (6 %).
- [x] Current best idea: `bait-combo` (lone Rs 10 from an operator; first-time whole-rupee PAY
      of Rs 1,000 or more to an operator): cleaned +0.00029 / +0.00017 / +0.00016, nine cells
      +0.00018 (lb +0.00005), 7 October rows legit -> phishing_collect. The plain PAY rule fails.
