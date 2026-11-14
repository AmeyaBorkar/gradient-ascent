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
- [ ] The blend, its one-command build, and the README (commands, seeds, versions, hardware).

## Screen of five more neural techniques (14 Nov)

Each technique is a variant of `python -m ringtrace.nnexp`, fitted once on July + August (seed 0)
and scored on `sep`, `s13`, `s14`, `s15`, alone and as the fraud-mass source, against the plain
TabM fitted on the same machine (`<job>-base-sd0`).

| technique | where | verdict (fraud-only role, x 1e-4 given) |
|---|---|---|
| soft self-training on the scored rows' features | laptop GPU | rejected: 0-2 calls, sep -0.1, s13 -1.2 |
| entity embeddings + strictly-past pooled history | box GPU | rejected: -4.2 to -4.7 in every cell |
| TabR-style retrieval | laptop GPU | rejected: +0.1 / +0.6 on sep / s13, -1.0 / -3.4 on s14 / s15 |
| TabR beside TabM (equal mean, no new fit) | scoring only | rejected: given positive everywhere, s14 cleaned -3.0 |
| shift reweighting toward the scored rows | box GPU | rejected, near-miss: positive in s13-15 on 1-5 calls, sep -0.2 |
| REINFORCE macro-F1 term | laptop GPU | rejected: sep -0.4, s13 -1.9 |
| TabM on generator-only labels | box GPU | rejected: -12 to -22 given, +15 to +30 gen-only |
| entity ablations (pooled history alone, embeddings alone) | box / laptop GPU | fitting |

Numbers are in `RESULTS.md` ("Neural techniques, second screen"), and each technique has a
finding in `docs/findings/nn-*.md`. Nothing passed, so nothing was promoted and no October file
was written.

Current best idea: none from this screen. No neural training technique moves more than a few
dozen October-like calls in the fraud-mass role without losing on given labels.
