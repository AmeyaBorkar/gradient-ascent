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

| technique | where | status |
|---|---|---|
| soft self-training on the scored rows' features (teacher: plain TabM, w 0.3) | laptop GPU | fitted; scoring against the laptop base |
| entity embeddings (sender, receiver, device) + strictly-past pooled history | box GPU | fitted; loses in every cell (fraud-only -0.0004 to -0.0006 given, cross-machine look) |
| TabR-style retrieval | laptop GPU | fitting |
| shift reweighting (domain classifier toward the scored rows) | box GPU | first fit stopped: domain AUC 1.0000 on all features (calendar columns); refit with those columns left out, queued |
| REINFORCE macro-F1 term (lam 0.5) | laptop GPU, after TabR | queued |
| TabM on generator-only labels rebuilt from July + August alone | box GPU | fitting |
| entity ablation: pooled history alone | box GPU, after shift | queued |

The plain TabM refitted on the laptop differs from the earlier screen's base (another GPU) by 70
September calls, so every comparison uses a base fitted on the same machine.

Current best idea: none yet; the bar is a non-negative fraud-only change in every cell and target.
