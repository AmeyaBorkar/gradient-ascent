# Refit test: one model on all months against the fold mean, October-like

Question: the October submission averages three fold models, each fitted on two months (the fold
mean). Would one refit on all three months be better?

On leave-one-month-out (`docs/findings/more-data.md`), one model on two months beats the mean of
two one-month models by +0.0002 to +0.0003. The test here is the same comparison on the
October-like frames, which are thinned like October.

Setup, per cell (thinning seeds 13-15 x seed sets r0-r2), champion recipe:
- **Fold mean:** the champion fitted on July alone and on August alone, with probabilities averaged.
- **Refit:** the champion fitted on July + August (the gate's existing champion cell).
- **Control:** July + August with the screen off.

One month cannot cross-fit the cascade's screening score, so the one-month fits run with the screen
off. The control separates the effect of more data from the effect of the screen.

Numbers: `RESULTS.md`, section "Refit test".

| comparison | given s13 / s14 / s15, LB | cleaned s13 / s14 / s15, LB | gen-only s13 / s14 / s15, LB | verdict |
|---|---|---|---|---|
| refit vs fold mean | +0.0006 / +0.0006 / -0.0000, -0.0003 | +0.0006 / +0.0006 / -0.0000, -0.0004 | +0.0007 / +0.0012 / +0.0002, -0.0003 | FAIL |
| July + August screen off vs fold mean | +0.0005 / +0.0003 / -0.0001, -0.0003 | +0.0008 / +0.0004 / +0.0001, -0.0001 | +0.0009 / +0.0004 / +0.0003, -0.0002 | FAIL |
| screen on vs screen off | +0.0000 / +0.0003 / +0.0001, -0.0006 | -0.0002 / +0.0002 / -0.0001, -0.0008 | -0.0002 / +0.0008 / -0.0000, -0.0009 | tie |

**Rejected.**
- The refit is ahead on two of three seeds, by +0.0006, and the mean is +0.0004 given and cleaned.
  But seed 15 is at zero, and every lower bound is negative.
- The gain is real in direction: the screen-off control shows the same pattern, so it comes from
  the second month, not the screen. It is too small and too uneven across thinnings to clear the
  bar.
- This matches leave-one-month-out: one model on more data is worth a few ten-thousandths,
  inside the noise.

Consequences:
- No October refit of `lgbm` or `catboost` was run.
- No refit five-way blend was built.
- The fold-mean blend stays.
- The existing refit files of the champion, clean-event and clean-l2 stay unused.

## Choices made
- **Base rounds:** the refit arm is the existing July + August cell at the champion's 300 rounds.
  The October refit module instead scales rounds by data size (1.5x). On leave-one-month-out,
  scaling neither helped nor hurt (`more-data.md`), so it was not tested again here.
- **Screen off in the one-month fits:** this is forced. The control row shows the screen itself is
  a tie on these frames.
- **Cell names:** the new cells keep the validator's naming (`champion-m7`, `champion-m8`,
  `champion-noscreen`, `_s<seed>_r<r>.npz`), so the fold mean is the blend `champion-m7+champion-m8`
  in the standard scorer.
