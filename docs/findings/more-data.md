# More data: learning curve, fold mean against refit, refit October

Model: the champion XGBoost cascade (300 rounds, eta 0.08, depth 8, seeds 0, 1, 2 averaged),
decoded as the run decodes (fixed point per month at fraud scale 0.875, then the anchored
template rule). Code: `python -m ringtrace.refit curve|october <recipe>`; tables:
`python scripts/refit_tables.py`.

## 1. Learning curve (screening off in every arm)

A single month cannot cross-fit the screen, so every arm here trains on the labels as they are
(all fraud rows are type rows, no legit row dropped).

| arm | given 7 / 8 / 9 (mean) | cleaned 7 / 8 / 9 (mean) | gen-only 7 / 8 / 9 (mean) |
|---|---|---|---|
| one month (mean score of the two single-month fits) | 0.5130 / 0.5084 / 0.5140 (0.5118) | 0.5646 / 0.5633 / 0.5706 (0.5662) | 0.9020 / 0.8990 / 0.9007 (0.9006) |
| two months, base rounds | 0.5136 / 0.5093 / 0.5145 (0.5125) | 0.5653 / 0.5643 / 0.5712 (0.5669) | 0.9031 / 0.9002 / 0.9011 (0.9015) |
| **gain from the second month** | +0.0006 / +0.0009 / +0.0005 (**+0.0007**) | +0.0007 / +0.0010 / +0.0006 (**+0.0007**) | +0.0011 / +0.0012 / +0.0004 (**+0.0009**) |

Single-month fits per held-out month (given): hold 7: Aug 0.5130, Sep 0.5129; hold 8: Jul 0.5092,
Sep 0.5076; hold 9: Jul 0.5144, Aug 0.5135.

Per class (mean over the three held-out months; the one-month figure averages both single-month
fits):

| class | given one / two / gain | cleaned one / two / gain | gen-only one / two / gain |
|---|---|---|---|
| legit | 0.987 / 0.987 / +0.000 | 0.988 / 0.988 / +0.000 | 1.000 / 1.000 / +0.000 |
| mule_account | 0.433 / 0.433 / +0.000 | 0.486 / 0.486 / +0.000 | 0.882 / 0.882 / +0.000 |
| velocity_bot | 0.517 / 0.517 / -0.000 | 0.581 / 0.581 / -0.000 | 0.933 / 0.933 / -0.000 |
| layering_chain | 0.551 / 0.551 / +0.000 | 0.611 / 0.611 / +0.000 | 0.926 / 0.926 / +0.000 |
| sim_swap | 0.499 / 0.500 / +0.001 | 0.552 / 0.553 / +0.001 | 0.868 / 0.869 / +0.001 |
| account_takeover | 0.434 / 0.434 / +0.000 | 0.485 / 0.485 / +0.000 | 0.836 / 0.836 / +0.000 |
| phishing_collect | 0.486 / 0.487 / +0.001 | 0.550 / 0.550 / +0.000 | 0.874 / 0.875 / +0.001 |
| qr_swap | 0.510 / 0.512 / +0.002 | 0.575 / 0.578 / +0.002 | 0.944 / 0.947 / +0.003 |
| refund_abuse | 0.529 / 0.530 / +0.002 | 0.594 / 0.596 / +0.002 | 0.949 / 0.953 / +0.004 |
| fake_cashback | 0.396 / 0.396 / +0.000 | 0.446 / 0.446 / +0.000 | 0.755 / 0.755 / +0.000 |
| loan_app_extortion | 0.413 / 0.414 / +0.001 | 0.473 / 0.474 / +0.001 | 0.906 / 0.904 / -0.002 |
| merchant_collusion | 0.386 / 0.389 / +0.002 | 0.453 / 0.456 / +0.003 | 0.935 / 0.939 / +0.004 |

**The curve is flat.** Doubling the training data (about 0.53M to 1.05M rows) adds +0.0007 to
+0.0009 macro-F1. No class gains more than 0.004. The small gains go to the rare, merchant-side
classes (collusion, refund, qr). Mule, bot, chain, takeover and cashback do not move: the model is
limited by label noise and the ordinary-looking fraud rows, not by the number of rows.

## 2. Fold-mean analogue against one model on more data

| arm | given 7 / 8 / 9 (mean) | cleaned 7 / 8 / 9 (mean) | gen-only 7 / 8 / 9 (mean) |
|---|---|---|---|
| mean of the two one-month models' probabilities | 0.5136 / 0.5088 / 0.5143 (0.5122) | 0.5652 / 0.5639 / 0.5709 (0.5667) | 0.9029 / 0.8998 / 0.9010 (0.9012) |
| one two-month model, base rounds | 0.5136 / 0.5093 / 0.5145 (0.5125) | 0.5653 / 0.5643 / 0.5712 (0.5669) | 0.9031 / 0.9002 / 0.9011 (0.9015) |
| one two-month model, 1.5x rounds | 0.5138 / 0.5089 / 0.5147 (0.5124) | 0.5654 / 0.5640 / 0.5715 (0.5669) | 0.9032 / 0.8996 / 0.9014 (0.9014) |

At base rounds the single model on more data is at least as good in all 9 month x target cells
(8 wins by up to +0.0005; given/Jul ties to 4 decimals). At 1.5x rounds it wins 8 of 9 (it loses
gen-only/Aug by 0.0002). On the means the margin is +0.0002 to +0.0003. That is far below the
month-to-month spread and the about ±0.002 bootstrap intervals of earlier comparisons. The
direction is consistent but the effect is negligible. Averaging two models recovers most of the
second month's gain, and scaling the rounds by the data neither helps nor hurts.

## 3. Refit October predictions

Each member is fitted once per seed (0, 1, 2) on all 1,564,060 train rows, with the screen
cross-fitted over the three months. Rounds are 300 x 1,564,060 / mean fold rows (1,042,707) = 450,
and the screen keeps its 150 rounds. The event-cleaned members clean the labels with all three
train months voting.

| member | refit: argmax changed | refit: decoded calls changed | hedge: argmax changed | hedge: decoded changed | October fraud calls fold / refit / hedge | wall |
|---|---|---|---|---|---|---|
| champion | 52 | 19 | 25 | 12 | 3945 / 3954 / 3952 | 5.3 min |
| clean-event | 48 | 15 | 24 | 9 | 3950 / 3953 / 3954 | 5.0 min |
| clean-l2 | 61 | 12 | 38 | 8 | 3948 / 3950 / 3951 | 7.1 min |

All changes are counted against each member's fold mean, out of 260,582 October rows. The refit
moves fewer than 0.03 % of argmax calls and at most 19 decoded calls per member. Given sections 1
and 2, expect a change of a few ten-thousandths at most. The files are there for the blend to use,
but they will not change the submission in any way that matters.

Outputs (box): `~/outputs/runs/{champion,clean-event,clean-l2}-{refit,hedge}/` hold `test.npy`
(n x 12 float32), `oof.npy` (the member's out-of-fold file, copied unchanged) and `summary.json`.
The learning curve is in `~/outputs/runs/refit-curve/summary.json`.

## Choices made
- The screen is off in sections 1 and 2 (as briefed). The two-month base arm (0.5125 given) scores
  like the screened champion (0.5122), so the comparison is not distorted by it.
- "One" in section 1 is the mean score of the two single-month fits, not the score of their
  probability mean (that is section 2's analogue).
- Refit rounds scale only the cascade boosters. The screen stays at 150 rounds: in the refit its
  models already train on two months, against one in a fold.
- The output directories use the member names (`champion-refit`, ...) as briefed, rather than a
  `refit-` prefix. None of them existed before.
