# Shift reweighting of TabM toward the scored month

**Verdict: rejected.** At seed 0 it looked like a near-miss. As the fraud-mass source it was
positive on every target in all three October-like cells, on 1-5 calls per cell, and -0.00002
given on September. Seeds 1 and 2 do not repeat it: across the three seeds the given change is
positive in 5 of 12 cell-seeds and negative in 5, always on 0-5 calls. It is seed noise, and
five calls of a mid-size class are worth about 0.0002 at most anyway.

## What was tested

`nnexp` variant `shift`, seed 0, box GPU, against the plain TabM refitted on the same GPU. The fit
rows (July + August) are weighted toward the scored rows' features (September on `sep`, the
thinned frame in s13-15, never October), and the scored rows' labels are never read:

1. **Domain classifier** (`nn_shift.domain_weights`). LightGBM (200 rounds, 31 leaves, balanced
   domains) tells fit rows from scored rows. Both sets are split in two seeded halves, and each
   half's fit rows are scored by the model trained on the other half. Weight = odds, clipped to
   [0.2, 5], divided by its mean.
2. **Training** (`nn_shift.fit_predict_weighted`). TabM's preprocessing, then a row-weighted
   cross-entropy with class weights from the weighted counts (`nn_selftrain.train_one_soft`).

**First attempt failed: every column.** On all 230 raw columns, the classifier told September
from July + August perfectly (domain AUC 1.0000), and 99.99 % of the weights sat on the clip
floor. The cause is the density-scaled window counts. `context.entity_view` divides each 1-day
and 7-day count by its month's rows-per-day factor, so a count of k becomes k / d_month: a value
on a lattice that names the month. The fit was stopped and the classifier now leaves out every
column whose own AUC between July + August and the unthinned September is more than 0.05 from
0.5. Between those months nothing but the calendar moves. That is 18 columns, fixed on `sep` and
reused in every cell: the sender, receiver, device and pair counts over 1 and 7 days, back and
ahead (gaps 0.05-0.50), and `flow_prev_in_dt` / `flow_prev_in_ratio` (0.21 / 0.16).

| cell | domain AUC | effective sample share | clipped |
|---|---|---|---|
| sep | 0.594 | 0.876 | 3.2 % |
| s13 | 0.786 | 0.662 | 26.7 % |
| s14 | 0.788 | 0.664 | 27.1 % |
| s15 | 0.788 | 0.664 | 27.3 % |

In the thinned frames the classifier finds the density shift, most likely through the
10-minute and 1-hour counts, which are not density-scaled. The unthinned September barely
differs from July + August once the month lattice is gone.

## Results

Change x 1e-4, given / cleaned / gen-only, with the calls changed. In the October-like cells the
event-bootstrap 95 % interval of the fraud-only given change follows.

| role | sep | s13 | s14 | s15 |
|---|---|---|---|---|
| fraud-only | -0.2 / -0.1 / +0.3 (3) | +0.4 / +0.5 / +2.5 (5) | +0.3 / +0.3 / +0.7 (1) | +0.2 / +0.3 / +2.1 (5) |
| fraud-only, 95 % interval (given) | - | [-1.9, +2.2] | [+0.0, +0.8] | [-1.9, +1.7] |
| alone | +0.6 / +0.7 / +2.0 (7) | +1.6 / -2.0 / +12.0 (27) | -3.5 / -4.5 / +0.3 (6) | -5.3 / -5.9 / +11.6 (16) |

**Seeds 1 and 2** (the same weights; only the TabM seed changes; each against the base of the
same seed on the box):

| seed | role | sep | s13 | s14 | s15 |
|---|---|---|---|---|---|
| 0 | fraud-only | -0.2 / -0.1 / +0.3 (3) | +0.4 / +0.5 / +2.5 (5) | +0.3 / +0.3 / +0.7 (1) | +0.2 / +0.3 / +2.1 (5) |
| 1 | fraud-only | -0.2 / -0.2 / -1.1 (4) | -0.6 / -0.5 / -0.2 (2) | -0.8 / -0.8 / -1.0 (1) | +0.0 / +0.0 / +0.0 (0) |
| 2 | fraud-only | +0.8 / -0.1 / +0.7 (5) | -0.3 / -0.3 / -0.8 (1) | +0.0 / +3.7 / +5.4 (2) | +0.6 / +0.5 / +0.2 (2) |
| 1 | alone | +0.4 / +0.4 / -1.0 (14) | -10.0 / -9.4 / +5.5 (17) | -6.8 / -6.5 / +3.4 (10) | -1.8 / -1.8 / -2.7 (11) |
| 2 | alone | -1.4 / -1.3 / -1.0 (10) | -3.1 / -4.7 / -10.1 (18) | +0.4 / +0.3 / -1.8 (5) | -3.9 / -4.1 / -1.6 (9) |

The seed-0 pattern does not repeat. Across three seeds and four cells the fraud-only given
change is positive in 5 of 12 cell-seeds, negative in 5 and 0 in 2, always on 0-5 calls. Alone,
seeds 1 and 2 lose given in 6 of 8 cells. The reweighting's effect is smaller than the seed
noise.

## What it means

- **Covariate reweighting toward a thinned month is a tiny lever.** The weights only reshuffle
  which training rows count, and the fraud mass moves on 0-5 rows per cell, with the sign set by
  the seed.
- **Alone it leans toward losing on given** (seed 0 s15 -0.00053, seed 1 s13 -0.00100, seed 2
  s15 -0.00039). Which reweighted rows cause this was not measured.
- **Real October has more shift than the frames** (amounts x1.166 on legit rows, dead columns),
  and a classifier pointed at October would weight harder. Nothing measured here says that would
  help, and the screen never uses October.
- **Any classifier on these features needs the month lattice removed first.** The same 18
  columns would identify October from the training months for any adversarial or reweighting
  use.
