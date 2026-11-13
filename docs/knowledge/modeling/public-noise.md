# The public-score noise model

**Summary.** The public leaderboard scores about 30% of the October rows (about 78,175 of 260,582).
It is a small, noisy sample of the full test, and the platform ranks the best public upload on the
private rest. For one upload the public score has a standard deviation of about 0.010-0.014 if
rows are split at random and 0.019-0.027 if whole events are; for two near-identical uploads the
difference has an sd of 0.002-0.008. Read a public gap under about 0.005 (row split) or 0.007-0.008
(event split) between near-identical files as zero. All numbers here are **earlier measurement**
(simulations on train months), which a later session can re-derive here.

## How the noise was measured
Take a train month, draw an October-sized test month (260,582 rows, whole events kept), split it
30/70 by row or by whole event, and score both parts; 400-2,000 draws per month and split type.
Truth is the cleaned labels (which favours the detector). Events were linked by label-free event
ids, fraud-block devices, or same-label links on device, sender, person-receiver and flow keys;
about 15,000 events per month, 99th percentile of 6-7 rows. The organisers' split rule (row or
event) is unknown, so both are given.

## One upload (macro-F1, range over the three train months)
| quantity | row split | event split |
|---|---|---|
| sd of the public score | 0.010-0.014 | 0.019-0.027 |
| sd of public minus whole-test score | 0.008-0.009 | 0.017-0.023 |
| sd of public minus private | 0.011-0.014 | 0.022-0.032 |
| public mean below the month's score (small-sample bias) | about 0.002 | about 0.006 |

A 90% interval for a champion forecast of 0.54-0.56 is about 0.536-0.579 (row) and 0.512-0.585
(event).

## Two similar uploads (about 300 rows differ)
Typical for variants that differ by a rule or a decoder tweak:
| quantity | row split | event split |
|---|---|---|
| sd of the public gain | 0.0037-0.0039 | 0.0047-0.0050 |
| sd of public gain minus whole-test gain | 0.0032-0.0033 | 0.0043-0.0046 |
| sd of public gain minus private gain | 0.0045-0.0047 | 0.0060-0.0065 |
| smallest public gain that is not noise (1.645 sd, one-sided 95%) | 0.005 | 0.007-0.008 |

Chance that public and private order the pair differently, by the true gap on the whole test:

| true gap | 0.002 | 0.005 | 0.010 | 0.015 |
|---|---|---|---|---|
| row split | 34% | 5.5% | 0.1% | 0% |
| event split | 42% | 10% | 0.5% | 0% |

A separate plain-model simulation gave 27-44% at a true gap of 0.003, 2-20% at 0.006 and 1-4% at
0.010-0.019. Chance that the public
shows no gain at all: 27% (row) and 28% (event) at a true gap of 0.002; 5.5% and 10% at 0.005; 0.1%
and 0.5% at 0.010. A true gap of +0.03 was never reversed in 6,000 draws.

For two different models (thousands of rows differ), the paired noise approaches the one-upload
numbers: up to 0.009 (row) or 0.022 (event) for public minus whole test. So "not noise" runs from
0.005 to 0.015 under a row split and 0.007 to 0.036 under an event split, depending on how different
the two files are. Knowing the organisers' split rule would change the threshold by about 1.4x for
near-identical files and 2.5x for one upload.

## Best of several uploads
Best public of eight candidates spread over 0.013-0.019 is the truly best one in 66-84% of draws and
costs 0.002-0.003 on private. Repeated adaptive uploads overfit the public slice; never select or
tune on the public score alone. Track the relationship between local scores and public scores after
each upload; if local validation stops ranking uploads the way the leaderboard does, find out why
(shift, a leak, a metric difference) before the next upload.

## Reading the all-legit upload
An all-legit file scores a macro-F1 set only by the test fraud share `s`:
`F = 2(1-s) / (12(2-s))`, inverted `s = (1-12F) / (1-6F)`. One point of fraud share moves it by
0.00043. Check which macro-F1 formula the organisers use; for a constant submission both common
formulas agree.

| fraud share | public score | fraud rows in public |
|---|---|---|
| 1.40% | 0.082746 | 1,094 |
| 1.50% (ordinary rows do not count) | 0.082703 | 1,175 |
| 2.00% | 0.082492 | 1,564 |
| 2.80% | 0.082150 | 2,189 |
| 3.60% | 0.081806 | 2,814 |
| 4.06% (ordinary rows count) | 0.081606 | 3,175 |

Four displayed decimals give the share to about +-0.12 points; five, +-0.012. The public share
itself has sd 0.06 points (row split) or 0.13-0.14 (event split), far below the 2.6-point gap
between the two hypotheses (about 0.0816 against 0.0827). Regime reading table:

| all-legit score | implied share | reading |
|---|---|---|
| <= 0.08170 | >= 3.85% | ordinary rows count, labels clean |
| 0.08170-0.08185 | 3.50-3.85% | ordinary rows count, train-style noise kept |
| 0.08185-0.08260 | 1.75-3.50% | unexplained: run the census audit, upload no model file |
| >= 0.08260 | <= 1.75% | ordinary rows do not count |

The first upload read ordinary rows as counting. A forecast of 0.541-0.544 for the model (and about
0.507-0.510 if train-style noise were kept) was made on that reading; the earlier XGBoost champion's
public score was 0.5470, inside the 0.54 +- 0.02 (row) / +- 0.04 (event) band. A second upload that
falls more than 0.045 below its forecast (2 sd, event split) means the October transfer failed:
compare per-class calls with the census before any further upload.

## Rules of thumb
1. A public gap smaller than the noise threshold above between near-identical files is zero.
2. Rank candidates by local intervals ([validation.md](validation.md)), not the public score.
3. Macro-F1 noise is dominated by the rarest classes: a class with about 30 scored rows and F1
   near 0.5 moves macro-F1 by about +-0.007 alone.
4. Probing row labels through the leaderboard is out of bounds; constants of a rare class are
   uninformative below about 0.1% prevalence.
5. The public score cannot see changes under about 0.005; a change that alters 16 of 260,582 rows
   (about 5 public rows) can never be resolved by it.
