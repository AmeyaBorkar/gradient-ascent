# Decoders and prior-shift methods

**Summary.** The champion decodes each month with a label-free fixed point (fraud scale 0.875), then
a counter-anchored one-class-per-look-alike-template rule. Every alternative decoder tested ties or
loses: per-class prior shifts, census-anchored counts, F/2 admission, cut-offs or multipliers tuned
on given or cleaned labels, bagged multipliers, EM / BBSE / calibration label-shift methods,
proportional splits of inseparable pools. The one tiny positive (one fraud class per label-free
event, +0.0009 October-like, +0.0003 LOMO) is already done by the template rule. The reason is in the
probability histogram: the cascade's output is almost bimodal, so no cut-off between about 0.2 and
0.95 changes a call.

Status key: **[repo]** = measured in this repository (`RESULTS.md`); **[earlier]** = earlier
measurement, to re-derive if needed. Back to the [index](index.md). Related:
[ensembles.md](ensembles.md), [counter.md](counter.md), [lookalikes.md](lookalikes.md).

## Overview

| experiment | effect (cleaned unless stated) | verdict |
|---|---|---|
| fixed point vs plain argmax | +0.0004 given (0.5122 vs about 0.5066 argmax before the rule) **[repo]** | baseline decoder |
| fraud scale 0.875 vs 1.0 | +0.0001 given **[repo]**; 0.60-1.10 within +-0.0004 [earlier] | tie, keep 0.875 |
| template rule (one class per template, anchored) vs none | +0.0059 cleaned / +0.0053 given [earlier]; anchoring itself +0.0029 cleaned **[repo]** | **helps** |
| plain (unanchored) template rule vs anchored | -0.0032 [earlier]; anchoring +0.0028 given / +0.0029 cleaned **[repo]** | anchoring helps |
| per-class prior shift to cleaned labels (a0-a4) | -0.0002 to +0.0002 | tie |
| census-anchored counts (b1-b3) | -0.0101 / -0.1248 / -0.0040 October-like | loses |
| F/2 admission (c1-c3) | -0.0006 to -0.0003 | tie / loses |
| cut-offs or multipliers fitted on cleaned or given labels (d0-d3) | -0.0022 to -0.0184 | loses |
| bagged multipliers or cut-offs (e0-e2) | -0.0024 to -0.0004 | loses |
| one class per event (f1-f6) | +0.0008 to +0.0009 October-like; +0.0003 LOMO | tie (free but tiny) |
| prior-shift methods EM / BBSE / BCTS / census | -0.0235 to +0.0005 | loses or tie |
| proportional split of an inseparable pool | -0.0001 to -0.016 per pair | loses |
| event-level unification at the old protocol | -0.0001 / -0.0003 | tie |
| decoder multipliers fitted on the other two months | +0.0004 given / +0.0002 cleaned | tie |
| anchors from event shapes with learned emissions (counter) | +0.0001 [-0.0001, +0.0003] | tie; see [counter.md](counter.md) |

## Why the decoder has no lever [earlier]

On leave-one-month-out months, 95.1% of rows have P(fraud) between 0.01 and 0.05 (2.4% of them are
labelled fraud: the ordinary rows), 1.6% have P(fraud) >= 0.95 (97.2% fraud on cleaned labels, 87.9%
on given), and only 17 rows per 100,000 lie between 0.2 and 0.95. The 0.05-0.2 band has 291 rows
per 100,000 at 5.3% fraud, below any class's F/2 threshold. So every fraud cut between about 0.2
and 0.95 calls the same rows. Moving the fraud scale from 0.70 to 1.00 changes 11-22 calls per
100,000 and cleaned macro-F1 by at most 0.0004. The train labels miss about 12% of generator fraud,
but those rows already score >= 0.95 because the cascade drops high-score legit-labelled rows from
its legit side. Calibration: the sum of P(fraud) is 3,843-3,910 per 100,000 rows against 3,672-3,702
given-label fraud, so the model is calibrated to the given labels, ordinary mass included, and that
mass (spread at about 0.023 on legit-looking rows) never crosses a threshold.

What is left for a decoder is the **fraud type inside called rows**, and the template rule settles
most of it.

## The champion's decoder and its settings

- **Fixed point per month [repo]:** re-estimates class priors on the scored month from its own
  probabilities, label-free. At fraud scale 0.875 the champion scores 0.5064 given / 0.5601 cleaned
  (fixed point) and 0.5066 / 0.5602 (plain argmax) before the rule; with the anchored rule 0.5122 /
  0.5665. A run with the rule off and scale 1.0 scores 0.5063 / 0.5600: scale 0.875 against 1.0 is
  a tie (+0.0001).
- **Template rule [repo]:** counter-anchored version changed 773 train and 137 test rows (+0.0058
  given, +0.0064 cleaned over the fixed point). The plain one-class-per-template version gives
  0.5094 / 0.5636, so anchoring adds +0.0028 / +0.0029.
- **Template groups [earlier]** (each template's majority class under macro-F1): mf1 -> layering_chain,
  drain 0.25 full signature -> sim_swap, mf2/3 -> mule_account, col7 -> fake_cashback (the minority
  class of its pair, 37-44% cashback; wins only because cashback has the lower F1), col1 ->
  phishing_collect, rng1 -> qr_swap. Nested leave-one-month-out the group choice agrees in all
  three months and no group is negative in any month (details in [rules.md](rules.md)).
- **Per-class cuts of the fixed point [earlier]:** per-class multipliers fitted on the scored month
  (an oracle) give 0.5096 / 0.5629 against the fixed point's 0.5069 / 0.5599, fitted on the other
  two months 0.5073 / 0.5601, split-half within the month 0.5057 / 0.5579 against 0.5056 / 0.5584 for
  the fixed point on the same halves. Half of the oracle's +0.0027 sits in sim_swap's drain
  look-alike decision, which the template rule takes without tuning.

## The six challengers (October-like, 3 seeds; deltas against the champion decoder) [earlier]

Baseline: 0.5396 October-like (event-cleaned labels), 0.5469 LOMO, 0.8876 generator-only LOMO.
"oct13" is one thinned September; "octfam" is a second thinning that protects label-free event
families (easier month, every delta moves toward 0).

| family | variants | oct13 cleaned delta | LOMO cleaned / generator delta |
|---|---|---|---|
| (a) per-class prior shift to the cleaned label counts | a0 uniform, a1 per class, a2 vs probability mass, a3 on top of 0.875, a4 fixed point with cleaned priors | -0.0002 to +0.0002 | -0.0002 to +0.0001 / +0.0001 to +0.0004 |
| (b) census-anchored per-class counts | b1 baseline's calls per census row; b2 cleaned label rows per census row; b3 halfway | -0.0101 / -0.1248 / -0.0040 | -0.0015 / -0.1334 / -0.0004 |
| (c) F/2 admission | c1 with achievable F1 from fit months; c2 label-free; c3 fixed point's F | -0.0003 / -0.0006 / -0.0006 | -0.0002 / 0.0000 / -0.0001 |
| (d) cut-offs tuned on labels | d0, d1 multipliers on given, cleaned; d2, d3 cut-offs on the fixed point's scores | -0.0022 / -0.0029 / -0.0184 / -0.0180 | -0.0007 / -0.0008 / -0.0004 / -0.0015 |
| (e) bagged versions (25 event bootstraps) | e0, e1 multipliers; e2 cut-offs | -0.0022 / -0.0024 / -0.0004 | -0.0004 / -0.0005 / -0.0001 |
| (f) one class per label-free event | f1 unify types; f2 plurality incl. legit; f3 half the rows fraud; f4 event-pooled probabilities; f6 | +0.0008 to +0.0009 [+0.0001, +0.0019] | +0.0002 to +0.0003 / +0.0005 to +0.0006 |
| (k) fraud scale 0.70 / 0.80 / 0.90 / 1.00 | | -0.0003 / -0.0000 / -0.0000 / -0.0002 | within +-0.0001 |

Lessons per idea:

- **(a) Prior shift:** the per-class ratios are real (cleaning adds rows unevenly), but they
  multiply probabilities that are either near 0.03 or near 1, so almost no call changes (1,670-1,683
  fraud calls per 100,000 rows against 1,681).
- **(b) Census anchoring:** loses. Anchoring to cleaned label counts asks for the ordinary rows (63%
  of each class) too and fills them with legit rows; holding the baseline's own calls per census
  row loses because a month's census moves for reasons the calls should not follow (thinning,
  event-size drift). Census is a diagnostic, never a count guard.
- **(c) F/2 admission:** with each class's F a few thousandths, the fixed point already admits a row
  when its probability exceeds F/2: the same cut up to conflict handling.
- **(d) Tuned cut-offs:** fits on cleaned and given labels agree within 0.0007 and both lose to the
  label-free fixed point. Cut-offs on the fixed point's scores lose -0.018 on one thinning because
  one seed fits a cut-off that does not transfer (-0.051 on that seed, -0.002 on the others).
- **(e) Bagging the multipliers** steadies the fits but does not make them better.
- **(f) One class per event** is the only positive family. Before the rule the event vote changes
  50-60 rows; after it, four events (28 rows) on one thinning and none on the other are still split:
  the template rule already does this job. All variants give the same calls to within a row.
- **Verdict:** keep the champion decoder. No count anchoring, no fitted or bagged multipliers, no
  prior shift. A decoder cannot reach the ordinary rows (P(fraud) 0.01-0.05, 2.4% fraud, no
  ranking inside them) or the look-alike types.

## Prior-shift (label-shift) methods [earlier]

Simulated October on each held-out train month by removing or duplicating whole thinning units
(events or non-event fraud rows) with scenario ratios: S1 (bot 0.53, collusion 0.53, mule 1.0), S2
(October's census ratios: mule 1.171, bot 0.564, chain 0.902, collusion 0.626, refund 1.065, ...), S3
(a 4x stress: bot 0.25, collusion 0.25, mule 1.5, chain 0.7). Deltas are cleaned macro-F1 in
units of 1e-4; every method ends in the champion fixed point and rule.

| method | unshifted | S1 | S2 | S3 |
|---|---|---|---|---|
| EM, source prior = out-of-fold column mean | -235 [-240, -229] | -235 | -231 | -234 |
| EM, source prior = label frequencies (textbook) | -2,731 | -2,908 | -2,846 | -3,031 |
| EM halfway / prior-only | -5.9 / -4.1 | -5.7 / -3.5 | -6.6 / -4.4 | +0.2 / +2.8 |
| BBSE hard | -1.0 | -7.5 | -2.3 | -106 |
| BBSE soft | -10.5 | -65 | -60 | -199 |
| census-based | -0.5 | -0.8 | -0.8 | +4.5 [+1.3, +7.6] |
| census, half-strength | +0.5 [-0.3, +1.5] | +0.7 | +0.4 | +4.2 [+1.9, +6.7] |
| BCTS (calibration only / with EM) | -4.9 / -258 | -10.2 / -251 | -8.4 / -259 | -28 / -194 |
| oracle (true ratio; ceiling for any estimator) | n/a | +0.3 to +0.6 | -0.3 to +0.3 | +6.7 to +7.5 |

- **Even a perfect prior gains almost nothing:** +0.3 to +0.6 x1e-4 on October-like shifts, about
  100 times below the 0.005 tie line; it needs a 4x shift to reach +7. The label-free estimators
  all lose or tie on unshifted months (EM -4 to -235, BBSE -1 to -10, BCTS -5 to -258). The one that
  never loses, half-strength census, changes **zero** October calls.
- **Why:** the fixed point already re-estimates the prior from each month's own column means
  (October's bot column mean is 0.69x the train out-of-fold mean, collusion 0.81x). Classes that shift
  in October (bot pool and bursts, collusion rings, mule bursts) sit on near-certain rows;
  reweighting by about 0.5 flips only rows within 2x of a boundary: the oracle changes 2-22 of 1.56M
  rows. The one shifted look-alike near a boundary, mule/bot in mf2+3, is overridden by the template
  rule anyway.
- **Failure mechanisms:** EM amplifies the mismatch between the cascade's column means and label
  frequencies (per 100k rows: mule 360 vs 656, qr_swap 821 vs 464, refund 427 vs 211, phishing 638
  vs 428), because the type model never sees low-score rows; the textbook version is catastrophic
  (+249k fraud calls). BBSE's 12x12 inversion is noisy for small classes (sd 22 on S1).
- **EM does recover October's prior shift** (bot 0.50, collusion 0.52, mule 1.22) matching the
  event-volume audit, so it is a fine report-only diagnostic. Never feed it to the decoder.
- Revisit only if a future month shifts by >= 4x (then use half-strength census).

## Proportional split of inseparable pools [earlier]

A theory-driven idea: for a pool of two indistinguishable classes that is the classes' whole
support, a deterministic proportional split beats assigning the whole pool to one class under
macro-F1. Tested by random event-level splits (30 draws, fractions 0.1 to 0.5 and the class share)
against the template rule: it **loses on every pair at every fraction**, mf1 -0.002 to -0.012, drain1
-0.001 to -0.008, mf2/3 -0.002 to -0.010, col7 -0.0001 to -0.001, col1 -0.002 to -0.016, rng1
-0.0005 to -0.008, because most of each class lies outside the pool.

## Lessons

- Keep the label-free fixed point at scale 0.875 and the anchored template rule; stop tuning
  thresholds (honest gains -0.0005 to +0.0004).
- Do not fit anything on given labels inside the decoder: they miss unreported generator fraud and
  contain noise that cleaned-label fits do not repair.
- Open decoder ideas (event-level expected-test-utility decoding, a disagreement-footprint screen)
  are in [open-ideas.md](open-ideas.md); expected gains <= +0.0003.
