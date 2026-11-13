# Getting more out of the data and the test month

Status convention. "Measured in this repository" means this repo's own runs (see its results
table). "Earlier measurement" means a previous research pass measured it and a later pass may
re-derive it here. Almost everything below is an earlier measurement; the repo-measured anchors are
listed first.

Summary. Nothing in this family moves macro-F1 by more than about +0.001 once it is judged on an
October-shaped September that keeps ring events whole and holds the device-counter columns at their
unthinned values. More training months, refitting on all months, self-training, pseudo-labels,
domain reweighting, prior-shift correction, October-shaped training copies and feature transforms
against drift all tie. The single apparent gain (October-shaped copies, +0.0065) was an artefact of
how the October-shaped month was built. The drift audits found October to differ from July-September
in density, a legit-only amount scale and a few split-length artefacts, none of which costs more than
about 0.001-0.003. The honest conclusion is that the remaining loss is the information limit inside
look-alike pools and ordinary rows (see [ordinary-rows.md](ordinary-rows.md) and
[lookalikes.md](lookalikes.md)), not a shortage of data or an unmodelled shift.

## Anchors measured in this repository
- Champion, leave-one-month-out (LOMO), given labels: 0.5122 (Jul 0.5130, Aug 0.5090, Sep 0.5146);
  cleaned labels 0.5665.
- October-like September on three thinned frames, three seed sets each (9 cells): champion mean
  0.5093 against 0.5097 for September at full density; seed sets within +-0.0003. Cleaned-label
  training variant 0.5091. So the construction, as built here, costs about -0.0004: a tie.
- Fraud and base rates for the stand-in: the test month keeps the train fraud share (4.06% against
  4.10%), i.e. October thins whole fraud events at the legit rate.

## Overview

| experiment | verdict | effect |
|---|---|---|
| Learning curve: one training month vs two | helps, barely | +0.0006 mean LOMO (every month positive); one LightGBM stand-in point: tie |
| Refit on all months vs mean of fold models | tie | -0.0010 [-0.0018, +0.0004] (analogue), +0.0001 (2x rounds); 0.01% of test calls differ |
| Three-month refit for the test month | tie / never fully run | about 0 |
| Guarded self-training (2 rounds, 0.9) | tie | +0.0011 LOMO, +0.0007 October-like (interval spans 0) |
| Self-training, 1 round at 0.8 | loses | -0.0052 October-like |
| Event pseudo-labels, confident-call pseudo-labels | tie | +0.0010 then +0.0000 on a second draw; +0.0002 |
| Domain reweighting toward October | tie | -0.0001 |
| Small positives stacked (bundle) | tie | +0.0003 cleaned, -0.0004 generator-only LOMO |
| Prior-shift correction of the decode (EM, BBSE, BCTS, census) | tie or loses | oracle ceiling +0.00003 to +0.00006; EM about -0.02 |
| October-shaped training copies, first stand-in | looked like helps | +0.004 to +0.009 (inflated prior) |
| October-shaped copies on events-thinned-at-legit-rate frames | tie | +0.0014 / +0.0016 (sd about 0.005) |
| Same copies with the real counter, default thinning mode | artefact | +0.0065, explained by cut ring events |
| October-like September construction | pitfall catalogue | -0.11 to -0.0006 depending on the build |
| Drift audits (four of them) | nothing to fix | -0.0009 to -0.003 expected cost |
| Drift transforms (drop / rank / scale amounts, drop counter columns) | tie | -0.0016 to +0.0005 |
| External data and public work | not usable / tie | no match found; pretrained stacker +0.0011 (tie) |

## 1. Learning curve: does more training help?
- Idea. If the model were data-starved, one training month versus two would show a steep gain, and
  Jul+Aug+Sep for the test month would be worth a refit.
- Measurement (earlier). A one-seed LightGBM stand-in, September held out: July alone 0.5090 given /
  0.5637 cleaned against July+August 0.5088 / 0.5635; the August-only point was not run and the
  masks came from a score that had seen September, so this is weak. A later cleaner run on the
  XGBoost cascade (filters off in every arm, three seeds): one month against two gains +0.0003 to
  +0.0009 on the held-out month (July 0.5641 to 0.5644, August 0.5629 to 0.5638, cleaned), mean
  +0.0006, positive in every month.
- Verdict. Helps by a hair; saturated. The curve is flat because the loss is information in
  look-alike pools, not sample size.
- Lesson. Do not budget effort for "more months". The test month's model is already at the plateau.

## 2. Refit versus the mean of fold models
- Idea. Predict the test month either as the mean of the leave-one-month-out fold models (test
  posteriors in the same regime as the out-of-fold ones the decoder was solved on) or with one refit
  on all months at scaled rounds (sharper posteriors, more data). The fold mean was chosen up front
  for regime consistency and the refit kept as a challenger.
- Measurement (earlier). October-like analogue, event-cleaned labels: mean of single-month models
  against one two-month model -0.0010 [-0.0018, +0.0004] (an exaggeration, since it averages weak
  one-month models); refit of July+August at twice the rounds +0.0001 [-0.0008, +0.0002]. On the
  real test rows the full refit (3 seeds, 450 rounds) and the fold mean disagree on 0.01% of calls,
  about 26 of 260,582 rows. Two disjoint 3-seed bags differ on 0.005% (about 13 rows); the 10-seed
  bag differs from seeds 0-2 on 0.004%. A refit October probability file was built for blending; its
  result was never recorded.
- Verdict. Tie. Fold mean stays.
- Lesson. The choice cannot change the submission by more than a few dozen rows, so it is not a lever.

## 3. Three-month refit for test
- Idea. Train on all of July-September for October, since a shorter lag between training data and
  test time is a known adaptation.
- Measurement (earlier). Same one-seed stand-in as the learning curve: about 0 (one month of training
  equals two within 0.0002).
- Verdict. Tie. Listed among the work that "cannot pay" in the gap analysis.
- Lesson. Held-out months are for choosing the pipeline; the shipped predictions can stay fold-mean.

## 4. Self-training and pseudo-labelling
- Idea. Add the model's confident October calls (or whole October events found without labels) to
  training. Literature: gains of +0.1 to +3.3 points in-distribution; under shift, pseudo-labels and
  entropy methods fell below the unadapted model unless the class mix was capped. Rare-class
  pseudo-labels are precise but low-recall.
- Measurement (earlier), sweep of 37 runs. Baseline seed sd on the October-like month 0.0036 for the
  sweep harness (0.0004 for the later three-seed protocol).
  - Guarded, two rounds, events with mean P >= 0.9: +0.0011 [+0.0002, +0.0020] LOMO cleaned;
    +0.0007 [-0.0025, +0.0040] October-like. Not confirmed there.
  - One round at 0.8: +0.0012 LOMO but -0.0052 [-0.0088, -0.0017] October-like: a lower threshold hurts transfer.
  - Event pseudo-labels (1,333 of the goal month's events labelled, 152 by counter zone and 1,181 by
    template, 199 left unlabelled; 87-89% equal the cleaned label, 97.6% are fraud), weight 0.3 or 1:
    +0.0010 / +0.0011 on one October-like draw, +0.0000 on a second draw (ensemble -0.0006
    [-0.0014, +0.0002]). The gains came from five event-shaped classes at +0.002 to +0.006 each.
  - Confident calls (top half of each class, about 127k rows, 97.7% correct): +0.0002. Training on
    the model's own surest calls adds nothing it does not already predict.
  - Event pseudo-labels on top of copies of the train months: +0.0024 against +0.0065 for copies alone,
    because pinning ring rows to the template's class undid the copy-trained model's better call.
  - Self-training as one piece of a stacked bundle (with pools, a 5-seed bag, one type per event):
    -0.0000 cleaned alone.
- Verdict. Tie.
- Lesson. The model already calls the events that pseudo-labels would teach. Use a gate that rejects
  steps whose gain is inside the seed and bootstrap noise, and never accept a lower confidence
  threshold on the strength of LOMO alone.

## 5. Domain reweighting
- Idea. Weight train rows toward October by the odds of a goal-versus-source classifier.
- Measurement (earlier). Classifier AUC 0.65 on the October-like month and 0.53 on a LOMO month, so
  LOMO weights stay near 1 (p1-p99 0.77-1.22); October-like weights spread 0.17 to 2.1 (effective
  share 0.80). Effect -0.0001 cleaned, -0.0015 generator-only.
- Verdict. Tie. Literature agreed: importance weights stop mattering once the model separates the data.
- Lesson. Resample the source (thin it) rather than reweight it; or do neither.

## 6. Small positives stacked
- Idea. Several wave-2 ties (TabPFN stacker +0.0011, one fraud type per event +0.0009, guarded
  self-training +0.0011, pool features +0.0002, 5-seed bag) might add up.
- Measurement (earlier). Gate-mode October-like (3 of 9 cells), bundle without the pretrained piece:
  +0.0003 cleaned [-0.0002, +0.0007], +0.0011 generator-only; LOMO +0.0005 given/cleaned. Full
  bundle LOMO generator-only -0.0004: the pretrained stacker turns the generator-only gain negative.
- Verdict. Tie; the pieces do not add. See [ensembles.md](ensembles.md) for blend results.
- Lesson. Stacking selected noise inflates the pooled gain; require each piece to still help when left out.

## 7. Test-month adaptation of the decode (prior shift)
- Idea. October has fewer bot bursts and collusion rings (roughly half the train event rate) and
  more mule bursts; re-estimate class priors on the test month (EM, black-box shift estimation,
  calibrated EM, an event census) and re-solve the decoder.
- Measurement (earlier), simulated October by removing or duplicating whole events on each held-out month:
  - Oracle given the true shift: +0.00003 to +0.00006 on October-like shifts (about 100x below the
    tie line); a 4x stress shift needs to be reached for +0.0007.
  - EM with the classifier's own column means: about -0.021 to -0.024 unshifted and shifted;
    textbook EM with label frequencies is catastrophic. EM half-step -0.0005; hard black-box
    estimation -0.0001 unshifted and up to -0.011 under the 4x stress; calibrated EM alone
    -0.0005 and, followed by EM, -0.026.
  - Census-half (the only method that never lost): +0.00005, changing zero October calls.
  - The fixed-point decoder already re-estimates the prior per month; shifted classes sit on
    near-certain rows, so re-weighting flips 2-22 of 1.56M rows even with an oracle.
- Verdict. Loses or ties. See [decoders.md](decoders.md).
- Lesson. EM does recover the event-volume shift label-free (bot ratio 0.50, collusion 0.52,
  mule 1.22), so keep it as a monitoring line, never in the decode.

## 8. October-shaped training copies
- Idea. October differs from train in density (legit rows 0.494 as dense, events whole), legit
  amounts x1.166 and a blank QR check on about 44% of QR-to-person payments. Train on thinned,
  density-matched copies of the train months so count features see October-like values.
- Measurement (earlier), first stand-in, forward fold Jul+Aug -> Sep, two seeds, XGBoost:

  | scored on | raw | October-shaped | both |
  |---|---|---|---|
  | October-shaped Sep, given, argmax | 0.5837 | 0.5909 | 0.5880 |
  | same, fixed point | 0.5898 | 0.5937 | 0.5922 |
  | same, cleaned, fixed point | 0.6550 | 0.6600 | 0.6585 |
  | raw Sep, given, fixed point | 0.4789 | 0.3463 | 0.4762 |

  October-shaped-only collapses by about -0.13 on raw September; the mix costs -0.003. The raw
  model's seed sd on the shifted month is 0.007-0.010, so two seeds gave "about one sd". Caveat that
  came later: the stand-in protected every generator row from thinning, so its months carried about
  7% fraud against test's 4.06%; training and scoring at the inflated prior explains part of the gain.
- Re-measurement on properly thinned frames (earlier): copies made with events thinned at the legit
  rate gave +0.0014 (one copy) and +0.0016 (two) on the October-like month (sd about 0.005), and
  -0.0010 / -0.0007 LOMO cleaned: a tie. A later three-seed rerun with the real device counter and
  the default thinning mode found +0.0065 on two independent October-like draws (all six seeds within
  0.0004; two copies +0.0066, seed ensemble +0.0069 [+0.0021, +0.0122]), all in merchant_collusion
  (0.424 to 0.473), loan_app_extortion (0.447 to 0.464) and refund_abuse (0.538 to 0.548), with LOMO
  cost -0.0008 / -0.0013 cleaned and -0.0028 / -0.0040 generator-only.
- Why +0.0065 is not real. The default mode thins merchant ring bursts row by row, so ring events
  shrink into smaller shapes: size-1 ring events 22.2 per 100k rows against 8.6 on raw September and
  10.0 on real October; size 9+ 3.97 against 2.94 and 1.15. Copies built in that mode teach the
  model to re-type cut rings, and a judge built the same way rewards it. Real October keeps rings
  whole, so the copies lose there. They were not sent to the upload gate. The protected mode
  (family events and label units protected together) was not used to re-run the copies, so this is
  closed by argument, not by a clean rerun. Not re-measured in this repository.
- Verdict. Tie (honest frames); artefact (default mode).
- Lesson. A gain that appears only on a judge built from the same transform as the training copies is
  the transform, not the data. Check event-size mixes on the thinned frame against real October before believing it.

## 9. The October-like September itself, and its pitfalls
- Idea. Stand in for October by thinning September (legit rows to about 0.494, fraud events whole),
  fit on July+August, score September; use it as the transfer check for every candidate.
- Pitfalls (earlier measurements, gap analysis):
  - Device counter refit on the thinned frame: thinning empties the train legit block, so the refit
    called all 7,336 September legit-block rows a fraud block (7,745 rows in the fraud block against
    517 under the real counter); bot calls rose from 374 to 7,515 and takeover F1 fell from 0.475 to
    0.089. Reading with label units and the counter refit: -0.1106 given / -0.1211 cleaned.
  - Label-unit thinning cuts merchant rings and refund/lending repeat events: with the counter held,
    -0.0240 / -0.0275 (collusion F1 -0.159, loan -0.065, refund -0.056).
  - Family-protected units with the counter's block and segment columns held at unthinned values:
    -0.0006 / -0.0006 (a tie), every class within +-0.003. Joining the real counter alone recovered
    +0.0092 on the champion at that time; the refit counter on thinned anchors still costs about
    -0.0014 (October's own anchors are thinner too).
  - Amount scale x1.166 adds -0.0005 / -0.0015; density-window features cost at most 0.0003 per group once the event families are in.
  - Thinning must skip legit rows next to events (8,829 such rows, 13.3% of event rows), or events shrink.
  - Single-seed noise is large: four baseline seeds on the sweep harness gave sd 0.0036; the three-seed
    protocol gives sd 0.0004 and the "gain worth noticing" bar moves accordingly.
- Measured in this repository: on three thinned frames the champion scores 0.5093 against 0.5097 at full
  density, consistent with the -0.0006 earlier reading.
- Lesson. Build the proxy from label-free events, hold the counter columns, check ring-size mixes,
  and always average seeds before ranking changes. See also [stress-test.md](stress-test.md).

## 10. Drift audits
All four used aggregates of the test month only.
- Broad October drift (earlier). Density 0.494 (8,400 rows a day against 17,000, events keep their
  size), legit amounts x1.166 (fraud amounts not scaled; only QR-to-person rows move, 467/479/435 to
  550), two columns that turned dead or reversed (a version column and a risk-score column), the
  QR check null on 45% of QR-to-person payments (never in train), bot bursts 7 against 13.5
  expected, chain-hop delay p75 24 to 80 minutes. About 35 of 234 features depended on the frame
  position and were dropped. Training on thinned months returned nothing; removing the frame-dependent
  features returned the whole 0.002 shift cost (0.0016 of it is the frame end alone).
- Generalisation audit (earlier). Adversarial AUC September v October 0.729 against 0.512-0.534
  between train months (evidence rows 0.893 against 0.825-0.833). Nearly half the separation is a
  device-counter column that counts "device seen once in its split": 36.8% of October rows against
  8.1% in train (a split-length artefact, 0.02% of the champion's gain). The 17 flagged features hold
  2.9% of the champion's gain. No event-family shape, size, span, delay or ratio drifts: delay medians x0.73-1.12, tails
  up to x1.6 well inside windows, exact constants intact (hop ratios identical, request median
  4,990, constant-keyed event rates x1.00-1.08). Widening any fragile tolerance finds zero extra
  October rows (0.4-2.5 per 100k in train). Expected generator-drift cost -0.003 (amounts), about 0
  for delays and bursts. Stress sizes: delays x3 -0.046 cleaned, slower delays x1.5 -0.0145, jitter
  of amounts with exact constants intact -0.003, breaking the exact constants -0.17 to -0.20.
- Drift audit wave 3 (earlier). October-like vs October adversarial AUC 0.724 (0.550 without the
  counter column); the 17 drifters hold 4.1% of gain and 11.7% of October |SHAP|. A frame that also
  has the festival amount scale and the one-month counter split (AUC 0.540) costs the champion
  -0.0009 cleaned (-0.0011 generator-only): the protocol is not blind to anything that matters.
- October checks on the final candidate (earlier). Template groups' keep-shares lie inside the
  July-September ranges; fraud-block phones are 100% called fraud in every month; per-class call rates sit
  inside forecast ranges except merchant collusion (31.5 against 37-40, about 15 rows, one ring);
  ring events are intact (size-1/2/3 at 10.0 / 10.4 / 13.4 per 100k against 8.6-13.3 / 8.6-10.3 / 9.5-11.8).
  A 495-check parameter audit found October out of the train range on 22% of checks against 17-24%
  for each train month against the other two: no family-wide drift. The one real structural change is
  two misplaced counter segment cuts (fraud-block call = segment on 95.2% of rows against 98.6-99.0%).
- Verdict. Nothing to fix; two zero-cost insurance widenings and one hygiene drop were recorded.
- Lesson. Run a label-free pre-upload audit (calls per 100k by class, splits inside look-alike
  templates, fraud-block rows called fraud, share of calls without generator evidence) on every
  candidate. See [counter.md](counter.md).

## 11. Drift transforms
- Idea. Drop, rank-transform, or month-scale the 17 drifting columns (mostly amount scale and sender
  windows) so the model sees them stably.
- Measurement (earlier), October-like cleaned, 3-seed mean (bar +0.007): drop the counter column
  -0.0004; drop its position partner -0.0003; drop both +0.0002; drop `ratio` -0.0008; drop the 16
  amount columns -0.0012; rank them +0.0005; scale them +0.0000; drop all 17 -0.0011. Rank and scale put each
  column at 1-D AUC 0.500 but make the month easier to tell (adversarial AUC to 1.00), because the
  transformed amount disagrees with exact-amount flags and with fraud amounts October does not scale.
  Gate check of the position drop: +0.00012 pooled [-0.00004, +0.00029], positive on every thinning
  seed, but fails the lower bound; it changes 7 takeover calls to sim_swap on October.
- Verdict. Tie; dropping is hygiene at no measured cost.
- Lesson. Do not normalise amounts by month: the festival scales legit amounts only.

## 12. External data and public work
- Source of the generator. No public generator with these class names was found; the nearest public
  synthetic-payment and money-laundering simulators use other mechanisms. A separate analytics
  datathon with a similar theme exists; it is a different dataset (not opened). Verdict: do not
  hunt further.
- External training data (any outside payment or fraud dataset): not usable. The rules say to train
  only on the provided data; papers may be read for typology ideas.
- Pretrained weights are allowed with a recorded source and licence. A pretrained tabular model as a
  stacker over candidate rows: +0.0011 October-like, +0.0003 LOMO; as a look-alike picker: -0.0002
  (fitted threshold), -0.0108 (fixed 0.5); in-shape AUCs 0.46-0.68 reproduce the capacity result.
- Provided hints worth recording: use the test month itself to build graph and aggregate features
  (allowed and expected), train labels are noisy but test labels are clean, and check every feature in
  October. The organisers do not say whether the 30/70 split is by row or by event.
- Lesson. All usable outside help is already in the problem statement's hints. See
  [labels.md](labels.md) for the noise side and [open-ideas.md](open-ideas.md) for what is left.
