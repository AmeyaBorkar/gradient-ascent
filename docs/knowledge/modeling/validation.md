# Validation

**Summary.** Folds are whole months with events kept whole (leave-one-month-out: July, August,
September), scored on three targets (given, cleaned, generator-only). Because no full-density fold
looks like October, a thinned "October-like" September is the rehearsal of the shift. Differences
are tested with a paired bootstrap over events. Selection among many options is nested. The rule for
deciding an upload is in [upload-rule.md](upload-rule.md); noise in the public score is in
[public-noise.md](public-noise.md).

## 1. Folds
- **Canonical:** leave one month out, three folds (train on two months, score the third); headline
  is the mean of the three months. Events stay whole: a row belongs to the month in which its event
  starts. 0.4-2.2% of multi-row events cross a month boundary (earlier measurement).
- **Drift check:** forward fold July+August to September.
- **Stopping:** fixed round counts, never validation log-loss.
- **Why months, not random folds** (earlier measurement): forward and leave-one-month-out folds rank
  six variants alike (Spearman 0.98, minimum 0.94) with means within 0.003-0.006; fold-to-fold sd of
  one variant 0.004-0.005 (given), 0.005-0.008 (cleaned); seed sd 0.0012 (given), 0.0025
  (generator-only), 0.0004-0.0005 for the full evidence model. A random third of rows scores 0.4661
  / 0.8576 (given / cleaned) against 0.4639 / 0.8441 for the month split, so random K-fold adds
  +0.006 / +0.015 (collusion +0.075, bot +0.038); a third of whole events scores 0.4602 / 0.8430.
- Each leave-one-month-out run records its fold fingerprint so runs can be compared and stacked.
- Measured in this repo: the three monthly scores of the champion are 0.5130 / 0.5090 / 0.5146
  given and 0.5645 / 0.5639 / 0.5712 cleaned (`python -m ringtrace.build recipes/champion.json`).
  Re-running from a clean checkout with an empty cache reproduced the same probabilities and
  submission hash.

## 2. Three targets
Every run is scored on: **given** labels (the headline), **cleaned** labels (the teacher's silver
labels) and **generator-only** (cleaned with ordinary-looking fraud set to legit). Definitions in
[training-labels.md](training-labels.md). A change counts only if it does not lose on any of the
three; the generator-only target is where genuine model gains show first (about 1.7x the given-label
gain). Per-class F1 of the champion pooled over the three months (measured in this repo):

| class | given | cleaned | generator-only |
|---|---|---|---|
| legit | 0.987 | 0.988 | 1.000 |
| mule_account | 0.433 | 0.486 | 0.882 |
| velocity_bot | 0.519 | 0.583 | 0.935 |
| layering_chain | 0.551 | 0.611 | 0.925 |
| sim_swap | 0.499 | 0.553 | 0.868 |
| account_takeover | 0.434 | 0.485 | 0.836 |
| phishing_collect | 0.488 | 0.551 | 0.876 |
| qr_swap | 0.511 | 0.576 | 0.943 |
| refund_abuse | 0.532 | 0.598 | 0.952 |
| fake_cashback | 0.396 | 0.446 | 0.755 |
| loan_app_extortion | 0.414 | 0.474 | 0.905 |
| merchant_collusion | 0.393 | 0.461 | 0.941 |

August is the weakest month because its label mix has more ordinary rows (bot 0.592 against
0.548 / 0.571 ordinary share), not because the model is worse: the gap to the per-month ceiling is
flat (given 0.0751 / 0.0754 / 0.0764; cleaned 0.0441 / 0.0446 / 0.0444) (earlier measurement).

## 3. The October-like month
October differs from every train month: the legit background is about 0.494 as dense (fraud
events whole), legit amounts are about x1.166, and the QR check is blank on about 44% of QR
payments to a person (never blank in train). The October-like month rehearses this.

**Construction** (September thinned; July and August stay as they are):
- **Thinning:** legit-labelled rows are thinned to about 0.494 density; fraud events are thinned
  whole at the legit rate so the fraud share stays near the train share (4.06% test against 4.10%
  train). An early transform that protected every generator row gave about 7% fraud and inflated
  gains; it was discarded.
- **Protected units:** two modes. The *default* mode protects "label units" only and thins merchant
  ring bursts row by row (nothing links their rows). The *families* mode protects label-free family
  events and label units together, so no ring burst is cut and the test rows are appended back.
- **Label-units-only bias:** the ring family types an event by its size (1-3 qr_swap, 4-8
  refund_abuse, 9+ collusion). In the default mode ring shapes are distorted (per 100,000 rows, size
  1 events 22.2 against 8.6 on raw September and 10.0 in real October; size 3: 7.9 against 11.8 and
  13.4). A model trained on such copies learns to re-type cut rings (+0.0065 gain, all in collusion,
  loan, refund), loses on leave-one-month-out (-0.0008 to -0.0013 cleaned, -0.003 to -0.004
  generator-only) and a judge built the same way rewards it. **Any comparison touching ring or
  repeat classes must use the families mode.** A tuning search on the default mode picks up the same
  artefact: top settings gained +0.0010 on fresh default thinnings and +0.0002 in the families mode.
  (earlier measurement)
- **Counter from the unthinned frame:** the counter's block finder sees no legit block once
  September is thinned, so refitting it on the thinned frame is a construction artefact. Earlier
  versions did this and showed a spurious -0.03; with the counter's columns computed on the unthinned
  frame and joined by stacked row, the shift costs about -0.0006 (earlier measurement), and 517
  fraud-block rows remain, not 7,745. In this repo every family is rebuilt on the frame's own ledger
  (frame train + original test), except the counter, whose columns come from the unthinned build,
  exported once and joined by stacked row; devices in the counter's legit block are renumbered, and
  the QR check is blanked on about 150 QR payments to a person. (decision 13, measured in this repo)
- **Fit and score:** fit on July+August of the frame; the screening score is cross-fitted between
  those two months; predict September; decoded with the fixed point (the template rule needs the
  counter's event view, which the thinned frame lacks, so it is off in this mode).
- **Seeds:** three thinning seeds x three cascade-seed sets (cascade seeds 3r..3r+2 for set r).

**Result in this repo** (`python -m ringtrace.octlike run ... champion:0,1,2 champion-cleantrain:0,1,2`;
September given-label macro-F1, fixed point):

| thinning seed | seed sets r0 / r1 / r2 | mean |
|---|---|---|
| 13 | 0.5060 / 0.5061 / 0.5058 | 0.5060 |
| 14 | 0.5094 / 0.5095 / 0.5098 | 0.5096 |
| 15 | 0.5126 / 0.5125 / 0.5124 | 0.5125 |

Mean over the nine cells 0.5093 against 0.5097 at full density in the month folds; cascade-seed
sets agree within +-0.0003, thinning seeds differ by about 0.0033-0.0065. Training on cleaned labels
rebuilt per frame: nine-cell mean 0.5091, a tie. One cell takes 4.5-5 minutes on a GPU with three
frames sharing it.

**What thinning costs** (earlier measurement): cutting off the future alone -0.001; half density
-0.0066 given / -0.0123 generator-only for a relational-only plain model (-0.0060 / -0.0077 for the
cascade), mostly cashback (-0.04 to -0.05), loan (-0.02 to -0.08), refund (-0.01 to -0.03); with
the evidence families and the counter, the champion loses nothing measurable (0.5093 against
0.5097, measured in this repo). Thinning the legit background of the training months to match is
worse (-0.007). The October-like month scores higher in absolute terms than raw September (0.59
against 0.48 for a relational probe) because thinning legit raises precision: compare arms within a
row, never across modes.

## 4. The event bootstrap
- **Unit:** the first family event of a row; rows outside events are singletons. Rows inside an
  event are correlated, so row bootstraps give intervals that are too narrow.
- **Method:** a paired Poisson(1) weight per unit, shared by both arms; 2,000 draws; macro-F1 per
  month averaged like the headline; one sparse product per month yields all draws. Both runs'
  saved probabilities are re-decoded (fixed point plus rules) instead of trusting stored calls.
  Intervals are 90% for routine comparison and 95% for the upload rule.
- **Command (this repo):** `python -m ringtrace.compare`. Earlier results: counter + rule over
  families alone +0.0333 [+0.0310, +0.0358] given; seven families over context only +0.0118
  [+0.0094, +0.0142] (both measured in this repo).
- **Reading:** the whole interval above zero is a gain; an interval spanning zero is a tie and the
  simpler option wins. Seed noise alone moves the score 0.002 (given) / 0.005 (generator-only) on a
  plain model but about 0.0004 on the full evidence model.

## 5. Nested selection
Choosing among many options on the same months inflates the winner. To check:
- **Declare, then run:** ranges, number of trials K, metric and selection rule fixed before running;
  a new space is a new sweep. Count every run, failures included.
- **Nested rule:** for each held-out month, pick the option that maximises the mean score on the
  other two months, then score the held-out month. Report the held-out scores.
- **Worked example** (earlier measurement): 176 pairs of tree settings x type-row cut; type-row
  cut 0.04 was picked in every fold; held-out gains +2.9 / +2.1 / +3.8 (given) and +3.4 / +1.7 /
  +2.5 (cleaned), in units of 1e-4; adding adaptive rounds did not inflate it (+2.92 / +2.79 before,
  +2.95 / +2.55 after). The partner setting changed from fold to fold and was not adopted.
- **Stacks:** several pieces each chosen because they looked positive on the same month inflate the
  pooled gain; each piece must still help when left out, on months not used to choose it.
- **Selection of model combinations:** per-class model choice gave +0.0005 / +0.0001 / -0.0001
  held out (in-sample +0.0003 to +0.0009), and the picks changed fold to fold: not real.
- **Ensembles:** fit blend or stack weights on k-1 folds, score on the held-out one; prefer means
  and rank means to a large meta-model.
- **Winner's tuned decoders:** half of a tuned gain is optimism (see [decoding.md](decoding.md)).

## 6. Leakage checklist (every new feature family)
- Target-aware transforms are fitted inside the training fold only.
- No feature uses information from after prediction time; no feature spans the train/validation
  boundary of a group or event.
- Duplicates between train and test are understood, not exploited blindly.
- A suspiciously large gain is a bug until it survives a time- or event-shifted split.
- Month-identifying columns are dropped or capped ([features.md](features.md)).
- Pin the feature cache per run: a counter rebuild between two runs once changed 3 October rows by up
  to 0.81 in probability (earlier measurement); in this repo the feature cache is keyed by the
  module's source hash and jobs run from a frozen copy of the code (decisions 3, 10).

## 7. Label-free audits before any upload (earlier measurement)
- Calls per class against other candidates and against the prior table; a class that only one model
  loses is drift.
- A census of rows by score and event size (score >= 0.5 on 1,368 rows per 100,000 in October
  against 1,517 in September).
- A classifier separating September calls from October calls (AUC 0.84-0.89 for six classes, 0.94-1.00
  for mule, bot, refund, loan, collusion; top features: raw `*_dt_next` gaps, device span, entity
  totals).
- October class mix, calls per 100,000 (model on all three months): mule 277, bot 103, chain 209,
  sim 108, ato 107, phishing 197, qr 189, refund 100, cashback 55, loan 35, collusion 28; total
  1,407 against 1,504 in September. October / September ratios: mule 1.18, bot 0.50, chain 0.92,
  sim 1.09, ato 1.04, phishing 1.01, qr 0.97, refund 0.87, cashback 1.08, loan 0.88, collusion
  0.57. Classify-and-count is within 8-9% per class on rehearsal; EM on raw probabilities is not.
