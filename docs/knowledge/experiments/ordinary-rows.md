# Ordinary rows: every attempt to separate them from legit

Status vocabulary used below. **measured in this repo** means this repository's own runs show it
(see `RESULTS.md`). **earlier measurement** means an earlier research pass measured it; a later pass may re-derive it here. **literature / belief** means reasoning or published claims, not a
run of ours.

## Summary

"Ordinary rows" are fraud-labelled rows that look exactly like legitimate traffic. They are about
63% of all fraud labels (about 35-36k of 1.54M train rows, about 2.3% of rows) and they carry the
same share of every fraud class (58-70%). Every angle tried to separate them from legit rows came
out at chance (AUC 0.49-0.52). They are not label noise: each fraud type's generator emits them,
and the October test month contains them at about the train rate, so the clean test labels very
probably count them as fraud. They set the score ceiling (about 0.58-0.61 macro-F1 for a perfect
detector of everything else), and about 90% of the gap between the champion and a perfect score
sits on them. The only handle is the roughly 2.8% of them that sit on a freshly numbered phone.
Nothing about them is worth more compute; see [open-ideas.md](open-ideas.md) for what remains.

Only these facts are **measured in this repo**: 62.9% of fraud labels judged ordinary-looking by
the silver-label builder; champion leave-one-month-out macro-F1 0.5122 on the given labels, 0.5665
on the cleaned labels, 0.9010 on the generator-only target (cleaned labels with ordinary-looking
fraud set to legit). Everything else in this file is an earlier measurement or belief.

## Overview

| experiment | verdict | effect |
|---|---|---|
| Size and nature: mixture on two train-only markers | design, not noise | 63.0% of fraud labels (36,318 +- 175) |
| Process and link checks (60 properties, pairs, entity reuse) | identical to legit | largest abs z 2.9 of 120 tests |
| Row-level classifier, July-August to September, 234 features | ties chance | AUC 0.499 / 0.501 |
| Clean single-use-device sample, 93 features | ties chance | AUC 0.491 (0.473-0.506) |
| 102 value-level fingerprints | ties chance | AUC 0.508 clean sample; 0.499 own-phone |
| Own-phone ordinary rows (32,037 positives) | ties chance | noise-corrected AUC about 0.506 bound |
| Other devices (1,617 positives) | ties chance | AUC 0.50-0.51 |
| Sequence model over account history | ties chance | seq minus row +0.0002 AUC |
| Joint structure: time, entities, file order, shards | ties chance | ratios 0.999-1.018 |
| Entity pools (receivers, merchants, senders, devices) | tie or loses | lift about 1.0; model +0.0002 / -0.0015 |
| Sender / receiver label history | leak or nothing | 0.619 with own label leaking, 0.519 honest |
| Generation-order device numbers (fraud phase) | helps, but small coverage | +0.016 to +0.026 on probe; 2.8% of ordinary rows |
| Cleaning training labels with marker weights | tie | +0.002 to +0.006, one seed |
| Treating ordinary rows as legit in training | tie to small gain | +0.010 probe (earlier); loses in one quick run |
| Ceiling and error-budget analyses | ceiling stands | 0.576-0.607; 90% of gap unreachable |
| All-legit upload reading | decisive, cheap | 0.0816 vs 0.0827 separates the two readings |
| Public leaderboard noise | thresholds | gains under 0.005-0.008 are noise |
| Champion given / cleaned / generator-only | measured here | 0.5122 / 0.5665 / 0.9010 |

## 1. Size and nature: are they designed or noise?

**Idea.** Decide whether the ordinary rows are (A) a designed part of each fraud generator that the
clean test labels also contain, or (B) noise present only in train. This fixes the ceiling, the
training target and how validation maps to the leaderboard.

**Measurement (earlier measurement).**
- Two train-only columns give a label-free handle. Rows built by a fraud generator have
  app_version below 4.0 in 61.6% and ip_risk_score above 0.5 in 39.5%, drawn per row and
  independently, the same for every fraud type and for unreported rows (10 label-free patterns,
  16,698 rows). Legit-generated rows: 15.0% and 2.0%.
- A two-component mixture on these gives 36,318 +- 175 ordinary rows, 63.0% of the 57,648 fraud
  labels (21,330 fraud-generated). By month 62.7 / 63.3 / 63.0%; 2,321 / 2,325 / 2,320 per 100,000
  rows. Per label: mule 69.4, velocity_bot 59.8, layering 57.6, sim_swap 62.4, takeover 62.1,
  phishing 60.5, qr_swap 61.3, refund 60.3, cashback 67.1, loan_app 69.5, collusion 69.9%.
- An older isolation rule had said about 17,000 rows (30%). It was a poor proxy: 16,211 ordinary
  rows collide on big merchants and look "core", while 5,322 fraud-generated rows (chain hops,
  single-row events) look "isolated". The "62% P2P" skew of ordinary rows was an artifact of that
  rule; the real mix is 36.4% P2P against 35.9% for legit.
- The count per label is flat by month and does not follow the label's event volume (slope -0.02):
  refund events grow 71 to 103 per 100,000 while ordinary refund rows stay 126 to 123.
- Measured in this repo: the silver-label builder judges 62.9% of fraud labels ordinary-looking,
  agreeing with the earlier 63%.

**Verdict.** Settled: design, not noise. The decisive evidence is in sections 2 and 9.

**Lesson.** A proxy rule built on structure (isolation) can be off by a factor of two. A mixture
over columns that are generated per row, independent of label, is the reliable way to size a
population you cannot see.

## 2. Process and link checks

**Idea.** If ordinary rows come from the legit row generator, they should match legit rows on every
property and have no tie to events, entities or each other.

**Measurement (earlier measurement).**
- Inside P2P and inside MERCHANT, ordinary rows match legit rows on 60 properties (pair recurrence
  13.7 vs 14.1% and 9.7 vs 9.6%, new payee, degrees, device kind, amount digits and quantiles,
  balance, hour, weekday, month, KYC, OS, account ages). Largest |z| over 120 comparisons: 2.9.
- Rate 2,357 +- 11 per 100,000 legit-generated rows in every large row type (2,301-2,407), week
  (2,300-2,420) and sender-activity band (2,304-2,441). The label is independent of the row:
  each label's ordinary rows have the legit row-type mix.
- Pairs of ordinary rows on one sender: 5,339 +- 192 against 5,328 expected; on one device 5,106
  against 5,160. Same label inside such pairs 12.4% against 11.4% chance.
- Ordinary rows sharing a sender, device or P2P receiver with an event row: 0.96-1.08 of the chance
  count at any time. Fraud-labelled rows within 24 h of an event on a shared entity are 95%
  fraud-generated, so those are missed event rows, not ordinary ones.
- "Fraud-side receivers recur" is entirely a property of fraud-generated rows: 45.0% of September
  fraud-generated rows with a P2P receiver go to a receiver that had a fraud label before, against
  2.9% for ordinary rows and 4.1% for legit rows.

**Verdict.** Ties legit on every property.

**Lesson.** The ordinary rows are independent single rows, uniformly spread in time, attached to no
event, victim, mule or ring. Any feature built from entity or event context sees them as legit.

## 3. Row-level classifier on 234 features

**Idea.** Train a gradient-boosted model on July-August and test whether it can tell the ordinary
component from legit in September.

**Measurement (earlier measurement).** AUC of the ordinary component 0.499 and 0.501 (two seeds).
The top 1% of September holds 21 +- 49 ordinary rows of 5,095. Predicting the type among ordinary
rows: accuracy 20.7% against 19.4% for the majority class; macro one-vs-rest AUC 0.52 (the 2.7%
fraud-generated rows left in the set explain that). A later silver-label analysis found the same:
AUC 0.497 against legit, type accuracy 17.9% against 19.7%. Identifier digits, file position,
seconds and fractional digits: AUC 0.500, except the device number (section 9).

**Verdict.** Chance.

**Lesson.** Neither presence nor type of an ordinary row is predictable from row-plus-context
features. Reading "58-71% of each fraud label predicted legit" as misses is wrong: the probe's
fraud predictions are 90-103% fraud-generated and it recalls 65-96% of fraud-generated rows per
label; the "missed" rows are the ordinary ones.

## 4. Clean single-use-device sample (label-free comparison)

**Idea.** Any row gets a brand-new single-use device with probability 0.03 in both generator
passes, and the device number says which pass wrote the row. So single-use devices in the fraud
block against those in the legit block is a labelled comparison that needs no train labels.

**Measurement (earlier measurement).** Sample after removing template rows with the same column
rules: train 1,103 fraud-block against 42,984 legit-block rows; October 201 against 7,105. The
survivors have app_version below 4 share 0.157 and mean ip_risk_score 0.203, against 0.150 and
0.200 for the legit block (removed template rows: 0.596 and 0.392). 11.2% carry a legit label, the
usual unreported rate. Gradient boosting on 93 features (trap columns, device number and label
excluded):

| check | AUC |
|---|---|
| train, 5-fold out-of-fold | 0.491 (95%: 0.473-0.506) |
| leave one month out (Jul / Aug / Sep) | 0.472 / 0.484 / 0.504 |
| best of all model settings | 0.519 (0.502-0.536) |
| permuted labels, three runs | 0.503 / 0.473 / 0.502 |
| October, within-test 5-fold | 0.516 (0.480-0.554) |
| positive control, template rows left in | 0.654 (0.638-0.668) |

Of 87 single features, 1 has p below 0.05 (4.4 expected) and none below 0.01. The best cut-off
implies about 3% precision against a 2.5% prior; a rule needs about 27% to pay in macro-F1. The
time gap to the nearest fraud-block row is 26.1 days against 25.8 for a random row.

**Verdict.** Chance. Idea closed.

**Lesson.** Always run a positive control (template rows left in gave 0.654) and permuted-label
nulls; they show the pipeline can find signal and how wide the chance band is.

## 5. Value-level fingerprints (102 features)

**Idea.** The fraud pass calls the row template from its own code, so values might differ in
rounding, random-number calls or arithmetic order.

**Measurement (earlier measurement).**
- Every value is a clean decimal with fixed places; no float artefacts. Both passes round the same.
  SIM age and device age are fresh draws per row, not per sender.
- 102 categorical fingerprints (digit patterns of amount, balance, ages, login seconds, timestamp
  parts, id hex digits, joint patterns, balance-against-amount ratios) on the clean sample:
  9 features with p below 0.05 against a permutation-null mean of 5.1 (P = 0.10); best feature
  family-wise p 0.50 in train; gradient boosting OOF AUC 0.508 (permuted 0.494 / 0.499 / 0.513);
  leave one month out 0.532 / 0.514 / 0.511; train to October 0.494.
- One feature (timestamp second mod 3) had AUC 0.525 / 0.523 / 0.521 in all three months, but
  about 0.08 such features are expected by chance, there is no mechanism, and it is flat in October
  (0.504).
- Twin counts: no row of either pass shares its whole tuple with another row. AUC of tie counts
  0.496-0.509 in train.
- Transaction ids: no repeats among 1,824,642 rows (independent 32-bit draws would give 387.6), so
  they were drawn without replacement. 1,070 seeded random streams and hash schemes were checked:
  no hits.
- October amount factor (legit October amounts are x1.165): ordinary October rows sit high, not
  low, so a skipped factor is refuted.

**Verdict.** Chance.

**Lesson.** Exhaustive digit-level hunting with a permutation null and a family-wise correction is
cheap, and it is how to prove a negative; a single "significant" feature among 100 means nothing.

## 6. Ordinary rows on the sender's own phone

**Idea.** 97% of ordinary rows are on the sender's own device, which the clean sample does not
cover. The only ground truth there is the noisy label, so test it adversarially.

**Measurement (earlier measurement).** Sample: 32,037 fraud-labelled rows (10,743 / 10,832 / 10,462
by month) outside every mark, against 1,319,643 legit-labelled rows. About 2.9% of the positives
are really legit (attenuation factor 0.97).

| feature set | 5-fold OOF AUC | permuted labels |
|---|---|---|
| 102 fingerprints | 0.4987 | 0.5016 |
| 93 clean-sample features | 0.5022 | 0.5004 |
| both | 0.5005 | 0.4998 |

Leave one month out 0.495-0.504 for all three. Of 195 features, 7 have p below 0.05 (null mean 8.8).
Sender-relative consistency features (share of the sender's other rows with the same category,
hour, receiver; gap to the previous same-category row): AUC 0.495-0.508. Per class (11 classes,
1,184-6,179 positives): 0.486-0.514; only qr_swap reaches 0.5135, because 4.1% of its rows carry a
merchant-QR mismatch flag (0.3% of legit), a residual event mark, not an ordinary-row trait.
Noise-corrected AUC upper bound 0.506, which gives about 2.65% precision in the top 1% against a
2.5% prior.

The one real effect: in the clean single-use sample, P2P amounts of fraud-block rows are higher than
legit (AUC 0.542 on 392 train rows, 0.551 on 80 October rows). It is absent on the own phone
(0.500 / 0.496 / 0.500). It is largest in the takeover, mule and sim_swap segments and is probably
residual single-row template events, already reachable by the device-number rule.

**Verdict.** Chance.

**Lesson.** A positive-looking effect in a tiny clean sample must be checked on the bulk
population; here it vanished, and its likely cause was a missed event pattern.

## 7. Other devices, sequence model, joint structure

**Idea.** Three angles left open: rows on devices that are neither the sender's own phone nor a new
phone; sequence models over each account's history; joint structure (time clusters, shared
entities, generation order, file order).

**Measurement (earlier measurement).** Sample: 1,422,828 unmarked train rows, 33,654 fraud-labelled.
- **Other devices** (1,617 positives; fraud rate 2.21% on shared, 2.48% on single-owner, 2.37% on
  own phone): OOF AUC 0.5025 (93 features), 0.5055 (device-sharing features), 0.4957 (both);
  permuted-label range 0.4915-0.5174. No feature stays 0.02 off 0.5 on one side in all months.
- **Sequence model** (2-layer transformer reading the sender's and receiver's timelines, 65 events
  each; leave one month out; control = same network with context masked): sequence minus row-only
  gain +0.0002 on average (+0.0002 / +0.0011 / -0.0006 by month); seed spread 0.005; inside the
  permuted-label range. Positive control on event rows: 0.9993 sequence, 0.9979 row-only, so the
  pipeline learns where sequences exist. Per class, only sim_swap and takeover gain in all months,
  by at most +0.008.
- **Joint structure:** pair counts within 60 s, 1 h, 1 day, same day, and sharing a sender, device,
  merchant or receiver all sit at the chance rate (ratios 0.999-1.018). The one excess is 12 pairs
  of same-class ordinary rows sending to one P2P receiver within a day (chance 0.6-1.5), all
  consecutive pairs layering chain: almost surely missed chain hops.
- **File order:** the train CSV is 6 shards and the test CSV 2; inside a shard rows are fully
  shuffled; only the legit-block device counter follows shards. Ordinary rows are no more adjacent
  than chance (698 against 723 +- 25). Transaction id carries no generation order. No stateful field
  exists (new_payee is a random 15% draw; balance never equals previous balance minus amount).

**Verdict.** Chance on all three. Nothing reached the gate (AUC above about 0.56, stable direction
per month, above the permutation null).

**Lesson.** Closing a whole family (here: sequence models) needs a masked-context control and a
positive control; the control gap, not the absolute AUC, is the result.

## 8. Entity pools and label history

**Idea.** The fraud pass picks receivers (and perhaps senders, merchants, devices) from fixed pools
reused each month. If so, pool membership could reach the ordinary rows.

**Measurement (earlier measurement).** Pool flags computed from other months only, or label-free
from the month's own marked rows. Ordinary rows: 35,382 in July-September, 61.4% of fraud labels.
- Reach: every pool type flags ordinary rows at the legit rate; largest lift of any rule with 20+
  ordinary rows is 1.29. A P2P receiver flag (earlier fraud on a marked row) is 40.9-48.7%
  fraud-labelled, but of 1,004 flagged rows per 100,000 only 554 are unmarked, and those are 2.8%
  ordinary. Merchant pools hold every large merchant (flag a third of all rows, lift 0.99). Ordinary
  rows do not pool by class either (rows with earlier ordinary activity: lift 0.82 / 0.99 / 1.00 /
  1.03 for receivers / merchants / senders / devices).
- As model features on the October-like September (3 seeds, cleaned labels, baseline 0.53963): the
  receiver-pool feature +0.0002 (tie), the extended pool feature set -0.0015 (generator-only
  -0.0035; leave-one-month-out cleaned +0.0002). Direct rules that re-type rows the model left
  legit: from -0.0029 to -0.0598 for ungated rules, and at best +0.0001 for a gated rule that
  changes only 1.1 rows per 100,000.
- Label history of a row's own sender or receiver: with the row's own label subtracted ("leave one
  out") the AUC rises to 0.619, but the same counts from other months give 0.5191 against 0.5194
  (nothing). The jump is the row's own label leaking, not signal. Sender label history on ordinary
  rows: 4.7% against a 3.7% base, worthless.
- A quick check in a repeat-payer analysis reached the same wall: ordinary rows are 58-71% of every
  label (36,445 of 57,648), a random sample of legit traffic.

**Verdict.** Tie or loses. Pools only look precise on rows that already carry generator evidence,
which the model already finds.

**Lesson.** Entity history of labels must come from other months only, never leave-one-out within a
month, or a model learns its own label. A precise flag on a rare subset can still be worthless if
the model already covers it.

## 9. The one handle: generation-order device numbers

**Idea.** New device numbers (above 420,000) are handed out in generation order in five blocks: bot
phones, a train fraud phase (one segment per fraud type in label-list order), a train legit block,
a test fraud phase, a test legit block. Every row on a fraud-phase phone is fraud and its type is
its segment.

**Measurement (earlier measurement).**
- Train fraud phase: 2,240 phones on 5,454 numbers (3,214 skipped). Legit block: 45,177 single-row
  phones, none skipped, only 32 carry a fraud label (0.07%). Label equals segment type on 97.0% of
  fraud-labelled phones.
- Ordinary rows on a new phone: 1,003 = 2.76% of them (legit rows 3.00%), all in the fraud phase,
  each in its own type's segment. So they are not relabelled legit rows: every type generator emits
  them. They are under-reported too (188 +- 14 legit-labelled single-row fraud-phase phones; 75
  expected if only event rows went unreported, 216 if all fraud rows do).
- Rows on fraud-phase phones: 215.7 / 205.4 / 198.6 per 100,000 in July-September, 216.4 in October
  (564 test rows, 215 single-row phones the probe does not flag); 87.7 / 88.9 / 88.6% labelled
  fraud by month.
- A likelihood decoder places segments without labels: type accuracy 0.90 on September phones the
  probe did not flag, 0.95 on the whole train phase with emulated predictions. It also separates
  sim_swap new-phone events (162 phones, 0.41 skipped numbers per phone) from takeover ones (361,
  2.81), which no row feature does.
- Overriding the probe on these rows lifts September macro-F1 from 0.471 to 0.487 (decoded) and
  0.497 (segments placed with labels); every class gains 0.014-0.02, sim_swap up to +0.09.

**Verdict.** Helps, but it reaches only about 2.8% of ordinary rows plus the event rows on new
phones. It is a decoding handle, not a way to separate ordinary rows in general. The device-number
structure is the basis of the counter family; see [counter.md](counter.md). The skipped numbers
(3,214 train, 508 test) are unexplained.

**Lesson.** The only signal a generator leaves for rows it builds from the legit template is in
side channels of generation order (identifier allocation), not in values.

## 10. October evidence for design (label-free)

**Idea.** Settle design (A) against noise (B) without test labels, using the test fraud phase.

**Measurement (earlier measurement).** The test fraud phase holds 320 single-row phones. Without
ordinary rows in test, 100-103 +- 10 are expected; with them 291-294 +- 17. Their rows have the
train mix: 20.3% in a pattern (train fraud-generated 56.8%, ordinary 0.2%), 14.1% whole-rupee not
10, 54.1% P2P, so 35.5 +- 4.0% fraud-generated (train 34.4%). Implied ordinary rows in test:
7,475 +- 622 = 2.87 +- 0.24% of rows (train: 2.32% labelled, 2.65% with the unreported part).
Single-row fraud-phase phones per 100,000 rows: 109 / 114 / 106 in July-September, 123 in October.
In October the two markers are gone (app version 5.0-5.2 uniform; ip_risk_score keeps a small
residue, mean 0.2356 on pattern rows against 0.2202, usable only as an aggregate meter). A
label-free October audit also found fraud calls with no generator evidence at only 0.23-0.45% of
fraud calls and no generator family drifting beyond month-to-month noise (October outside the train
range on 22% of 495 shape-quantile checks; each train month is outside the others' range on
17-24%).

**Verdict.** Design (A) holds: the clean test labels very probably contain the ordinary rows.
Belief: this rests on a count of phones, and the all-legit upload (section 12) is the direct check.

**Lesson.** When a count can be predicted under two hypotheses (100-103 against 291-294) from
label-free structure, use it before spending leaderboard uploads.

## 11. Cleaning or relabelling ordinary rows in the training target

**Idea.** Give ordinary rows a low weight, drop them, or call them legit so the model stops
trying to learn noise-like targets. Details of label construction live in [labels.md](labels.md).

**Measurement.**
- Marker-weight cleaning (earlier measurement; probe, 300 rounds, one seed, September; weight of a
  fraud-labelled row = P(fraud-generated | marker cell, structure)):

  | training labels | noisy | clean A | fraud-generated truth | train rows |
  |---|---|---|---|---|
  | as given | 0.4690 | 0.5096 | 0.8275 | 241,662 |
  | soft weights | 0.4713 | 0.5105 | 0.8295 | 241,662 |
  | soft + drop legit-labelled rows in or next to a pattern | 0.4722 | 0.5115 | 0.8310 | 240,854 |
  | drop rows with P below 0.5, plus the same legit drop | 0.4721 | 0.5119 | 0.8337 | 215,230 |

  Gain 0.002-0.006: the trees already ignore ordinary rows. A tie on the noisy metric; a choice of
  speed and cleaner type learning, not score.
- Training them as legit (earlier measurement, probe, clean reference): +0.010 [0.007, 0.014]
  paired event bootstrap over the silver-labelled target; dropping them instead is a tie (-0.001
  [-0.004, +0.002]).
- In a quick earlier run in another analysis, setting 22,844 ordinary rows to legit before training
  lost: 0.4675 against 0.4759.
- In the mature pipeline (earlier measurement) the 14-variant noise-robust study never moved the
  score by more than +0.0002 and the remaining 12% of unreported fraud sits on ordinary rows where
  no relabelling can rank anything; see [labels.md](labels.md).

**Verdict.** Tie (small gains on a weak probe do not survive on the champion).

**Lesson.** Decoder class priors must still count the ordinary rows. Do not relabel them legit in a
way that changes the priors the decoder uses: per-class F1 is 2 TP / (predicted + about 2.7 x the
fraud-generated truth), so a row group is worth predicting when its clean precision exceeds F1/2
(about 0.20-0.29).

## 12. The all-legit upload reading and public noise

**Idea.** An all-legit file scores 2(1-s) / (12(2-s)) macro-F1 for test fraud share s, so one upload
reveals s and therefore whether ordinary rows count as fraud.

**Measurement (earlier measurement; the public part has about 78,175 rows).**

| test fraud share | public score | fraud rows in public |
|---|---|---|
| 1.50% (ordinary rows do not count) | 0.082703 | 1,175 |
| 2.40% | 0.082321 | 1,876 |
| 3.20% | 0.081978 | 2,502 |
| 4.06% (ordinary rows count, 10,582 fraud-pass rows) | 0.081606 | 3,175 |
| 4.20% | 0.081546 | 3,283 |

One point of fraud share moves the score by 0.00043: four decimals give the share to about +-0.12
points, five to +-0.012, six to +-1 row. Sampling of the public share (sd 0.06 points by row split,
0.13-0.14 by event split) is far below the 2.6-point gap between the two readings (0.0816 against
0.0827). The expected fraud share is about 4.1-4.3% if ordinary rows count (1.44% structured +
2.6-2.9% ordinary).

Noise of the public score for a rules-only detector, simulated on each train month (October-sized
month, 30/70 split by row or by whole event, 2,000 draws, truth = labels cleaned by the detector's
own events): sd of the public score 0.013-0.014 (row) / 0.023-0.025 (event); sd of public minus
private 0.012-0.013 / 0.030-0.032. For two uploads differing on about 300 rows (zones added, true
gap +0.03): sd of the public gain 0.0037-0.0039 (row) / 0.0047-0.0050 (event); smallest public gain
that is not noise 0.005 (row) / 0.007-0.008 (event); public and private order the pair differently
in 0 of 6,000 draws. For true gaps of 0.002 / 0.005 / 0.010 / 0.015 the public ordering disagrees
with the private one 34 / 5.5 / 0.1 / 0% of the time (row split) or 42 / 10 / 0.5 / 0% (event
split). Belief: for two quite different models (thousands of differing rows) "not noise" is
0.005-0.015 (row split) or 0.007-0.036 (event split).

**Verdict.** Decisive and cheap: one upload separates the two readings; a public gain under about
0.005-0.007 between near-identical files is zero.

**Lesson.** Read leaderboard = noisy-label macro-F1 + about 0.01, minus drift. Take thresholds for
"real gain" from a simulation of the split, and do not rank candidates 0.005 apart on the public
slice (see [stress-test.md](stress-test.md)).

## 13. Ceilings and the error budget

**Idea.** Quantify how much macro-F1 any detector could still win, and where.

**Measurement.**
- Earlier measurement, probe, September: macro-F1 0.471 on the noisy labels; on clean labels 0.510
  if ordinary rows stay fraud (0.484 with their unreported part added, 0.560 if every in-group
  type disagreement is a label error); 0.827-0.913 if they were legit. Ceiling for a perfect
  detector of fraud-generated rows only: 0.607 (0.576 with the unreported part). The structure
  ceiling quoted by later studies is 0.577.
- Earlier measurement, error budget of the earlier XGBoost champion (leave-one-month-out given
  0.5126, cleaned 0.5469; October-like September cleaned 0.5302): fixing every error would gain
  +0.4531; the 35,345 ordinary fraud rows called legit (unreachable) hold +0.4080, about 90% of
  the gap. The reachable part is 3,178 rows, +0.0630: +0.0565 is the wrong type inside template
  events (2,292 rows), mostly look-alike pairs that no feature separates (see
  [lookalikes.md](lookalikes.md)); the decoder or template lane holds only +0.0103; lowering the
  fraud cut-off is worth nothing (only 50 event rows are missed).
- **Measured in this repo:** the champion scores 0.5122 given and 0.5665 cleaned on leave-one-month-
  out and 0.9010 on the generator-only target, where ordinary-looking fraud is legit. Per-class
  F1 on the generator-only target is 0.76-0.95 (fake_cashback lowest at 0.755, takeover 0.836) and
  0.39-0.55 on the given labels; the 0.4-0.5 gap per class is the ordinary rows. Cleaned labels in
  this repo are the silver labels of the silver module: 3,629 train rows changed (2,944 legit to
  fraud, 685 wrong types).
- Earlier measurement: the October-like September cut breaks the device counter (7,282 September
  legit rows put in the fraud block instead of 115), so counter-dependent changes read low there;
  see [stress-test.md](stress-test.md). This does not affect the ordinary-row conclusions.

**Verdict.** The ceiling stands: about 0.58-0.61 for a perfect structured detector, about 0.50 as a
realistic top score. Expected gain per ordinary-row idea: zero.

**Lesson.** Report the structured-only (generator) score next to the cleaned and given scores:
only that part can move, and it is what a model change actually changes.

## 14. Capacity versus information on look-alike types

Not an ordinary-row experiment, but it supports the same conclusion. Stronger models (deeper
boosters, about 1,400 event-level aggregates, other-month label history, a pretrained tabular
foundation model) did not separate look-alike type pairs beyond the cascade by more than 0.05 AUC
(earlier measurement); the single exception (cashback against phishing, 0.74 against 0.62) came
from other-month label history and is worth at most +0.0005 macro-F1. A perfect separator for all
six pairs caps at +0.030. Details and the ordinary-row sibling result (sender label history
worthless) are in [lookalikes.md](lookalikes.md).

## What to carry forward

- Do not spend compute on separating ordinary rows from legit; every family (values, entities,
  sequences, order, pools, label history, device sharing) is closed at AUC about 0.50.
- Keep ordinary rows counted in the decoder priors and validate on both the noisy and the
  generator-only score.
- The device-number handle is real but covers about 2.8% of ordinary rows and is already in the
  counter family ([counter.md](counter.md)); decoding strategy lives in [decoders.md](decoders.md).
- Open questions: what consumes the skipped device numbers; the exact October ordinary rate (2.87%
  from the phone count against 2.64% expected). The all-legit upload settles the second.
- See [index.md](index.md) for the other ledgers and [labels.md](labels.md) for how noise and
  cleaning interact with the ordinary rows.
