# Ordinary fraud rows: 63% of fraud labels that no model can find

**Summary.** About **63% of all fraud labels (about 36,300 of 57,648 train rows; 58 to 70% of each
class)** sit on rows written by the legit row template and labelled with a fraud class by the fraud
pass. They are designed fraud, present in October at the train rate, and nothing in them or in their
connections separates them from legit traffic (AUC 0.50). The consequence is a hard ceiling: a perfect
structure-only model scores about **0.57** macro-F1 on the clean test labels. Every later search for a
separator (values, timestamps, ids, pools, sender history, ties) has been negative. The only handle is
the 3% of ordinary rows that got a new single-use device in the fraud block.

Back to the [index](index.md). Related: [generator](generator.md), [device counter](device-counter.md),
[label noise](label-noise.md).

Status: **earlier measurement** unless marked "measured in this repo".

## 1. Size

- Fraud-labelled train rows: 57,648 = 21,330 fraud-generated plus **36,318 (plus or minus 175) ordinary
  (63.0%)**; by month 62.7 / 63.3 / 63.0%, or 2,321 / 2,325 / 2,320 per 100,000 rows.
- Per class (percent of the class's labels that are ordinary): mule 69.4, velocity_bot 59.8, layering
  57.6, sim_swap 62.4, takeover 62.1, phishing 60.5, qr_swap 61.3, refund 60.3, cashback 67.1, loan app
  69.5, collusion 69.9 (standard errors 0.7 to 1.6).
- Estimated from a mixture on the two train-only columns, which the fraud pass draws per row (app
  version below 4.0 on 61.6%, IP risk above 0.5 on 39.5%; legit-generated rows 15.0% and 2.0%). A
  second route (an independent label-free census) gives 61.4% (35,382 rows). An early estimate of 30%
  came from an isolation rule that mislabelled ordinary rows on big merchants as "core"; it was wrong,
  as was the accompanying "62% person-to-person" bias (ordinary rows are 36.4% person-to-person against
  35.9% for legit).
- Per 100,000 legit-generated rows the rate is 2,357 plus or minus 11, flat across every large row type,
  week and sender-activity band. A class's monthly ordinary count does not follow that class's event
  volume (slope -0.02): for refund, ordinary rows are 127 / 127 / 126 per 100k while its structured rows
  grew 76 / 96 / 116.
- Per 100,000 rows the ordinary labelled rows per class are in the table in
  [classes and templates](classes-and-templates.md).

## 2. They are written by the fraud pass, not by label noise

- Inside the person-to-person group and inside the merchant group they match legit rows on 60
  properties (pair recurrence 13.7 against 14.1% and 9.7 against 9.6%, new-payee, degrees, device kind,
  amount digits and quantiles, balance, hour, weekday, month, KYC, OS, account ages); largest |z| of 120
  comparisons is 2.9.
- Largest ordinary group (QR to a merchant, flag 0, amounts with cents): fraud-label rate is flat at 2.1
  to 2.5% across merchant sizes and categories; senders link to other fraud as often as legit senders
  (5.2% against 5.3%).
- The label is independent of the row: each label's ordinary rows have the legit row-type mix.
- **Generation order decides it** ([device counter](device-counter.md)): ordinary rows that got a new
  single-use device all sit in the fraud block, in the segment of their own class (1,003 rows, 2.76% of
  ordinary rows; legit rows 3.00%). Only 32 of 45,177 legit-pass single-use devices carry a fraud label
  (0.07%). Fraud-pass single-use devices with no template mark are 84 to 87% labelled with their zone's
  class and 10.8% `legit`, the same unreported rate as template rows.
- Any base row gets a brand-new single-use device with probability 0.03 in both passes, which is why
  ordinary fraud rows appear in the fraud block at all.
- Links: pairs of ordinary rows on one sender 5,339 plus or minus 192 against 5,328 expected under
  independence; on one device 5,106 against 5,160; links to event rows 0.96 to 1.08 of chance.

## 3. They are in October

- Test fraud block holds 320 single-row devices; without ordinary rows 100 to 103 plus or minus 10
  would be expected, with them 291 to 294 plus or minus 17. Implied ordinary rows in test: **7,475 plus or
  minus 622 = 2.87 plus or minus 0.24% of rows** (train 2.32% labelled, 2.65% with the unreported part).
  Single-row fraud-block devices per 100,000 rows: 109, 114, 106 in July to September, 123 in October.
- Single-row fraud-block devices number 320 in test against 286 expected from train, and about 97 if
  absent.
- The all-legit upload (public 0.0816) confirmed the clean test labels count them ([label
  noise](label-noise.md)).

## 4. Nothing separates them (negative results)

| Test | Result |
|---|---|
| clean single-use-device sample (fraud block against legit block, template rows removed), 93 features, gradient boosting | train 5-fold AUC 0.491 (95% 0.473 to 0.506); leave one month out 0.472 / 0.484 / 0.504; October 0.516 (0.480 to 0.554); permuted labels 0.503 / 0.473 / 0.502; positive control with template rows left in 0.654 |
| sample sizes | train 1,103 fraud-block against 42,984 legit-block rows; October 201 against 7,105; 11.2% of survivors carry a `legit` label |
| 102 value-level fingerprints (decimals, last digits, mod classes, round-together patterns, timestamp seconds, id hex digits) | 9 features with p below 0.05 against 5.1 expected by chance; boosting OOF AUC 0.508 (null 0.494 to 0.513); October 0.494 |
| own-phone ordinary rows (97% of them), 32,037 rows | noise-corrected AUC 0.4987 to 0.5023 on 102 / 93 / both feature sets; upper bound about 0.506, i.e. about 2.65% precision at the top 1% against a 2.5% prior (a rule needs about 27% to pay) |
| per class, ordinary against 100,000 legit | AUC 0.486 to 0.514 over 11 classes; only `qr_swap` at 0.5135, from residual merchant-QR mismatch rows (4.1% against 0.3% legit) |
| ties and duplicates (19 tuples) and sender-relative consistency | AUC 0.495 to 0.509; no ordinary row is a copy of a base row |
| record-level classifier, 234 relational features, July to August then September | AUC of the ordinary component 0.499 and 0.501; top 1% of September holds 21 plus or minus 49 ordinary rows of 5,095; class type among ordinary rows: accuracy 20.7% against 19.4% for the majority class |
| entity pools (receivers, merchants, senders, devices), history or label-free | flag precision on rows the model does not already see is 1.4 to 4.1% against a 2.3% base rate; lift at most 1.29; every rule loses on generator-only rows |
| `txn_id`, file position, seconds, fractional digits | AUC 0.500; ids drawn without replacement over both files at once, no seeded stream reproduces them |

Details worth keeping:
- Every CSV value is a clean fixed-decimal number; both passes round to the same number of places.
  SIM age and device age are fresh per-row draws, not per-sender (median sd of implied SIM install time
  across one sender's rows is 7,165 hours), so no cross-row consistency exists to break.
- A hypothesised "fraud pass skipped the October amount factor" is refuted (ordinary October rows sit
  high, not low).
- The one real effect: on single-use fraud-block devices, person-to-person amounts are higher than legit
  (train AUC 0.542 on 392 rows; October 0.551 on 80); it is absent on the sender's own phone. It belongs
  to the 3% with a new device and those rows are already reached by the device-zone rule.
- A leak to avoid: leave-one-out label counts on the same receiver or sender lift AUC to 0.619, but the
  same counts from other months add nothing (0.5191 to 0.5194); the leave-one-out version leaks the
  row's own label.
- Not tested: ordinary rows on devices that are neither own-phone nor new single-use (1,617
  fraud-labelled unmarked train rows), sequence or neural models over a sender's whole history, joint
  structure among several ordinary rows of one event, the organisers' generator code.

## 5. The one handle: fraud-block single-use devices

Every row on a fraud-block device is fraud (87.7 / 88.9 / 88.6% labelled fraud by month, the rest
unreported) and its class is the segment ([device counter](device-counter.md)). That reaches about 200
to 320 October rows, roughly 3% of the ordinary rows; type accuracy 0.90 on unflagged September phones,
0.95 on the whole train block with emulated predictions.

## 6. Consequences for targets and decoding

- Train to find fraud-generated rows and their class. Ordinary rows carry no signal, so train them as
  legit (or weight 0 to 0.3); cleaning gains only 0.002 to 0.006 in a probe because trees already ignore
  them, and the choice is one of speed and cleaner type learning.
- Validation: keep ordinary rows as fraud in the validation target (they count in clean test labels)
  and report a structure-only score beside it, because only that part moves.
- Ceiling for a perfect detector of fraud-generated rows is about 0.607 (0.576 if the unreported part
  of ordinary rows is added as misses); the probe's fraud predictions are 90 to 103% fraud-generated and
  recall 65 to 96% of fraud-generated rows per label: the "58 to 71% of each fraud label predicted `legit`" is
  the ordinary rows, not misses.
- Per-class recall is capped near 0.35 to 0.40 by this alone.

## 7. Measured in this repo

Command: `python -m ringtrace.build recipes/champion.json` (`RESULTS.md`).
- The frozen cleaned labels judge **62.9%** of fraud labels ordinary-looking (3,629 rows changed:
  2,944 legit to fraud, 685 wrong types), in line with 63%.
- Score on "generator-only" labels (ordinary-looking fraud set to legit) is 0.9010 against 0.5665 on
  cleaned labels and 0.5122 on given labels: the drop from 0.90 to 0.57 is almost entirely these rows
  being counted.
