# Decisions that matter to modelling and validation

Summary: whole-month folds; the device counter is used (as features everywhere, as a rule only in
the first seven segments, fitted per split); uploads need consistent evidence, not a big effect. Status: **earlier measurement** for the evidence quoted; the
repository's own design choices at the end are **measured in this repo** where a number is given
(see [RESULTS.md](../../../RESULTS.md)).

## 1. Canonical folds are whole months

Context: the test set is the month after training; events do not cross the split; rows of one
event share a label.

Decision:
- Out-of-fold predictions: leave one month out, three folds (July, August, September). Train rows
  dated October belong to September.
- Drift check: July + August -> September ("forward").
- Transfer check: an "October-like" September: legit-labelled rows thinned to 0.494 of their count
  with event rows protected, and the QR check blanked on 44% of QR payments to a person.
- Two targets on every run: the labels as given, and the event-cleaned labels.
- Comparisons: paired bootstrap over events, never rows. Stopping: fixed round counts, never
  validation log-loss.

Evidence (earlier measurement):
- Forward and leave-one-month-out folds rank six variants the same (Spearman 0.98, minimum 0.94);
  their means differ by at most 0.003.
- A random K-fold cuts events and inflates macro-F1 by 0.006 on given labels and 0.015 on generator
  fraud (merchant_collusion +0.075).
- Only 0.4-2.2% of multi-row events cross a month boundary.
- Log-loss against the noisy labels is lowest at 10-30 rounds while macro-F1 still rises at 100.
- Fold-to-fold sd of one variant 0.004-0.005; seed sd 0.0012.

Consequences: every run carries the fingerprint of the three folds. No fold on the full-density
store looks like October, so a change must also hold in the October-like rehearsal. Rejected:
random or stratified K-fold (leaks events); forward-only (never validates the first month, wasting
a third of the rare-type events).

## 2. The device counter is used

Context: device numbers from 420,000 up are handed out by one counter in the order the generator
wrote its rows: a bot block, then a fraud block written label by label, then a legit block in which
every number is used once; then the same blocks for the test rows. A row on a fraud-block phone is
fraud and its position names the label. Using it reads an identifier rather than behaviour.

Decision:
- Used as features everywhere: block, position, segment fitted to label-free anchors, sim/takeover
  cut and a confidence, computed by one function that runs on train and test alike.
- Used as a rule only where reliable: the first seven segments (mule to QR swap). In the last four,
  small and poorly located in October, a fraud-block row stays "fraud" and takes its type from the
  model. A fresh one-row phone in the legit block is legit.
- Segments are fitted in each split, never copied from train positions. Hard-coded boundaries are
  forbidden (a second plan measured -0.003 with them). Nothing is fixed for a particular test row.

Evidence (earlier measurement): September macro-F1 0.471 -> 0.496 with two columns; rules-only
0.469 -> 0.497. Fraud-block rows are 216 / 205 / 199 per 100,000 in the train months and 215 in
October; 88-89% carry a fraud label in train. The sim/takeover cut is right on 95% of labelled
multi-row phones; a fit on October-sized subsamples keeps 92 +/- 2% of the labels, while copying
train positions misplaces 19% of October devices. It is the only handle on ordinary rows (2.8% of
them) and on the true label of look-alike events with a fresh phone.

Measured in this repo: the counter and the template rule are worth +0.0333 given / +0.0343 cleaned
over the same recipe without them (paired event bootstrap, 90% interval [+0.0310, +0.0358]); the
ablation without counter features loses 0.028 (RESULTS.md runs 4, 10, 12).

Consequences: the rules allow features written in code, and the code is submitted, so the feature
is in the open. If the organisers rule it out, it is one family to switch off; keep the counter-free
model as the fallback. Not using it was rejected as giving up the largest single measured gain; a
rule for all eleven segments was rejected because the last four are poorly located in October.

## 3. The upload rule and how it evolved

The final entry is the best public upload, so every upload is a potential final.

1. **Initial plan:** upload only candidates that beat the champion on CV; never noisy variations;
   only the validated candidates and the few calibration questions the plan names.
2. **First additions:** write a forecast and a read-out rule into the record before each upload and
   score the forecast afterwards; run a change-footprint screen (rows changed per class pair; a
   candidate whose bound on the public change is under about 0.005 is a lottery ticket); treat any
   validator warning as a blocker for the final upload; nothing that could become best public by
   luck goes up (all-legit excepted). Expected-score stop line: more than 0.045 below forecast.
3. **"+0.007 on the October-like month":** set against validation noise (seed sd about 0.0036,
   thinning-seed spread about 0.004). In practice no change reached it; the best candidates were
   stacks of about +0.001 gains.
4. **"Consistently better, not only +0.007"** (final form). Upload candidate B against champion A
   only when all hold, one upload per candidate:
   1. October-like: event-bootstrap 95% lower bound of B - A above 0 (3 thinning seeds x 3 model
      seeds, counter from the unthinned frame, ring events kept whole), and the seed-averaged gain
      positive on every thinning seed.
   2. LOMO: B loses in none of July, August, September, on given or cleaned labels.
   3. Generator-only rows: B does not lose.
   4. Stacks: each piece still helps when left out, measured on months not used to choose it
      (nested selection); several pieces each chosen because they looked positive inflate the pooled
      gain.
   5. Unchanged: pre-upload audit 0 FAIL, October footprint recorded, forecast written first, record
      kept.
   Evidence: the six-seed + LightGBM variant failed (lower bound -0.0000); the best single parameter
   change shrank from +0.0004 (one seed) to +0.0001 / 0.0000 (two seeds averaged).
5. **Rule lifted for later uploads:** the later uploads (3, 4, 5 in [uploads.md](uploads.md)) were
   made without the gate. Upload 3 failed it afterwards (lower bound at or below 0, one thinning
   seed negative, August -0.00002). Result: +0.0017, +0.0001, 0.0000 on public: ties. The lesson is
   that the gate was right about the evidence; skipping it cost slots, not a measurable gain.

Final selection (if more than one pick were allowed): the best on CV and the most robust (for
example the best blend), picks that fail differently. With one pick, the champion unless CV-to-LB
evidence says CV is broken. In practice the platform picks the best public automatically.

Also in force: a tie goes to the simpler option; declare a sweep (ranges, K, metric, rule) before
running it and count every run; among candidates within one standard error take the simplest;
fit blend weights out of fold; run the leakage checklist on every new family (target-aware
transforms fitted inside the fold; no future information; no group straddling a fold boundary; a
suspiciously large gain is a bug until proven otherwise).

## 4. Repository design choices that bear on validation (this repo)

Measured in this repo where a number appears; full text in [decisions.md](../../decisions.md).
- The cleaned-label teacher is a different, simpler model than the scored one, and the cleaned labels
  are frozen on first build; scoring a model against labels cleaned by itself would turn its own
  false positives into true positives. The cleaning changes 3,629 train rows (2,944 legit -> fraud,
  685 wrong types).
- The screening score inside the cascade is cross-fitted nested inside each month fold, so no
  filter ever saw the held-out month's labels.
- The bootstrap resamples events and re-decodes saved probabilities (fixed point plus rules);
  October-like mode uses thinned September frames with the counter columns taken from the unthinned
  build; the template rule is not applied in that mode, so those scores are fixed-point only.
- The counter's emission table is fixed (from measured unmarked shares per zone plus three cross
  terms); only the anchors and the number line are fitted per split.
- Scores are reported on the repo's own cleaned labels; different cleanings give comparable but not
  identical numbers (a few thousandths).
- Jobs run from a frozen copy of the code so a late edit cannot change a running job.
