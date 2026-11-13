# Labels: noise model, cleaning and noise-robust training

Status vocabulary. **measured in this repo** means this repository's own runs show it (see
`RESULTS.md`). **earlier measurement** means an earlier research pass measured it; a later pass may re-derive it here. **literature / belief** means published claims or reasoning, not a
run of ours.

## Summary

Train labels carry two small per-row noises and one large design effect; test labels are clean.
Per row, a fraud row is filed `legit` with probability about 12% (unreported) and filed under a
wrong fraud type with about 3%; a legit row carries a fraud label with about 0.07%. The large
effect is not noise: about 63% of fraud labels sit on ordinary-looking rows that the generator
emits on purpose (see [ordinary-rows.md](ordinary-rows.md)), and look-alike fraud types are
confused at event level by design. From the noise model, "silver" (cleaned) labels were built;
every cleaning and noise-robust training scheme tried afterwards either tied or lost against the
plain champion, because the model already calls almost every unreported row that carries any
generator evidence, and the rest of the unreported fraud sits on rows that nothing can rank. The
practical value of cleaning is as a scoring target (a truer validation), not as a training target.

Measured in this repo: cleaned labels are the silver labels of the silver module (3,629 train rows
changed: 2,944 legit to fraud, 685 wrong types; 62.9% of fraud labels judged ordinary-looking);
champion leave-one-month-out macro-F1 0.5122 on the given labels and 0.5665 on the cleaned labels;
a champion trained on cleaned labels ties the given-label champion on October-like September
(0.5091 against 0.5093, mean over 9 cells); cleaning changes about 2,410 of 1,053,917 rows per
thinned frame. Everything else here is an earlier measurement or literature.

## Overview

| experiment | verdict | effect |
|---|---|---|
| Noise model (rates, matrices, event structure) | measurement | 12.2% unreported, 3.0% wrong type, 0.07% false reports |
| Silver (cleaned) labels: build | measurement | 0.231% of train rows changed (earlier); 3,629 rows (here) |
| Cleaned versus given versus generator-only targets | measurement | here: 0.5122 / 0.5665 / 0.9010 on leave-one-month-out |
| Probe trained on silver / structured / dropped labels | small gain on a weak probe | +0.004 to +0.010 clean (earlier, one seed) |
| Marker-weight cleaning of fraud labels | tie | +0.002 to +0.006 (earlier, one seed) |
| Champion trained on cleaned labels (cleantrain) | tie | 0.5091 against 0.5093, 9 cells (measured here) |
| Prior scale: fraud scale 0.875 against 1.0 | tie | +0.0001 (measured here) |
| Forward loss correction (binary, type) | tie | -0.0005 to -0.0000 (earlier) |
| Noise-posterior weights and soft labels | tie | -0.0005 to -0.0002 (earlier) |
| Small-loss selection / two-round co-teaching | loses | -0.0018 to -0.0039; generator-only -0.0075 to -0.0138 |
| In-event relabelling plus prior correction | tie | +0.0002 / +0.0001 (earlier) |
| Hard repair of unreported flips (early prototype) | helps cleaned, hurts raw | +0.0507 cleaned, lower raw (earlier) |
| Privileged train-only columns as model inputs | loses | -0.0294 in an early pilot (earlier) |
| Label-noise-robust architectures (literature review) | not run | ranked ideas; mostly superseded |

## 1. The label-noise model

**Idea.** Train labels miss some fraud and file some under the wrong type, test labels are clean.
Size each noise, say whether it is per row or per event, and decide which rows to relabel, drop or
down-weight, and what a clean validation set should contain.

**Measurement (earlier measurement).**
- Label-free handle on "made by a fraud generator": the two train-only columns are drawn per row
  given only that fact. app_version below 4.0 on 61.8% of generator rows against 15.0% of
  legit-generated rows; ip_risk_score above 0.5 on 39.0% against 1.94%; jointly 24.2% (product
  24.1%): the same for every type, month and for unreported rows. The generator share of any set of
  train rows is (old-app rate - 0.150) / 0.468. A QR-consistency flag (8,156 train rows, all
  generator-made) and 14 rules together flag 1,447 / 1,384 / 1,421 / 1,316 rows per 100,000 (train
  months and October), 99.5-99.7% generator-made.
- **Unreported fraud, per row:** u = 12.2% [11.5, 12.9] (12.2 / 11.5 / 12.9% by month; 9.6-15.4% by
  pattern, every interval holds 12.2; 11.0% outside the rules). Dispersion against a binomial
  0.83-1.05 where the event key is exact, so noise is per row. Of 713 events with 6+ rows none is
  silent; whole events are never unreported. Legit-labelled rows inside rules are 90-98%
  generator-made.
- **Wrong type, two layers.** Row level (noise): 3.0% [2.4, 3.7] of fraud-labelled rows in bot
  bursts, 3.4% refund bursts, 3.0% drains, 1.9% chain hops, 2.8% cashback pairs; the wrong label is
  uniform over the other types. Rows agree with their event's majority in 96.5-97.9%. Event level
  (design): whole events carry a partner type, e.g. new-device 2-5 row events are 70.2% takeover and
  29.2% sim_swap; chain hops 82.0% layering and 18.0% mule; bait-then-collect pairs 66.8% cashback
  and 32.4% phishing; fan-in 68.4% mule, 24.9% layering, 6.7% bot. Partner-labelled events differ
  systematically from the majority (e.g. all rows away from home in 91.3% of takeover-labelled
  events and 100% of 153 sim_swap-labelled ones). Event-level AUC to tell pairs apart: 0.57
  (takeover/sim), 0.49 (chain/mule), 0.47-0.52 (collect pairs), 0.76 (bait pairs), 0.93-1.00
  (fan-in size, QR to a person, round merchant rings). Belief: the type is set at generation;
  look-alike events keep their true labels in the clean test set.
- On fraud-phase phones, where the device-number segment gives the true type, wrong-type noise is
  3.0% (fraud-generated 2.7%, ordinary 4.1%); by segment qr_swap 8.4%, cashback 8.8%, refund 6.1%,
  others 0.5-3.3%.
- **Legit rows with a fraud label:** 32 of 45,177 legit-block phones = 0.071 +- 0.013% (about 1,000
  train rows if it holds everywhere).
- **Ordinary rows:** 36,241 fraud-labelled rows (62.9%; 12,146 / 12,270 / 11,825 by month), flat by
  month. They are design, not noise (see [ordinary-rows.md](ordinary-rows.md)).
- The 12% quoted elsewhere is the upper end; a later study uses 0.120 +- 0.010 unreported, 0.023
  wrong type in total, 0.0007 legit filed fraud.

**Verdict.** A well-measured noise model; two small per-row noises, two large design effects.

**Lesson.** A confusion that looks like 13-30% noise inside a pattern is mostly generator overlap
(two generators run one after the other). Only noise measured where an independent truth exists (a
phone segment) is noise. Do not "correct" partner types inside look-alike events.

## 2. Silver (cleaned) labels

**Idea.** Build one cleaned label per train row from independent evidence, to use as validation
target and, optionally, training target.

**Measurement.**
- Earlier measurement: an out-of-fold relational teacher over months (234 relational plus 15 rule
  columns, no train-only columns; macro-F1 0.457 / 0.459 / 0.464 on noisy labels) is calibrated
  with isotonic regression to q = P(fraud label | x); g = (q - rho)/(1 - u - rho) with rho = 0.0238
  gives P(generator-made | x); posterior odds then multiply likelihood ratios for the label (37.0
  fraud, 0.125 legit), for the app version (4.12 old, 0.45 new) and for the ip bin (0.39 below 0.15,
  6.9 at 0.50, 137 at 0.70). Held-out check: a posterior built without the app version matches the
  share implied by the old-app rate (0.003 / 0.84 / 0.9997 against 0.000 / 0.92 / 0.997 per bin).
  The posterior is bimodal: 58.0% of fraud-labelled rows below 0.01, 36.6% above 0.99, 472 rows in
  0.2-0.8. The train-only columns move only 214 rows across 0.5; they are the measuring device.
- Earlier measurement, rows changed: 3,607 (0.231%): 2,930 legit to a fraud type (1,769 by event
  majority, 1,161 by teacher type; 1,008 / 949 / 973 by month) and 677 wrong types (341 by event
  majority, 336 where the label is at the uniform noise floor, p(label | x) below 0.0068). Partner
  labels of look-alike events are kept.
- **Measured in this repo:** the silver labels, frozen once built, change 3,629 train rows (2,944
  legit to fraud, 685 wrong types); 62.9% of fraud labels judged ordinary-looking. A clean-checkout
  rebuild gave the same 3,629 rows and the same submission file.
- Earlier measurement, event-cleaned labels (a simpler variant used in a later study): a
  legit-labelled row inside an event takes the event's class. A "generator-only" proxy sets every
  fraud row that has no label-free evidence (or a cross-fitted score below 0.10) to legit; it is a
  proxy, not the silver labels.

**Verdict.** Works as designed; the earlier and the in-repo counts agree within 1%.

**Lesson.** Relabel only with an independent witness (event consensus, rule, marker posterior), and
only `legit` to fraud or noise-floor type flips, never among look-alike partners.

## 3. Given versus cleaned versus generator-only targets

**Idea.** The same predictions can be scored on three targets. Knowing how the numbers move tells
you what a change can really do.

**Meaning.**
- **given:** labels as shipped; ordinary rows count as fraud, 12% of fraud is filed legit. Maps to
  the leaderboard as noisy macro-F1 plus about 0.01, minus drift (earlier measurement).
- **cleaned:** silver labels (or event-cleaned labels); unreported rows restored, noise-floor type
  flips fixed; ordinary rows still fraud. A proxy for the clean test labels.
- **generator-only:** cleaned labels with ordinary-looking fraud set to legit; isolates what a
  model can change. The ordinary part is constant, a ceiling the model cannot touch.

**Measurement.**
- **Measured in this repo (champion, leave-one-month-out):** given 0.5130 / 0.5090 / 0.5146 (mean
  0.5122); cleaned 0.5645 / 0.5639 / 0.5712 (mean 0.5665); generator-only 0.9010. Cleaning adds
  about +0.054 because precision is under-stated by noisy labels; the generator-only score is about
  0.39 higher still because the ordinary rows are removed from truth. Per-class F1 (pooled): given
  0.39-0.55, cleaned 0.45-0.61, generator-only 0.76-0.95; fake_cashback is lowest on all three
  (0.396 / 0.446 / 0.755).
- Earlier measurement, probe, September (with silver labels): trained on noisy labels, noisy 0.470,
  silver 0.525, clean (ordinary rows weighted 1/0.878) 0.500, structured-only 0.869. Noisy-label
  validation ranks the four training variants the same way and tunes the same decoder, but reads
  about 0.030 low (0.470 against 0.500). Multipliers tuned on noisy labels reach the same clean
  score as those tuned on clean (0.4996 / 0.4995).
- Earlier measurement, the earlier XGBoost champion: leave-one-month-out 0.5126 given, 0.5469
  event-cleaned, 0.8876 generator-only; October-like September 0.5396 cleaned. Different "cleaned"
  definitions (event-cleaned against silver) give different cleaned values for similar models: do
  not compare cleaned numbers across definitions.
- Earlier measurement, why noisy-label decoding under-predicts fraud: unreported fraud lowers
  noisy precision to c x clean precision, so the noisy-optimal threshold F~/2 exceeds the
  clean-optimal c x F/2. In practice the decoder's fraud scale of 0.875 (= 1 - 0.125) against 1.0
  is a tie.
- **Measured in this repo:** rescoring the champion with no template rule and fraud scale 1.0
  gives 0.5063 given / 0.5600 cleaned against 0.5064 / 0.5601 at 0.875, a tie (+0.0001).

**Verdict.** Report all three; the cleaned target is for choosing models, the generator-only
target for seeing what changed, the given target for the leaderboard.

**Lesson.** A tie on the given labels and a difference on the cleaned ones are different things;
the cleaned labels' value is in making validation less noisy, not in raising scores by themselves.
See [decoders.md](decoders.md) for decoding on noisy versus clean targets.

## 4. Training on cleaned labels (cleantrain)

**Idea.** Train the champion on cleaned labels (relabelled unreported rows and noise-floor type
flips) instead of the given labels.

**Measurement.**
- **Measured in this repo:** October-like September, three thinned frames, three cascade seed sets
  each, fit on July + August (cleaned labels built on each frame from July + August only, teacher
  fitted on the other month), scored on September given labels after the fixed-point decoder (no
  template rule). Champion 0.5093 mean over 9 cells (frames 0.5060 / 0.5096 / 0.5125); cleantrain
  0.5091. Cleaned labels change 2,419 / 2,421 / 2,408 of 1,053,917 rows per frame (about 1,950
  legit to fraud and about 470 wrong types each). The cleaned target cannot be scored on September
  here, because September has no cleaned labels by protocol.
- Earlier measurement, probe, one seed: silver-trained +0.003 over noisy-trained on the same
  target; fixing both noises +0.004 [0.001, 0.007].

**Verdict.** Tie: -0.0002 on the given labels.

**Lesson.** The champion's trees already ignore the noise that cleaning would remove. Cleaning
costs a teacher fit and brings nothing on the leaderboard-like target; see
[ensembles.md](ensembles.md) for what did move scores.

## 5. Training with ordinary rows weighted, dropped or called legit

See [ordinary-rows.md](ordinary-rows.md), section 11, for the full table. In short (earlier
measurement, probe, one seed): marker-posterior weights and dropping rows with P below 0.5 gain
+0.002 to +0.006 (noisy 0.4690 to 0.4721, generator-only truth 0.8275 to 0.8337); treating
ordinary rows as legit gains +0.010 [0.007, 0.014] on the clean reference; dropping them is a tie.
Verdict: tie on the champion. Decoder priors still need the ordinary count.

## 6. Noise-robust training against the earlier champion

**Idea.** Use the measured transition matrix in four families: (a) forward loss correction, (b)
noise-posterior weights or soft labels, (c) small-loss selection across folds (including two-round
co-teaching), (d) in-event relabelling plus the 0.12 prior correction. The champion handles noise
bluntly: legit-labelled rows with a cross-fitted fraud score of 0.70 or more leave the binary
model's legit side, and the decoder divides fraud by 0.875.

**Measurement (earlier measurement).** Fourteen variants, 3 seeds on the October-like September
(event-cleaned labels, counter held at its unthinned values; baseline 0.5396 cleaned, generator-only
0.8629), and one seed leave-one-month-out (baseline 0.5124 given, 0.5467 cleaned, 0.8873
generator-only). Bar to adopt: +0.007.

| family | October-like cleaned delta | leave-one-month-out cleaned delta | generator-only |
|---|---|---|---|
| (a) forward correction, binary / type / both | -0.0005 / -0.0002 / -0.0001; with the 0.70 drop -0.0000 | -0.0003 / +0.0002 / -0.0002 | -0.0013 to +0.0000 |
| (b) posterior weights and soft labels | -0.0005 / -0.0005 / -0.0002 | -0.0003 / -0.0002 / +0.0004 | -0.0016 to -0.0000 |
| (c1) small loss: drop 5,299 top-score legit and 711 low-score fraud | -0.0018 | -0.0021 | -0.0075 |
| (c2) legit side only | -0.0019 | -0.0022 | -0.0078 |
| (c3) type stage: drop 2.3% lowest P(filed type) | +0.0001 | +0.0002 | -0.0005 |
| (c4) two-round co-teaching | -0.0039 | -0.0037 | -0.0138 |
| (d1) relabel 1,543 in-event legit rows fraud | +0.0002 (bootstrap +0.0004 [-0.0001, +0.0009]) | -0.0001 | -0.0001 |
| (d2) d1 plus those rows join the type stage | +0.0001 (bootstrap +0.0007 [0.0000, +0.0015]) | +0.0001 | -0.0002 |

Calls changed against the baseline: 3-15 per 100,000 for every principled variant; 51-88 for the
small-loss ones. Fraud scale 1.0 instead of 0.875 moves every variant by at most 0.0005.

Why they tie or lose:
- The champion already calls all 375 legit-labelled in-event rows of the October-like September
  fraud (2,337 of 2,337 in leave-one-month-out). The 0.70 drop removes 1,985 legit training rows,
  of which 1,494 sit inside detected events.
- The rest of the unreported fraud sits on ordinary rows (AUC about 0.5), where a correction can
  only shift the prior, and the fixed point absorbs prior shifts.
- Posteriors are near 1 for almost every row (filed fraud with a low ordinary-row score gives P
  about 0.97), so weights re-weight only the roughly 2,000 rows the drop already removes.
- Small loss at the measured noise rate asks for 5,299 legit rows to go against the drop's 1,985,
  pushing the cut down to a score of 0.061; the extra 3,300 rows are true legit rows in the
  0.06-0.70 band, so 50-87 more rows per 100,000 are called fraud.
- The 2.3% uniform type flips cost trees nothing: correcting them changes 3 calls per 100,000.

**Verdict.** Eleven variants tie (|delta| 0.0005 or less), three small-loss variants lose, best
+0.0002 against a bar of +0.007. Keep the 0.70 drop and the 0.875 fraud scale.

**Lesson.** The 12% unreported fraud is largely unrecoverable because it lives on rows no model can
rank. A principled correction buys the same calls as a blunt rule with more machinery. Never remove
fraud-labelled rows by loss or margin: rare rows are high-loss by construction. Untested: a softmax
head with forward correction on the cascade's leaf embeddings; over-corrected noise rates as a
sensitivity check.

## 7. Hard repair of unreported flips (early prototype)

**Idea.** Overwrite the 12% unreported legit flips with the structure-repaired label during
training, and blend a model trained on raw labels with one trained on repaired labels.

**Measurement (earlier measurement, early weak baseline around 0.44-0.47).** Adding about 30 exact
detector features lifted raw September macro-F1 from 0.4424 to 0.4648 and cleaned from 0.4648 to
0.4889; pooling probabilities inside episodes (0.65 row, 0.35 episode) plus tuned multipliers gave
0.4712 raw and 0.4955 cleaned. Hard-repairing the flips raised cleaned September macro-F1 to 0.5155
(+0.0507) but lowered raw to 0.4336 (0.4524 when gated to templates of 88% precision or more),
because predicted fraud probability on templates overshoots the 88% noisy prior if the test labels
keep the same unreported flip. Proposed hedge (belief, not run): blend 0.75 raw-trained and 0.25
repaired-trained models and check the legit multiplier with one upload.

**Verdict.** Helps the cleaned target, hurts the raw target; superseded by the decoder that handles
the prior (section 3). Hard repair without recalibrating the legit multiplier costs -0.018 to
-0.037 raw.

**Lesson.** The test labels' cleanliness decides which target to trust; the leaderboard evidence
(all-legit upload reading, see [ordinary-rows.md](ordinary-rows.md)) points to ordinary rows being
counted as fraud, so the cleaned target is the better proxy but the decoder must still be tuned
against both.

## 8. Privileged train-only columns

**Idea.** app_version and ip_risk_score are informative in train and dead in October. Use them as
model inputs, or only as measuring devices.

**Measurement (earlier measurement).** Feeding them to the model in an early pilot lowered
September macro-F1 from 0.2079 to 0.1785 (-0.0294). app_version below 4.0 falls from 15.70% in
July-September to 0.00% in October; ip_risk_score above 0.5 from 2.52% to 0.52%. In the label
model they are used only to measure the generator share of train rows and to build silver labels
(they move 214 rows across 0.5). In October, ip_risk_score keeps a small residue (+0.0154 +-
0.0021 on pattern rows) that is usable only as an aggregate precision meter.

**Verdict.** Loses as inputs; useful as measuring devices.

**Lesson.** Never give them to a model. Keep one privileged column out of every cleaning step as an
independent referee, or nothing validates the cleaning.

## 9. Research notes on noise-robust architectures (literature, not run)

Three literature reviews fed the experiments above. Claims here are **literature / belief**, not
measurements in this repo.
- **Privileged information.** Distillation from a teacher that sees extra training-only columns
  helps most where the extra columns separate clean from wrong labels; a teacher that sees only the
  privileged columns lost on two of three sets (-5.4%, -15.0%); with a baseline trained to
  convergence, distillation transfers nothing on four real tabular sets. Own arithmetic: here the
  columns measure the true class per row only weakly (likelihood ratio about 2.9 for an old app) and
  work at event or aggregate level.
- **Soft labels and weak supervision.** Boosting students on teacher soft labels beat hard labels by
  about +1.0 point (binary) and +0.6 (multiclass) across 30 datasets; learned label models fail in
  the regime of low label density and correlated votes, which is ours.
- **Selection and relabelling.** Boosters shrug off random noise up to 40%; biased noise under 10% is
  what hurts. Correction helped small sets and hurt the largest imbalanced one. Co-teaching and mix
  methods depend on neural training order and showed no tabular or graph evidence at our scale.
- **Robust losses inside boosting.** Generalised and symmetric cross-entropy become cross-entropy
  with per-round weights in a Newton booster; from a cold start the generalised loss starves rare
  classes (weight 0.97 for legit against 0.017 for a 0.3% class, a 57x gap). Verified by finite
  differences: cross-entropy against a soft label equals one weighted copy of the row per class.
- **Macro-F1 under noise (derivation, belief).** Unreported fraud scales precision by c and leaves
  recall unchanged, so thresholds tuned on noisy labels under-predict fraud on clean labels; the
  fixed-point decoder has the optimal threshold F/2 per class. Class-conditional noise estimates
  (positive-unlabelled) assume reporting is independent of features, which is probably false here.
- **Ranked ideas from the notes** and their fate: rules as test-time evidence (done, the evidence
  families); event-consensus soft labels (equivalent to the silver and in-event relabelling above,
  tie); privileged teacher into soft labels (the silver teacher, tie); forward correction (tie, section
  6); margin relabelling (not run); a gated student on a neural tabular model (not run). "Not to
  do" list: no robust loss from cold start, no learned label model, no label-derived teacher feature,
  no removal of fraud-labelled rows by loss, no co-teaching port, no trust in uniform-flip injection
  tests. Section 6 confirms the first and the removal rule empirically.

## What to carry forward

- Use silver or event-cleaned labels as the validation target and report the generator-only score
  beside it; train on the given labels with the 0.70 legit drop and the 0.875 fraud scale.
- Do not spend more time on noise-robust objectives; the reachable noise is already called.
- Relabel only with an independent witness and never among look-alike partner types.
- Related ledgers: [ordinary-rows.md](ordinary-rows.md) (the 63% that is not noise),
  [decoders.md](decoders.md), [lookalikes.md](lookalikes.md), [stress-test.md](stress-test.md),
  [more-data.md](more-data.md), [open-ideas.md](open-ideas.md) and the [index](index.md).
