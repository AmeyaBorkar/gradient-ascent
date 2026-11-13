# Training labels: given, cleaned and event-cleaned

**Summary.** The train labels are noisy in two small per-row ways (about 12% of generator fraud is
filed as legit; about 2-3% of fraud rows carry a random wrong type) and one large structural way
(about 63% of fraud-labelled rows are ordinary-looking rows that no model can tell from legit).
There are three label sets: the labels as given; cleaned labels from an independent teacher; and
event-cleaned labels. A fourth view, generator-only, scores only the part that can be learned.
Training on the given labels with the cascade's built-in handling is as good as anything more
elaborate.

## The noise, measured (earlier measurement unless noted)
| quantity | value |
|---|---|
| Fraud-labelled train rows that are ordinary-looking | 62.8-62.9% (36,224 of 57,648; 2,304 / 2,326 / 2,318 per 100,000 in Jul / Aug / Sep) |
| What ordinary rows look like | legit rate on the old-app (15.1%) and high-IP (2.0%) columns; a 234-feature model separates them from legit at AUC 0.495-0.502; type accuracy 0.17 against a 0.19 majority share |
| Ordinary rows per fraud type | 58-71% (mule 68, bot 60, chain 58, sim 62, ato 62, phishing 60, qr 62, refund 60, cashback 65, loan 68, collusion 71) |
| Unreported fraud (generator fraud filed legit) | 12.2% [11.5, 12.9], drawn per row, same for every class and month (9.6-15.4% by pattern, every interval contains 12.2); whole events are never unreported |
| Wrong type | 3.0% [2.4, 3.7] of fraud-labelled rows inside events (uniform over the other types); 2.3% in total; events are type-pure |
| Legit rows filed fraud | 0.0007 |
| Generator-only share | fraud 1.51% of rows (noisy 3.68%, with ordinary rows 3.83%) |

Two label-free handles on "made by a fraud generator": old app version (61.8% of generator rows
against 15.0% of legit) and IP-risk above 0.5 (39.0% against 1.94%). The generator share of any
row set is (old-app rate - 0.150) / 0.468. These columns measure noise; they are not features.

**Look-alike events carry a partner type by design.** A pattern's events carry a sibling type 9-32%
of the time (takeover pattern 70.2% takeover / 29.2% sim_swap; chain hop 82.0% layering / 18.0%
mule; bait pair 66.8% cashback / 32.4% phishing; QR to a person 88.8% qr_swap). This is set at
generation, not noise, and is expected in October. Partner labels are never "corrected".

**Consequence for the ceiling.** A perfect detector with perfect types scores 0.547-0.552 against
the noisy labels (0.589-0.597 with unreported rows restored); per-class recall is capped at
0.31-0.45.

## The three label sets
### Given
The labels as shipped. Used to train and as the headline target (the test labels also hold the
ordinary rows). Leave-one-month-out macro-F1 of the champion: 0.5122 (measured in this repo).

### Cleaned (the teacher's silver labels)
Built to score runs on a less noisy reference; frozen on first build.
- **Teacher:** a different model from the scored one (a plain multiclass booster on context
  features and row marks, no cascade, no train-only columns), out-of-fold across months. Reason: a
  model scored against labels it cleaned itself would turn its own false positives into true
  positives (decision 4).
- **Calibration:** isotonic q = P(fraud label | x), then g = (q - rho) / (1 - u - rho) with
  rho = 0.0238 and u = 0.122 (the unreported share).
- **Posterior:** odds = g/(1-g) times likelihood ratios for the label (37.0 if filed fraud, 0.125 if
  filed legit), the old app version (4.12 old, 0.45 new) and the IP-risk bin (0.39 below 0.15, 6.9
  at 0.50, 137 at 0.70). A held-out check, building the posterior without the app version, matched the
  share implied by the old-app rate bin by bin.
- **Rewrites:** legit-labelled rows judged generator fraud become fraud (event majority class, else
  the teacher's type); fraud-labelled rows at the uniform noise floor take the event majority or the
  teacher type; partner labels are kept.
- **Measured in this repo:** 3,629 train rows changed: 2,944 legit to fraud, 685 wrong types; 62.9%
  of fraud labels judged ordinary-looking. A rebuild from a clean checkout reproduces the same
  statistics (3,629 rows) and the same submission hash. Small differences in which rows change move
  the cleaned score by a few thousandths, so cleaned scores from different cleanings are comparable,
  not identical. Earlier build: 3,607 changed (2,930 + 677), p_gen bimodal (58.0% of fraud-labelled
  rows below 0.01, 36.6% above 0.99, 472 rows in 0.2-0.8).
- **Per-frame cleaning** for the October-like month builds the labels from July and August only,
  with the teacher scoring each month from a model fitted on the other (2,408-2,421 rows changed
  per frame, about 1,950 legit to fraud and 470 wrong types; measured in this repo).

### Event-cleaned
A lighter rule used in the earlier sweeps: a legit-labelled row inside a detected label-free event
takes the event's class. It is not the same as the teacher's cleaned labels; the two must not be
read on one scale (the earlier harness' baseline of 0.5126 cleaned equals the champion's given score
by coincidence only).

### Generator-only (the diagnostic target)
The cleaned labels with ordinary-looking fraud rows (fraud label, cross-fitted fraud score < 0.10)
set to legit. It shows where the learnable loss is: sim 0.68, ato 0.69, cashback 0.72, phishing
0.81 and others 0.85-0.94 in an early model; the champion scores 0.9010 here (measured in this
repo). Variants rank identically on given, cleaned and generator-only labels inside a fold
(Spearman 1.00 in four folds, 0.94 in one). A gain on generator-only is about 1.7 times the gain on
the given labels with 1.8 times the sd, so neither is more sensitive.

## Which target for what
- **Headline:** given labels (the test labels carry the ordinary rows). Also report cleaned and
  generator-only on every run.
- **Per-class diagnosis:** generator-only.
- Noisy-label validation ranks variants the same as clean validation, and tunes the same decoder; it
  reads about 0.030 lower (0.470 against 0.500).

## What training choices were tested (earlier measurement)
Plain model, July+August to September, argmax; given / cleaned / generator-only deltas:
- base 0.4639 / 0.5040 / 0.8441;
- ordinary rows dropped +0.0036 / +0.0028 / +0.0032 (sd about 0.002); weighted 0.2: -0.0004 /
  -0.0005 / -0.0019; relabelled legit +0.0007 / +0.0009 / -0.0002: inside noise, the trees ignore
  them. Calling ordinary rows legit gave +0.010 [0.007, 0.014] on a clean-reference probe; dropping
  them was a tie;
- unreported rows dropped too: +0.0070 / +0.0071 / +0.0090.

Cascade-era noise handling (October-like September, three seeds; baseline = built-in 0.70 drop plus
fraud scale 0.875):
| variant | delta cleaned | delta generator-only |
|---|---|---|
| forward-corrected fraud loss, with or without the 0.70 drop | -0.0005 / -0.0000 | -0.0006 / -0.0013 |
| forward-corrected type loss (uniform 2.3% flip) | -0.0002 | -0.0004 |
| full forward correction | -0.0001 | +0.0000 |
| posterior weights, soft labels (three variants) | -0.0005 to -0.0002 | -0.0016 to 0.0000 |
| small-loss selection (drop 5,299 legit + 711 fraud rows) | -0.0018 | -0.0075 |
| small-loss, legit side only | -0.0019 | -0.0078 |
| small-loss on the type stage | +0.0001 | -0.0005 |
| two-round small-loss | -0.0039 | -0.0138 |
| relabel 1,543 in-event legit rows as fraud (and join type stage) | +0.0002 / +0.0001 | -0.0001 / -0.0002 |

Reading: the built-in handling already uses the noise that can be reached (375 of 375 unreported
in-event rows are called; the 0.12 prior correction is the decoder's fraud scale; posteriors are
near 1 for almost every row, so weights move only about 2,000 rows the drop already removes; the
2.3% type flips cost trees nothing). Only the ordinary rows remain unreported and they cannot be
ranked. Do not adopt any of these.

## What to do with them
- Train the cascade on the given labels with the screening-score recipe in [cascade.md](cascade.md).
- Cleaned-label training ties on the given-label score (nine October-like cells: 0.5091 against
  0.5093; measured in this repo).
- Never "correct" partner types inside look-alike events; never score a model against labels cleaned
  by itself.
