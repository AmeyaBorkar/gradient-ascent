# The two-stage cascade

**Summary.** The model is P(fraud) x P(type | fraud), both stages XGBoost on the GPU, fixed round
counts, three seeds, test predictions averaged over the month-fold models. A cross-fitted
screening score decides which fraud-labelled rows train the type model and which legit-labelled
rows are removed from the legit side (they are probably unreported fraud). The cascade beats a
plain 12-class model on every fold. Once the evidence families were in, no tuning of this machinery
moved the score by more than about +0.001.

## Why a cascade
- Each stage sees one kind of label noise: stage one the unreported fraud (fraud filed as legit),
  stage two the wrong-type labels, a small near-uniform flip on fraud rows only.
- The type model is small and fast, and the test prior enters through one number.
- Earlier measurement (relational features only, third-of-a-month folds, argmax decoding):
  cascade gain over a plain multiclass model +0.0101 given / +0.0099 cleaned / +0.0170 generator-only
  (about 5 sd, first in all five folds; mean +0.008 / +0.009 / +0.015). The cascade with unreported
  rows removed from the legit side added more (+0.0153 / +0.0160 / +0.0247).

## The recipe
| piece | setting | status |
|---|---|---|
| Stage one | fraud vs legit | measured in this repo |
| Stage two | 11-way type model on fraud-labelled rows | measured in this repo |
| Screening score | cross-fitted P(fraud), computed 2-fold over the fold's two training months, so no filter sees the held-out month's labels | measured in this repo (decision 5) |
| Type rows | fraud-labelled rows whose screening score is >= 0.10 (ordinary-looking rows sit near 0.02-0.03 and are excluded) | earlier measurement for the cut |
| Dropped legit rows | legit-labelled rows with screening score >= 0.70 are removed from stage one's legit side | earlier measurement |
| Trees | depth 8, 300 rounds at learning rate 0.08 per stage; fixed rounds, 3 seeds x 300 | measured in this repo (runs 1, 4: "3 seeds x 300 rounds") |
| Test prediction | mean of the three month-fold models | measured in this repo |
| Stopping | never on validation log-loss | earlier measurement |
| Product | P(type k) = P(fraud) x P(k | fraud); legit = 1 - P(fraud) | measured in this repo |

Why not early-stop: log-loss against the noisy labels is lowest at 10-30 rounds while macro-F1
keeps rising (flat for the plain model from 40 to 200 rounds; the cascade on the generator-only
target rose 0.8611 at 100 rounds to 0.8641 at 200 and 0.8677 at 400). Stopping on log-loss costs
0.003-0.005 for the plain model and 0.02-0.15 when ordinary rows are excluded (earlier measurement).

## What the dropped legit rows do
- On an October-like September the cascade calls all 375 legit-labelled rows inside detected events
  as fraud, and the 0.70 drop removes 1,985 legit-labelled training rows, 1,494 of them inside
  detected events (earlier measurement).
- So the fraud rows the train labels miss are already called: they score >= 0.95. This is why the
  decoder has nothing to fix on under-calling ([decoding.md](decoding.md)).
- Removing the drop entirely lowers September by 0.0008 (a screen over two seeds); drop thresholds
  0.6 and 0.8 tie with 0.7 (earlier measurement).
- Small-loss selection at the measured noise rate (dropping 5,299 legit rows instead of 1,985)
  loses 0.0018-0.0039 cleaned and 0.0075-0.0138 generator-only because it removes true legit rows
  scoring 0.06-0.70 (earlier measurement). See [training-labels.md](training-labels.md).

## Things that tied (earlier measurement unless noted)
- **Learner, seeds, blends** (leave-one-month-out, cleaned labels, deltas against the champion):
  six seeds +0.0001; seeds 3-5 +0.0002; LightGBM +0.0002; CatBoost +0.0002; XGBoost + LightGBM
  +0.0002; all four +0.0002 [+0.0000, +0.0005]. A 5-round model decoded to the same score as the
  900-round one: the evidence features near-determine the decoded calls. Per-class model choice,
  per-class blend weights and a type-stage stacker, all selected nested, gave +0.0002 / -0.0001 /
  -0.0003; none passes the upload rule.
- **Settings search:** 160 trials over 24 settings (both stages' depth, learning rate, rounds,
  child weight, sampling, regularisation, bins, class weights, thresholds). Best +0.0013 on the month
  it was tuned on, +0.0010 on fresh default-mode thinnings, +0.0002 in the families mode and on
  leave-one-month-out (losing August by 0.0001). Only one cliff: type-stage minimum child weight
  above about 8 (mean -0.0024 at 8-16, -0.0167 at 16-50); flat from 1 to 4. Importance of that
  setting was about 0.55 of the explained variance; everything else was flat. Binary fraud weight
  0.9-2 is flat. Keep it at 5 or below.
- **Type-row cut:** lowering 0.10 to 0.03-0.04 (431 more type rows, 22,744 against 22,313) was the
  only recipe change that held: +0.0002 to +0.0003, at or above the base in every month on all
  three targets; chosen in every nested fold. October effect: 16 of 260,582 rows change. It is a
  tie-level change, taken only for simplicity-neutral precision on borderline calls.
- **Noise-robust training** (forward loss correction, posterior weights, small-loss selection,
  in-event relabelling): eleven variants within 0.0005, three lose; see
  [training-labels.md](training-labels.md).
- **October-shaped training copies** (the train months thinned to October density, mixed with raw):
  +0.0065 on a default-mode October-like month, but all of it came from re-typing merchant ring
  bursts that the default thinning cuts; they lost on leave-one-month-out (-0.0008 to -0.0013
  cleaned, -0.003 to -0.004 generator-only). An artefact; not used ([validation.md](validation.md)).

## Where capacity cannot help
For the look-alike pairs (mule/chain, ato/sim, cashback/phishing, qr/refund/collusion) deeper or
longer boosted trees, about 1,400 event-level aggregates, other-month label history, linear and
nearest-neighbour models and a tabular foundation model all land at or below the cascade's AUC
inside the shape (0.59-0.68 for the pairs; one pair, phishing collects, reached 0.74 from label
history, worth at most +0.0005). Verdict: an information gap, not a capacity gap. A positive
control (events with a fraud-block phone) separates at 0.90-1.00, so the test can find signal.
