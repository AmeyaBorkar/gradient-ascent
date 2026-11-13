# Features: families, counter and leakage hygiene

**Summary.** Three feature groups feed the model: relational window features, seven label-free
evidence families, and the device-number counter. Evidence plus counter lifts macro-F1 from 0.467
to 0.512 (given labels, measured in this repo). A set of "hygiene" rules removes columns that
identify the month; they cost 0.0095 without the evidence families, and were kept because the
evidence families carry the affected classes. Everything is built label-free on the stacked
train+test ledger and cached per family by a hash of the module's source.

## Families
| family | what it is | status of the claim |
|---|---|---|
| Relational / context | window counts and sums over sender, receiver, phone, pair and money flow; 123 features in the hygiene-free build | measured in this repo (run 1) |
| Row marks | exact constants and stale field combinations (e.g. exact balance steps of 0.25 and 0.35 of the previous balance, matched to the cent) | measured in this repo (decisions) |
| Drains | chained balance step-downs; connected components so one chance 3-row chain merges (607 vs 608 events, same rows) | measured in this repo (decisions) |
| Bot bursts | bursts on a reserved device pool; "farm" = more than 6 accounts with accounts >= 0.9 x rows; a 2-6-account burst is scripted when most logins take <= 6 s | measured in this repo (decisions) |
| Money flow | chains and fan-in/fan-out sweeps | measured in this repo (run 13: a tie) |
| Collect requests | round-limit-minus-1/10/50 collect amounts; bait payments of 1/2/5/10; two-bait sessions without requiring a 10 | measured in this repo (decisions) |
| Merchant rings | merchant burst events whose size names the type (1-3 rows qr_swap, 4-8 refund_abuse, 9+ merchant_collusion) | earlier measurement |
| Repeat payers | repeated payee patterns | earlier measurement |
| Device counter | block, position, fitted segment, sim/takeover cut and confidence (see below) | measured in this repo (runs 10, 12) |

The full pipeline has 230 features in the repo's champion (run 4); the earlier build carried 329-339.

## The device counter
- Device numbers from 420,000 up are issued by one counter in the generator's writing order: a bot
  block, then a fraud block written label by label, then a legit block in which every number is
  used once; the same again for the test rows. A fraud-block row is fraud and its position names
  the label (earlier measurement).
- Features: block, position, segment (found by a left-to-right HMM fitted to label-free anchors per
  split, never copied from train positions), sim/takeover cut, confidence.
- The emission table is a fixed table (measured unmarked shares per zone plus three cross terms:
  bot-zone marks look like mule, sim-zone like takeover, cashback-zone like phishing). It reproduces
  the segment agreement of 0.8456 / 0.9566 (fraud-block / fraud-labelled train rows) against
  0.8447 / 0.9555 for a fitted version, without a fit step (measured in this repo, decision 8).
- A richer observation model (event-shape tags with learned emissions) improves the counter's own
  accuracy (+0.0064 stability accuracy, loan recall 0.61 to 0.77) but ties end to end (+0.0001
  [-0.0001, +0.0003]); not adopted (earlier measurement). The counter's best contiguous 11-segment
  partition of the train labels scores 0.8527 / 0.9646, so the fit is within 0.9 points of its ceiling.
- The counter's rule use is restricted to the first seven segments (mule to QR swap); in the last
  four, small and poorly located in October, a fraud-block row stays "fraud" and takes its type
  from the model; a fresh one-row phone in the legit block is `legit` (earlier decision).
- October fraud-block rows run 215 per 100,000, against 216 / 205 / 199 in the train months; 88-89%
  of train fraud-block rows carry a fraud label (earlier measurement).

## Leakage and drift hygiene (month-identifying columns)
Rules, all measured against the October shift:
- Drop columns that identify the month: weekday, account and device ages, whole-frame entity totals,
  raw device numbers, the raw `*_dt_next` / `*_dt_prev` gaps (blank for 68-84% of October calls
  against 37-53% in September).
- Replace raw gaps by versions capped at 24 h, 72 h and 7 days.
- Drop the app-version and IP-risk columns: they are used only to measure label noise and to build
  cleaned labels (see [training-labels.md](training-labels.md)), never as model inputs.
- Drop dead columns (the repo drops two; one earlier example is a flow-share column that is
  infinite or missing on about 93% of rows).
- Blank receiver windows on merchant receivers.
- Keep amounts in rupees (October legit amounts run about x1.166 on train).
- Event ids must not be row indices, and per-entity totals must not come from the whole frame:
  both leak the frame (earlier finding).
- Any target-aware transform (labels per entity, teacher scores) must be built inside the training
  fold; label-derived features in test are up to a month stale, so validation imposes the same
  staleness.

**Cost of hygiene.** Without the event families it costs 0.0095 given (0.4671 to 0.4576; refund
-0.064, collusion -0.037, because those classes lose the merchant-receiver windows). Kept, because
the merchant and repeat families carry those classes (measured in this repo, runs 1 and 2).

## Ablations (leave-one-month-out mean; measured in this repo)
| change | given | cleaned | note |
|---|---|---|---|
| context only | 0.4671 | 0.5192 | run 1 / 5 |
| context with hygiene | 0.4576 | 0.5084 | runs 2 / 6 |
| seven families, no counter, no rule | 0.4788 | 0.5322 | run 10 |
| champion without counter (rule on) | 0.4846 | 0.5388 | run 12: counter is -0.028 / -0.028 at the fixed point |
| champion without money-flow | 0.5116 | 0.5661 | run 13: tie (-0.0006 / -0.0004) |
| champion | 0.5122 | 0.5665 | run 4 / 9 |

Paired event bootstrap (2,000 draws, 90% interval): seven families with hygiene over context-only
+0.0118 [+0.0094, +0.0142] given, +0.0130 cleaned; counter plus template rule over families alone
+0.0333 [+0.0310, +0.0358] given, +0.0343 cleaned. Command: `python -m ringtrace.compare`.
Other family ablations (drains, merchant, requests, marks, bursts, repeats, context) were queued
but never ran; they remain to be re-derived with `scripts/ablate.sh`.

## Open feature ideas (earlier measurement, all small)
- Explicit QR flags: the QR check was never blank on a QR payment to a person in train but is blank
  on 44% of them in October; a raw column would mislead. Window counts for cashback, loan and refund
  should be density-free.
- History of other months' labels for the same accounts lifts a look-alike pair (phishing vs
  cashback collects) from AUC 0.62 to 0.74, worth at most +0.0005 macro-F1: not built.
- A neighbour stage (re-scoring a row from its neighbours' predictions) costs -0.0003 and adds
  about 41 spurious QR calls in October: do not use.
