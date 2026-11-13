# Rule and evidence-family experiments

Ledger of every experiment on label-free rules and evidence families: what was tried, what it
measured, whether it helped, and what it taught. Companion files: [index](index.md),
[counter](counter.md), [lookalikes](lookalikes.md), [ordinary rows](ordinary-rows.md),
[decoders](decoders.md), [ensembles](ensembles.md), [labels](labels.md),
[stress test](stress-test.md), [open ideas](open-ideas.md).

## Status labels

- **measured in this repo**: this repository's own runs show it (its results ledger).
- **earlier measurement**: an earlier session measured it, with a different code base and
  usually a different scoring frame; a later session may re-derive it here. Treat the numbers as
  strong leads, not as this repo's facts.

Scoring frames used below. LOMO = leave-one-month-out over July, August, September (macro-F1 over
12 classes per held-out month, then the mean). "Given" labels are as shipped; "cleaned" labels
give legit-labelled rows inside a detected event the event's class (or, in this repo, the frozen
silver labels). "Generator-only" is a proxy that keeps only rows with label-free generator
evidence and treats other fraud labels as legit. "October-like" thins September's legit rows to
October's density and scores it with a model fitted on July and August. A change needed about
+0.007 on the October-like month to count (single-seed noise there is about 0.005, see the
sweep section).

## Summary

The hand-built evidence families (drains, money flow, requests, merchant rings, bot devices,
repeat payers, marks) are the reason the model reaches 0.51 on given labels. They were fully
mapped: every structure was found label-free, and the classifier already calls every row inside
them as fraud and types it right except for look-alike label choices. After they were in, nothing
further paid. 38 request variants, 12 drain variants, 11 money-flow variants, 12 merchant
variants, 8 marked-row variants, pool features and 9 pool rules, a 160-trial cascade search and
a 37-run sweep all tied or lost, with best gains of +0.0001 to +0.0013 against a bar of +0.007.
What remains is information the data does not contain: about 63 percent of every fraud class is
ordinary-looking rows no feature reaches, and the choice between look-alike classes inside a
template is not predictable from any label-free feature.

The two things that did pay were both about the device-number counter: the counter features
(+0.028, measured in this repo) and the per-event, anchor-protected one-class-per-template rule
(+0.0058 given / +0.0064 cleaned over the fixed point, measured in this repo).

## Overview

| experiment | verdict | effect |
|---|---|---|
| Counter features (device-number blocks and segments) | helps | +0.028 LOMO given and cleaned (measured in this repo); +0.021 to +0.025 on September in the first probe (earlier) |
| Seven evidence families vs context-only | helps | +0.0118 given [+0.0094, +0.0142], +0.0130 cleaned (measured in this repo) |
| Counter features plus template rule vs neither | helps | +0.0333 given / +0.0343 cleaned (measured in this repo) |
| One class per look-alike template, anchored variant | helps | +0.0028 given / +0.0029 cleaned over the plain rule (measured in this repo); +0.0064 LOMO cleaned over no rule (earlier) |
| Money-flow family removed | tie | -0.0006 given / -0.0004 cleaned (measured in this repo) |
| Money-flow rule variants (wave 2) | tie / loses | best +0.0015 on a flawed frame, 0.0000 once fixed |
| Drain rule, cut-off and detector variants (wave 2) | tie / loses | 0.0000 to -0.0077; detector saturated |
| Drain cross-template model | loses | AUC 0.56 on events whose label follows a phase |
| Payment-request rules and cut-offs (wave 2) | tie / loses | best +0.0002 across 38 variants |
| Merchant-ring rule variants (merchant classes saturated) | tie / loses | best +0.0007, changes 2 test rows |
| Marked-row class model | loses | -0.0011 to -0.0073 when it changes anything |
| Entity-pool features and pool rules | tie / loses | +0.0002 features; rules best +0.0001, most lose 0.003 to 0.06 |
| Entity history features (pools by label) | tie | 0.4713 -> 0.4675 to 0.4666 on the probe (earlier) |
| Velocity-bot rules, October missing bots | tie | bot-device rule already used; no moved variant found |
| Repeat-payer features (refund, loan) | tie | macro-F1 +0.006 / -0.000 / +0.000 on the probe |
| Rules-only detector prototype | works as a baseline | 0.476 raw September; +0.028 with zones |
| Detector prototype fix (time unit) | bug fix | every earlier window was 1000 times too short |
| Hybrid detector features plus smoothing | helps (first model) | +0.0224 over relational GBDT on raw September (earlier) |
| Parameter sweep (tree settings) | tie | one recipe change, +0.0002 to +0.0003 |
| Sweep ranking (37 runs) | tie | only dropping the rings family loses, -0.0067 LOMO |
| Cascade hyperparameter search (160 trials) | tie | +0.0013 on the tuning month, +0.0002 on the honest frame |
| Validation and decoding for the rule layer | method | thresholds half optimism; fixed-point decoder kept |

## 1. Counter features and the counter-anchored rules

**Idea.** New-device numbers are issued in generation order: a bot range, then a fraud block laid
out class by class, then a legit block. A row on a fraud-block phone therefore reveals its class.
Features: block, position in the block, segment, confidence.

**Measurement.**
- Removing the counter features from the champion (rule left on): 0.4846 given versus 0.5122,
  and 0.5388 versus 0.5665 cleaned. At the fixed point 0.4788 versus 0.5064 given and 0.5322
  versus 0.5601 cleaned: worth +0.028 on both targets (measured in this repo). The counter
  family plus the template rule together are +0.0333 given [+0.0310, +0.0358] and +0.0343
  cleaned [+0.0319, +0.0369] (measured in this repo).
- First probe on July+August -> September: the zone label of the row's own phone plus the bot
  block lifted macro-F1 from 0.4627 (argmax) / 0.4713 (tuned) to 0.4833 / 0.4960; sim_swap F1
  0.362 -> 0.455 (earlier measurement). A rules-only detector gained +0.028 from zones (below).
- Zone rule precision on train: rows on a zone phone get their segment label 83 percent of the
  time, any fraud label 88 to 89 percent; 79 to 86 rows per 100k are ordinary-looking rows
  reached this way (earlier). Dropping the sim/takeover cut accuracy: 95 percent of
  fraud-labelled fresh-phone rows correct against 62 to 69 percent for "always takeover".
- Fitting the segments to anchors per split beats carrying train positions to October: positions
  misplace 19 percent of October's zone devices (earlier).

**Verdict.** Helps; the single largest family contribution.

**Lesson.** Fit boundaries per split from label-free anchors; never carry train positions over.
Details in [counter](counter.md).

### 1a. Richer anchors and learned emissions for the counter (tie)

**Idea.** Replace hard anchors (the template's predicted class) by a per-zone emission over the
row's shape tag, learned once on the train fraud block, to place segment boundaries better.

**Measurement (earlier).** Train segment accuracy 0.8447 -> 0.8468 (all rows), within 0.9 points
of the best contiguous partition. At test size, refit accuracy +0.0064 [0.0033, 0.0095], won 19
of 30 draws; loan-zone recall 0.61 -> 0.77. October: the refund/cashback cut moved 7 devices,
10 rows changed segment. End to end: decode-only 0.0000 in-sample, a full pipeline rerun +0.0001
[-0.0001, +0.0003] on both targets, 19 October rows changed. Shape-only anchors with no template
class broke (accuracy 0.289).

**Verdict.** Tie. **Lesson.** The champion's template rule reads anchors only inside six
look-alike templates, and the misplaced cut touched rows the model already ignored. Adopt only if
a rule starts reading segments outside the templates.

## 2. One class per look-alike template (the template rule)

**Idea.** Six templates are shared by two or three classes (chain/mule, sim/takeover,
mule/bot, one-bait fast payback phishing/cashback, just-under collect phishing/cashback,
round-amount merchant qr/refund). No label-free feature separates the classes inside, so call the
class with the higher break-even share for every row of the template, except where a phone-anchored
row names the class.

**Measurement.**
- Break-even mix per template is well below the train mix for five of six (train mix minus break
  even +0.29 to +0.56), so the fixed choice is right every month; only the one-bait fast-payback
  group sits near break-even (+0.03 to +0.11, earlier).
- Plain rule vs no rule and the anchored variant: the anchored rule changed 773 train and 137 test
  rows and scored +0.0058 given / +0.0064 cleaned over the fixed point; the anchored rule beats
  the plain one by +0.0028 given / +0.0029 cleaned (0.5122 vs 0.5094 given, 0.5665 vs 0.5636
  cleaned); fraud scale 0.875 against 1.0 with no rule is a tie, +0.0001 (all measured in this
  repo).
- Earlier ladder, LOMO mean over no rule, cleaned: fixed rule +0.0032; anchor rows keep their
  segment class +0.0040; anchor protection plus propagating the anchor class to the other rows of
  its event +0.0064 (above the fixed rule in every month: +0.0051, +0.0073, +0.0069); anchor
  class gate at confidence 0.8 identical on train, a 0.9 gate drops to +0.0052.
- Choosing the class each month from that month's anchors loses (+0.0023 to +0.0029 against
  +0.0040): anchors are about 3 percent of each template (9 to 50 rows a month), their mix has
  a bias of +0.003 to +0.061 and a month-to-month sd of 0.03 to 0.05; the one-bait group's sd is
  0.39. The oracle monthly class equals the fixed class every train month.
- A rule-only detector gave sibling-pair ceilings: all five confusable pairs together hold about
  +0.029 macro-F1 and no label-free feature reaches it (pair AUCs 0.42 to 0.62, null p 0.02 to
  0.86; model gain +0.000 per pair). Only the zone class helps, +0.003 in all, already in the
  baseline (earlier). The one decoder choice that pays is giving the one-bait fast-payback group
  to cashback (+0.002 on train).
- October (aggregates, earlier): the fixed rule re-types 73 rows, protection plus propagation
  re-types 99; it fixes the drain split (98 percent sim in October against 82 in September) and
  the one-bait group (30 against 50 percent cashback).

**Verdict.** Helps. **Lesson.** Decide the look-alike class per event, from anchors, never per
month; do not build sibling sub-models. See [lookalikes](lookalikes.md).

### 2a. The October-like frame breaks the counter and biased every rule comparison

The October-like frame thins legit rows, so the legit block of device numbers is no longer filled
once each and the refitted counter puts 7,745 September rows in the fraud block instead of 517
(7,282 of them legit-labelled; earlier). The model ignores the phantom block but the anchored
rule does not: it flipped about 70 real layering chain rows to mule. Scored with the counter built
on the full frame and joined by row, the baseline moved from 0.5299 to 0.5391 (+0.0092).

**Lesson.** Any October-like comparison that touches counter columns is biased against the
counter unless the counter is taken from the unthinned frame. Several "wins" below (+0.0056 for
the plain template, +0.0015 for a slow-chain guard) were artefacts of this and vanished once the
counter was fixed. See [stress test](stress-test.md).

## 3. Drains (sim_swap, account_takeover)

**Finding (earlier).** A drain is 2 to 4 PAY rows by one victim; between consecutive rows the
balance is multiplied by an exact constant, 0.25 (sim mechanism) or 0.35 (takeover mechanism).
Links per month: 384/359/313/173 (0.25) and 417/336/328/177 (0.35); no other constant exceeds 23
links; no single-row drain exists. Rows are about 110 per 100k each constant, stable into
October. Labels follow the generator phase, not the mechanism: a takeover-mechanism event that sits
on a phone numbered in the sim phase is labelled sim_swap (and conversely 27 to 28 percent of
events). 63 percent of both labels (2,565 of 4,072 sim, 2,572 of 4,049 takeover) are ordinary
rows on no chain; probe AUC 0.507 and 0.503 against legit. Noise on chain rows: 9.5 to 12.7
percent unreported per row, 2 to 3 percent random type.

**Rule measurements (earlier).** Chain -> its constant's class: precision 0.62 to 0.67 against
the noisy label, 0.86 to 0.91 for any fraud, coverage 0.25 to 0.29. Adding the sim-phase phone:
precision 0.69, coverage 0.36 to 0.39. Extra device with 2 to 4 rows -> takeover: 0.59 to 0.64
precision. Single-row phone in the sim or takeover sub-block: 0.81 to 0.91 precision (rare, 4 to 6
per 100k).

**Wave 2 (earlier).** Baseline October-like cleaned 0.5391 (given 0.5063), LOMO 0.5125 given,
0.5468 cleaned. Where the two classes lose: 58 percent of each class is ordinary rows (costing
0.034 and 0.039 of macro-F1); the one reachable cell is takeover-labelled rows called sim inside
sim-template events on the victim's own phone (401 rows LOMO, 80 October-like), worth +0.0114 /
+0.0146 if fixed perfectly, and nothing marks the phase there.

| variant | Oct-like cleaned | LOMO cleaned |
|---|---|---|
| drain rows called legit take the event class | -0.0000 | +0.0000 |
| all drain rows take the event class | -0.0000 | -0.0003 |
| event zone extended to device-template events, strong or weak zones | +0.0000 | +0.0000 |
| single-row fraud-block phones take the segment class | +0.0000 | up to +0.0001 |
| night start (22-06) -> sim | -0.0010 | -0.0005 |
| daytime sim-template start -> takeover | -0.0077 | -0.0051 |
| 2-row sim events -> takeover | -0.0002 | -0.0020 |
| class cut-off multipliers 0.5x to 3x | -0.0002 to +0.0001 | about 0 |
| multipliers 5x / 10x / 30x | -0.0014 / -0.0225 / -0.0725 | -0.0003 / -0.0356 / -0.0783 |

- Detector knobs: looser tolerances or hop counts add nothing or only legit rows (a relative
  tolerance of 1 percent adds 1,255 rows, 1,127 of them legit, and loses 57 chain rows); tighter
  ones lose real chains (a 45-minute span loses 859 rows, a 3-row minimum 931). 2-row drains are
  real (264 sim, 145 takeover, 49 legit). The detector finds 4,041 rows in 1,407 events and was
  not retrained on variants.
- Cross-template model (XGBoost on all features plus event aggregates, 495 sim-template events
  without a fraud-block row): AUC 0.557 pooled; event size alone gets 0.613. Calling the minority
  at 0.5 flipped 97 events and got 34 right.
- October (label-free): seven rows of drain events on sim-segment phones are called takeover
  because the position feature's class cut drifts (0.4584 in train, 0.4845 in test); extending the
  zone rule fixes exactly these and cannot be scored by the protocol (+0.001 expected, belief).

**Verdict.** Tie or loses; the drain decisions of the champion stay. **Lesson.** The drain ceiling
is information: the phase of an own-phone sim-template event is not in the data. A position
feature whose class cut moves between splits is a liability; the per-split segment carries the
same information.

## 4. Money flow (mule_account, layering_chain)

**Finding (earlier).** Three exact event shapes, all label-free: chains (3 to 6 hops, each 96.5 to
99.5 percent of the previous, 20,000 to 180,000), mule bursts (8 to 25 senders, then 1 to 3 onward
transfers), fan-ins (3 to 6 payers, layering). 578 chains, 204 bursts, 229 fan-ins in train;
October 96, 42, 27. Burst -> mule, chain and fan-in -> layering. The cross-labelling is a
whole-event relabel (of 429 fast chains, 339 layering-only, 86 mule-only; of 149 slow chains none
is mule; 177 of 204 bursts are mule, 18 are bot). 69 percent of mule and 59 percent of layering
rows are ordinary rows outside every event (AUC 0.505 and 0.492). Nothing moved in October; chain
rate 36.8 per 100k against 35.3 to 38.9.

**Measurement (earlier).**
- Rules by month, precision against the noisy label: chain or fan-in -> layering 0.76 to 0.78
  (any fraud 0.87 to 0.89, coverage 0.40 to 0.42); burst -> mule 0.74 to 0.83. Rule layer on a
  September probe: mule F1 0.400 -> 0.409, layering 0.493 -> 0.532, macro 0.4628 -> 0.4671.
- The 31 money-flow features added to the probe: 0.4628 -> 0.4618 (tie), 0.4605 with ring
  features.
- Removing the family from the champion: -0.0006 given / -0.0004 cleaned; fixed point 0.5065 given
  / 0.5601 cleaned (measured in this repo). Same tie in the 37-run sweep (-0.0013 LOMO,
  -0.0029 October-like, interval spans zero October-like).

**Wave 2 (earlier).** Baseline October-like 0.5299 (0.5391 with the real counter); LOMO 0.5125 /
0.5468 / 0.8875. 89 to 98 percent of the false negatives are ordinary rows; the rest are the
look-alike relabels (mule-filed fast chains, bot-filed bursts). Event rows are already all called
fraud and never an unrelated type.

| variant | Oct-like cleaned | LOMO cleaned |
|---|---|---|
| force legit-called event rows to the event class (any cut-off) | 0.0000 | 0.0000 |
| fan-in template / bot template | 0.0000 | 0.0000 |
| anchor confidence 0.5-0.7 / 0.9-0.95 | 0.0000 | 0 / -0.0013 |
| plain template, all groups | +0.0056 (artefact) | -0.0033 |
| slow-chain guard | +0.0015 (artefact) | 0.0000 |
| event vote of model probabilities | -0.0049 | -0.0033 |
| cut mule/layering/velocity calls outside shapes to legit | -0.0041 | -0.0040 |
| same, to the best other class | -0.0056 | -0.0049 |

With the real counter the plain template loses -0.0014 (mf groups) / -0.0036 (all) and the guard
is 0.0000. A perfect label choice inside look-alike chains and bursts is worth +0.0126
October-like, +0.0086 LOMO; no label-free signal reaches it.

Relaxed shapes found nothing: stealth small runs (45 train rows, 98 percent legit), bursts split
by gap, off-pool bot sessions (velocity at most 0.003), hop look-alikes outside chains (layering
0.00). Stealth mules (new-payee share under 0.4, 26 events) are already 84 percent mule and caught
by the burst rule. No fan-in of size 7 exists. Account pools do not choose the look-alike label (a
chain sharing an account with a burst's onward receiver is 32 percent mule-filed against 17).

**Verdict.** Family is a tie; no rule variant wins. **Lesson.** The three shapes are saturated;
the remaining loss is ordinary rows and relabels with no signal. Do not chase calls outside the
shapes: cutting them loses 0.004 to 0.006.

## 5. Mule bursts and the missing October bots (velocity_bot)

**Finding (earlier).** Bot bursts live on devices 420,001 to 422,999: 84 in train (33/26/25 per
month) and 7 in October where about 13.8 were expected (Poisson p 0.035). Per burst: 20 to 60
rows in 2 to 15 minutes, 1 to 3 ecommerce or gaming merchants, PAY of 10 to 499. The label has
three parts: device bursts 34.9 percent, fan-in events written by the bot generator 3.5 percent
(look like mule), ordinary rows 61.6 percent (AUC 0.50 to 0.52). Noise inside bursts: 84.6
percent bot, 12.8 legit, 2.6 other type. Rules: device range 420,001-422,999 precision
0.84 to 0.85, coverage 0.31 to 0.39; the probe already calls 97 to 99 percent of burst rows bot
and 270 October rows, 269 of them on bot devices.

**Experiments on the October shortfall (earlier).** Scans for an adapted variant (sessions at
5 minutes to 24 hours, devices with 4+ accounts, merchant sessions on small merchants, 10-minute
bins, signature rates) found nothing: no bot variant moved. October has 7 bot-pool bursts against
14.0 expected and 37 all-new mule-like bursts against 29.7: -7.0 / +7.3. If the generator moved,
about 28 percent of the all-new bursts would be bot, still a minority and separable only through
the counter. The mule phase wrote more bursts (new devices in the mule zone 16 against 13.0
expected, bot zone 1 against 0.8); had 288 missing rows moved to any template, 8.6 more template
rows would sit in that zone (P(<=1) = 0.0007). Rule applied: predict bot on every row of a bot
device and nowhere else; class multipliers cannot move bot onto other rows.

**Verdict.** The device rule is already in; every adaptation variant ties. **Lesson.** Keep the
October bot prior at about half of train; do not reassign mule bursts to bot. The 12.8 percent
legit-labelled bot rows are unreported fraud. App version and risk score are dead in October
(62 percent old version in train, 0 percent in October).

## 6. Requests and collect scams (phishing_collect, fake_cashback, loan_app_extortion)

**Finding (earlier).** Only 38 percent of phishing and 34 percent of cashback labels are generator
rows; 4,136 phishing rows and 1,670 cashback rows are ordinary traffic. Five label-free shapes
on a pool of about 6,500 ring accounts: just-under-limit COLLECT requests (limits 2,000 / 5,000 /
10,000 / 25,000 minus 1, 10 or 50), whole-rupee requests, a single bait of Rs 1, 2 or 5, two baits,
and baits answered within 72 hours. Type rule: unanswered baits and requests -> phishing; two baits
or a late answer -> cashback; the one-bait answered within 4 hours is a 59/41 mix nothing
separates. Unreported share 13 percent per row. October: no adaptation (request rate 125 per
100k against 121 to 136; no new bait amount or limit).

**Rule measurements (earlier).** Core request rule -> phishing precision 0.76, any fraud 0.86 to
0.87, 84 to 86 percent of the label's real rows; answered baits -> cashback 0.86 to 0.90; the
one-bait fast-payback group -> phishing 0.46 to 0.54. On proxy-clean labels the rules reach
phishing F1 0.886 and cashback 0.723; sending the one-bait group to cashback wins the combined F1
in all three months (+0.031, +0.005, +0.023) against labels as given.

**Wave 2 (earlier).** Baseline October-like cleaned 0.5391 (real counter), given 0.5063,
generator-only 0.8605; LOMO 0.5125 / 0.5468 / 0.8875. 87 to 93 percent of the misses are ordinary
rows. Reachable errors are sibling labels inside two shapes (one bait with payback within 4 hours:
phishing vs cashback; just-under collect: phishing vs cashback vs loan), worth about +0.017 if
chosen perfectly. A per-shape model sees every label-free feature: AUC 0.529 (just-under collect)
and 0.419 (one-bait group; shuffled labels give 0.48 and 0.52). 38 variants:

| variant | Oct-like cleaned | LOMO cleaned |
|---|---|---|
| one-bait group -> phishing | -0.0032 | -0.0018 |
| one-bait group left to the model | -0.0029 | -0.0013 |
| one-bait group split by event mean P(cashback) at 0.335 / 0.5 | -0.0025 / -0.0034 | -0.0013 / -0.0016 |
| bait 5 -> cashback, others phishing (best) | +0.0002 | +0.0004 |
| shape class forced on calls (fraud-only or all) | +0.0000 | +0.0001 |
| PAY/QR under-limit requests -> phishing | -0.0004 | -0.0003 |
| bait-only Rs 10 from a ring account / all | +0.0001 / -0.0040 | +0.0001 / -0.0039 |
| unify a victim's two requests by summed probability | +0.0000 | +0.0001 |
| scale phishing 0.7 to 2.0 | -0.0009 to 0.0000 | -0.0004 to +0.0001 |
| scale cashback 0.7 to 2.0 | -0.0001 to +0.0000 | about 0 |
| scale loan 0.7 to 2.0 | -0.0008 to +0.0001 | -0.0003 to +0.0001 |
| scale all three 0.85 / 1.2 / 1.5 | -0.0003 / +0.0002 / -0.0000 | about 0 / +0.0001 / -0.0001 |

The 205 October cashback calls against 104 in a rule census are the template choice, not an
error. Loan rows outside 20:00-23:59 do not exist (64 train lending PAYs outside the window, 0
loan-labelled); loan labels on repeat victims are not evening rows either (20 percent against 21
for phishing). Break-even cashback share of the one-bait group is 0.335 against a train share of
0.32 to 0.55 by bait amount, so the class stays cashback. Bootstrap on the best variant:
+0.0002 [-0.0028, +0.0032].

**Verdict.** Every variant ties or loses. **Lesson.** Keep the one-bait group on cashback; no
class cut-offs for these classes (the fixed-point decoder already sits at the F1 optimum).

## 7. Merchant rings (qr_swap, refund_abuse, merchant_collusion)

**Finding (earlier).** Two row patterns carry all three classes: QR to a person that still has a
merchant category (qr_swap 82.9 percent of those rows) and round-amount payments (1000, 2000,
2500, 5000, 10000) to pool merchants. Event size names the type: 1 to 3 rows qr_swap (151 of 152
events at size 1), 4 to 8 refund_abuse, 9 or more merchant_collusion (36 of 36); row purity
96 to 100 percent. Label composition: qr_swap 62.9 percent outside every pattern, collusion 73.5
percent. Pools: about 360 to 410 merchants, about 5,200 to 6,000 accounts. Odd quirk state: a
mismatch value on a non-QR row or blank on a QR row marks generator output (8,156 train rows,
87.8 percent fraud, 537 to 544 per 100k, stable into October). Ring events keep their shape
in October (3 collusion events of 20, 23, 28 rows).

**Rules (earlier).** Rule precision by month: QR-to-person 0.81 to 0.85; 1-3 row events -> qr
0.81 to 0.82; 4-8 -> refund 0.77 to 0.83; 9+ -> collusion 0.82 to 0.85; coverage 0.08 to 0.27. A
rule override on the probe lifted September collusion F1 0.362 -> 0.406, refund 0.521 -> 0.531.
Counts for October: 515 qr, 71 collusion, 76 refund rows from rules.

**Merchant classes saturated (wave 2, earlier).** Baseline October-like cleaned 0.5391, LOMO
0.5126 / 0.5468 / 0.8878. The champion already calls every structure as its rule says (185 of
185, 76 of 76, 71 of 71 on October ring events; QR-to-person all qr_swap).

| variant | Oct-like cleaned | LOMO cleaned |
|---|---|---|
| ring size class forced on every ring row | +0.0003 | +0.0000 |
| QR to a person forced to qr_swap | 0.0000 | 0.0000 |
| QR to shop merchant, blank check, not a ring amount -> qr_swap | 0.0000 | 0.0000 |
| whole-amount event -> collusion | +0.0004 | 0.0000 |
| all of the above | +0.0007 | +0.0001 |
| size-3 bursts -> refund | -0.0034 | -0.0081 |
| size-4 bursts -> qr_swap | -0.0022 | -0.0013 |
| size-9 bursts -> refund | -0.0043 | 0.0000 |
| linking gap 12 h (24 h now) | -0.0073 | -0.0044 |
| linking gap 18 / 36 / 48 h | +0.0003 / +0.0001 / +0.0001 | -0.0007 / 0.0000 / -0.0002 |
| merchant-class calls outside structures -> legit | -0.0053 | -0.0041 |
| class cut-offs 0.5x to 2.0x (16 runs) | best +0.0003 | all <= 0.0000 |
| oracle: three classes right on every generator row | +0.0272 | +0.0141 |

The union of the structural rules changes 2 test rows. Losses are label noise inside structures
(269 unreported one-row QR-to-person rows, 48 of them filed collusion, 14 of 166 size-3 bursts
refund-filed with AUC 0.42 for telling them apart) and ordinary rows (4,463 of 7,335 qr_swap rows
have no evidence). Merchant history does not choose a burst's class (after a 9+ burst 17 / 1 / 2
qr / refund / collusion; base mix 492 / 66 / 36). No October ring row links back to September, so
month-edge chaining changes nothing.

**Frame flaw (earlier).** October-like thinning without protecting event families cuts ring
bursts: raw September has 148 / 21 / 15 bursts of size 1-3 / 4-8 / 9+, the thinned frame 111 / 11
/ 10 and files collusion rows in 4-8 bursts, inflating the ceiling (+0.027 against +0.014 on
LOMO). Deltas stay valid because both arms see the cut bursts.

**Verdict.** Tie. **Lesson.** Do not move the 3|8 size cut-offs, the 24-hour linking gap, or add
merchant-history features; for merchant classes use LOMO or a family-protected October-like
frame.

## 8. Marked singles and marks

**Idea.** Rows carrying a near-certain generator mark (stale QR check, balance 1.1x or 1.02x the
amount, amount above balance) but in no event were expected to number about 1,100 in the test
month; their class would come from fields the routine overwrote, so a dedicated class model
might beat the champion.

**Measurement (earlier).** The population is tiny: 14,785 marked rows in train, but only 72 in no
event of the six families (27 of 2,339 in test); 2,918 sit in no event of 2+ rows and 450 in
test. The champion calls none of the fraud-labelled marked rows legit and types them 0.892
(single), 0.935 (in no 2+ event), 0.873 (in no 3+ event), 0.898 (all) on LOMO. An XGBoost
11-class model on these rows (all champion columns plus amount grids, hour, counter zones, nearby
marked rows, receiver history from other months) gets 0.908 / 0.935 / 0.872 / 0.890: a tie.

| override (re-type) on tier | Delta LOMO given / cleaned / generator | Delta Oct-like cleaned |
|---|---|---|
| single | +0.0000 (1 row) | -0.0011 (7 rows) |
| in no 2+ event | +0.0000 (5 rows) | -0.0012 |
| in no 3+ event | -0.0011 / -0.0011 / -0.0008 (303 rows) | -0.0029 |
| all marked | -0.0029 / -0.0032 / -0.0046 (496 rows) | -0.0073 |

50/50 blends tie or lose by the same amounts. Training only on the in-no-3+-event tier is worse
(-0.0012 to -0.0073 per tier, -0.11 on all marked rows). Error budget: 120 mark-only train rows,
14 fraud errors.

**Verdict.** Loses. **Lesson.** The marks are almost always already inside an event; do not add a
marked-row class stage. The "about 1,100 test marked singles" estimate was wrong (27).

## 9. Pools: entity pools and pools in ordinary rows

**Finding (earlier).** Two disjoint hidden pools of fraud-side accounts, about 6,300 each (1.5
percent of 420,000), fixed over all four months (pool of money receivers: drain beneficiaries,
mules, cash-outs; pool of operators: bot accounts, QR and collect receivers, bait senders, refund
abusers, collusion payers). Episodes per account fit uniform draws with replacement; account
pairs sharing two events match chance (bot 44 against 50.3 +- 8.3; drain 0 against 0.3): no
rings in the sense of repeated groups. 35 percent of October's operator-pool accounts and 20
percent of the receiver pool appeared in one earlier month. Row-level lift looks large (P2P rows
to a known receiver: 43 to 45 percent fraud against 4.9 base; with new payee and whole rupees not
a multiple of 10, 87 percent). City, bank, KYC, OS and age carry nothing (hub-city lift 1.02 to
1.08; the only city signal is device city differing from home, 93 to 96 percent on takeover
phones).

**Measurement (earlier).**
- Entity-history features on the probe (10 features): macro-F1 0.4713 -> 0.4675 (time split),
  0.4666 (leave-own-month-out), 0.4676 (leave-own-episode-out): no gain; the relational features
  already find pattern rows (recall 70.3 percent before, 70.0 to 70.6 after).
- Wave 2: could pools reach the ordinary rows? Every pool type (receivers, merchants, senders,
  devices; every threshold; label history from other months or label-free same-month marked
  rows) reaches ordinary fraud at the legit rate: largest lift of any rule with 20+ ordinary rows
  is 1.29 (P2P receiver with a marked fraud row); entities that had an ordinary row before
  receive them at lift 0.82 to 1.03. Among rows the model does not already see, a pool-flagged row
  is fraud 1.4 to 4.1 percent of the time against a 2.3 percent base. The only precise cell
  (43 to 49 percent fraud) is rows that already carry generator evidence.
- Pool features on the October-like month (fixed counter), baseline 0.53963: + receiver pools
  +0.0002 (LOMO +0.0002 given and cleaned, -0.0001 generator); + receiver pools + extended pools
  (sender, device, merchant history; same-month marked-row counts) -0.0015 (generator -0.0035).
- Direct pool rules (give a model-legit row the top fraud type when flagged):

| rule | rows changed per 100k | Oct-like cleaned | LOMO cleaned |
|---|---|---|---|
| P2P, 1+ marked row, no gate | 572 | -0.0266 | -0.0287 |
| P2P, 3+ marked rows, no gate | 46 | -0.0029 | -0.0029 |
| P2P fraud share >= 0.5, no gate | 939 | -0.0377 | -0.0422 |
| sender fraud share >= 0.5, no gate | 1,328 | -0.0598 | -0.0673 |
| device fraud share >= 0.5, no gate | 1,203 | -0.0555 | -0.0625 |
| P2P label-free same-month marked, no gate | 237 | -0.0134 | -0.0220 |
| P2P 1+ marked, gated at fraud score 0.03 | 98 | -0.0056 | -0.0055 |
| P2P 1+ marked, gated at 0.10 | 3.0 | +0.0001 | +0.0001 |
| best (sender or device share, gate 0.10) | 1.1 | +0.0001 | +0.0001 |

Every rule loses on generator-only rows. Merchant pools hold every large merchant, so the flag
covers a third of all rows at lift 0.99.

**Verdict.** Tie (features) and loses (rules). **Lesson.** Pool membership is one more field the
fraud pass did not change on its ordinary rows; any precision it shows comes from rows that
already have marks. Keep a pool feature only as a backup against pattern shift, built label-free.
See [ordinary rows](ordinary-rows.md).

## 10. Repeat payers (refund_abuse and loan_app_extortion)

**Finding (earlier).** Refund abuse: one abuser sends 3 to 8 REFUND rows to 1 or 2 ecommerce or
electronics merchants within a 10-day window (amounts log-normal, median 2,263 to 2,327); variant:
round-set merchant events. Loan: 3 to 6 PAY rows to different lending merchants, 20:00 to 23:59,
1,500 to 15,000, always new payee, within 7 days; a P2P variant is 8 to 10 percent of events.
Ordinary rows are 59.9 percent of refund and 67.3 percent of loan labels (no pattern account;
clustering equals random legit). Per-row noise inside events: 84.7 to 84.8 percent own label,
12.4 legit, 2.8 other type. October: no displacement (refund events 5.9 rows, loan hours 20 to 23
only, amounts not scaled by the festival factor).

**Measurement (earlier).** Rules: refund rule precision 0.82 to 0.87, covering 97 to 98 percent of
labelled REFUND rows; loan rule 0.82 to 0.87; on September the rules alone reach refund F1 0.550
(probe 0.521) and loan 0.408 (probe 0.364); overriding the probe lifts macro-F1 0.4713 -> 0.4772.
Adding 29 event features to the probe: refund F1 +0.021 / +0.003 / +0.037, loan +0.047 / +0.032 /
+0.018, collusion +0.036 / +0.007 / +0.014, but macro-F1 +0.006 / -0.000 / +0.000 over three
seeds (a tie; the probe stops after 16 to 22 rounds). Setting 22,844 ordinary rows to legit
before training does not help (0.4675 against 0.4759).

**Verdict.** Rules kept as overrides; features tie. **Lesson.** The 60 to 67 percent ordinary rows
cap these classes at F1 about 0.55 and 0.41 on the noisy labels; cleaning them out of the
training labels brings nothing.

## 11. Rules-only detector prototype and detector fixes

**Idea.** Compose the class templates into 17 prioritised rules with no training.

**Measurement (earlier).** September macro-F1: rules 0.476 on raw labels (the first 234-feature
model got 0.463, 0.471 tuned); rules plus zones 0.502. Pooled over three train months: 0.469
rules, 0.497 with zones; on event-cleaned labels 0.515 and 0.543 (this cleaning uses the
detector's own events, so it flatters it). Per class (rules plus zones, cleaned, pooled):
precision 0.93 and recall 0.405 on average; recall, not precision, limits every class; weakest
are cashback 0.383, loan 0.408, collusion 0.399. Letting a rule see the whole split rather than
one month adds about 0.003. Zone boundary estimator agrees with label-fitted zones on 99.0
percent of train block rows; at test scale 94.5 percent (62 percent for the loan zone, 75 percent
for collusion). October predicted rows against a sixth of train: bot 0.48, collusion 0.65, others
0.91 to 1.23.

A second prototype (E1 to E5): pure rules 0.4372 on raw September; adding a hard device-zone
assignment cost -0.0032; a relational GBDT 0.4424 -> 0.4648 with 30 detector features (+0.0224;
cleaned 0.4889); plus episode smoothing 0.4675 and multipliers 0.4712 (cleaned 0.4955). A repaired
target scored 0.5155 cleaned but only 0.4336 on raw labels: training on repaired labels
over-calls fraud against the noisy floor.

**Detector fix.** The window code assumed nanosecond timestamps while the data library parsed
microseconds, so every time window and lag in the detectors was 1,000 times too short; also the
class order was aligned to the statement. Every detector number produced before the fix is
invalid.

**Verdict.** Rules alone equal the first model; zones add 0.028. **Lesson.** Check units of every
timestamp before trusting window features; recall limits every class; the structure-only 0.57
estimate is not reached by rules alone.

## 12. Parameter sweep and cascade settings

**Parameter sweep around the early champion (earlier).** Tree settings per stage, the fraud-score
cut for the type model, the drop-legit threshold, and blends of settings. Screen with two seeds,
then nested selection (176 pairs), then six-seed confirmation with an event bootstrap.
- Every tree setting ties (within +-3e-4 with no sign that holds across months); type colsample
  0.8 is worse. Dropping no legit rows loses -8e-4 in September (confirms the rule); 0.6 and 0.8
  tie with 0.7.
- One recipe change counts: lowering the type-model score cut from 0.10 to 0.03 to 0.04 (431 more
  type rows, 22,744 against 22,313). The nested rule picked 0.04 in every fold (held-out gain
  +2.1e-4 to +3.8e-4). Full recipe: given 0.5134 / 0.5096 / 0.5158 against 0.5130 / 0.5095 /
  0.5157, cleaned 0.5646 / 0.5638 / 0.5713 against 0.5642 / 0.5637 / 0.5712; pooled +1.7e-4
  given [-0.1, +3.5], +2.3e-4 cleaned [+0.5, +4.1]; 57 train calls differ (candidate right on 37,
  old right on 15); October changes 16 of 260,582 rows, too few for any public score to resolve.
- Blending different tree settings instead of seeds ties.

**Sweep ranking (37 full-data runs, earlier).** Baseline: relational plus all seven families.
Four seeds of the same baseline scored 0.4972 / 0.4910 / 0.4891 / 0.4946 on the October-like
month (sd 0.0036), so a single-seed difference has sd about 0.005, while LOMO cleaned is about
seven times steadier (seed sd 0.0005). Only clear signals: dropping the rings family -0.0067
LOMO [-0.0089, -0.0048]; dropping the device family -0.0012 LOMO and -0.0054 October-like;
dropping money flow -0.0013 and -0.0029; 150 rounds -0.0014; guarded self-training with two
rounds +0.0011 LOMO (not confirmed October-like, +0.0007 [-0.0025, +0.0040]); one round at
threshold 0.8 hurts transfer (-0.0052). Ties: adding pools, refunds, density or detectors (each
-0.0009 to +0.0008), all families, LightGBM (+0.0008 / +0.0006), depth 6 or 10, raw or reweighted
ordinary rows, October-shaped copies (+0.0014 / +0.0016, sd about 0.005), dropping bots, repeat,
drain, flow, marks, collect, sender, receiver, pair or row (-0.0013 to +0.0002 LOMO). Keeping
ordinary rows as fraud is the worst target October-like (-0.0047 [-0.0085, -0.0010]).

**Cascade hyperparameter search (160 trials, earlier).** 24 settings (per-stage depth, rate,
rounds, child weight, sampling, regularisation, bins; class weights; fraud weight; drop and floor
thresholds) over three seeds each, best-guess sampler with pruning. Best trial +0.0013 cleaned on
the tuning month (90th percentile +0.0010, median +0.0004, 83 of 130 above baseline, worst
-0.0301); top-5 re-runs: three fresh default thinnings +0.0003 to +0.0011 (pooled lower bound
above 0 for two), but in the family-protected thinning pooled +0.0002 [-0.0001, +0.0005], and on
LOMO +0.0001 to +0.0003 with a small August loss of 0.00003 to 0.00034. Importance was driven by
the type stage's minimum child weight, the only cliff (mean delta -0.0024 at 8 to 16, -0.0167 at
16 to 50; flat from 1 to 4); fraud weight 0.9 to 2 is flat. A variant of the type-cut recipe with
one counter feature dropped: +0.0001 [-0.0002, +0.0003] family-protected, -0.0001 default,
generator-only +0.0010 [+0.0004, +0.0016].

**Verdict.** Tie on every frame that matters. **Lesson.** Tree and tuning settings are exhausted;
the small default-thinning gain is the same ring-cut artefact as the October-shaped copies. Keep
the type stage's minimum child weight at about 5 or below; tune on the family-protected frame in
future. See [ensembles](ensembles.md).

## 13. Validation and decoding rules for this layer

**Measurements (earlier).**
- Folds: forward and leave-one-month-out rank six variants alike (Spearman 0.98); cutting events
  across folds leaks (+0.006 noisy / +0.015 clean for a random third of rows).
- Cleaner target: variants rank identically on noisy, cleaned and generator-only labels (Spearman
  1.00 in four folds); a gain on generator-only is 1.7 times the noisy gain with 1.8 times the sd.
- Training under noise (July+August -> September, argmax, 100 rounds): dropping ordinary rows
  +0.0036 noisy; weighting them 0.2 -0.0004; relabelling them legit +0.0007; the cascade
  (fraud against legit, times an 11-type model without the ordinary rows) +0.0101 noisy,
  +0.0170 generator-only, first in all five folds. Early stopping on log-loss costs 0.003 to 0.005.
- Decoding fitted on one month and applied to the next: multipliers +0.0028 / +0.0035 / +0.0022,
  thresholds +0.0045 / +0.0056 / +0.0016 against +0.0055 to +0.0081 on the month fitted: half a
  tuned gain is optimism; on another model they lose (-0.0078). The label-free fixed point:
  +0.0047 / +0.0057 / +0.0052 on the cascade with fraud probabilities divided by 0.875. Unifying
  only the fraud type among called rows: +0.0025 to +0.0037; one label per whole group loses.
- Leaderboard noise: public-slice sd 0.010 (rows) to 0.022-0.027 (whole events); gains under about
  0.005 are invisible.

**Lesson.** The label-free fixed-point decoder and per-event type unification are the only
decoder steps that held; see [decoders](decoders.md) and [labels](labels.md).

## What the ledger says

1. Every rule family is saturated. The champion already calls every structured row as fraud and
   types it as the rule says; further rules change 0 to 5 rows in October or lose.
2. Gains came from information the data holds that the first model lacked (device-number
   counter, seven families together), not from tuning, cut-offs, pooling or extra classifiers.
3. Costly mistakes were measurement artefacts: the October-like frame breaks the counter and cuts
   ring bursts; single-seed October-like differences under about 0.007 are noise.
4. What is left (about 63 percent of every fraud label, and look-alike class choice) is not
   reachable by any label-free signal tested; see [ordinary rows](ordinary-rows.md),
   [lookalikes](lookalikes.md) and [open ideas](open-ideas.md).
