# The device-number counter, its zones, and the bot pool

**Summary.** Device ids are `DV` plus a 7-digit number. Numbers up to 419999 are account home phones.
**Numbers from 420000 up come from one global counter** that the generator advances as it writes rows, so
the number records which pass and which class routine wrote the row. There are five blocks: a bot pool
(420000 to 422999), a train fraud block, a train legit block, a test fraud block, a test legit block.
Inside a fraud block the classes come in a fixed order, so **position gives the class** (85.2% of train
fraud-block rows carry their zone's class; the rest is label noise). This is the single most valuable
non-row signal: it is worth about +0.028 macro-F1 in this repository.

Back to the [index](index.md). Related: [generator](generator.md),
[ordinary rows](ordinary-rows.md), [classes and templates](classes-and-templates.md).

Status: the block map and zone table are an **earlier measurement**; items marked "measured in this
repo" come from this repository's runs.

## 1. The blocks

| Block | Device numbers | Rows | Contents |
|---|---|---:|---|
| home pool | 1 to 419999 | 1,767,992 | account home phones (device = account on 92% of legit rows); 3.41% fraud-labelled |
| bot pool | 420000 to 422999 | 3,421 (2,959 train, 462 test) | reserved for `velocity_bot`; 542 devices used; every row on such a device is in a burst; 84.6% `velocity_bot` (about 96% before noise) |
| train fraud pass | 423000 to 428454 | 3,232 rows on 2,240 devices | classes in the label-list order; 68.4% fraud-labelled |
| train legit pass | 428455 to 473631 | 45,177 | every id used exactly once; 99.93% `legit` |
| test fraud pass | 473632 to 474551 | 564 rows on 410 devices | October fraud routines |
| test legit pass | 474552 to 482013 | 7,462 | one row per id |

- Any base row gets a new single-use device with probability 0.03, in both passes. That is why ordinary
  fraud rows occur in the fraud block ([ordinary rows](ordinary-rows.md)).
- Block edges are visible without labels: id occupancy is about 41 to 45% inside the fraud block (mean
  gap 2.24; about 60% of ids unused, cause not established) and exactly 100% in the legit block with every row on
  a single-use device. The boundary between the fraud and legit blocks is identifiable without labels.
- Bot, train-fraud and test-fraud numbers are **never reused across splits**: no extra device is used
  in two months (2 of 55,377). Raw device numbers shift between splits, so use the device kind and the
  zone, never the number.
- Fraud-block devices per 100,000 rows: 349 (train) and 353 (test) ids; rows on them 216 / 205 / 199 in
  July to September and 216 in October.

## 2. The zones (class order inside the fraud block)

Train zone starts, in the order the classes were written: `mule_account` 423000, `velocity_bot` 423640,
`layering_chain` 424125, `sim_swap` 424600, `account_takeover` 425040, `phishing_collect` 426750,
`qr_swap` 427140, `refund_abuse` 427800, `fake_cashback` 428030, `loan_app_extortion` 428160,
`merchant_collusion` 428290.

Rows per device and label precision by phase position (position = (number - block start) / block length;
precision July / August / September):

| Phase | Position | Rows per device | Label precision |
|---|---|---:|---|
| `mule_account` | 0.000 to 0.121 | 1.0 | 0.87 / 0.79 / 0.87 |
| `velocity_bot` | 0.122 to 0.209 | 1.0 | 0.83 / 0.89 / 0.86 |
| `layering_chain` | 0.211 to 0.297 | 1.0 | 0.88 / 0.86 / 0.83 |
| `sim_swap` | 0.297 to 0.374 | 2.0 | 0.82 / 0.87 / 0.77 |
| `account_takeover` | 0.374 to 0.693 | 2.5 | 0.85 / 0.86 / 0.85 |
| `phishing_collect` | 0.693 to 0.770 | 1.0 | 0.87 / 0.81 / 0.86 |
| `qr_swap` | 0.770 to 0.891 | 1.0 | 0.72 / 0.79 / 0.80 |
| refund, cashback, loan app, collusion | 0.891 to 1.000 | 1.0 | 0.59 to 1.00 |

- Position gives the class: 85.2% of train fraud-block rows carry their zone's class (84.9 / 85.6 / 85.1%
  by month), 11.6% say `legit`, 3.2% another type. That is exactly the train noise, so under clean labels
  the zone is close to exact (belief).
- Three zone peculiarities: the `sim_swap` phase uses about 0.41 skipped numbers per phone and the
  takeover phase 2.81, which is how the two are told apart; the `sim_swap` segment is 162 multi-row phones
  (96.7% of labels agree), the takeover segment 361 (99.5%).
- **An earlier map ordered the zones differently** (it put `sim_swap` first, with the single-row classes
  last); the later, finer map above lists the mule block first and was checked against label anchors. If a new run finds a disagreement, re-derive the order from label-free markers; the block edges are
  robust, the class order inside is the part to verify.

## 3. What the zones buy

1. **`sim_swap` against `account_takeover` for device-template events.** 147 of 148 `sim_swap`-labelled
   device events lie in the `sim_swap` zone and 336 of 337 `account_takeover`-labelled ones in the other.
   No row column can make this split (both kinds of phone are new, device age under 2 days). Cut accuracy
   on fresh phones is 95.8 / 95.0 / 94.6%, against "always takeover" at 65.0 / 62.3 / 69.3%.
2. **Ordinary rows on new devices.** About 200 to 320 October rows on single-use fraud-block devices have
   no template mark; nothing else reaches them, about 3% of the ordinary rows.
3. **Ground truth without labels.** Fraud-block rows against legit-block rows is a labelled comparison
   that needs no train labels (used for the noise rates and the separability experiment).
4. **Event propagation.** A single zone row names its event: 40.2% of mule fan-ins have one (label right
   89.9% against 78.0% for "always mule"), chains 11.6% (98.2 against 85.7), own-phone drains 7.5% (100
   against 72.0), bait pairs 6.5% (94.7 against 50.0), round-amount merchant pairs 6.8% (100 against 56.0).
5. Rows on single-use devices in the legit block are legit (about 2,864 per 100,000 October rows; 99.93%
   `legit` in train).

Effect sizes:
- A probe model with just the zone label of the row's own phone plus the bot-pool flag rose from 0.4627
  to 0.4833 argmax (0.4713 to 0.4960 tuned) on September; `sim_swap` F1 went 0.362 to 0.455. A later,
  fuller counter run lifts the leaderboard-style score by about 0.021 to 0.025.
- Overriding a probe on fraud-block rows by decoded segment raised September macro-F1 from 0.471 to 0.487.
- **Measured in this repo** (`python -m ringtrace.build recipes/champion.json`, `RESULTS.md` runs 4, 10,
  12): removing the counter features costs **-0.028** at the fixed point on both given and cleaned labels
  (0.5122 to 0.4846 given; 0.5665 to 0.5388 cleaned, rule still on). The counter plus the template rule
  together are worth +0.0333 given [+0.0310, +0.0358] and +0.0343 cleaned [+0.0319, +0.0369] in a paired
  event bootstrap (`python -m ringtrace.compare`). The template rule alone is +0.0058 given / +0.0064
  cleaned over the fixed point (773 train and 137 test rows changed).

## 4. Locating zones in October (cautions)

- **Locate zones per split, from label-free markers.** Train zone fractions do not transfer: October
  markers sit about 0.016 later in relative position. In test the `sim_swap` / `account_takeover`
  boundary is the gap at 473991 to 473996; `sim_swap` spans 473922 to 473990 and `account_takeover` 473997
  to 474279. COLLECT rows start near 474280, QR rows with a mismatch flag near 474360, REFUND rows near
  474440. The last three zones are only about 20 ids each, so their boundaries are uncertain. Cutting the
  zone at train positions instead of fitting to anchors misplaces 19% of the October zone devices.
- Segment fitting by dynamic programming on anchors keeps 91.8% plus or minus 1.8% of labels on
  October-sized train subsamples (94.6% on the full zone).
- **Two misplaced cuts in October** (mule/bot and refund/cashback): fraud-block calls agree with the
  segment on 95.2% of October rows against 98.6 to 99.0% in train; the refund segment holds 11.1 rows per
  100k (train 5.7 to 7.8), cashback 1.9 (4.6 to 5.5), mule 30.7 (20.4 to 24.5). Template rows are not
  affected (single-class templates sit 100% in their expected segment). Likely cause: the October
  segment model is fitted with about half the anchors. Do not let rules read the segment on non-template
  rows without checking.
- The October sim/takeover cut rests on only 127 devices (20 phones before the cut, 27 expected).
- Anchor-based adaptive class choice per template per month **loses** (anchors are about 3% of rows,
  7 to 50 a month per template); what pays is per event: protect anchor rows, propagate the anchor's class
  to its event, and give the template class to events without an anchor (+0.0064 cleaned in
  leave-one-month-out against +0.0032 for a fixed rule).
- **Measured in this repo:** the counter's emission table is fixed (measured unmarked shares per zone
  plus three cross terms: bot-zone marks look like mule, sim zone like takeover, cashback zone like
  phishing); only anchors and the number line are fitted per split. It reproduces the target segment
  agreement (0.8456 of fraud-block and 0.9566 of fraud-labelled train rows against 0.8447 / 0.9555).
- Thinning a month to October's density breaks the block finder (no legit block is visible), so the
  counter columns are built on the unthinned frame and joined by row; without that fix 7,745 September
  rows landed in the fraud block instead of 517.
- A column "this device number has one row in its split" differs by construction between train (three
  months, full density) and test (one month, half density): 36.8% of October rows against 8.1% of an
  October-like September. Dropping it and the raw position column is hygiene at no measured cost.

## 5. The bot pool

- Numbers 420000 to 422999 are a pool of 3,000 phones, shared across July to October, used only by
  `velocity_bot` bursts (see burst anatomy in [classes and templates](classes-and-templates.md)).
- 84 bursts in train on 81 devices (33 / 26 / 25 by month), **7 in October** where the train rate predicts
  13.8 (Poisson p = 0.035; 0.061 at the September rate) with the same shape (about 38 rows, 15 s apart,
  login about 4 s). No adapted variant was found on any scan with bot devices removed; the bot phase
  still writes ordinary rows at the train rate.
- Accounts on bots come from a ring pool: 2,044 accounts over 91 bursts, 364 in two or more bursts
  (4.6 expected if drawn from all accounts); 58 of 178 October bot accounts were in a train burst;
  capture-recapture gives a pool of 6,400 to 6,750 accounts. Outside bursts, 20 to 21% of rows they send
  and 39 to 42% of the person-to-person rows they receive carry a fraud label (others 3.3%). Fan-in
  (mule) receivers are never in this pool (0 of 289).
- Noise inside bursts: 84.6% `velocity_bot`, 12.8% `legit`, 2.6% another fraud label; row flips are
  independent. No burst is wholly unreported or filed under another type (lowest bot share 0.69).
- Rule: predict `velocity_bot` on every row of a bot-pool device and nowhere else; the unreported 12.8%
  are bot rows. Bot-pool rows were called bot 100% in every month. Rule rates per 100,000 rows July /
  August / September / October: bot-device rule 248 / 184 / 209 / 103; precision 0.850 / 0.841 / 0.846;
  coverage of the label 0.392 / 0.312 / 0.339.
- Expect `velocity_bot` near half its train rate in October (bursts 0.50 times; bot-pool rows 103 against
  184 to 248 per 100k).
- **A team decision** governed whether to use the counter at all: it is a coded rule on the provided
  data and the written rules allow "rules and features you write in code", but it exploits an id artefact
  and the top teams' code is checked. It was adopted and is part of the champion.
