# A close look at the best file: per-class anatomy, call volume, look-alike pairs, October

**Question.** The best public file (`blend-tabmF-bq`, 0.5499354) leaves 0.030 macro-F1 to the
cleaned-label ceiling. Is any class's call volume, look-alike assignment or October count off by
enough rows to be worth +0.0005 or more on October?

**Answer (measured in this repo).** No.
- **Call volume is already F1-optimal and rigid.** Moving any class's admission cut by half or
  by 1.4x changes 0-25 train rows a month.
- **The look-alike pools already go to the class that maximises the pair's F1 sum.** The
  partner shares sit well below the switching point. The model's within-pool score does not rank
  well enough to split a pool (AUC 0.44-0.64).
- **October's call counts match its label-free evidence.** Every class's anchored rows are called
  fraud at the train rate. No fraud-pure predicate is called less often in October.
- **The one lever that never loses on leave-one-month-out fails the bar.** That lever is a lower
  qr_swap cut. October-like given is negative on all three thinning seeds.

No candidate passes. The +0.0025 the round needs is not in the decoder, the call volume or the
template assignment of this file.

All commands are `python analysis/closelook.py <command>`. Code: `analysis/closelook.py`,
`analysis/closelook_core.py`, tests in `tests/test_closelook_core.py`. Outputs:
`$RINGTRACE_OUT/closelook/<command>.json`. October is only ever counted.

## Setup (`prep`)

- **Rebuild.** The best file's recipe is `recipes/final-half.json`:
  - the five tree members set the type split;
  - TabM (3 seeds, given labels) takes 1/2 of the fraud mass and each tree 1/10;
  - then the fixed point at 0.875, the anchored template rule and `blank_merchant_qr`.

  Rebuilt from the saved arrays, it differs from the uploaded file on **0 of 260,582 October
  rows**. Leave-one-month-out (LOMO):
  - given 0.51372 / 0.50935 / 0.51500;
  - cleaned 0.56536 / 0.56439 / 0.57179;
  - gen-only 0.90311 / 0.89998 / 0.90185.

  Wall time 25 s.
- **Fast rules.** The anchored template rule and the blank-QR rule read only each row's own call.
  `prep` stores their per-row targets once. Re-applying them reproduces the recipe's calls on
  every train and October row (0 differ), so the volume and pair tests below decode a month in
  about 0.3 s.
- **A-clean proxy.** This is the cleaned labels plus the expected 3,482 hidden ordinary fraud
  rows, shared out per month in proportion to the ordinary-looking labels. It is scored as an
  exact expected confusion matrix (`closelook_core.expected_confusion`). The best file scores
  0.54620 / 0.54509 / 0.55254.

## 1. Per-class anatomy (`anatomy`)

Pooled over July, August and September (1,564,060 rows). The ceiling is the gen-only labels used
as calls, scored on cleaned labels (as in `headroom-per-class.md`). Precision, recall and the FP
split are on cleaned labels.

| class | F1 given | F1 cleaned | ceiling | gap | F1 gen-only | calls | true | precision | recall | FP: other fraud | FP: legit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mule_account | 0.433 | 0.486 | 0.518 | 0.032 | 0.882 | 3,873 | 10,778 | 0.919 | 0.330 | 280 | 33 |
| velocity_bot | 0.519 | 0.583 | 0.601 | 0.018 | 0.936 | 3,580 | 8,587 | 0.991 | 0.413 | 10 | 21 |
| layering_chain | 0.551 | 0.611 | 0.624 | 0.013 | 0.925 | 3,748 | 7,228 | 0.894 | 0.464 | 380 | 16 |
| sim_swap | 0.500 | 0.554 | 0.579 | 0.025 | 0.869 | 2,226 | 4,243 | 0.805 | 0.422 | 424 | 11 |
| account_takeover | 0.435 | 0.486 | 0.577 | **0.091** | 0.837 | 1,386 | 4,236 | 0.986 | 0.322 | 7 | 13 |
| phishing_collect | 0.490 | 0.553 | 0.605 | **0.051** | 0.877 | 3,129 | 7,128 | 0.907 | 0.398 | 254 | 38 |
| qr_swap | 0.511 | 0.576 | 0.582 | 0.006 | 0.944 | 3,253 | 7,666 | 0.967 | 0.410 | 88 | 20 |
| refund_abuse | 0.533 | 0.600 | 0.602 | 0.003 | 0.953 | 1,550 | 3,497 | 0.976 | 0.433 | 20 | 17 |
| fake_cashback | 0.397 | 0.447 | 0.541 | **0.095** | 0.755 | 1,140 | 2,603 | 0.733 | 0.321 | 300 | 4 |
| loan_app_extortion | 0.414 | 0.474 | 0.497 | 0.023 | 0.905 | 639 | 2,023 | 0.987 | 0.312 | 3 | 5 |
| merchant_collusion | 0.394 | 0.462 | 0.461 | -0.001 | 0.943 | 795 | 2,603 | 0.987 | 0.302 | 3 | 7 |
| **macro** (with legit 0.987-0.988) | 0.5136 | 0.5683 | 0.5980 | 0.030 | 0.9022 | | | | | | |

Per month (`anatomy.json`) the picture is the same. A class's precision moves by at most 0.06
between months, and recall ranges 0.23-0.48 over classes and months.

**What it shows.**
- **Legit false alarms are 4-38 rows per class in three months.** Every class's FPs are almost
  all another fraud class's rows: the look-alike partner.
- **Two separate regimes:**
  - **Volume-limited classes** (bot, account_takeover, qr, refund, loan, collusion): precision
    0.97-0.99 at recall 0.30-0.43. The calls stop where the generator rows stop. The next rows
    down are ordinary rows at the base rate (2-3 % fraud), far below the class's F/2 of about
    0.25. Calling more cannot help, and calling fewer drops true positives at 0.99 precision.
  - **Type-limited classes** (sim_swap 0.805, fake_cashback 0.733, chain 0.894, phishing 0.907,
    mule 0.919): their precision gap is the look-alike pool the template rule assigns them
    (section 3).
- **Under A-clean**, every class's true count rises by its share of the hidden rows (for
  example mule_account 3,593 to 3,814 in July). TP and calls stay put. F1 falls 0.01-0.02 per
  class and the ranking of classes does not change. The hidden rows are spread evenly over
  legit-called rows, so they lower every class's F/2 slightly but give no row a reason to move.

## 2. Per-class volume scaling (`volume`)

The lever: one admission multiplier m_k per class in the fixed point. A row's gain for class k
becomes (2 p_ik - m_k F_k) / (N_k + P_k); m < 1 calls more, m > 1 fewer. It is now an optional
`class_mult` in `ringtrace.decode.fixed_point`. Its default is the plain fixed point, and that is
tested.

Each class alone, m in {0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.15, 1.4, 2.0}, change in 1e-5
(given / cleaned / A-clean / gen-only per month):

| class | what m does | best setting |
|---|---|---|
| mule, bot, chain, sim, takeover, cashback, collusion | 0.3-1.4 change 0-4 rows a month (most settings 0), with changes of -7.9 to +6.9 cleaned; 2.0 changes up to 126 rows a month and loses up to -0.0032 cleaned | none; volume is rigid |
| loan_app_extortion | every m < 1 loses in every month (-2 to -38 cleaned, -21 to -122 gen-only); m > 1 changes nothing | none |
| refund_abuse | -6 to +12.5 cleaned, sign changes by month | none |
| phishing_collect | 0.4-0.85 gain in July (+6 to +32 cleaned) and lose gen-only or cleaned in August (0.5: cleaned -1.5, gen-only -18.1) | none (fails August) |
| **qr_swap** | **0.5: given +10.4 / +13.4 / +9.8, cleaned +8.9 / +20.0 / +13.0, A-clean +8.9 / +19.8 / +12.9, gen-only +5.2 / +10.0 / +10.2; 7 / 13 / 8 rows; 4 October rows** | the only never-loses setting; 0.4 and 0.3 lose gen-only, 0.6 is the same calls minus a few rows |

**Nested choice per class.** For each held-out month, each class's m is chosen on the other two
months (best A-clean change, kept only if neither fit month loses on any target) and scored on
the held month:
- held July: sim 1, qr 0.5, refund 1.4, cashback 0.3. Change given +0.6, cleaned -0.4, A-clean
  -0.1, gen-only **-8.9** (1e-5).
- held August: given +7.8, cleaned +17.6, A-clean +18.1, gen-only **-3.6**.
- held September: +9.8 / +13.0 / +12.9 / +10.2.
- October (chosen on July + August): sim_swap 1.4, qr_swap 0.5, refund_abuse 1.15; 7 October rows.

It fails "never loses" (gen-only in July and August). qr_swap 0.5 is chosen in every fold.

**The bar for qr_swap** (`gate`, against the best file's recipe; the full harness of
`ringtrace.final`):

| candidate | given Jul / Aug / Sep [95 % boot] | cleaned [95 % boot] | gen-only | A-clean | October-like given per seed (pooled lb) | October-like cleaned (lb) | rows train / Oct | verdict |
|---|---|---|---|---|---|---|---|---|
| qr_swap m 0.5 (`closelook-qrvol`) | +0.00010 / +0.00013 / +0.00010 [+0.00002, +0.00021] | +0.00009 / +0.00020 / +0.00013 [+0.00005, +0.00024] | +0.00005 / +0.00010 / +0.00010 | +0.00009 / +0.00020 / +0.00013 | **-0.00012 / -0.00010 / -0.00003** (-0.00015) | +0.00012 / +0.00008 / +0.00000 (-0.00003) | 28 / 4 | **FAIL** |
| qr_swap m 0.4 | +0.00003 / +0.00008 / +0.00001 [-0.00010, +0.00017] | +0.00001 / +0.00014 / +0.00022 | -0.00015 / -0.00011 / +0.00009 | +0.00001 / +0.00014 / +0.00022 | -0.00007 / -0.00018 / -0.00007 (-0.00023) | +0.00028 / +0.00009 / +0.00002 (-0.00002) | 54 / 6 | FAIL |
| qr_swap m 0.6 | +0.00005 / +0.00009 / +0.00004 [-0.00001, +0.00013] | +0.00004 / +0.00012 / +0.00008 | +0.00001 / +0.00009 / +0.00005 | +0.00004 / +0.00012 / +0.00008 | -0.00003 / -0.00003 / +0.00000 (-0.00007) | +0.00009 / -0.00004 / +0.00000 (-0.00005) | 17 / 4 | FAIL |

The qr_swap gain exists only at full density. On September thinned to October's density it
loses on given labels in every thinning seed. That is exactly the case the October-like check
exists to catch. Rejected. Its October file, built as a labelled near-miss, changes 4 rows
against `blend-tabmF-bq` (legit -> qr_swap 2, refund_abuse -> qr_swap 2):
`$RINGTRACE_OUT/candidates/closelook-qrvol.csv`, sha256
`91b918f0615a71b4435ad40aa1dd5fcd7794d8e502d35a0eab6659f943a1586e`, valid against the sample
submission.

## 3. Look-alike pairs (`pairs`)

Template rows split by whether their template event holds a counter anchor of the pair ("held":
the anchored rule sets the class from the device counter) and by event size. Train, cleaned
labels:

| template (rule class / partner) | held rows: truth rule class / partner; calls rule class / partner | unheld rows by size: partner share | October unheld rows |
|---|---|---|---|
| drain1 (sim_swap / account_takeover) | 146 rows: 117 / 29; calls 117 / 29 | size 2: 416, 0.38; size 3: 501, 0.35; size 4: 504, 0.15 | 221, all sim_swap |
| col7 (fake_cashback / phishing_collect) | 52 rows: 21 / 31; calls 22 / 30 | size 2: 484, **0.59** | 94, all fake_cashback |
| col1 (phishing_collect / fake_cashback) | 84 rows: 79 / 3; calls 81 / 3 | 1,924 rows, 0.08-0.09 | 310, all phishing |
| mf1 (layering_chain / mule_account) | 308 rows: 273 / 35; calls 273 / 35 | 2,325 rows, 0.14-0.17 | 361, all chain |
| mf2+mf3 (mule_account / velocity_bot) | 1,679 rows: 1,588 / 89; calls 1,607 / 72 | 2,007 rows, 0.12 | 422, all mule |
| rng1 (qr_swap / refund_abuse) | 73 rows: 70 / 3; calls 70 / 3 | 899 rows, 0.00-0.09 | 175, all qr_swap |

**Switching point.** Moving a pool of n rows with partner share q from class A to class B
changes F1_A + F1_B at first order by
(2n / D_B)(q - F_B / 2) - (2n / D_A)((1 - q) - F_A / 2), with D = true + calls. The pool should
move when

q > q* = ((1 - F_A / 2) / D_A + F_B / (2 D_B)) / (1 / D_A + 1 / D_B).

Pooled counts from section 1:

| template | q* | partner share | margin |
|---|---|---|---|
| drain1 -> account_takeover | 0.466 | 0.15-0.38 | keep sim_swap |
| col7 -> phishing_collect | 0.643 | 0.59 (0.55 / 0.62 / 0.60 by month) | keep fake_cashback (the smaller, lower-F1 class wins the coin flip) |
| col1 -> fake_cashback | 0.357 | 0.08-0.09 | keep phishing |
| mf1 -> mule_account | 0.501 | 0.14-0.17 | keep chain |
| mf2+mf3 -> velocity_bot | 0.503 | 0.12 | keep mule |
| rng1 -> refund_abuse | 0.430 | 0.00-0.09 | keep qr_swap |

The decoder and rule are already at the pair optimum for every pool. Upload L7 tested `col7`
-> phishing on October: -0.0078 public from 94 rows (`ledger.md`). It confirms the col7 call,
the closest one.

**Splitting a pool by the model.** Within the unheld pools, the blend's partner score
p_B / (p_A + p_B), averaged over the template event, has AUC 0.61 / 0.57 / 0.61 (drain1),
0.61 / 0.63 / 0.64 (mf1), 0.56 / 0.58 / 0.59 (col1), 0.51 / 0.52 / 0.45 (col7), 0.31 / 0.62 /
0.62 (mf2+mf3) and 0.57 / 0.90 / 0.62 (rng1) per month. Giving the top 2-20 % of a pool's events
to the partner class loses in almost every month and template. The exceptions are single months
of drain1 at the top 2 % (+12.8 / +23.9 / -14.2 cleaned, 1e-5) and of col7. The nested choice
picks "no split" in every fold for mf1, mf2+mf3, col1 and rng1. For drain1 it picks the top 2 %
in two folds, for a held-month total of -10.4 given / -1.4 cleaned (1e-5). The coin-flip part of
each pair is a coin flip for this model, and the assignment is already the best fixed one.

## 4. October aggregates (`october`, `scan`)

Calls per 100,000 rows (October holds 260,582 rows):

| class | Jul / Aug / Sep | October | October calls vs September-scaled | fraud mass per 100k, Sep / Oct | label-free anchors per 100k, Sep / Oct (share called fraud in October) |
|---|---|---|---|---|---|
| mule_account | 246 / 237 / 261 | 303 | 789 vs 680 (**+109**) | 354 / 380 | 247 / 282 (1.000) |
| velocity_bot | 257 / 209 / 220 | 122 | 318 vs 574 (**-256**) | 325 / 244 | 209 / 103 (1.000) |
| layering_chain | 244 / 247 / 228 | 213 | 556 vs 594 | 284 / 283 | 222 / 213 (1.000) |
| sim_swap | 154 / 143 / 130 | 142 | 371 vs 338 | 142 / 160 | 105 / 111 (1.000) |
| account_takeover | 99 / 82 / 86 | 88 | 230 vs 223 | 151 / 174 | 103 / 111 (1.000) |
| phishing_collect | 213 / 199 / 188 | 193 | 502 vs 489 | 414 / 425 | 212 / 214 (1.000) |
| qr_swap | 210 / 213 / 201 | 203 | 530 vs 523 | 493 / 491 | 191 / 190 (1.000) |
| refund_abuse | 80 / 99 / 119 | 110 | 286 vs 310 | 316 / 321 | 115 / 101 (1.000) |
| fake_cashback | 72 / 66 / 81 | 79 | 207 vs 210 | 93 / 101 | 38 / 40 (1.000) |
| loan_app_extortion | 40 / 40 / 42 | 38 | 100 vs 110 | 93 / 126 | 40 / 38 (1.000) |
| merchant_collusion | 34 / 59 / 59 | 32 | 84 vs 154 (**-70**) | 97 / 90 | 55 / 27 (1.000) |

**Flags.** Three classes' October volume is off the September scale: velocity_bot (-256 calls),
merchant_collusion (-70) and mule_account (+109).
- **What it would cost if these were misses.** From `public-score-noise.md`, one correct public
  row is worth 0.00023 (bot), 0.00078 (collusion) and 0.00016 (mule). If the shortfalls were
  generator rows the model missed, they would cost about 256 x 0.3 x 0.00023 = 0.018 (bot) and
  70 x 0.3 x 0.00078 = 0.016 (collusion) of public macro-F1. That is far more than any lever
  measured tonight.
- **They are not misses.**
  - The label-free evidence behind each class moves the same way: bot anchors 209 -> 103 per
    100k, collusion 55 -> 27, mule 247 -> 282.
  - In October every anchored row of these classes is called fraud (1.000), and called its anchor
    class at the train rate (bot 1.000, collusion 1.000, mule 0.985 against 0.94-1.00).
  - So October holds fewer bot bursts and collusion rings and more mule bursts. This is the pool
    change already recorded in `docs/knowledge/data/october.md` ("bursts of bots and collusion
    rings roughly halved, mule bursts up"). The calls follow the evidence, and nothing is left to
    recover.
- **October priors from the probabilities** (fraud mass per 100k) agree with the train months
  within the month-to-month spread. The exceptions are velocity_bot (244 against 288-376, fewer
  bursts) and loan_app_extortion (126 against 93-113). Loan's extra mass is not reflected in its
  calls or anchors (38 per 100k). It sits below the cut-off, and lowering loan's cut-off loses in
  every train month (section 2), so there is no lever.
- **Blank-QR-like pockets** (`scan`). We checked 1,318 label-free predicates:
  - one or two of: transaction and receiver type, app, KYC levels, OS, the merchant-category null
    state, the QR/VPA check state, new payee, PIN retries, whole amounts, the family shapes, the
    marks, the counter block and the template;
  - each with cleaned fraud rate >= 0.8 on 30 or more train rows and 5 or more October rows.

  **None** has an October fraud-call share even 0.05 below its train share. Ranked by drop x
  October rows, the top groups lose 0.001-0.006 of their share: for example 134 COLLECT rows with
  a QR/VPA check of 0, 0.999 in train against 0.993 in October. The blank-QR rule's October case
  was the only pocket of its kind.

## What this leaves

- The best file has no lever in its decoder, its per-class volume, its pair assignment or its
  October counts.
  - The remaining cleaned gap to the ceiling (0.030) is the look-alike pools: account_takeover,
    fake_cashback and phishing_collect hold 0.237 of the 0.36 per-class gap sum. These are coin
    flips for every model measured.
  - The rest of the loss to 1.0 is ordinary rows, which no score ranks.
- A +0.0025 October gain would need about 10 more correct rows of a small class on the public
  slice, or 35 on the whole of October. Every measured lever moves 0-7 October rows, most of them
  at about 50 % precision.
- **Recommendation: keep the final selection between `blend-tabmF-bq` and `final.csv`.**
  - They differ on 2 rows. `final.csv` passes the bar against the blend; `blend-tabmF-bq` misses
    gen-only September by one row.
  - `closelook-qrvol` is a near-miss, not an upload. It fails October-like given on every seed,
    and its 4 changed rows are inside public noise (1 public row of qr_swap is 0.00022).

**Commands.** `prep` (25 s), `anatomy` (10 s), `volume` (about 4 min), `pairs` (1 min),
`october` (20 s), `scan` (40 s), `gate recipes/cand/closelook-qrvol*.json` (8.7 min; October-like
218 s), `build recipes/cand/closelook-qrvol.json` (30 s); laptop CPU, 8 threads.
