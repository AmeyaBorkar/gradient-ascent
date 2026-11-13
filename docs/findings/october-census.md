# The October census: expected against actual calls per label-free unit

**Question.** Where do October's calls in the blend with the blank-check merchant QR rule fall
short of, or exceed, what the out-of-fold behaviour predicts? Which pockets of fraud-like
structure are called legit in October, like the 20 blank-check merchant QR rows?

**Method (measured in this repo; October as aggregates only).** `analysis/census.py` rebuilds
the blend's out-of-fold calls from its five member runs (its own October decode matches the
submission file on all rows). Each census unit is a label-free mask: anchor class, look-alike
template, event-family shape, counter block, segment and confidence band, mark, payment kind.
For each unit and class, the train rows give the out-of-fold call rate and the cleaned-label
rate. The expected October calls are that rate times the unit's October rows. The surprise is
z = (actual - expected) / sqrt(expected). October is half a month: its row count is 0.49x a train
month.

## 1. Coarse units (95 units, 1,183 unit x class cells)

Largest deficits and surpluses, |gap| >= 3 rows. "prec" is the out-of-fold precision of the
call on cleaned labels.

| unit | class | train rows | Oct rows | oof % | cleaned % | prec | expected | actual | z |
|---|---|---|---|---|---|---|---|---|---|
| PAY to merchant | velocity_bot | 380,480 | 62,698 | 0.89 | 1.19 | 0.998 | 557.1 | 278 | -11.83 |
| PAY to merchant | any fraud | 380,480 | 62,698 | 1.37 | 3.57 | 0.994 | 859.2 | 543 | -10.79 |
| mrk_odd | velocity_bot | 8,156 | 1,418 | 15.79 | 16.87 | 0.997 | 223.9 | 97 | -8.48 |
| counter block 0 | merchant_collusion | 1,512,310 | 252,287 | 0.05 | 0.17 | 0.996 | 121.1 | 70 | -4.64 |
| mrk_bal110 | merchant_collusion | 2,360 | 397 | 10.51 | 10.51 | 0.996 | 41.7 | 15 | -4.14 |
| flows event | layering_chain | 7,349 | 1,291 | 49.35 | 44.21 | 0.896 | 637.2 | 537 | -3.97 |
| merchant-ring event | merchant_collusion | 4,224 | 641 | 17.66 | 18.37 | 0.996 | 113.2 | 71 | -3.97 |
| blank-check merchant QR | merchant_collusion | 678 | 144 | 35.99 | 35.99 | 1.000 | 51.8 | 26 | -3.59 |
| mrk_spill | mule_account | 147 | 46 | 19.73 | 19.73 | 1.000 | 9.1 | 0 | -3.01 |
| collect event | legit | 5,115 | 814 | 21.86 | 21.09 | 0.948 | 177.9 | 145 | -2.47 |
| counter block 2, conf >= 0.9 | fake_cashback | 3,178 | 522 | 2.14 | 1.95 | 0.868 | 11.2 | 3 | -2.44 |
| mrk_odd | **qr_swap** (surplus) | 8,156 | 1,418 | 5.92 | 5.68 | 0.940 | 84.0 | 255 | +18.66 |
| counter block 2, mule segment | velocity_bot (surplus) | 358 | 80 | 1.68 | 2.23 | 1.000 | 1.3 | 10 | +7.48 |
| PAY to person | mule_account (surplus) | 516,022 | 86,036 | 0.72 | 1.17 | 0.922 | 620.6 | 761 | +5.64 |
| template mf1 (chain hop) | mule_account (surplus) | 2,633 | 431 | 1.33 | 15.46 | 1.000 | 5.7 | 18 | +5.13 |
| flows event | mule_account (surplus) | 7,349 | 1,291 | 49.64 | 51.04 | 0.924 | 640.8 | 743 | +4.04 |
| QR to merchant | refund_abuse (surplus) | 553,283 | 92,584 | 0.03 | 0.16 | 0.946 | 31.1 | 55 | +4.28 |

**Fraud calls inside the structure.** Every template, event-family shape and counter zone
gets exactly its expected number of fraud calls in October; |z| <= 0.07 on every any-fraud cell.
For example, flows events: 1,291 expected and 1,291 actual; counter block 2: 563.1 against 562.
No unit with a cleaned fraud rate >= 90 % has more than 2 October rows called legit. The largest
are counter block 2 (2 rows, 0.9 expected) and mrk_odd (2 rows, 0.3 expected).

## 2. The coarse deficits are October's event mix

October's event counts per family shape, against the mean train month (expected ratio 0.49):

| unit | Jul | Aug | Sep | Oct | Oct / month |
|---|---|---|---|---|---|
| burst events (velocity_bot) | 1,307 | 968 | 1,066 | 269 | 0.24 |
| merchant-ring shape 3 (collusion) | 136 | 254 | 266 | 71 | 0.33 |
| merchant-ring events | 1,354 | 1,496 | 1,374 | 641 | 0.46 |
| flows shape 4 (chain) | 405 | 338 | 286 | 124 | 0.36 |
| flows shape 2 (mule) | 1,067 | 1,086 | 1,123 | 656 | 0.60 |
| blank-check merchant QR | 223 | 238 | 217 | 144 | 0.64 |
| QR to person | 763 | 748 | 688 | 319 | 0.44 |

October holds half as many bot bursts as a half-month would, fewer collusion rings and chain
flows, and more mule flows. A coarse unit such as "PAY to merchant" or "flows event" mixes these
shapes, so its expected calls carry train's mix. This is the known "few events" shape of
`fresh-october-profile.md` and the per-class census ratios of the earlier measurement (bot 0.56,
collusion 0.63, mule 1.17).

The qr_swap surplus in mrk_odd is the QR-to-person blank-check pocket. 140 of the 255 October
calls are QR to a person; train has 0 such calls, and `fresh-blank-merchant-qr.md` already
explains them.

## 3. Composition-controlled census (fine cells)

Each row is assigned to one cell of anchor x template x first family shape x counter block (with
the segment on fraud-block phones) x payment kind x blank QR check x any mark. That gives 236
cells; cells with fewer than 20 train rows are listed separately.

| class | expected | actual | ratio |
|---|---|---|---|
| legit | 256,610.8 | 256,611 | 1.000 |
| mule_account | 776.1 | 781 | 1.006 |
| velocity_bot | 311.6 | 317 | 1.017 |
| layering_chain | 553.8 | 543 | 0.981 |
| sim_swap | 363.2 | 365 | 1.005 |
| account_takeover | 223.9 | 223 | 0.996 |
| phishing_collect | 485.6 | 481 | 0.990 |
| qr_swap | 369.2 | 368 | 0.997 |
| refund_abuse | 250.6 | 252 | 1.006 |
| fake_cashback | 185.1 | 189 | 1.021 |
| loan_app_extortion | 94.0 | 94 | 1.000 |
| merchant_collusion | 70.0 | 70 | 1.000 |

Once the event mix is held fixed, every class is within 0.98-1.02x of expectation. The
`fake_cashback` 1.97x of the earlier census is a template choice measured against a rule
census, and here it is 1.02x. The largest remaining cell gaps (|gap| >= 2 rows):

| cell | class | train | Oct | oof % | cleaned % | prec | expected | actual | z |
|---|---|---|---|---|---|---|---|---|---|
| mf1 template, PAY to person, block 0 | mule_account | 1,632 | 279 | 0.61 | 15.38 | 1.000 | 1.7 | 10 | +6.34 |
| no anchor, block 2 mule segment, PAY to merchant | velocity_bot | 55 | 10 | 3.64 | 3.64 | 1.000 | 0.4 | 4 | +3.64 |
| no anchor, block 2 mule segment, QR to merchant | velocity_bot | 80 | 24 | 3.75 | 3.75 | 1.000 | 0.9 | 4 | +3.10 |
| no anchor, block 2 mule segment, PAY to merchant | mule_account | 55 | 10 | 96.36 | 78.18 | 0.811 | 9.6 | 6 | -1.17 |
| mf2 template, PAY to person, mark | velocity_bot | 1,558 | 314 | 2.05 | 9.37 | 0.875 | 6.4 | 4 | -0.96 |

No cell has a deficit beyond |z| 1.2. The mf1 surplus moves towards the cleaned rate (15 % of
chain-hop rows are mule in train), so it is not a loss.

**October rows in cells that train lacks (< 20 train rows): 288.** They are called
5 / 8 / 1 / 13 / 5 / 7 / 17 / 162 / 34 / 18 / 5 / 13 in class order (legit first).

| cell | train rows (cleaned) | Oct rows | Oct calls |
|---|---|---|---|
| QR to person, blank check, merchant-ring shape 4, qr anchor | 0 | 136 | 136 qr_swap |
| QR to merchant, blank check, **no anchor, template or family event** | 0 | 20 | 13 qr_swap, 7 refund_abuse (legit before the rule) |
| QR to person, blank check, merchant-ring shape 4, block 2 | 0 | 4 | 4 qr_swap |
| 17 further cells, 1-19 train rows each | 228 | 128 | 5 legit; every other row takes the cell's majority train class |

**The blank-check pocket, explained.** All 678 train blank-check merchant QR rows sit inside an
anchor, a template or a family event. In October, 21 of the 144 rows sit in none of them, and 20
of those are on block-0 phones. With no structure, the model's legit call follows the
non-structure feature mix. That is the only structural novelty in October, and the baseline rule
already covers it.

## Verdict

No unexplained deficit, so no fix.
- Every deficit in a coarse unit is October's event mix, which section 2 shows with per-month
  counts.
- Inside the fine cells, no class departs by more than 2 % and no cell by more than |z| 1.2.
- The only October rows that the structure marks as fraud-like but that were called legit are
  the 20 blank-check merchant QR rows that the baseline already rescues.
- The 5 legit calls in thin cells sit in cells with at most 19 train rows, too few to show
  97 % precision. The one listed, counter block 2 with no anchor (PAY to person), is 11 of 13 fraud on cleaned labels (85 %).

No candidate file was written. For the signature sweep: the units checked here are every
`anchor=*`, `tmpl=*`, `<family>:shape<k>`, `ctr:block<b>[:seg*|:conf*]`, `mrk_*` and `pay:*`
unit in `analysis/census.py`. None holds a legit-called pocket beyond the blank-check merchant
QR rows.

**Commands.** `python analysis/census.py table --sub <blend-blankqr.csv>`;
`python analysis/census.py cells --sub <blend-blankqr.csv>` (laptop, 8 threads, 21 s and 39 s;
writes `$RINGTRACE_OUT/census/table.json` and `cells.json`).
