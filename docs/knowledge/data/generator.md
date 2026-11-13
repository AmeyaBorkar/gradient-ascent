# How the data was generated

**Summary.** The data comes from a two-pass synthetic generator: a legit pass writes ordinary rows,
then a fraud pass writes the 11 fraud classes. Fraud rows are of two kinds: **template rows**, where a
class routine overwrites a few fields of an ordinary row (or adds a multi-row event) and leaves the
dependent fields stale, and **ordinary rows**, which the fraud pass writes with the legit template and
labels with the class. About 37% of fraud labels are template rows and 63% are ordinary. Label noise
(12% unreported, 2 to 3% wrong type) is added after generation, on train only. The stale fields are
deterministic and hold in October; they are the foundation of every label-free detector.

Back to the [index](index.md). Related: [classes and templates](classes-and-templates.md),
[ordinary rows](ordinary-rows.md), [device counter](device-counter.md), [label noise](label-noise.md).

Status: everything here is an **earlier measurement** unless a line says "measured in this repo".

## 1. Two passes, row accounting

| Split | Rows | Legit pass | Fraud pass | Notes |
|---|---:|---:|---:|---|
| train | 1,564,060 | 1,500,000 | 64,060 (4.10%) | 57,648 rows carry a fraud label (3.69%) |
| test | 260,582 | 250,000 | 10,582 (4.06%) | labels hidden |

- Test is one sixth of train: a **half-volume month**, not a random half of the rows and not a subset
  of accounts. Legit rows per day in test are 0.49 times train.
- Fraud events stay whole in October: label-free signature counts in test are 0.87 to 1.05 times
  the half-month expectation, except bot bursts (0.50). The fraud-pass share (4.06% of rows against
  4.10%) is the same.
- Events are assigned to a split by start month, so there are 147 October-dated train rows and 46
  November-dated test rows (event tails).
- Inside train and test the generator ran as one continuous simulation of July to October; there is
  no trend within July to September, and October is a step change on 1 October (see [October](october.md)).

## 2. Template events against overwritten rows

A fraud row starts as an ordinary legit row. A class routine then does one of two things:
1. **injects an episode** (a burst, a chain, a drain, a ring) of several linked rows, or
2. **overwrites** some fields of one base row (type, receiver, amount, device) **without recomputing
   the fields that depend on them**.

Roughly 62% of fraud-pass rows are episode rows and 38% are single-row overwrites, by an early
count. The later census (see [ordinary rows](ordinary-rows.md)) refines this: only 37% of fraud labels
carry anything detectable, because most overwrite-free fraud rows are plain legit-looking rows.

### The stale-field contradictions (hold in October)

In the legit pass the QR/VPA mismatch flag is filled exactly when the row is a QR payment to a
merchant, and the merchant category is filled exactly when the receiver is a merchant. Overwriting the
type or receiver breaks that. After a corrected recount, the contradictions are:

| Flag | Train rows | Fraud-labelled | Test rows | Main classes |
|---|---:|---:|---:|---|
| non-QR row with the mismatch flag set (`c1`) | 7,478 | 87.8% | 1,134 | all 11 classes |
| QR to a merchant with the flag blank (`c2`) | 678 | 88.5% | 144 | `qr_swap` 275, `merchant_collusion` 206, `refund_abuse` 106 |
| person receiver with a merchant category (`c3`) | 2,123 | 87.3% | 309 | `qr_swap` (83% of rows, 95% of fraud) |
| QR to a person with no category (`c4`) | 76 | 85.5% | 10 | `phishing_collect` (61 of 65 fraud rows), no `qr_swap` |

An early map gave different numbers for `c2` and `c4` (850 / 275 rows, wrong classes); the table is
the corrected recount. The mismatch flag on non-QR rows is 0.06% on legit rows, so the union of the
structural rules has **zero hits** in 45,177 train and 7,462 test rows that sit on single-use devices
from the legit pass.

| Other generator constants | Train rows | Test rows | What writes it |
|---|---:|---:|---|
| `balance_before` = 1.1 x amount | 2,340 to 2,360 | 397 | phishing, loan app, cashback paybacks, collusion; 86 to 87% fraud-labelled |
| amount above `balance_before` (the legit pass never produces it) | 1,889 | 297 | mule, drains, refund; 88.5 to 89% fraud |
| `balance_before` = 1.02 x amount | 244 to 246 | 53 | mule onward transfers; 98% clean precision |
| sender balance step of exactly 0.25 x the previous balance | 1,137 | 184 | `sim_swap` drain routine (and its sibling) |
| sender balance step of exactly 0.35 x the previous balance | 1,126 | 188 | `account_takeover` drain routine (and its sibling) |
| amount above 100,000 to a person | 1,345 | 227 | merchant rows above 100,000 are not legit-free in October |

Rate checks per 100,000 rows, July / August / September then October: `balance = 1.1 x amount`
144 / 161 / 153 then 154; amount above balance 121 / 116 / 129 then 114; whole-rupee amounts that are
not multiples of 10: 1,458 / 1,383 / 1,381 then 1,271 (0.77% of legit rows, 24 to 33% of seven fraud
types).

### Timestamp shifts that leave the account age stale

Two routines move the timestamp without recomputing the sender account age, leaving an off-by-one
(a midnight crossing): `sim_swap` shifts rows back into 00:00 to 05:30 (age error of minus 1 day on
10.98% of its rows against 3.64% for legit, 3.0 times enriched); `loan_app_extortion` shifts rows into
20:00 to 23:59 (age error plus 1 day on 25.3% against 18.1%, 1.4 times). Counts of the error rows
overall are large (57,892 and 283,183 in train) because the legit pass produces them too; only the
enrichment is informative. Ordinary fraud rows of these classes do **not** get the shift (night share
0.031 against legit 0.032).

### Fields drawn per row on fraud routines

Fraud-generated rows carry the train-only columns at fixed per-row rates (app version below 4.0 on
about 62%, IP risk above 0.5 on about 39 to 40%, jointly 24.2%, the product of the two), independently
per row and identically for every class, every month and for unreported rows. Legit-generated rows have
15.0% and 1.9 to 2.0%. This is the measuring device used to count ordinary rows; see
[ordinary rows](ordinary-rows.md).

## 3. The pools

- **Fixed pools of fraud-side accounts.** Two disjoint pools of about 6,300 accounts each (1.5% of
  420,000), fixed for all four months: "money receivers" (drain beneficiaries, mules, cash-out
  accounts) and "operators" (bot accounts, QR receivers, collect receivers, bait senders, refund
  abusers, collusion payers). Episodes per account fit uniform draws with replacement from 6,300. The
  accounts look like everyone else on every attribute (city, bank, KYC, age); there are no rings
  beyond that (pairs sharing two events: 44 for bots against 50.3 expected, 0 for drains).
- **Merchants:** a pool of about 330 low-volume merchants receives every round-amount row (420
  merchants in train; 198 of 268 October burst merchants, 73.9%, were in train's pool; 77.6% of
  October collusion rows are on merchants seen in train). Bot targets are random ecommerce (86%) and
  gaming merchants; refund targets are ecommerce (76%) and electronics.
- **Victims** are drawn in proportion to activity and hit once, except the collect victim list (see
  [October](october.md)).
- **The bot device pool** (device numbers 420000 to 422999) is a pool of phones, not accounts; see
  [device counter](device-counter.md).
- Entity pools reach ordinary rows at the legit rate, so they are no handle on them
  ([ordinary rows](ordinary-rows.md)).

## 4. Label noise is added afterwards

On train only: 12% of fraud rows are relabelled `legit`, about 2.3 to 3% get a random other fraud type,
and 0.07% of legit-pass rows get a random fraud label. Details and the test-label question are in
[label noise](label-noise.md). Note the order: the device counter shows the fraud rows were
numbered under their true class **before** noise was applied.

## 5. What the generator does not touch

Nothing on an ordinary fraud row differs from a legit row: not values, rounding, timestamps, the
transaction id, ties and duplicates, or sender-relative consistency (noise-corrected AUC about 0.506
at most). Both passes round values with the same fixed number of decimal places before writing, so no
float artefacts exist. Details: [ordinary rows](ordinary-rows.md).

## 6. Where this repository stands

- **Measured in this repo:** the evidence families (row marks, drains, bot bursts, chains, collect
  requests, merchant rings, repeat payers, device counter) reproduce the target train and October
  event counts; the champion built with them scores 0.5122 given / 0.5665 cleaned in leave-one-month-out
  (`python -m ringtrace.build recipes/champion.json`, `RESULTS.md` run 9), and 0.9010 on "generator-only"
  rows (ordinary-looking fraud set to legit), run 4.
- Exact readings used where the specification was ambiguous (balance steps matched to the cent,
  which reproduces 1,592 and 1,602 rows; the stale-QR mark kept in two variants) are in the repo's
  decision record.
