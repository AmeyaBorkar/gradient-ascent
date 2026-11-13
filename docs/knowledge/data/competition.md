# The competition and the tables

**Summary.** A 48-hour live round. Label every October 2026 UPI transaction with one of 12 classes
(`legit` plus 11 fraud types); the score is macro-F1 over the 12 classes. There are 20 uploads a day,
30% of test is public and 70% private, and the platform ranks each team by its best public upload
scored on the private part. The organisers gave four hints about the data (next month, noisy train
labels, dead columns, relational signal); every one proved true. This file also lists the tables, the
columns and the train-only columns that are dead in October.

Status labels: **earlier measurement** = from the prior research pack, a later session can re-derive
it; **measured in this repo** = this repository's own runs (stated with the command). Back to the
[index](index.md).

## 1. Competition facts (stated by the organisers)

| Item | Value |
|---|---|
| Task | one of 12 labels per row of the test file: `legit` (about 96%) and `mule_account`, `velocity_bot`, `layering_chain`, `sim_swap`, `account_takeover`, `phishing_collect`, `qr_swap`, `refund_abuse`, `fake_cashback`, `loan_app_extortion`, `merchant_collusion` |
| Split | train = July, August, September 2026 (labelled); test = October 2026 (labels hidden, clean) |
| Metric | macro-F1 over all 12 labels, each weighted equally. All-`legit` scores about 0.08. The organisers call 0.40 excellent and say they do not expect 0.50 |
| Submission | CSV with header `id,label`, one row per test transaction, labels spelled as above, at most 15 MB |
| Uploads | 20 per team per day (raised from 10 on the first day), reset at midnight IST. Uploads failing validation do not count |
| Public / private | public = 30% of test, private = 70% |
| Final ranking | each team's **best public upload, scored on the private part**. The team does not pick a final file |
| Code | a link plus a short approach note must be submitted before the round closes; the top teams' code is checked by reproducing their best upload |
| Round | 13 Nov 2026 10:00 IST to 15 Nov 2026 10:00 IST; results the same day at 17:00 IST |

Rules that shape work (earlier reading of the statement):
- Train only on the provided data: no external data. Public pretrained weights are allowed.
- No hand-labelling of test rows. Rules written in code are fine, and using the test file itself to
  build connections (same account, same device, same receiver) is explicitly allowed and expected.
  Working convention here: test rows are used through code and aggregates only.
- Sharing code or predictions between teams is forbidden. Probing the leaderboard is not forbidden,
  but the statement says not to tune to it.

### What the best-public rule implies (earlier measurement, reasoning)
- The upload the team believes in most must also be its highest public score; a weaker model that is
  lucky on the public 30% would be the one ranked. Upload only candidates that beat the champion on
  cross-validation.
- Macro-F1 gives each fraud type 1/12 of the score: a class with a few hundred test rows is worth as
  much as `legit` with 250,000. How many rows to call each class matters as much as ranking quality.
- Train under-reports fraud and test does not, so decision thresholds tuned on train-month validation
  are not the best ones on test (see [label noise](label-noise.md)).

### The organisers' four hints
1. **Test is the next month.** October is festival season, spending rises, the app shipped a new
   version, and fraud rings adapt. See [October](october.md).
2. **Training labels are noisy; test labels are clean.** Some fraud was never reported (labelled
   `legit`) and some was filed under the wrong type. See [label noise](label-noise.md).
3. **Some columns stop working in October.** See section 3.
4. **Rows are shuffled and the signal is relational.** Patterns live in how rows connect: same account,
   same phone, same receiver, money moving on minutes later.

## 2. Tables (earlier measurement; the row counts also appear in this repo's results)

| Table | Rows | Columns | Notes |
|---|---:|---:|---|
| train | 1,564,060 | 22 | 1 Jul to 30 Sep, plus 147 rows dated up to 10 Oct (event tails); `legit` 96.31% |
| test | 260,582 | 21 | 1 Oct to 31 Oct, plus 46 rows dated up to 9 Nov; no label column |
| accounts | 420,000 | 6 | every sender and every person-to-person receiver is in it; 26,566 never transact |
| merchants | 30,000 | 4 | 29,528 receive payments |
| sample submission | 260,582 | 2 | ids in the same order as the test rows |

Transaction columns: a random id, timestamp, sender, receiver, receiver type (merchant or person),
device, amount, transaction type (PAY 57.3%, QR 35.5%, COLLECT 3.3%, AUTOPAY 3.1%, REFUND 0.7%),
merchant category (null on person receivers, 12 values), device city (40 cities), SIM age in hours,
device age in days, sender account age in days, a new-payee flag (23.1% ones), PIN retries (0 / 1 / 2 =
90.0 / 8.0 / 2.0%), login-to-transaction seconds, balance before, app (five apps), app version,
IP risk score and a QR/VPA mismatch flag (null on 64% of rows). Train adds the label.

Properties that matter later:
- **The split is by event start.** The 147 train rows dated October are 83 to 87% fraud: tails of
  September events. The 46 test rows dated November include 36 REFUND rows from ten-day refund
  windows. So no fraud event straddles train and test.
- **Entity overlap:** 91% of test senders, 79% of test receivers and 87% of test devices appear in
  train (95%, 91% and 92% of test rows).
- **Labels per month are flat:** fraud is 3.69% of train rows. Rarest classes: `loan_app_extortion`
  1,991 rows, `fake_cashback` 2,532, `merchant_collusion` 2,549 (the largest, `mule_account`, has 10,261).
- **Identifiers do not leak.** `txn_id` is random (drawn without repeats over both files; no seeded
  generator stream reproduced it); account and merchant numbers, file row order and the digits of
  amounts carry nothing about the label. The exception is device numbers above 420,000: see
  [the device counter](device-counter.md).
- **Columns drawn independently per row for ordinary traffic** (they do not follow the account or
  the device): SIM age, device age, balance, new-payee, app version, IP risk. They carry signal only
  where a fraud routine sets them. Sender account age grows by 30 days a month (no account opened
  after late June), so it is a calendar proxy, not a fraud signal.

## 3. Train-only columns that are dead in October (earlier measurement)

| Column | Train | October | Verdict |
|---|---|---|---|
| `app_version` | 3.0 to 4.8; below 4.0 on 15.7% of all rows (15.0% of legit-generated rows; 61.6 to 61.8% of fraud-generated rows) | only 5.0, 5.1, 5.2, close to uniform; below 4.0 on 0.000% | **drop**. A rank transform does not rescue it |
| `ip_risk_score` | above 0.5 on 2.0% of legit-generated rows, about 39% of fraud-generated rows; template-fraud mean 0.41 against legit 0.20; AUC 0.76 to 0.80 | above 0.5 on 0.52% of rows (train 2.5%); mean 0.236 on pattern rows against 0.220; AUC 0.53 to 0.56 | **drop** |
| QR/VPA mismatch on QR to a person | 1 / 0 / blank = 59 / 41 / 0% | 35 / 21 / 44%: a new blank state | not a `qr_swap` trigger; the flag-on-non-QR contradiction (see [generator](generator.md)) still works |

- Adversarial validation (a classifier telling September from October): with both columns the AUC is
  1.000; with `ip_risk_score` only 0.664; without both 0.586; without the amount and age trends 0.550.
- The two columns are stable inside train (month-to-month decay about 0.001 to 0.003), so **no
  train-only check exposes them**; only the October comparison does.
- Both stay useful for one purpose: as evidence about **train** rows when estimating how many fraud
  labels are on ordinary-looking rows and when cleaning labels, because a fraud routine draws them per
  row. They never enter a model that predicts October.

## 4. Where this repository stands (measured in this repo)

- The champion recipe uses no app version or IP risk and scores, in leave-one-month-out on September
  and the other months, **0.5122 given / 0.5665 cleaned** (mean of three months). Command:
  `python -m ringtrace.build recipes/champion.json`; see `RESULTS.md`, run 9.
- Its October submission has 260,582 rows with ids and order identical to the sample (VALID),
  3,945 fraud calls out of 260,582.
- Upload results themselves (public scores) predate this repository and are an **earlier
  measurement**: the all-legit file scored 0.0816 public and the best-model upload 0.5470 public (see
  [label noise](label-noise.md) and [October](october.md)).
