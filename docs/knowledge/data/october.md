# October: what changes in the test month

**Summary.** October is a **step change on 1 October**, with no trend inside July to September. Four
things differ: (1) the legit background is **half as dense** (0.494 times the rows per day) while fraud
events stay whole; (2) legit amounts are **scaled by 1.166** and fraud amounts are not; (3) two train-only
columns are **dead** and the QR/VPA check gains a new blank state; (4) a few fraud pools move (bursts of
bots and collusion rings roughly halved, mule bursts up, a new collect-victim list). Everything
structural (stale fields, balance steps, class parameters, round amounts) holds at the train rates.
Measured cost of the drift the model cannot see is small (about 0.001 macro-F1).

Back to the [index](index.md). Related: [competition](competition.md) (the dead columns),
[generator](generator.md), [device counter](device-counter.md).

Status: **earlier measurement** unless marked "measured in this repo". October statistics are
aggregates only; no test row was inspected.

## 1. Density and the half-volume month

- Test is 1/6 of train rows: 260,582 against 1,564,060. Per day: 8,400 rows against 17,000 (ratio 0.494);
  35% of accounts active against 52%; 0.62 rows per account against 1.25. Rows per active sender fell
  from 2.41 to 1.76.
- Devices with 6 or more rows a month: 19,194 per 100k rows in train against 9,553 in October.
- **Fraud events keep their size**: rows per event July / August / September / October: bot device 43 /
  37 / 43 / 38; mule receiver 13.2 / 13.8 / 13.9 / 13.9; refund sender 6.4 / 5.9 / 6.0 / 6.2; takeover
  phone 2.95 / 2.80 / 2.93 / 2.71; chain hops per 100k 102 / 110 / 101 then 96.
- October is flat day to day: 8,211 to 8,642 rows a day (cv 1.3%; train 0.8 to 1.0%), weekday volume
  within 0.994 to 1.011 of the mean, hour, receiver type, transaction type and category mix within 5%.
  The file holds 46 rows dated 1 to 9 November (event tails, 36 of them REFUND).
- A thinning that mimics density: window counts shift (for example `r_n_f72h` 409 against 206), the
  adversarial AUC without leak and calendar columns falls 0.9725 to 0.9245 (natural 0.918) once train is
  thinned to 0.494. Thinning must keep fraud events whole: 8,829 legit-labelled rows (0.59% of legit,
  13.3% of event rows) sit next to a fraud-labelled row, and thinning them shrinks events (bot rows per
  device 46.3 to 43.6; 6% of chain hops lose their onward transfer).
- Effect of the density shift on a probe model: about 0.002 macro-F1, of which 0.0016 is "the last month
  of the frame" alone. Thinning the train months returns nothing; removing the 35 frame-dependent features
  returns it all. Only `sim_swap` needs a lag.
- Do not divide counts by a density factor, rescale amounts or sums, or thin the train months. Use
  density-safe counts, fixed windows, and capped gaps.

## 2. Amount scale: the festival

- Legit amounts are multiplied by **1.165 to 1.166** at every quantile from p1 to p90 (median 464 to 541;
  1.17 to 1.19 in every week, 1.14 to 1.21 in every category); one factor for merchant and person
  payments; balance x 1.019. Quantile ratio at p99.9 (the fraud tail) is 0.998.
- **Fraud template amounts are not scaled** (medians train to October: 1/1.1 rows 9,761 to 9,343; chains
  105k to 107k; bait answers 11,003 to 10,333; bots 247 to 253; mule 2,600 to 2,595); integer or round
  amounts stay a tell. Only QR-to-person rows, which are redirected purchases, scale (467 / 479 / 435 to
  550; the p10 of QR-ring amounts 86 to 100 against 130).
- A month-level division or rank transform makes the month **easier** to tell (adversarial AUC to 0.998
  to 1.000) because the transformed amount stops agreeing with exact-amount flags and the unscaled fraud
  amounts. Only dropping removes drift, and dropping the 16 amount columns loses 0.0012 to 0.0016: the
  amounts carry more than their drift costs. So amounts stay in rupees.

## 3. Dead and moved columns

See [competition](competition.md) section 3 for the full dead-column table. Summary:
- `app_version`: only 5.0 to 5.2 in October; `ip_risk_score` above 0.5 on 0.52% of rows; drop both and
  every rank of them. Inside bot bursts, IP risk above 0.5 falls from 40.5% to 1.5%.
- QR to a person with a blank VPA check: 0 per 100k in train, 53.7 in October (45% of QR payments to a
  person have a null check); mismatch = 1 on those rows halves (86 / 84 / 79 to 43). The QR type and
  mismatch "out of step" group is stable (537 / 506 / 522 to 544 per 100k). So `qr_swap` is not
  triggered by the mismatch value alone.
- Weakened: QR mismatch = 1 overall 199 / 189 / 196 per 100k to 146.
- Stable: SIM age under 48 h (0.919 / 0.878 / 0.880 / 0.906%), device age under 7 days (2.40 / 2.41 /
  2.44 / 2.37%), new-payee (23.2 / 23.1 / 23.1 / 23.0%), login under 6 s, PIN retries, night share,
  away from home, whole-rupee-not-10 share, balance.
- Calendar-like drifters to drop: weekday, sender and receiver account age (accounts under one year 29.6%
  to 22.6% of rows; sender age grows 30 days a month), cumulative ranks, unbounded previous/next lags and
  whole-frame totals (35 of 234 features), and a "this month's median amount" term that is a month id.

## 4. Fraud signals in October (label-free groups; 82 to 90% fraud-labelled in every train month)

Rows per 100,000, July / August / September, then October and ratio:

| Group | Jul | Aug | Sep | Oct | Oct / train |
|---|---:|---:|---:|---:|---:|
| amount / balance = 1/1.1 | 144 | 161 | 153 | 154 | 1.01 |
| amount above balance | 121 | 116 | 129 | 114 | 0.93 |
| extra device, 2 to 5 rows | 106 | 92 | 93 | 94 | 0.97 |
| extra device, 6+ rows (8+ rows in 10 min) | 248 | 184 | 209 | 103 | 0.48 |
| chain hop forwarded within 1 h / 72 h | 102 / 127 | 110 / 137 | 101 / 129 | 96 / 129 | 0.92 / 0.98 |
| answered bait | 27.5 | 26.4 | 30.4 | 28.4 | 1.01 |
| refund burst | 47 | 61 | 84 | 68 | 1.07 |
| lending, whole rupees, 19 to 23 h | 35 | 40 | 40 | 35 | 0.90 |
| mule fan-in, 6+ senders in 72 h | 238 | 229 | 246 | 273 | 1.15 |
| round merchant repeats | 40 | 60 | 66 | 47 | 0.84 |
| QR to a person with a merchant category | 140 | 137 | 130 | 119 | 0.87 |

- Chain hops forwarded within 1 h: 80 / 79 / 78% in train to 74% (p75 of the delay 24 to 80 minutes;
  within 6 h still 100%).
- `sim_swap` night share 0.70 (00:00 to 05:30) to 0.84 in October (22:00 to 06:00; single pass, not
  re-tested): widen the night window.
- **Prior shift in event volume**: mule bursts about 1.19 times train, bot 0.52, collusion 0.62, chain
  0.92; the other classes within 15% (about 0.97 to 1.07). All-fraud calls 1,527 per 100k against
  1,594 to 1,653. These are event volume on confident rows, not model drift, so no decoder adaptation
  is needed: even a perfect prior oracle gains only 0.00003 to 0.00006 macro-F1, and every label-free
  estimator loses or ties on unshifted months (expectation-maximisation loses 0.0004 to 0.0235).
- Per-class shape checks: 495 distribution checks give 108 (22%) outside the train range by more than
  5%, against 17 to 24% for each train month against the other two: no family-wide drift. Credible moves:
  shorter collect spans (end-of-frame truncation of requests that straddle the split), more Rs 10 baits
  (0.68 against 0.37 to 0.44), higher QR amounts (the festival).
- **Bots:** 7 bursts, 269 rows (14 expected, p = 0.03 to 0.04); same shape, no replacement pattern.
  Bursts of 8+ rows in 10 min: 103 per 100k against 184 to 248, 7 devices against about 12 to 13.5
  expected. A guess: part of the generator moved to fan-ins.
- **Merchant collusion:** 3 bursts, 71 rows against 6 expected (p = 0.14).
- **No new behaviour:** a scan of 26 attributes (96 values, 4,200 pairs, 107,583 triples) finds 8 cells
  absent from train with 20+ October rows, 5 of them the QR-to-person null-check rows (140 rows) and 3 one
  night-time bot burst; every other large deviation is density, amount scale, age or the bot deficit.

## 5. Entity pools and rings in October

- The two hidden pools of about 6,300 fraud-side accounts persist (shares of a month's pool-O accounts
  seen in one earlier month: 35.8 / 32.6 / 35.2%, then 34.3 / 31.6 / 29.0% for October against
  July / August / September; pool M about 20%). October pattern rows whose fraud-side account was
  fraud-side in train: money receivers 48.4%, operators 74.6%, chain accounts 70.3% (with only August
  and September as history: 37.7 / 59.7 / 51.5%). Role counts in October are lower because the month is
  half-volume (drain beneficiaries 538 against 983 to 1,153).
- Collusion rows on merchants seen in train: 77.6% (September with two months: 39.5%); pool P2P
  receiver "known" flags per 100k: 1,101 / 1,100 / 1,113 then 1,085 (1,390 with three months of
  history), 44 to 45% fraud-labelled, 68 to 69% with `new_payee` = 1, 87 to 88% with whole rupees not a
  multiple of 10. History from labels alone is half as precise, and entity history adds nothing to a
  relational model (a tie), so pools are a backup feature only.
- **No rings:** account pairs sharing two events match independent draws (bots 44 against 50.3 expected,
  drain 0 against 0.3, mule 0, chain 1 against 0.2); an entity does not keep its label (43.2% for
  collusion merchants against a shuffled 40.5).
- **One victim list changes.** Collect-request victims come from a list of about 3,000 accounts in train
  (20.0% of August's and 36.7% of September's victims were hit before, median 26 days earlier). October
  uses a new list of about 510: 0.9% are train victims and 32.8% are hit twice inside the month (6.5 to
  11.6% in a train month). So sender fraud history fails for phishing (recall 0.49 to 0.00).
- Victim-label history is worthless (4.7% against a 3.7% base) and recurrence of
  people-receiving-fraud holds: 45% of September's and 41% of October's sure-fraud person-to-person rows
  go to people who received a fraud-labelled payment in the two months before.
- Entity pools reach ordinary rows at the legit rate ([ordinary rows](ordinary-rows.md)).
- Merchants: about 330 low-volume merchants receive all round-amount rows (80 of 92 October merchants
  are in train's pool); no account, bank, city or KYC attribute separates pool members.

## 6. Cost of the drift

- Adversarial validation: relational features reveal the month even between train months (AUC 0.936 for
  August against September) because they encode position in the frame; raw columns outside the two traps
  sit at about 0.55 to 0.59. Behaviour: nothing at feature level (with train thinned to October's density,
  row features give 0.504 against 0.502 natural).
- An October-like September (thinned to 0.494 with fraud events whole, legit-block devices renumbered, QR
  check blanked on about 150 QR-to-person rows) is close to October except for the device-counter split-length
  artefact and the amount scale (adversarial AUC 0.724, 0.550 without one column, 0.540 with both drifts
  emulated). A frame emulating both costs the champion about **0.0009 cleaned / 0.0001 given / 0.0011
  generator-only**, below noise. No challenger that drops or transforms the drifting columns clears the
  +0.007 adoption bar.
- Time consistency inside train: the 197 ordinary features decay by 0.000 plus or minus 0.003 on average;
  decaying ones are receiver age, relative amount, rank on device.

### Measured in this repo
October-like September (`python -m ringtrace.octlike run ... champion:0,1,2 champion-cleantrain:0,1,2`):
champion mean over nine cells **0.5093** (frames 0.5060 / 0.5096 / 0.5125; seed sets within plus or minus
0.0003) against 0.5097 for September at full density in the month folds; training on cleaned labels
0.5091. So the density shift costs essentially nothing on the given-label score. Champion October calls
(`RESULTS.md` run 9): 3,945 fraud of 260,582; mule 789, bot 317, chain 556, sim 356, takeover 242,
phishing 497, qr 518, refund 280, cashback 207, loan 99, collusion 84.

Earlier measurement of the first model uploaded: public 0.5470 against forecast about 0.54; pre-upload
audit 44 pass, 6 warn, 0 fail.
