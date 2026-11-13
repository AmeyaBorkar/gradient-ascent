# The 11 fraud classes: template signatures and look-alikes

**Summary.** Each fraud class has a label-free template signature that holds in July to September and
in October. Precision against the noisy train labels is 62 to 93%; about 12 points of every shortfall
are unreported fraud that clean test labels count. Classes come in sibling groups that share generator
subroutines, so one row predicate has a 70 to 78% primary-class ceiling on noisy labels (82 to 97%
after allowing for the 12% unreported flip). A partner type is written **by design** on 9 to 32% of a
pattern's events (see section 3); those events are not mislabelled, and it is a belief that clean test
labels keep them.

Back to the [index](index.md). Related: [generator](generator.md), [device counter](device-counter.md),
[label noise](label-noise.md).

Status: all numbers are an **earlier measurement** (a later pass can re-derive them) unless marked
"measured in this repo".

## 1. Signatures

"Precision" is the fraud-labelled share of rows the rule flags, by month where given. "Oct" is how the
same label-free detector behaves on the test month.

| Class | Template | Train precision | October |
|---|---|---|---|
| `mule_account` | 8 to 25 distinct senders pay one account: PAY, amounts multiples of 10 in 400 to 4,900, within 48 h, new-payee rate 0.89. Then 1 to 3 integer onward transfers within 2 h with balance = 1.02 x amount. 16% of bursts are a stealth variant. About 69 events a month | row rule 93% pure; reaches about 28% of mule labels | bursts about 20% above the half-month expectation |
| `layering_chain` | 3 to 6 linked PAY rows (the receiver becomes the next sender), integer amounts 20,000 to 180,000, each hop keeps 96.4 to 99.6%, gap 2 to 25 min (74% of chains) or up to 360 min; balance = amount x 1.00 to 1.15. Or a fan-in of 3 to 7 new-payee senders into one account | chain rows 72.6 / 71.5 / 75.0% layering, the rest mule (21% of fast chains, 0 of 149 slow) and legit; fan-in 0.84 / 0.86 / 0.85 | hops 0.92 times train; fan-in 27 events against 38 |
| `velocity_bot` | device number 420000 to 422999; 20 to 60 integer PAY of Rs 10 to 498 in 2 to 15 minutes to ecommerce (85 to 86%) or gaming merchants; many accounts (65%) or exactly 3 accounts | 85.0 / 84.1 / 84.6%; recall 0.39 / 0.31 / 0.34 | 7 bursts, 269 rows (0.50 times train) |
| `sim_swap` | 2 to 4 PAY to new person receivers; each `balance_before` is exactly 0.25 x the previous one; within 40 min; SIM age 0.5 to 30 h; login 3 to 25 s; night | balance-linked rows 0.86 / 0.87 / 0.86 (`sim_swap` plus `account_takeover`); 0 of 22,930 legit pairs match | 110.5 per 100k rows against 118 / 110 / 105 |
| `account_takeover` | the same drain with factor 0.35 within 90 min; one new device per event; device age under 1.5 days; city differs from home (93 to 96% of takeover-phone rows); PIN retries 0 / 1 / 2 = 0.40 / 0.37 / 0.23 | as above | multi-row new-device rows 0.97 times expected |
| `phishing_collect` | COLLECT to a person, new-payee = 1, amount = 2,000 / 5,000 / 10,000 / 25,000 minus 1 / 10 / 50; one victim gets 1 or 2. Also bait-only rows of Rs 1, 2 or 5 | 0.74 / 0.73 / 0.72 (the rest: unreported, `fake_cashback` 7.7%, `loan_app_extortion` 3.5%) | 326 rows = 125 per 100k against 136 / 128 / 121 |
| `fake_cashback` | 1 or 2 baits of Rs 1 / 2 / 5 / 10 from a pool account, then an integer payback of 1,500 to 20,000 the other way within 3 h (68%) or 4 to 48 h | "one bait + payback within 3 h" is 47 to 54% phishing and 28 to 36% cashback; the two cannot be separated (CV AUC 0.41) | 96 rows in that mixed group |
| `qr_swap` | QR to a person with the merchant category still set (85% of events, one row); or 1 to 3 round-amount sales (1,000 / 2,000 / 2,500 / 5,000 / 10,000) at a pool merchant | 0.81 / 0.83 / 0.85; bursts of 1 to 3: 0.82 / 0.83 / 0.81 | 309 rows (0.87 times) plus 185 |
| `refund_abuse` | one sender, 3 to 8 REFUND rows at 1 or 2 ecommerce or electronics merchants within 10 days; or a round-amount burst of 4 to 8 rows | 0.87 / 0.86 / 0.82; catches 935 of 950 template REFUND rows | 190 rows = 73 per 100k against 52 / 67 / 78; plus 76 |
| `loan_app_extortion` | 3 to 6 integer PAY of 1,500 to 15,000 to distinct new lending merchants, 20:00 to 23:59, within 7 days; balance guard 1.1 x | 0.86 / 0.81 / 0.87 | 98 rows = 37.6 per 100k against 38.7 / 38.7 / 38.2 |
| `merchant_collusion` | round-amount burst of 10 to 28 rows at a pool merchant from 2 to 5 payers (8 to 18 distinct senders in a looser reading), gaps about 1.5 h | bursts of 10 or more are collusion-labelled | 3 bursts, 71 rows (6 expected, p = 0.14) |

Corrections made when the table was re-tested: mule amounts are multiples of 10 in 400 to 4,900 and a
burst needs 8 senders (sessions of 7 are layering fan-ins); the chain gap limit is 360 minutes, not
340; bait-only must exclude Rs 10 (1,089 legit Rs 10 payments exist); "fraud amounts are integers, not
multiples of 10" is too broad (mule bursts and fan-ins are multiples of 10, and merchant rows on that
grid are only 25 to 29% fraud-labelled).

## 2. Rules that cut across classes

- **Round-amount bursts at a merchant** (rows linked with gaps up to 36 h): burst size sets the class.
  1 to 3 rows: `qr_swap` (92 to 93% clean precision); 4 to 8: `refund_abuse` (87 to 97%); 10 or more:
  `merchant_collusion` (about 97%). **No burst of size 9 exists in train.**
- **Drains:** SIM-template events are 72% `sim_swap`, device-template events 73% `account_takeover`.
  For device-template events the device-counter zone gives the class (147 of 148 `sim_swap`-labelled
  device events lie in the `sim_swap` zone, 336 of 337 `account_takeover` ones in the other); row
  columns cannot make this split. See [device counter](device-counter.md).
- **Pools:** fraud receivers are reused across months and classes (see [generator](generator.md)).
- **Victim repetition inside a split:** 66 to 69% of just-under COLLECT rows have a second one from the
  same sender, in train months and in test.
- **Bot burst anatomy:** 84 bursts in train (33 / 26 / 25 by month) on 81 devices; per burst 20 to 60
  rows, a window of 2 to 15 minutes independent of size, 1 to 3 merchants (none of them large), device
  age uniform 0 to 60 days; three sub-types: one row per account with 20 to 60 accounts (65%), exactly 3
  accounts with login 1 to 6 s (17%), exactly 3 accounts with ordinary logins (18%). Every row on a bot
  device is in a burst.
- **Look-alike label parts of `velocity_bot`** (train, 8,106 rows): device bursts 2,827 (34.9%, findable
  exactly); fan-in events written by the bot generator 285 (3.5%, look like `mule_account`); ordinary
  rows 4,994 (61.6%, not findable, AUC 0.50 to 0.52).

## 3. Sibling and look-alike pairs, and the partner-label share by design

Four groups share generator subroutines: `sim_swap` with `account_takeover`; `phishing_collect` with
`fake_cashback`; `layering_chain` with `mule_account` (and `velocity_bot` for fan-ins); and `qr_swap`,
`refund_abuse` and `merchant_collusion` (round-amount merchant bursts). An event uses a sibling's
template and keeps its own label on 20 to 30% of events.

Events with two or more fraud-labelled rows, by majority label (monthly range in brackets):

| Pattern (label-free) | Events | Own type | Partner types |
|---|---:|---|---|
| extra device, 2 to 5 rows | 459 | takeover 70.2% | `sim_swap` 29.2% (25 to 32) |
| own device, 2+ rows with SIM age at most 36 h | 457 | `sim_swap` 70.5% | takeover 28.0% (27 to 29) |
| chain hop | 489 | layering 82.0% | mule 18.0% (17 to 19) |
| fan-in or fan-out on a person account | 269 | mule 68.4% | layering 24.9%, bot 6.7% |
| bait up to Rs 10 and its answer | 358 | cashback 66.8% | phishing 32.4% (26 to 36) |
| collect ending in 99, or balance = 1.1 x amount | 147 | phishing 78.9% | cashback 13.6%, loan 4.1% |
| lending, whole amount, 3+ payments | 134 | loan 91.0% | collusion 5.2%, refund 3.7% |
| QR to a person | 251 | `qr_swap` 88.8% | phishing 5.6%, collusion 4.4% |
| round merchant amount, pair of at most 2 rows | 227 | `qr_swap` 79.7% | refund 16.3%, collusion 2.6% |
| round merchant amount, pair of 3+ rows | 82 | collusion 40.2% | refund 40.2%, `qr_swap` 19.5% |
| bot burst; refund burst | 84; 168 | 100; 100 | none |

Why these are design and not noise:
- A partner-labelled event is not a random flip. In the takeover pattern, every row is away from home in
  91.3% of takeover-labelled events and in 100% of 153 `sim_swap`-labelled ones; in the sim pattern, every
  payee is new in 93.6% against 100% of 146 (p = 0.0005). Fan-ins labelled layering have 4 to 7 senders,
  those labelled mule 8 to 33. Bait pairs: cashback has 1 or 2 baits, phishing always 1.
- Event-level AUC of telling the pair apart: 0.57 (takeover against sim), 0.49 (chain against mule),
  0.47 to 0.52 (phishing against cashback collects), 0.76 (bait pairs), 0.93 to 1.00 (fan-in size, QR to
  a person, round merchant rings).
- The device counter shows the partner-labelled events were generated in the labelled class's own turn
  (new phones labelled `sim_swap` sit in the `sim_swap` zone: 366 of 367; labelled takeover: 913 of 922).
  The proposal "relabel sibling events to the template's class" was therefore refuted.
- Random wrong-type noise is separate and small (about 3% of rows, uniform over the other ten types).
- **Belief, not known:** clean test labels keep the partner labels, because the organisers' statement says
  some fraud types look like each other.

On structured rows alone, per-class F1 is 0.89 to 0.96 except `sim_swap` 0.70, `account_takeover` 0.73
and `fake_cashback` 0.76: the look-alike pairs are where the remaining structured loss is.

## 4. Class sizes and priors

Structured (non-ordinary) rows per 100,000 rows, including unreported (earlier estimate):

| Class | Jul | Aug | Sep | Oct estimate | Ordinary labelled rows |
|---|---:|---:|---:|---:|---:|
| `mule_account` | 244 | 235 | 243 | 274 to 288 | 446 |
| `velocity_bot` | 259 | 213 | 238 | 124 to 129 | 311 |
| `layering_chain` | 214 | 211 | 200 | 184 to 191 | 252 |
| `sim_swap` | 121 | 110 | 101 | 97 to 110 | 160 |
| `account_takeover` | 120 | 104 | 105 | 107 to 109 | 160 |
| `phishing_collect` | 212 | 197 | 194 | 193 to 196 | 258 |
| `qr_swap` | 208 | 207 | 200 | 194 | 287 |
| `refund_abuse` | 76 | 96 | 116 | 98 to 100 | 127 |
| `fake_cashback` | 61 | 54 | 63 | 64 to 68 | 104 |
| `loan_app_extortion` | 41 | 42 | 42 | 38 to 46 | 86 |
| `merchant_collusion` | 34 | 56 | 55 | 35 to 38 | 116 |
| all | 1,589 | 1,523 | 1,556 | 1,438 | 2,307 |

Use the October column as the prior, not the September label share (bots about halved, mule up). A
fraud class is scored against 2.2 to 3.4 times its structured rows once the ordinary rows are counted
([ordinary rows](ordinary-rows.md)).

## 5. This repository (measured in this repo)

Per-class F1 of the champion (leave-one-month-out, pooled; `python -m ringtrace.build
recipes/champion.json`, `RESULTS.md` run 4): on given labels 0.393 (`merchant_collusion`) to 0.551
(`layering_chain`); on generator-only rows 0.755 (`fake_cashback`), 0.836 (`account_takeover`), 0.868
(`sim_swap`), 0.882 (`mule_account`) up to 0.952 (`refund_abuse`). The weakest classes on structure are the
look-alike pairs, matching section 3.
