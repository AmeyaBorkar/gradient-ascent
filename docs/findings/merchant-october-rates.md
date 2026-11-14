# Mule, refund and collusion calls in October: real mix change, not drift

**Question.** The best file (`blend-tabmF-bq`) calls October at unusual rates for three classes:
mule_account, refund_abuse and merchant_collusion. Are the extra or missing calls a real change
in October's fraud mix, or errors from drift? Festival shopping could imitate refund abuse and
mule bursts, and adapted collusion rings could be missed. Is there a trim or an addition that
pays?

**Method.** `python analysis/merchant.py` (12 s on the laptop). It reads the best file's calls:
out of fold on the train months, and October, where it agrees with the uploaded file on
260,582 of 260,582 rows. It writes `$RINGTRACE_OUT/merchant/report.json`. October is only ever
counted. A month is the row's split period: train rows dated 1 October are tails of September
events and count as September.

## Answer

**1. Most of the refund_abuse surplus was a counting error.** Counting by calendar month puts
the 147 train rows dated October into October. They are tails of September events: 87 refund
chain rows, 29 mule, 15 collusion. Calls per 100k rows, July / August / September / October:

| class | by split period | October / train | by calendar month | October / train |
|---|---|---|---|---|
| mule_account | 245.7 / 236.5 / 261.1 / 302.8 | 1.22 | 245.7 / 236.5 / 255.5 / 313.7 | 1.28 |
| refund_abuse | 79.9 / 99.1 / 119.0 / 109.8 | **1.11** | 79.9 / 99.1 / 102.0 / 143.1 | 1.53 |
| merchant_collusion | 34.3 / 59.2 / 59.2 / 32.2 | 0.63 | 34.3 / 59.2 / 56.3 / 38.0 | 0.76 |

Refund calls rise through train (80, 99, 119), and October (110) sits inside that range.

**2. October's calls sit in the high-precision bands, at band rates seen in train.**

Rows per 100k: July / August / September / October. Train precision is pooled over the three
months.

| class: band | rows per 100k | train precision, given / cleaned |
|---|---|---|
| refund: REFUND chains of 3+ claims | 51.9 / 66.7 / 91.6 / 73.0 | 0.81-0.92 / 0.97-1.00 |
| refund: round-amount ring of 4-8 rows | 23.0 / 27.0 / 23.5 / 29.2 | 0.79-0.84 / 0.95-1.00 |
| collusion: round-amount ring of 9+ rows | 25.8 / 48.2 / 52.1 / 27.2 | 0.82 / 0.995 |
| collusion: whole-rupee repeat chain | 6.1 / 8.4 / 2.7 / 0.0 | 0.87 / 1.00 |
| mule: inflow burst rows | 202 / 206 / 220 / 252 | 0.70-0.83 / 0.82-0.97 |
| mule: onward transfers | 25.8 / 26.2 / 26.9 / 30.7 | 0.79 / 0.93 |

- Every pattern row is called in October, as in train, so there is nothing to add.
- Every band holds precision far above the F1/2 trim point, so there is nothing to cut.

**3. Independent volume check: the device counter.** A fraud-pass row gets a new single-use
phone with probability 0.03 whatever its kind. So fraud-block rows per 100k, by counter segment,
count each class's whole October volume, ordinary rows included, without labels.

| segment | Jul | Aug | Sep | Oct | Oct / train |
|---|---|---|---|---|---|
| mule_account | 24.5 | 23.7 | 20.4 | 30.7 | 1.34 |
| refund_abuse | 7.6 | 6.8 | 5.7 | 8.4 | 1.26 |
| merchant_collusion | 2.8 | 4.7 | 4.7 | 4.6 | 1.12 (12 rows) |
| velocity_bot | 9.1 | 12.0 | 11.0 | 11.1 | 1.04 |

Mule and refund volume rise in October along with their template calls. This is a mix change,
not festival imitation.

**4. October's missing velocity_bot bursts are not hiding in mule fan-ins.**
- The two kinds of fan-in look the same on every row column, including the new-payee share, the
  amount grid and the sender pools. The counter tells them apart.
- October has 42 inflow events. 16 hold a mule-segment row and 1 holds a bot-segment row
  (that one is called velocity_bot).
- Train has about 2 bot-segment fan-ins a month at full volume.
- If the roughly 11 events above train's rate were bot fan-ins, about 4 to 5 bot-segment events
  would be expected. One was seen.

**5. Collusion has fewer rings, not adapted ones.**
- *Repeat-payer chains on other amount grids.* Merchant chains of 5+ PAY/QR rows from at most
  half as many senders were searched on these grids: multiples of 500 outside the ring set,
  multiples of 100, whole rupees, and not-whole amounts. There are 0 such events in any month
  except the known whole-rupee form (0.38 / 0.38 / 0.20 per 100k, then 0 in October).
- *Ring-amount chains.* They occur at 3.2 / 5.7 / 4.5 events per 100k in train and 4.6 in
  October. October's rings are smaller: 3 rings of 9+ rows (sizes 20-28) and 11 of 4-8 rows.
- *The rings of exactly 8 rows.* Five October rings have exactly 8 rows, against 3 / 4 / 3 a
  month in train (Poisson p about 0.03). Two of them hold a fraud-block row, and both rows sit in the
  refund segment.
- *How reliable the segment is.* In train, every fraud-block row inside a ring event sits in its
  label's segment: 10 of 10 for refund rings and 18 of 18 for collusion rings.

So the size-8 rings are refund_abuse, as in train.

**6. Rules tried.** Macro-F1 change of the best file per held-out month, given / cleaned /
gen-only:

| rule | October rows | July | August | September |
|---|---|---|---|---|
| refund calls on 8-row rings -> collusion | 40 | -0.0021 / -0.0027 / -0.0073 | -0.0028 / -0.0033 / -0.0068 | -0.0022 / -0.0023 / -0.0048 |
| mule calls in inflow bursts of 24+ senders -> legit | 124 | -0.0030 / -0.0032 / -0.0045 | -0.0020 / -0.0023 / -0.0025 | -0.0047 / -0.0056 / -0.0075 |
| refund calls outside REFUND chains and rings -> legit | 20 | -0.0018 / -0.0019 / +0.0018 | -0.0014 / -0.0012 / +0.0018 | -0.0012 / -0.0011 / +0.0011 |
| collusion calls outside rings -> legit | 13 | -0.0016 / -0.0015 / +0.0028 | -0.0015 / -0.0014 / +0.0019 | -0.0018 / -0.0017 / +0.0030 |
| mule calls outside flows events -> legit | 46 | -0.0019 / -0.0018 / +0.0025 | -0.0015 / -0.0014 / +0.0022 | -0.0016 / -0.0015 / +0.0020 |

Every rule loses on given and cleaned labels in every month. The last three gain only on the
gen-only target, which treats the model's reachable ordinary rows (counter-segment phones,
84-86 % fraud on given labels) as legit.

**Other checks, train only, all negative.**
- *Time.* Ordinary fraud rows sit no closer in time to the same class's generator rows than
  legit rows do. Median gap for mule: 959 s against 927 s.
- *Balance trajectory.* There is no per-sender balance trajectory to break: the ratio to the
  previous balance has the same quantiles for both kinds of row.
- *Ids and phones.* The own-phone share is 0.921 / 0.924 and the transaction-id hex quantiles
  match.

**Verdict.** No candidate.
- mule_account (1.22x), refund_abuse (1.11x) and merchant_collusion (0.63x) are real October mix
  changes.
- The calls sit in high-precision template bands, and the counter's volume count agrees for mule
  and refund.
- No trim or retype passes the bar, and October has no uncalled pattern rows in these classes.
- Expected public effect of leaving them as they are: none.

**What it changes.**
- Count train rows by `period`, not by calendar month, in any October-versus-train rate.
- The counter segment volumes give a label-free check on each class's October volume.
