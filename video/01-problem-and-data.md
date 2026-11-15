# The problem and the data

## Task and scoring

| Item | Value |
|---|---|
| Task | label every October 2026 UPI payment: `legit` or one of 11 fraud types |
| Fraud types | mule_account, velocity_bot, layering_chain, sim_swap, account_takeover, phishing_collect, qr_swap, refund_abuse, fake_cashback, loan_app_extortion, merchant_collusion |
| Metric | macro-F1 over all 12 labels, each weighted 1/12 |
| Train | July to September 2026: 1,564,060 rows (527,065 / 526,852 / 510,143) |
| Test | October 2026: 260,582 rows, labels hidden |
| Leaderboard | public = 30 % of October; the final entry is the best public upload, scored on the rest |
| Round | 13 Nov 10:00 to 15 Nov 10:00 IST; the best upload must rebuild from the submitted code |
| Data rules | given data only; rules in code are fine; no test row labelled by hand; using `test.csv` to build connections is allowed |

## The October twists

October is a step change on 1 October, not a trend.

| Twist | What changes |
|---|---|
| Half density | 8,400 rows a day against 17,000 (ratio 0.494). Fraud events keep their size. |
| Festival amounts | legit amounts scaled by about 1.166; fraud template amounts are not |
| Dead columns | `app_version` and `ip_risk_score` carry no signal in October |
| New blank state | about 45 % of QR payments to a person have a blank QR/VPA check |
| Fraud mix moves | fewer bot bursts, more mule bursts, collusion rings at larger sizes |
| Smaller fraud phase | the counter's October fraud phase: 920 numbers and 410 phones, against 5,455 and 2,240 in train |

## Noisy train labels, clean test labels

- About 12 % of train fraud is unreported, filed as legit.
- The cleaned labels used for scoring change 3,629 train rows: 2,944 legit → fraud, 685 type fixes.
- An all-legit upload scored **0.0816** public. That matches clean test labels with a 4.06 %
  fraud share, ordinary-looking rows included.

## Most fraud looks ordinary

| Period | Share of fraud on ordinary-looking rows |
|---|---|
| July | 0.607 |
| August | 0.627 |
| September | 0.602 |
| October (estimate, no labels) | 0.648 |

- These rows look exactly like legit traffic. AUC against legit rows: 0.4904 in train, 0.4896 in
  October.
- The generator-made rest is found almost perfectly: macro-F1 0.90 on generator-only labels.
- So October's reachable macro-F1 is about 0.556 to 0.565.

## How the fraud is written, reverse-engineered without labels

Each type follows a routine that leaves traces across rows:
- layering hops and mule fan-ins, with onward transfers;
- bot bursts on shared phones;
- balance drains stepping by exactly 0.25x (sim_swap) or 0.35x (account_takeover);
- collect requests just under round limits, and cashback baits of Rs 1 / 2 / 5 / 10;
- repeat refund chains and lending payments;
- round-amount merchant rings, typed by ring size.

The device counter is the one clue that is not behaviour: see `03-device-counter.md`.
