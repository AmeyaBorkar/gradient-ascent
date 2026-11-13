# The decoder on the fused fraud mass, judged on an A-clean proxy

**Verdict: rejected.** No `hidden_factor` / `fraud_scale` setting, and no rare-class-only
multiplier, never loses on the cleaned target and the A-clean proxy. The incumbent decoder (fixed
point, `fraud_scale` 0.875, `hidden_factor` 1) stays. No October file was written.

Code: `src/ringtrace/aclean.py` (commit 85c2d30); numbers in `RESULTS.md` ("A-clean proxy and
the decoder on the fused mass").

## The A-clean proxy

The test labels count every fraud-pass row as fraud. The cleaned labels still leave about 3,482
unreported ordinary-looking train rows legit, and nothing tells those rows apart from legit
traffic. So the proxy scores their expectation instead of guessing which rows they are. Each
month's hidden count of class k is the 3,482 shared out in proportion to that month's
ordinary-looking class-k labels (cleaned fraud, generator-only legit). Every cleaned-legit row then
carries the same chance of being hidden class k. Macro-F1 is read off the exact expected
confusion matrix. A fraud call on a cleaned-legit row therefore earns a small expected true
positive, which is what a lower cut-off would be buying.

The proxy reads 0.0192 below cleaned on average, matching the knowledge base's 0.019.

| file (train-month analogue, after its rules) | public | proxy Jul / Aug / Sep (mean) |
|---|---|---|
| `blend-col7-phishing` | 0.5408 | 0.5424 / 0.5433 / 0.5504 (0.5454) |
| champion | 0.5470 | 0.5453 / 0.5446 / 0.5520 (0.5473) |
| five-way blend | 0.5474 | 0.5455 / 0.5448 / 0.5521 (0.5475) |
| TabM full (arm c + QR rule) | 0.5483 | 0.5460 / 0.5450 / 0.5525 (0.5478) |
| `blend-blankqr` | 0.5486 | 0.5455 / 0.5448 / 0.5521 (0.5475) |
| fraud-only TabM (`blend-tabmF-bq`) | 0.5499 | 0.5462 / 0.5451 / 0.5525 (0.5479) |

- The proxy reproduces the public order, with one swap: it puts TabM full above
  `blend-blankqr`. That pair is 0.0003 apart on public, which is noise, and the QR rule changes a
  single train row, so no train-month judge can separate the five-way blend from `blend-blankqr`.
- The order is the same as on cleaned labels. Given labels also agree. The generator-only target
  does not: it ranks col7-phishing first.
- So the proxy is a sound offline judge. On these files it adds nothing that cleaned labels miss.

## Probability mass

Fraud mass = 1 - P(legit), train out-of-fold rows, per 100,000 rows:

| band | <0.01 | 0.01-0.05 | 0.05-0.2 | 0.2-0.5 | 0.5-0.95 | >=0.95 |
|---|---|---|---|---|---|---|
| five-way | 98,338 | 29 | 12.2 | 5.1 | 13.8 | 1,603 |
| fused (fraud-only TabM) | 5,739 | 92,614 | 25.6 | 4.7 | 884 | 733 |

- Five-way: 0.031% of rows and 0.9% of the fraud mass lie in 0.05-0.95.
- Fused: 0.91% of rows and 30% of the mass lie in that band.
- The increase is almost all confident fraud squeezed from >0.95 into 0.5-0.95. That band is 99.4%
  fraud on cleaned labels (five-way: 92%). TabM's probabilities are less extreme; they are not
  more uncertain.
- The decision-relevant band, 0.05-0.5, holds 30 rows per 100k (five-way 17), at 24% / 57%
  cleaned fraud rate in its two halves.
- The ordinary mass moved too: 92.6% of fused rows sit at 0.01-0.05 (2.4% fraud, the base rate),
  where the five-way puts 98.3% under 0.01. That changes no call.

Reliability per class: class-k probability band against the cleaned precision of class k, with
rows and mean probability per band, from `python -m ringtrace.aclean mass`.

| class | p 0.05-0.2: n, prec | 0.2-0.5 | 0.5-0.8 | 0.8-0.95 | >=0.95 |
|---|---|---|---|---|---|
| mule_account | 1124, 0.21 | 358, 0.25 | 346, 0.74 | 3138, 0.91 | 436, 0.91 |
| velocity_bot | 492, 0.12 | 182, 0.24 | 35, 0.54 | 2693, 0.99 | 795, 1.00 |
| layering_chain | 20, 0.05 | 73, 0.85 | 582, 0.81 | 2765, 0.89 | 362, 1.00 |
| sim_swap | 38, 0.29 | 190, 0.63 | 607, 0.69 | 1379, 0.86 | 58, 0.98 |
| account_takeover | 733, 0.25 | 469, 0.33 | 131, 0.39 | 1135, 0.98 | 231, 1.00 |
| phishing_collect | 191, 0.25 | 210, 0.60 | 604, 0.72 | 2635, 0.90 | 179, 0.97 |
| qr_swap | 243, 0.21 | 36, 0.22 | 62, 0.81 | 3157, 0.97 | 30, 1.00 |
| refund_abuse | 103, 0.14 | 35, 0.31 | 52, 0.94 | 1050, 0.97 | 433, 1.00 |
| fake_cashback | 631, 0.18 | 265, 0.39 | 137, 0.43 | 348, 0.96 | 281, 0.99 |
| loan_app_extortion | 127, 0.12 | 14, 0.07 | 9, 1.00 | 525, 0.99 | 105, 0.99 |
| merchant_collusion | 25, 0.04 | 1, 1.00 | 29, 0.93 | 493, 0.99 | 272, 1.00 |

Mean probability per band is about 0.10 / 0.32 / 0.68 / 0.91 / 0.96. Above 0.8 the classes are
calibrated or under-confident. At 0.2-0.5 chain, sim_swap and phishing are under-confident
(0.60-0.85 cleaned precision), but on 73-210 rows over three months, and those rows are usually
called a competing fraud type already.

## Decoder sweep (fused mass, rules `anchored_templates, blank_merchant_qr`)

Deltas against the incumbent (hf 1, fs 0.875), Jul / Aug / Sep, in units of 1e-5:

| hidden_factor | fraud_scale | given | cleaned | proxy | Oct calls changed |
|---|---|---|---|---|---|
| 1 | 0.8 | -7 / -1 / 0 | -4 / -2 / -1 | -3 / -1 / 0 | 2 |
| 1 | 1.0 | 0 / 0 / 0 | 0 / +1 / -4 | 0 / +1 / -4 | 2 |
| 1.5 | 0.8 | +3 / -11 / -8 | +4 / -14 / -6 | +5 / -12 / -5 | 5 |
| 1.5 | 0.875 | -4 / -13 / -20 | -3 / -15 / -17 | -1 / -14 / -15 | 4 |
| 1.5 | 1.0 | -12 / -11 / -16 | -9 / -18 / -17 | -8 / -16 / -16 | 2 |
| 2 | 0.8 | +13 / -9 / -13 | +7 / -9 / -12 | +8 / -7 / -10 | 9 |
| 2 | 0.875 | +5 / -5 / -19 | -1 / -9 / -17 | +1 / -8 / -15 | 9 |
| 2 | 1.0 | -3 / -11 / -12 | -2 / -14 / -10 | -1 / -12 / -8 | 6 |
| 2.7 | 0.8 | +19 / -16 / -20 | +10 / +10 / -2 | +12 / +12 / 0 | 15 |
| 2.7 | 0.875 | +15 / -4 / -19 | +7 / +5 / -4 | +9 / +6 / -3 | 12 |
| 2.7 | 1.0 | -2 / -6 / -15 | -8 / -11 / -14 | -5 / -9 / -12 | 11 |
| 1.5, 4 rarest only | 0.875 | -9 / -10 / -16 | -10 / -16 / -16 | -9 / -15 / -15 | 0 |
| 2, 4 rarest only | 0.875 | -9 / -10 / -32 | -10 / -16 / -33 | -9 / -15 / -31 | 3 |
| 2.7, 4 rarest only | 0.875 | -13 / -10 / -32 | -14 / -16 / -33 | -13 / -15 / -31 | 4 |

The 4 rarest classes are picked label-free, by fused probability mass.

- Only hf 2.7 / fs 0.8 is non-negative on the proxy (+0.00012 / +0.00012 / +0.000004). It loses
  September on cleaned (-0.00002) and 0.00016-0.00020 on given in two months. The generator-only
  target loses 0.0005-0.0007 in every month. All of it is 10-20 rows' worth: noise, and not
  never-loses. **Rejected.**
- Raising the expected class size lowers the fixed point's F and lets in more rows, as the theory
  says. But the rows it lets in have precision below even the lowered cut-off, and on the proxy
  a cleaned-legit row earns only about 0.0002 hidden-fraud credit per class, far below any class's
  F/2.
- The rare-only multiplier loses everywhere. The October-like check was not run, because no
  setting passed leave-one-month-out.

## Why there is still no lever

TabM moves confident fraud into the 0.5-0.95 band without making it less right, and the band that
holds the calls a cut-off could change (0.05-0.5) is still about 30 rows per 100,000. The "lower
F means a lower cut-off" argument holds. But under A-clean the extra true class size is ordinary
rows that no score ranks, so a lower cut-off only adds rows at the base rate.
