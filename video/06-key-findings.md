# Key findings (one slide each)

## 1. Two kinds of fraud

- Generator-made fraud is found almost perfectly: macro-F1 **0.9013** on generator-only labels.
- On given labels the blend scores **0.5123**, on cleaned labels **0.5667** (month folds).
- The gap is ordinary-looking fraud: 0.607 / 0.627 / 0.602 of fraud labels in July / August /
  September, about 0.648 estimated for October.
- AUC against legit rows: 0.4904 in train, 0.4896 in October. Links through the same sender,
  receiver or phone: ratio 0.95 to 1.02. September against October: AUC 0.5024.

## 2. A hard ceiling

- October's reachable macro-F1 is about **0.556 to 0.565**, counting every reachable row as right.
- A public score of 0.6338 would need **1,006** more correct October rows: 13.9 % of the 7,249
  hidden ordinary rows. Nothing in the data reaches them.
- The call-count forecast works: 0.5475 expected for the five-member blend, 0.5474 public.

## 3. Noisy train, clean test

- About 12 % of train fraud is unreported. The all-legit upload (0.0816) shows the test labels are
  clean, with a 4.06 % fraud share.
- So the decoder divides fraud probabilities by 0.875. Two members train on event-cleaned labels.
  Every change is scored on cleaned labels too.

## 4. The device counter

- Phones are numbered in creation order. The fraud phase is written class by class.
- Counter features are worth +0.028 macro-F1 on held-out months. Counter plus template rule:
  +0.0333 given.
- The refined counter cuts boundary error from 82 to 58 phones on four labelled passes.

## 5. Look-alike types

- Some pairs share one template: sim_swap / account_takeover, mule_account / velocity_bot,
  layering_chain / mule_account, phishing_collect / fake_cashback, qr_swap / refund_abuse.
- Inside a template, the second type is close to random per event: AUC at most 0.61.
- The anchored template rule types the whole event by its counter zone or anchor. Anchoring adds
  +0.0028 given over the plain one-class-per-template rule.

## 6. October drift is a mix shift, not missed fraud

| Class | October call rate against train | Traced to |
|---|---|---|
| velocity_bot | 0.52x | 7 bot phones against about 13 expected |
| mule_account | 1.24x | mule bursts 1.24x; the counter's mule zone 1.21x wider |
| layering_chain | 0.89x | pass-through hops inside the law at train rates |
| refund_abuse | 1.11x | within noise; refund rings pile up at their 8-row cap |
| merchant_collusion | 0.63x | rings moved to 20 rows or more |

- A scan of 16 near-miss bands found none that rises in October.
- Correction: two velocity_bot events hid in the mule-burst shape, with no fraud-phase phone. That
  became `fresh_bursts`.

## 7. One event, one answer

- An event's rows share one class. So a public score reads the whole event, private rows included.
- That is how the burst steps were settled: 0.5516 → 0.5531 → 0.5532 → 0.5556.
- A single-row border move cannot do that: a rise only says the row was public.

## 8. Byte-for-byte rebuilds

- The best file rebuilds to sha256 `c2784900…86eb2efc`.
- A clean tree and an empty cache rebuilt the six member runs and `counter-news` in 5,086 s,
  byte-identical.
