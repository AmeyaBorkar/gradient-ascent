# The device counter: the key insight

## In one sentence

Phone numbers are handed out in order, and the fraud routines' phones were created together, so a
phone's place on the number line tells you whether its rows are fraud, and which type.

## How it works, simply

1. **Three blocks per split.** Fresh phones above a fixed number fall into a reserved bot pool, a
   fraud phase and a dense legit block.
2. **The fraud phase is written class by class,** in the label-list order: mule_account,
   velocity_bot, layering_chain, sim_swap, account_takeover, phishing_collect, qr_swap,
   refund_abuse, fake_cashback, loan_app_extortion, merchant_collusion.
3. **No labels at run time.** An 11-state left-to-right HMM reads the block from evidence anchors.
   Its emission table is fixed, not refitted per split. The borders are found separately in train
   and in October.
4. **The refined counter** adds each phone's allocation geometry: the run of unused numbers just
   below it, and its row count. sim_swap phones are dense (mean gap 0.41). account_takeover phones
   are sparse (mean gap 2.81). Those constants were fitted once on train and are fixed in the code.

## Why it matters

- It certifies fraud on rows that look ordinary, and gives their type.
- It anchors the look-alike templates: events that two types share take the class of their zone.
- It reads October on its own terms. October's mule zone is 1.21x wider than train's.

## Evidence

| Check | Result |
|---|---|
| Counter features, held-out months | +0.028 macro-F1 given, +0.028 cleaned (ablation) |
| Counter + template rule, paired event bootstrap (90 % interval) | given +0.0333 [+0.0310, +0.0358], cleaned +0.0343 [+0.0319, +0.0369] |
| Counter-free fallback against the blend | given −0.030 / −0.033 / −0.027 per month |
| Label-free segments against labels | 0.8456 of fraud-block rows and 0.9566 of fraud-labelled train rows agree |
| Refined against old counter, four labelled passes | boundary error 82 → 58 phones; mule \| bot better on all four passes |
| Where the two counters disagree, posterior ≥ 0.7 | refined right on 25 of 27 phones, old counter on 2 |
| October-like frames, `counter_news` moves | sim_swap → account_takeover right on 18 of 18 (cleaned); velocity_bot → mule_account 7 of 8 |

## On the leaderboard

| Step | Public |
|---|---|
| `mule_edge`: October's wider mule zone, 10 rows velocity_bot → mule_account | 0.5502 → 0.5506 |
| `counter_news`: the refined counter, 3 rows on one phone plus 1 mule-edge row | 0.5506 → 0.5516 |

## Limits, stated honestly

- October's fraud phase is about a sixth of train's, so every border is uncertain. Its most likely
  position holds 0.21 to 0.98 of the probability.
- The border posteriors are calibrated when pooled, but biased at two borders.
- The counter cannot reach the ordinary-looking fraud: only 2.82 % of ordinary fraud rows drew a new
  phone. The rest sit on owners' own phones.
- Whether gaps come from discarded or overwritten numbers cannot be told from the data.
