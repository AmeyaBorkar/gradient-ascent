# October drift: the class-volume shifts are the generator's October mix, not missed fraud

**Question.** October's call rates depart from the train months for five classes. Does any of it
come from fraud whose generator law changed in October, so that the exact-law evidence misses it
or types it wrongly?

**Answer.** No. Every departure is a change in how much of each fraud type October holds, or an
effect of half density and festival amounts on legit traffic. No evidence family shows an
October-only excess of traffic just outside one of its thresholds. Measured on the calls of
`bait-edge` (train calls out of fold), aggregates only.

## Call rates per 100k rows (October test rows / mean of the July-September periods)

| class | ratio | traced to |
|---|---|---|
| velocity_bot | 0.52 | 7 bot phones in October against about 13 expected at half density. Every device in every month with 3 or more ecommerce or gaming PAYs within 30 minutes is a bot-range phone, and all of them are called bot: fewer bot events, none missed. |
| mule_account | 1.24 | Mule bursts 1.24x and layering fan-ins 0.71x. The counter's mule zone is 1.21x wider in October, and the public mule probe's F1 (0.544 against 0.46 predicted) says the mule calls are precise: more mule events. |
| layering_chain | 0.89 | Pass-through hops inside the law: 130 per 100k against 127-139 in train. The in-law hops not called chain are mule rows in train and October alike. |
| refund_abuse | 1.11 (z +1.7, within noise) | Merchant ring events: refund rings have 3-8 rows and collusion rings 10-28 in train (all pure). October has 5 rings of exactly 8 rows (about 1.7 expected), none of 9-19 (about 3.7 expected), and 3 of 20 or more. No chain splits (no consecutive rings at a merchant share a sender), no other payments by the ring's senders to its merchant, and no non-size feature separates the two types (AUC 0.52). An upward shift of event sizes (refund piling at its 8-row cap, collusion moving to 20+) fits; the refund calls stand. |
| merchant_collusion | 0.63 | The same ring sizes; the whole-amount collusion form has 0 October events against about 1 expected. |

## Systematic scan (`analysis/drift.py`)

`python analysis/drift.py --calls <full-ledger calls .npy>` tabulates, per evidence family and
shape, events and event rows per 100k rows by month, size histograms with a chi-square against
row-scaled train counts (and a flag for pile-ups next to a cut that splits two classes), and 16
near-miss bands: traffic that meets a generator law except for one threshold, relaxed on its far
side. Largest departures (Poisson z, test against the mean train rate):

| z | what | train / October per 100k |
|---|---|---|
| -12.2 | bot event rows | 213.6 / 103.2 |
| -10.7 | near-miss balance steps within 0.02 of the drain constants (6 % generator-made in train) | 156.8 / 73.7 |
| -7.3 | P2P PAYs of 3, 20 or 50 (0 % generator-made in train) | 537.9 / 432.5 |
| +6.0 | mule calls | 248.2 / 306.6 |
| +4.7 | mule burst inflow rows | 209.5 / 251.7 |
| -4.2 | collusion calls | 50.9 / 32.2 |
| -3.6 | layering fan-in rows | 65.7 / 47.6 |
| +2.6 | ring events of 8 rows (pile-up beside the 8 / 9 cut) | |

The two near-miss drops are legit traffic thinned by half density (fewer consecutive rows per
sender) and by festival amounts (fewer exact small round payments). No near-miss band rises in
October: hop ratios and delays outside the law, burst and fan-in sender counts one or two below
their minimum, bot sessions with longer gaps or 8-9 rows, drain steps off the constants or slower
than 3 h, collect offsets and bait amounts outside the generator's sets, slower bait sessions and
answers, repeat chains with longer gaps, lending payments before 20:00 and other round ring
amounts all sit at or below their train rates.

**Limits.** The scan sees only laws written into the evidence code, and a law that moved but
stayed inside its window (a hop ratio from 0.97 to 0.99) does not show. Rates move with month
length, density, festival amounts and noise, and the z-scores treat rows as independent although
fraud comes in events, so they overstate certainty.

**What it changes.** Nothing to build: no October-only pocket is left for a rule. The October
differences that matter are the class mix (handled by the per-split decoder and counter) and the
device counter's density (`docs/findings/density-counter.md` on `feat/noise-density`).

**Command.** `python analysis/drift.py --calls <calls.npy>` (about 4 minutes on a laptop CPU with
the families cached); report in `$RINGTRACE_OUT/drift/drift.json`.
