# Generalisation stress test and the October drift audit

**Summary.** Does the champion survive a generator that moves between train and October? We
perturbed the labelled generator rows of the train months (slower or faster delays, shorter bursts,
amount jitter, broken exact constants), refitted nothing, and measured end to end. Findings:
**faster events are harmless, slower events cost a lot (x3 delays: -0.046 cleaned), amount jitter
with exact constants intact is small (-0.003), and breaking the exact constants is catastrophic
(-0.17 to -0.20).** October's measured drift is small and keeps every exact constant, so the
expected private-set risk from generator drift is about **-0.003**, inside the noise. An adversarial
check (September vs October AUC 0.729) is explained by density, amount scale and one split-length
artefact that the model hardly uses.

Status: all numbers are **[earlier]** (measured before this repository's runs; this repository's own
runs are in `RESULTS.md`). The October-like frame construction and its pitfalls are in
[more-data.md](more-data.md). Back to the [index](index.md).

## Overview

| experiment | effect on champion macro-F1 (cleaned) | verdict |
|---|---|---|
| event delays x0.5 | -0.0001 | harmless |
| delays x1.5 / x2 / x3 | -0.0145 / -0.0336 / -0.0456 | fragile (rings, repeat, chain, mule, drain) |
| bursts: keep first half (vs mechanical reference) | -0.0244 | fragile |
| bursts: drop every second row | -0.0549 | fragile |
| amounts +-20% per row, constants intact | -0.0028 | small |
| amounts +-20% per event, constants intact | -0.0027 | small |
| amounts x1.2 on every row, constants intact | -0.0121 | mule inflow ceiling at 5,000 |
| amounts jittered, constants broken (per row / per event) | -0.1961 / -0.1708 | catastrophic by design |
| drain step constant jittered | -0.0295 | catastrophic for drains |
| widen tolerances (insurance) | 0 change on October rows | free, tie expected |
| adversarial validation Sep vs Oct | AUC 0.729 (train months 0.51-0.53) | density / scale artefacts |

## Setup [earlier]

- The champion's nine fold models (three held-out months x three seeds) were refit on unperturbed
  features and reproduced the out-of-fold probabilities to 3e-8; the unperturbed path scored the
  champion's 0.5126 given / 0.5662 cleaned.
- 24,337 labelled generator rows were perturbed and all 14 feature families rebuilt in memory for
  each variant, predicted by the fold model that never saw the month, then decoded with the fixed
  point and the anchored template rule. Rows are keyed to their family event (98.6%), else the
  event of the cleaned labels, else alone; events split at gaps over seven days (8,764 sub-events).
- Delay x k stretches times inside each sub-event around its start (multi-day patterns scale day
  offsets and keep time of day). "Half bursts" keeps the first half in time or drops every second
  row. Amounts: independent factors per row or per sub-event, "intact" keeps the exact constants
  (round-5, classic-12, under-limit, bait, ring amounts, the 1.1x and 1.02x balance clamps, whole
  rupees), "broken" scales to cents.
- Row-dropping variants are scored against a mechanical reference (unperturbed predictions on the
  kept rows): of the raw -0.098 for first-half bursts, -0.074 is mechanical, -0.024 the model's.

## Results by perturbation [earlier]

End to end, cleaned (given in brackets):

| variant | cleaned | delta cleaned | delta given |
|---|---|---|---|
| delays x1.5 | 0.5517 | -0.0145 | -0.0128 |
| delays x2 | 0.5326 | -0.0336 | -0.0301 |
| delays x3 | 0.5206 | -0.0456 | -0.0409 |
| amounts +-20% intact, per row | 0.5634 | -0.0028 | -0.0024 |
| amounts x1.2 intact | 0.5541 | -0.0121 | -0.0105 |
| amounts broken per row | 0.3701 | -0.1961 | -0.1721 |
| drain step constant jittered | 0.5367 | -0.0295 | -0.0269 |

The per-month ranking is the same in every month.

**Which families break (row recall lost, in points):**

| family / class | first breaks | worst | cause |
|---|---|---|---|
| merchant_collusion rings | delay x1.5 (-24) | -54.7 at x2-3; -95.6 broken | 24 h gap chains the ring; 2+ day gaps split a 9+ ring into refund-shaped pieces |
| refund rings | x1.5 (-5.4) | -23.8 broken | same chaining; ring size band 4-8 |
| layering_chain | x1.5 (-3.5); amounts x1.2 (-7.0) | -92 broken; -58 drop every second row | 6 h hop-delay window; hop ratio band 0.9649-0.9951 |
| mule_account | x2 (-5.9); amounts x1.2 (-17.4) | -88.5 broken | inflow ceiling exactly at the generator's 5,000 maximum; 8-sender minimum; 24 h run gap; multiples-of-10 inflows |
| loan_app_extortion / refund repeat | x2 (-7.2 / -6.6) | -90.8 broken | 7-day repeat gap; 3-row minimum; whole-rupee lending |
| sim_swap / takeover drains | x3 only (-6.2 / -19.4) | -99 with step constant jittered | 3 h span; exact 0.25 / 0.35 balance steps |
| phishing / cashback collect | only broken amounts (-95 / -97) | | exact under-limit and bait amounts; timing robust |
| qr_swap rings | only broken amounts (-29) | | round-5 amounts |
| velocity_bot, marks, counter | never beyond 2 points (counter -6.1 broken amounts, -4.8 drop every second row) | | 300 s device session gap, 10-row minimum: robust to x3 and half bursts |

## October's measured drift [earlier]

Label-free quantiles of the families' own events, same code for every month:

- **Delays:** medians moved x0.73 to x1.12; tails up to x1.61 (bait answer p90: 29 h of a 72 h
  window); chain hop delay p75 moved x3.3 as a mixture shift (more hops from the slow mode) while p50
  is x1.03 and p90 x1.21 (4.0 h of a 6 h window). Ring and repeat gaps got shorter, not longer.
- **Bursts:** not shorter, except bot p90 length x0.75 (the family is robust to x0.5); collusion
  rings (9+ rows 23 vs 17-19), refund and lending runs got longer.
- **Amounts:** drain x1.17, mule onward x1.30, QR-to-person x1.19 (legit amounts x1.166); every exact
  constant intact (hop-ratio quantiles 0.968 / 0.980 / 0.992 identical; request median exactly 4,990;
  constant-keyed event rates x1.00-x1.08). The rate drops (bots x0.50, 9+ rings x0.50, fan-in x0.71)
  are volume, not parameters.
- **Decisive check:** widening any fragile tolerance finds **0 extra October rows** in every family,
  against 0.4-2.5 per 100k in train months: no October event sits beyond a current window.
- **Expected private-set risk:** amounts about -0.003, delays about 0, bursts about 0.

## Adversarial validation on the champion's own matrix [earlier]

329 columns, XGBoost depth 6, 200 rounds, three-fold out-of-fold AUC:

| pair | all rows | evidence rows |
|---|---|---|
| Jul v Aug / Aug v Sep / Jul v Sep | 0.528 / 0.512 / 0.534 | 0.833 / 0.829 / 0.825 |
| **Sep v Oct** | **0.729** | **0.893** |
| Aug v Oct / Jul v Oct | 0.732 / 0.735 | 0.881 / 0.890 |

- Almost half the gain is `ctr_dev_single` (device number seen once in its split): 8.1% of train
  rows vs 36.8% of October rows, a split-length artefact (train is three months, test one month at
  half density) carrying 0.02% of the champion's gain. The fraud rate by that flag is the same in
  both splits, so it does not move scores; it is the cleanest single October identifier.
- The rest is density (sender 7-day counts) and the known legit amount scale. 17 flagged features hold
  2.9% of the champion's gain (1.0% on evidence rows). **No event-family shape, size, span, delay or
  ratio column drifts.** Evidence rows are fingerprinted even between train months (0.83) because a
  month has few events; October adds only +0.06.
- An earlier gap analysis found that event ids (row-index minima) separate September from October
  perfectly (AUC 1.000) and a whole-frame device total takes half the adversarial gain; dropping
  the ten hygiene columns is a tie on September (0.5086 / 0.5632 vs 0.5088 / 0.5635).

## Insurance tolerance widening (all zero change on October rows) [earlier]

| widened constant | train effect | restores under stress |
|---|---|---|
| mule inflow ceiling 5,000 -> 6,000 | none (3,687 rows, precision 0.9040) | amounts x1.2: mule -17.4 -> -0.2 |
| chain hop delay 6 h -> 12 h | none (same 3,662 rows) | x3: -12.1 -> -1.0 |
| drain span 3 h -> 6 h | +6 rows; precision 0.9879 -> 0.9862 | x3: sim -6.2 -> 0, takeover -19.4 -> 0 |
| ring chaining gap 24 h -> 72 h | qr_swap ring recall -0.5 pt | collusion x3: -54.7 -> 0 |
| repeat gap 7 d -> 21 d | +12 rows, refund recall +0.6 pt | x3: refund -15.1 -> +0.6, loan -19.9 -> 0 |

Not run through retraining; expected tie. Gate item "delays x3 within 2 points" fails as written
(eight family-classes lose more than 2 points); the suggested restatement is to test x1.5, which
brackets every October median and tail. Mule would still lose about 8 points at x3 even with all
fixes. Do nothing for broken constants: the champion keys on them by design, and a generator change
that broke them would be visible label-free immediately (event rates would collapse).

## Lessons

- The pipeline is a bet on exact generator constants, and October kept all of them; check the
  constant-keyed event rates and hop-ratio quantiles on any new month before trusting a score.
- Slower, not faster, drift is what hurts; ring and repeat day gaps and the 24 h ring chain are the
  most fragile points.
- Keep the split-length and density artefacts out of the model matrix (hygiene); they identify
  October but carry no gain.
- The repository's own October-like check (nine cells, 0.5093 vs 0.5097 at full density
  **[repo]**, [more-data.md](more-data.md)) agrees: transfer to a thinned month costs almost nothing
  when the counter is held at its unthinned values.
