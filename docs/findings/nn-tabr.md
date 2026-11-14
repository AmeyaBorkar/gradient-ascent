# A TabR-style retrieval member

**Verdict: rejected.** In TabM's slot of the fraud mass it gains on September and s13 and loses on
s14 and s15 (s15 given -0.00034, s14 cleaned -0.00037). On its own it is far weaker than TabM:
-0.0082 to -0.0089 given in every cell.

## What was tested

`nnexp` variant `tabr` (`tabr.fit_predict`, `TabRSpec` defaults), seed 0, laptop GPU, against the
plain TabM refitted on the same GPU:

- inputs are TabM's quantile map and gap flags (337 inputs);
- a 2-layer MLP encoder with a linear skip, L2-normalised to 256 dims;
- the memory is every fraud row of the fit months (38,858) plus 200,000 seeded legit rows, held
  fixed within an epoch;
- each query attends over its 32 nearest memory rows (exact L2), with values
  `W E(x_c) + embed(y_c) + T(E(x) - E(x_c))`;
- a training row never retrieves itself; scored rows are never in the memory;
- 10 epochs, batch 2,048, AdamW 2e-3, the same warm-up and decay as TabM.

Wall time 8.7-9 min per cell on the laptop RTX 5070 Ti (about 49 s per epoch on 1.05 M rows); the
final training loss is 0.170, the same as TabM's.

## Results

Change x 1e-4, given / cleaned / gen-only, with the calls changed. In the October-like cells the
event-bootstrap 95 % interval of the fraud-only change follows.

| role | sep | s13 | s14 | s15 |
|---|---|---|---|---|
| fraud-only | +0.1 / +1.5 / +1.9 (16) | +0.6 / +2.5 / +3.8 (11) | -1.0 / -3.7 / -5.4 (6) | -3.4 / -1.1 / -0.2 (13) |
| fraud-only, 95 % interval (given) | - | [-3.7, +4.7] | [-4.5, +2.2] | [-9.2, +1.2] |
| alone | -88.6 / -103.2 / -139.7 (180) | -85.5 / -105.4 / -135.9 (110) | -81.9 / -104.1 / -143.5 (91) | -86.9 / -101.6 / -136.9 (99) |

**TabR beside TabM instead of in its place** (scoring only, no new fit). The TabM slot gets the
equal mean of the two seed-0 members (`wildrun-tabmtabr-sd0`):

| role | sep | s13 | s14 | s15 |
|---|---|---|---|---|
| fraud-only | +0.9 / +1.8 / +2.8 (6) | +1.3 / +3.1 / +4.3 (5) | +0.7 / -3.0 / -4.5 (3) | +0.4 / +0.5 / +1.4 (6) |
| alone | -7.1 / -6.1 / -17.0 (43) | -10.3 / -14.4 / -22.7 (44) | -4.3 / -3.4 / -4.2 (23) | -20.1 / -20.1 / -25.7 (36) |

Given is positive in every cell, but s14 loses on cleaned and gen-only, and only 3-6 calls move.
It fails the bar, and an October file from it would differ from the current one on a handful of
rows.

## What it means

- **Retrieval does not find what the features miss.** The member reaches the same training loss
  as TabM but scores about 0.0085 below it alone in every cell. The likely reason is that the
  feature families already summarise each row's neighbourhood (bursts, flows, repeats), so the
  retrieved labels add little beyond them.
- **In the fraud mass it is a different but not a better partner.** It changes 6-16 calls per
  cell, with mixed signs and every interval across 0. This is the same pattern as the RealMLP
  member: a second neural family moves calls, but not one way.
- **It was the only variant this round to pass two cells.** `recipes/wildrun-tabr.json` makes the
  3-seed member with one command (`python -m ringtrace.run recipes/wildrun-tabr.json`). It was not
  fitted, because the screen failed.
