# Entity embeddings and strictly-past pooled history for TabM

**Verdict: rejected.** It loses in every cell, on every target, in both roles. As the fraud-mass
source it costs 0.0004-0.0005 given, 0.0005-0.0006 cleaned and 0.0011-0.0018 gen-only. On its
own it costs 0.003-0.006 given. The embeddings learn the fit months' labels of each account, and
those labels do not carry over to the next month's rows of the same accounts.

## What was tested

`nnexp` variant `entity` (`nn_entity.fit_predict_entity`), seed 0, fitted on July + August and
scored on September (`sep`) and the r0 October-like frames s13 / s14 / s15:

- **Embeddings.** The `sender`, `receiver` and `device` ids are each hashed (one salt per column)
  into 2^18 slots of an 8-dim table, and the three vectors are concatenated to the TabM inputs.
  During training each id is replaced by a reserved unknown row with probability 0.2.
- **Past-pooled history.** For each row, two means of up to 8 strictly earlier rows: one over the
  same sender, one over the same receiver. Each covers amount, new payee, PIN retries,
  login-to-transaction seconds, SIM age, device age, balance and person-receiver, plus how many
  rows entered the mean: 18 inputs on top of the 337 TabM inputs.

Everything else is the plain TabM recipe (`recipes/tabm.json`).

## Results (box GPU, against the plain TabM refitted on the same GPU)

Change x 1e-4, given / cleaned / gen-only, with the calls changed. In the October-like cells the
event-bootstrap 95 % upper bound is also given (given / cleaned / gen-only).

| role | sep | s13 | s14 | s15 |
|---|---|---|---|---|
| fraud-only | -4.2 / -6.1 / -13.1 (46) | -4.5 / -5.1 / -17.6 (37) | -4.6 / -6.0 / -16.3 (25) | -4.7 / -4.7 / -11.2 (23) |
| fraud-only, upper bound | - | +0.6 / +0.0 / -7.3 | +0.1 / -1.3 / -7.5 | -0.1 / -0.0 / -1.6 |
| alone | -62.4 / -70.5 / -121.1 (169) | -25.3 / -34.9 / -62.4 (106) | -48.8 / -57.7 / -138.1 (112) | -45.5 / -53.2 / -96.5 (95) |

The training loss ends at 0.105, against about 0.17 for the plain TabM: the embeddings fit the
fit months' labels of each id.

ABLATION_PENDING

## Leakage checks

- **Ids first seen in the scored month carry nothing.** An id's embedding is trained only on the
  fit rows' labels (July + August). At prediction, an id absent from the fit rows of its column
  goes to the reserved unknown row (`nn_entity.unseen_to_unknown`). So neither an untrained slot
  nor a hash collision with a training id can reach the model. Shares of September rows whose id
  was seen in July + August: sender 0.906, receiver 0.862, device 0.878 (s13: 0.905 / 0.862 /
  0.877).
- **Seen ids carry only earlier months' labels.** The fit months are strictly before the scored
  month in every cell, so an embedding never encodes a label from the scored month or later.
- **Pooling is strictly earlier in time.** `past_pool` takes only rows with a smaller timestamp
  (a tie is not history) and reads features, never labels. The pool is the cell's own ledger, so
  in the October-like frames it is the thinned frame.

## What it means

The loss is not leakage: every check above holds, and leakage would have inflated the score, not
cut it. The low training loss points to memorization. With 86-91 % of September's ids already
seen, the model can lean on an id's July-August labels, and that history does not predict the
id's September rows as well as the row's own features do. Using ids for the full
leave-one-month-out bar would also need earlier-months-only training for July and August, which
the screen did not need.
