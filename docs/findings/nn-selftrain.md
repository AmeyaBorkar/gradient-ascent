# Soft self-training of TabM on the scored rows

**Verdict: rejected.** As the fraud-mass source it changes 0-2 calls per cell and loses on
September (given and cleaned -0.00001) and on s13 (given -0.00012). On its own it moves 8-16
calls with mixed signs (s13 -0.00037 given, s15 -0.00021 given, s14 +0.00019). It adds nothing
the plain TabM does not already give the blend.

## What was tested

`nnexp` variant `selftrain` (`nn_selftrain.fit_predict_selftrain`), seed 0, laptop GPU, scored on
September (`sep`) and the r0 October-like frames s13 / s14 / s15 against the plain TabM refitted
on the same GPU:

- **Teacher.** The plain seed-0 TabM's probabilities of the scored rows (`base-sd0`, fitted on
  July + August). It never saw those rows' labels.
- **Pseudo rows.** Every scored row (510,143 on `sep`, about 252,000 in each October-like frame)
  keeps the teacher's full probability vector as a soft target, with weight 0.3 against 1 for a
  labelled row. Only the scored cell's features are used: the held-out month on `sep`, the
  thinned frame in s13-15, never October.
- **Caps.** `min_conf` 0. Each fraud class is capped at the floor of the teacher's expected count
  `floor(sum_i P[i, c])`. The cap never binds there: the teacher's fraud argmax counts sit well
  below its mass (September: 8,215 fraud argmax calls against 19,945 fraud mass). Legit is left
  uncapped on purpose. Its cap (490,197 against 501,928 legit argmax rows on `sep`) would drop
  the 11,700 least confident legit rows. Those rows hold most of the ambiguous fraud mass, so
  dropping them would tilt the pseudo prior toward legit.

## Results

Change x 1e-4, given / cleaned / gen-only, with the calls changed:

| role | sep | s13 | s14 | s15 |
|---|---|---|---|---|
| fraud-only | -0.1 / -0.1 / +0.2 (2) | -1.2 / +2.3 / +2.7 (2) | +0.0 / +0.0 / +0.0 (0) | +0.0 / +0.0 / +0.0 (0) |
| alone | -0.1 / -0.0 / +1.7 (10) | -3.7 / -3.6 / -3.1 (12) | +1.9 / +3.0 / +8.1 (8) | -2.1 / -1.3 / -5.1 (16) |

Event-bootstrap 95 % intervals contain 0 in every October-like cell (for example s13 alone given
[-9.3, +0.8]). Wall time 7-8.5 min per cell on the laptop RTX 5070 Ti (1.31-1.56 M training rows).

## What it means

- **Self-training on the model's own soft output is close to a no-op for the blend.** The student
  is pulled toward the teacher on exactly the rows it is scored on. So its legit column moves
  only where the teacher was unsure, and at half weight in the fraud mass that moves 0-2 calls.
- **The differences alone are the size of fit noise.** Two plain fits of the same seed on two
  GPUs differ by 70 September argmax calls. The self-training loop draws its batch order
  differently from `tabm.train_one`, so part of its 8-16 changed calls is that noise.
- **Not tried:** hard pseudo-labels or a confidence cut. Both shift the pseudo prior toward legit,
  which is the direction the decoder punishes, and the soft version already shows no gain to
  sharpen.
