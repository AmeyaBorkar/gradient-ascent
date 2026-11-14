# TabM with a REINFORCE macro-F1 term

**Verdict: rejected.** As the fraud-mass source it changes 0-5 calls per cell and loses on
September (-0.00004 given) and s13 (-0.00019 given). Alone it moves 4-8 calls with mixed signs.
The term barely changes the model: the final training loss is 0.1688, against 0.1689 for the
plain TabM.

## What was tested

`nnexp` variant `metric` (`nn_metric.fit_predict_metric`, `metric_loss="reinforce"`, `lam` 0.5),
seed 0, laptop GPU, against the plain TabM refitted on the same GPU. Each batch's loss is the
class-weighted cross-entropy plus 0.5 x a score-function term. One label per row is sampled from
the head-averaged softmax, the batch macro-F1 of the samples is the reward, and the term is
`-(reward - baseline) x mean log p(sample)`, with an exponential moving average (0.9) as
baseline. The loop, preprocessing, optimiser and batch order are `tabm.train_one`'s.

## Results

Change x 1e-4, given / cleaned / gen-only, with the calls changed:

| role | sep | s13 | s14 | s15 |
|---|---|---|---|---|
| fraud-only | -0.4 / -0.4 / -0.5 (1) | -1.9 / +1.4 / +0.3 (5) | +0.6 / +0.5 / +0.2 (2) | +0.0 / +0.0 / +0.0 (0) |
| alone | +2.0 / +1.9 / +2.0 (6) | -2.4 / -2.6 / -6.6 (8) | +1.4 / +2.2 / +1.7 (7) | +0.0 / +1.0 / +1.7 (4) |

s13 fraud-only given: event-bootstrap 95 % interval [-4.2, -0.3]. Wall time 5.6-5.9 min per cell
on the laptop RTX 5070 Ti, the same as the plain TabM.

## What it means

- **A score-function estimate of batch macro-F1 is mostly noise at this batch size.** A 2,048-row
  batch holds about 75 fraud rows (3.7 % of the fit rows) across eleven classes, so the sampled batch macro-F1 swings
  widely between draws. Centred by its baseline, the term averages to about zero, and the
  training loss ends where the plain loss ends.
- **The calls that move come from noise, not from the metric.** The sampling draws from the GPU
  generator that also drives dropout, so from the first batch on, the variant's dropout masks
  differ from the base's. That is a seed change, and the 0-8 moved calls with mixed signs look
  like one.
- **The decoder already optimises macro-F1 on calibrated probabilities.** A training-time metric
  term can only help by changing the ranking inside a class, and this one does not change it.
