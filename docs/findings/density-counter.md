# The device counter at October's density

Question: October's device counter is computed on the October test file alone, at about half of
a train month's density. The October-like frames thinned September but kept its counter columns
at their values on the shared, full-density pass of all three train months. How much does that
overstate the best recipe, and do two fixes help: (a) dropping the absolute position column
`ctr_pos`, (b) training on a counter rebuilt at October's density?

Numbers and commands: `RESULTS.md`, section "Density: the counter at October's density".

## How the counter is rebuilt

- `ringtrace.density` renumbers one month's fraud-phase and legit-block phones as a pass of their
  own (own gaps kept, other months and dropped units removed) and decodes its fraud phase from its
  own anchors. On a thinned September: block assignment unchanged, segment agreement with the held
  counter 0.92 to 0.96, mean segment confidence 0.971 to 0.975 (October's own: 0.975; held: 0.993
  to 0.997). So the rebuilt counter looks like October's; the held one is cleaner than October's.
- A cell fits July + August once and predicts September twice: with the held counter and with the
  rebuilt one. Only September's counter columns differ (and what the template rule reads).
- `--train-counter half<d>` rebuilds every train month in two halves (protected units kept whole),
  each half a pass of its own, so the training rows see the counter at October's density.
- Month folds also predict each held-out month with its own standalone pass (`oof_alone.npy`) and
  in two halves at October's density (`oof_half0.npy`).

## Results

**The held counter overstates the best recipe by about 0.0035 macro-F1.** Same fits, only
September's counter rebuilt: given 0.5160 -> 0.5125 (-0.00354, 95 % [-0.00528, -0.00192]),
cleaned 0.5737 -> 0.5699 (-0.00388 [-0.00588, -0.00194]), gen-only -0.00297. Every member loses
0.002 to 0.006 per cell, TabM the most (about -0.005 given). Only 25 to 69 calls change per cell
of about 252,000 rows: the loss sits in the small classes, where a few calls move a class F1.
The October-like cells were optimistic by roughly this amount, uniformly enough that most past
comparisons between production-trained recipes still stand: the anchored template rule gains
more under the rebuilt counter (+0.0046 given against +0.0042 held), and TabM at half the fraud
mass still beats a third (+0.00024 [+0.00005, +0.00050]).

**(a) Dropping `ctr_pos` does not help.** The champion without it loses on every seed under the
rebuilt counter (-0.00032 given [-0.00088, +0.00023], cleaned -0.00035) and ties under the held
one. Rejected; the other members' variants and its month folds were not run.

**(b) Training on the counter at October's density helps, and the old cells hid it.**

| | October-density cells, given / cleaned | held cells, given | month folds at October's density, given / cleaned | month folds, production counter, given |
|---|---|---|---|---|
| champion alone | +0.00227 [+0.00109, +0.00363] / +0.00238 [+0.00116, +0.00379], every seed | -0.00100 | +0.00086 / +0.00063 / +0.00080; +0.00081 / +0.00118 / +0.00128 | -0.00078 / -0.00173 / -0.00096 |
| best recipe, champion only (chalf0) | +0.00001 (1 call per cell) | +0.00013 | -0.00001 / -0.00005 / -0.00001 | -0.00001 / -0.00004 / +0.00017 |
| best recipe, three XGBoost members (chalf3) | +0.00155 [+0.00072, +0.00252] / +0.00162 [+0.00076, +0.00266] (s15 -0.0002) | -0.00029 | +0.00025 / +0.00046 / +0.00092; +0.00025 / +0.00073 / +0.00127 | -0.00059 / -0.00028 / -0.00028 |

- Under the held counter the half-density members lose: the old validation would have rejected
  them. Under the rebuilt counter they win on every seed as members and on two of three seeds in
  the blend, with positive lower bounds.
- The production-counter month folds put the held-out month on the same full-density pass as the
  fit months. That is the held cells' flaw again, and they reject (b) for the same reason. The
  month folds that predict each held-out month in two halves, each its own pass (October's
  situation), are positive for chalf3 in every month on given and cleaned
  ([+0.00001, +0.00108] and [+0.00018, +0.00137]); gen-only is -0.0006 in July.
- The champion alone moves 1 October call of the blend; the three members move 11 (8 sim_swap ->
  account_takeover, 1 velocity_bot -> mule_account, 1 qr_swap -> phishing_collect, 1 qr_swap ->
  refund_abuse). Candidate: `candidates/density-half3.csv`.
- TabM is half the fraud mass and still trained on the full-density counter. On one frame (s13,
  both fits on the same GPU) TabM trained at October's density gains +0.0044 given / +0.0047
  cleaned on the rebuilt counter (0.5173 -> 0.5217). TabM on another GPU moves the rebuilt-counter
  cell by 0.0024, so TabM comparisons must stay on one device. This is the largest open lever.

## Verdicts

- Step 1: the held counter overstates the best recipe by 0.0035 given / 0.0039 cleaned.
- (a) no `ctr_pos`: rejected (loses on every seed under the rebuilt counter).
- (b) three XGBoost members at October's density: passes the October-density cells and the
  October-density month folds; fails the production-counter month folds, which carry the same
  density mismatch as the held cells. Whether that counts as passing the bar is a judgement on
  which month folds are honest; this finding argues for the October-density ones.

## Not done (time, one shared 8 GB GPU)

- The no-position variants of clean-event, clean-l2, LightGBM and CatBoost, and (a)'s month
  folds (the run failed for GPU memory while TabM was training).
- LightGBM and CatBoost at October's density; the second training draw beyond the champion's
  cells; the October-mix frames.
- TabM at October's density beyond one like-for-like seed: its s15 cell, its baseline cells on
  this GPU for s14 and s15 and its month folds are queued on the box after this report (the s14
  baseline failed for GPU memory once).
