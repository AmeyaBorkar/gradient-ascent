# TabM trained on generator-only labels

**Verdict: rejected.** Training the fraud-mass source on generator-only labels trades the given
and cleaned targets for the gen-only target. As the fraud-mass source it loses 0.0012-0.0022
given and 0.0011-0.0025 cleaned in every cell, with event-bootstrap upper bounds below 0 in every
October-like cell.
It gains 0.0015-0.0030 gen-only in the three cells whose target is sound. The leaderboard sits
on the given / cleaned scale (public 0.55, against 0.90 for a gen-only target), so this moves
calls the wrong way.

## What was tested

`nnexp` variant `gentarget`, seed 0, box GPU, against the plain TabM refitted on the same GPU.
The plain TabM recipe, but the fit rows' target is their generator-only label: fraud labels that
look ordinary become legit, unreported generator-made rows become fraud and off-event types are
corrected.

**The target is rebuilt per cell from July + August alone.** The cached silver labels cannot be
used for training. Their teacher is out of fold by month, so the July and August labels come from
a booster that saw September. `octlike._cleaned_target(cell, which="structured")` cross-fits the
teacher between July and August, and every estimate in `silver.build` (isotonic calibration,
evidence ratios, event majorities) uses the fit rows only. The scored cell's labels are never
read.

| cell | fraud labels kept as fraud (of 38,858) | legit-labelled rows made fraud | ordinary share of fraud | old-app likelihood ratio |
|---|---|---|---|---|
| sep | 16,369 | 1,951 | 0.629 | 2.77 |
| s13 | 16,385 | 1,950 | 0.629 | 5.11 |
| s14 | 16,443 | 1,959 | 0.627 | 10.47 |
| s15 | 6,321 | 782 | 0.857 | inf |

## Results

Change x 1e-4, given / cleaned / gen-only, with the calls changed. In the October-like cells
the event-bootstrap 95 % interval follows (given / cleaned / gen-only).

| role | sep | s13 | s14 | s15 |
|---|---|---|---|---|
| fraud-only | -20.3 / -18.5 / +29.5 (45) | -21.7 / -18.9 / +25.5 (22) | -11.9 / -10.8 / +15.2 (12) | -21.0 / -24.5 / -4.9 (61) |
| fraud-only, 95 % interval | - | [-36.5, -9.5] / [-32.6, -8.0] / [+12.7, +44.1] | [-22.4, -3.6] / [-20.9, -3.0] / [+5.3, +29.4] | [-43.8, -4.7] / [-50.4, -6.5] / [-43.7, +25.7] |
| alone | -155.3 / -141.9 / +198.6 (404) | -156.6 / -146.1 / +195.2 (218) | -163.1 / -153.1 / +180.8 (216) | -523.2 / -553.1 / -283.3 (746) |

## What it means

- **The trade is real and large, not noise.** Every given and cleaned interval lies below 0. The
  gen-only intervals lie above 0 in s13 and s14. A detector that never sees ordinary fraud rows
  as fraud gives them less fraud mass. Those rows are fraud in the given and cleaned targets and
  legit in the gen-only target, so the result follows the target.
- **The remaining detection error does not come from noisy fraud mass.** Removing the ordinary
  rows from training does not sharpen the findable rows enough to pay for the ordinary rows it
  gives up: every given and cleaned change is negative. It would pay only if the test labels
  were generator-only, and the public scores rule that out.
- **`silver.build` is fragile on two months.** The old-app likelihood ratio is a ratio of two
  small rates (old app among confident generator-made rows and among confident legit rows). Fit
  on July + August alone, it swings from 2.77 to 10.47, and in s15 no confident legit row has an
  old app version. There the ratio is infinite and 86 % of fraud labels are judged ordinary
  instead of 63 %. Any label built with `silver.build` on a subset of months (the October-like
  `cleantrain` cells use the same path, with `which="label"`) inherits this. A floor on that rate
  would fix it. The shared label code is left unchanged here.
