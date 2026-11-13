# The upload rule: consistently better, not just large

**Summary.** A candidate B is uploaded against the champion A only when it is better *everywhere it
can be checked*: the event-bootstrap lower bound of B minus A is above zero on the October-like
month and the gain is positive on every thinning seed; B loses in no leave-one-month-out month on
given or cleaned labels, nor on generator-only rows; each piece of a stack still helps when left out,
chosen on other months; and the pre-upload checks pass. This replaced an earlier "at least +0.007"
bar that no single change could reach. Nothing tried since the evidence layer was built has passed it
(earlier measurement). This repository makes no uploads itself.

## Why the rule changed
- The first bar was "+0.007 or more on the October-like month", set against validation noise (seed
  sd about 0.0036 and thinning-seed spread about 0.004 in the early harness).
- No single change reaches +0.007; real gains are stacks of about +0.001.
- The platform scores the best *public* upload on the private slice, so an upload can only hurt when
  the candidate is truly worse than the champion and gets lucky on public. Deciding by "is B better
  than A, consistently" is therefore enough; one upload per candidate.
- Public differences between near-identical files are mostly noise ([public-noise.md](public-noise.md)),
  so the decision rests on local intervals, never on the public score.

## The conditions (all must hold)
1. **October-like month.** Event-bootstrap 95% lower bound of B - A above 0, on three thinning
   seeds x three model seeds, with the counter taken from the unthinned frame
   ([validation.md](validation.md)). The seed-averaged gain must be positive on **every** thinning
   seed. Use the families mode for anything that touches ring or repeat classes.
2. **Leave-one-month-out.** B does not lose in any of July, August or September, on given labels or
   on cleaned labels.
3. **Generator-only rows.** B does not lose.
4. **Stacks.** Each piece still helps when left out, measured on months not used to choose it
   (nested selection). Pieces chosen because each looked positive on the same month inflate the
   pooled gain.
5. **Housekeeping.** The pre-upload audit has no failure; the October footprint (calls per class)
   against the champion is recorded; a forecast is written before the upload; the record is filed.

Older gate wording (earlier measurement): a change is real when it is more than 2 sd on the headline
fold with the same sign on at least four of five folds; single-fold gains under 0.004 (given) or
0.008 (generator-only) are ties; ties go to the simpler option (fewer features, parameters,
moving parts). A champion changes only if the challenger beats it on the canonical folds, is no worse
than noise on any fold or flagged slice, rebuilds to the same hash twice, and CV and leaderboard are
consistent.

## What the rule rejected (earlier measurement)
| candidate | outcome |
|---|---|
| Build with later changes stacked (pooled event-bootstrap vs earlier champion) | lower bound -0.0000 on both targets, though it loses in no month: fails condition 1 |
| Type-row cut 0.05 (best single sweep change) | shrinks from +0.0004 (one seed) to +0.0001 / 0.0000 (two seeds averaged): fails 1 |
| Best of a 160-trial settings search | +0.0013 on the tuning month, +0.0002 in the families mode; loses August on leave-one-month-out by 0.00003-0.00034 for all five top settings: fails 1 and 2 |
| Type-row cut 0.04 with the stale-column drop and no LightGBM | families mode +0.0001 [-0.0002, +0.0003]: fails 1 (generator-only +0.0010 [+0.0004, +0.0016]) |
| Model families, seeds, blends | +0.0001 to +0.0002, intervals touching zero |
| Noise-robust training, decoder variants | ties or losses (see [training-labels.md](training-labels.md), [decoding.md](decoding.md)) |
| October-shaped training copies | artefact of the default thinning mode: not uploaded |
| Per-class model choice, per-class blends, type stacker | held-out Sep negative or interval touching zero: fail |

## Consequences and cautions
- Small real gains can reach the leaderboard; the bar is consistency, not size.
- **Winner's curse:** when several passing candidates are uploaded, the best public one is chosen
  partly on noise. Because each passed conditions 1-4, a truly worse one is unlikely to be among
  them. Best public of eight candidates spread over 0.013-0.019 is the truly best in 66-84% of
  draws and costs 0.002-0.003 on private.
- **Do not upload:** decoder variants of the same model; two near-identical files in the hope of
  picking one by public score; anything chosen by a public gap under the noise threshold
  ([public-noise.md](public-noise.md)); anything that changes labels on specific test rows.
- The first upload of an all-legit file is a probe: it can never become the best upload, and it
  reads the test fraud share ([public-noise.md](public-noise.md)). If more than one final entry is
  allowed, pick two that fail differently: the champion and the most robust candidate (for example
  the best blend or the best under the October shift).
- A better-looking public score never overrides local intervals. Rules for this repo: train rows may
  be inspected, test rows used only through code and aggregates; no external data; no uploads.
