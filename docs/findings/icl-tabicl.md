# TabICL v2 as a blend member: rejected

**Question.** Does a pretrained in-context tabular model (TabICL v2, decision 16 in
`docs/decisions.md`), refitting the champion's fraud stage on its most uncertain rows, add
anything alone or inside the five-member blend?

**Answer (measured in this repo).** No. Alone it loses to the champion everywhere except the
generator-only target. Inside the blend it moves the fourth decimal: up in July and August, down
by a hair in September. It fails the "never loses" rule at both weights, and it ties on the
October-like gate.

## Design

`python -m ringtrace.icl run --name icl-tabicl --estimators 8 --max-context 100000` (module
docstring; decision 16 has the details).

- **Candidates.** The champion's fraud probability `1 - p(legit)` must be at least
  tau = 2.18e-5. That threshold puts 5.0 % of the train rows in the candidate set.

  | month | rows | candidates | fraud-labelled rows | share of fraud covered |
  |---|---|---|---|---|
  | July | 527,065 | 26,977 | 19,513 | 41.6 % |
  | August | 526,852 | 22,901 | 19,345 | 40.2 % |
  | September | 510,143 | 28,418 | 18,790 | 41.1 % |
  | October | 260,582 | 15,013 | — | — |

  30.2 % of the train candidates are fraud-labelled. The other 59 % of fraud rows lie below tau:
  the champion is already sure of them.
- **Model.** Binary fraud vs legit with the top 100 of the champion's 230 columns (by
  `|AUC - 0.5|` on the context) and 8 estimators. The context is every candidate row of the other
  months (49,878-78,296 rows), so no subsampling was needed. The checkpoint supports 10 classes,
  so there is no 12-class variant.
- **Wall time.** Single RTX 4090, held to 12 GB: 48 / 35 / 34 s for the three folds and 69 s for
  October, 194 s for the whole run. Each October-like cell took about 30 s.
- **Determinism.** Fold September run twice gives bit-identical probabilities, also identical to
  the full run.

## Ranking on the candidates (out-of-fold, given labels)

| month | AUC TabICL | AUC champion | log-loss TabICL | log-loss champion | correlation |
|---|---|---|---|---|---|
| July | 0.9465 | 0.9478 | 0.205 | 0.468 | 0.994 |
| August | 0.9453 | 0.9463 | 0.215 | 0.497 | 0.994 |
| September | 0.9434 | 0.9450 | 0.226 | 0.446 | 0.964 |

TabICL ranks the candidates slightly worse than the champion. Its probabilities are better
calibrated on this slice, but the decoder sets its own operating point, so calibration buys
nothing. The two scores correlate at 0.96-0.99.

## Leave-one-month-out (fixed point at 0.875, then the anchored template rule)

Change per month (Jul / Aug / Sep) and the mean:

| comparison | given | cleaned | gen-only | event bootstrap, given / cleaned (5-95 %) |
|---|---|---|---|---|
| member alone vs champion | −0.0026 / −0.0018 / −0.0149 (−0.0064) | −0.0024 / −0.0019 / −0.0141 (−0.0061) | +0.0003 / −0.0004 / +0.0134 (+0.0044) | [−0.0072, −0.0057] / [−0.0069, −0.0054] |
| six-member blend, member at 1/6, vs the blend | +2.8e-4 / +5.1e-5 / −4.0e-6 (+1.1e-4) | +2.6e-4 / +4.9e-5 / −3.3e-6 (+1.0e-4) | +3.5e-4 / +8.1e-5 / +8.2e-5 (+1.7e-4) | [+0.0000, +0.0002] / [+0.0000, +0.0002] |
| member at 1/3 (others 2/15) vs the blend | +4.8e-4 / +8.1e-5 / −2.2e-6 (+1.9e-4) | +4.4e-4 / +7.4e-5 / −4.2e-5 (+1.6e-4) | +6.5e-4 / +1.7e-4 / +3.1e-5 (+2.9e-4) | +0.0002 [+0.0001, +0.0003] given, +0.0002 [+0.0000, +0.0003] cleaned |

The blend scores 0.5123 given / 0.5667 cleaned here, as before.

## October-like (9 cells, seeds 13-15 x sets r0-2)

Change per thinning seed (s13 / s14 / s15) and the mean of the 2.5 % bounds:

| challenger vs champion | given | cleaned | gen-only | gate |
|---|---|---|---|---|
| member alone | −0.0229 / −0.0212 / −0.0185, bound −0.0235 | −0.0235 / −0.0217 / −0.0194, bound −0.0242 | −0.0406 / −0.0381 / −0.0278 | fail |
| champion x 5/6 + member x 1/6 | +0.0000 / +0.0001 / +0.0000, bound −0.0001 | +0.0001 / +0.0001 / +0.0000, bound −0.0000 | +0.0001 / +0.0002 / −0.0000 | fail |
| champion x 2/3 + member x 1/3 | +0.0000 / +0.0001 / +0.0000, bound −0.0001 | +0.0002 / +0.0002 / +0.0000, bound −0.0001 | +0.0002 / +0.0001 / −0.0001 | FAIL (tie) |

The other four members have no October-like cells, so the six-member blend cannot be gated
there. The champion + member mixes stand in for it, which assumes the other members match the
champion. They nearly do: `docs/findings/member-disagreement.md` counts at most 95 differing
rows per member.

The member alone loses about twice as much on the thinned September as on the full one. It
was trained on full-density context, and its fraud probabilities do not survive October's
lower density.

## Verdict

**Rejected.** Alone, it loses under the bar on every count. Inside the blend, the gain is about
1e-4 and September is negative at both weights, so it fails "non-negative in every month and
target". The October-like gate is a tie at best. The extra member is not worth its cost.

Reproduce: `python -m ringtrace.icl run ...` and `python -m ringtrace.icl octlike ...` on the
GPU, then `python -m ringtrace.blend recipes/icl-tabicl-{solo,blend6,blend3}.json`,
`python -m ringtrace.compare <base> <challenger>` and
`python -m ringtrace.octlike score champion "<mix>"`.
