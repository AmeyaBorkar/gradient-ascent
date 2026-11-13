# Knowledge base: the data and how it was generated

**Summary.** The UPI Fraud Rings data is synthetic. A two-pass generator wrote 1.5 million legit-pass
rows and 64,060 fraud-pass rows for July to September (train), and 250,000 plus 10,582 for October
(test, one sixth of the size). Only about 37% of fraud labels sit on rows with a detectable template; the
other 63% are ordinary-looking rows nothing can find, which caps a structure-only model at about 0.57
macro-F1. Train labels are noisy (12% unreported, 2 to 3% wrong type) and test labels are clean. October
changes in four ways: half the legit density, legit amounts x1.166, two dead columns, and some fraud
pools shifting (bots about halved). A device-number counter above 420000 records which class routine
wrote each row and is the strongest non-row signal.

## Read in this order

| File | What it holds |
|---|---|
| [competition.md](competition.md) | split, metric, upload limits, public/private, final ranking; tables and columns; dead train-only columns |
| [generator.md](generator.md) | two passes, template against overwritten rows, stale-field contradictions, fixed pools |
| [classes-and-templates.md](classes-and-templates.md) | every class's signature, sibling and look-alike pairs, partner-label share by design, priors |
| [label-noise.md](label-noise.md) | unreported and wrong-type rates, regime A against B, what the all-legit upload showed |
| [ordinary-rows.md](ordinary-rows.md) | the 63% of fraud labels on ordinary rows, why no model finds them, the ceiling |
| [device-counter.md](device-counter.md) | the counter and its zones, locating them in October, the bot device pool |
| [october.md](october.md) | density, amount scale, dead columns, bot and collusion volume, entity pools and rings, drift cost |

## The numbers to remember

| Fact | Value | Status |
|---|---|---|
| Train / test rows | 1,564,060 / 260,582 | earlier measurement |
| Fraud-pass rows (train / test) | 64,060 (4.10%) / 10,582 (4.06%) | earlier measurement |
| Fraud-labelled train rows | 57,648 (3.69%) | earlier measurement |
| Ordinary rows among fraud labels | 63.0% (36,318 plus or minus 175); this repo's cleaned labels: 62.9% | earlier measurement; second figure measured in this repo |
| Unreported fraud (train only) | 12.0 to 12.2% per row | earlier measurement |
| Random wrong type (train only) | 2.3 to 3.0% per row | earlier measurement |
| Public / private split; uploads | 30% / 70%; 20 a day | competition statement |
| All-legit public score | 0.0816, i.e. test fraud share about 4.06% (ordinary rows count) | earlier measurement |
| Perfect structure-only model | about 0.57 macro-F1 | earlier measurement |
| Champion, leave-one-month-out | 0.5122 given / 0.5665 cleaned; generator-only 0.9010 | measured in this repo |
| Counter features | worth 0.028 macro-F1 | measured in this repo |
| First model upload, public | 0.5470 (forecast about 0.54) | earlier measurement |
| Legit amount scale in October | x1.165 to 1.166; fraud amounts unchanged | earlier measurement |
| Legit density in October | 0.494 times train | earlier measurement |
| Bot bursts October | 7 (train 25 to 33 a month; 14 expected) | earlier measurement |

## Status labels used in these files

- **Measured in this repo**: this repository's own runs, logged in `RESULTS.md`; each such claim states
  the command (the main one is `python -m ringtrace.build recipes/champion.json`).
- **Earlier measurement**: from the earlier research pack. A later pass may re-derive it; the
  pack's analysis scripts are not in this repository, so re-deriving means rewriting a short query.
- **Belief** and **guess** mark things inferred, not shown.

## Known disagreements in the earlier measurements (handle with care)

- Noise rates: 12.0% (first map) against 12.2% (interval 11.5 to 12.9) in the later model; this base uses
  the range.
- The stale-field table (`c2`, `c4`, age-error counts) was corrected after a recount; use the corrected
  figures in [generator.md](generator.md).
- The first device-counter map ordered the zones differently from the later map; see
  [device-counter.md](device-counter.md), section 2.
- "30% of fraud labels are ordinary" was a first estimate and is wrong; 63% is the figure.
- The fraction of structure present in each class is a class-level range (58 to 70%), not a constant.

## Open questions

- Zone order and boundaries inside the October fraud block (the last three zones are about 20 ids each).
- Whether clean test labels keep the partner labels of look-alike events (belief: yes).
- Why about 60% of the ids inside the fraud blocks are unused.
- Whether the collapse of bot bursts to 7 is a generator change or chance (p about 0.03).
- Ordinary-row structure not yet tested: devices that are neither the sender's own phone nor a new
  single-use device, whole-history sequence models, joint structure of several ordinary rows.
