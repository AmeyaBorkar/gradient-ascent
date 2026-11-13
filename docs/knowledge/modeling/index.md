# Modelling, decoding and validation: overview

**Summary.** The pipeline turns label-free event evidence plus relational window features into
P(fraud) x P(type | fraud) with a two-stage XGBoost cascade, decodes the probabilities for macro-F1
with a label-free fixed point, then settles look-alike event types with a counter-anchored template
rule. Validation is leave-one-month-out on whole events, scored on three targets, backed by an
October-like month and a paired event bootstrap. After the evidence layer was built, every model,
tuning, noise-handling and decoder variation tied; the remaining gap is information (rows that look
legitimate), not modelling. Changes are judged by consistent, interval-backed gains, never by the
public score.

## Status labels used in this folder
- **[measured in this repo]**: shown by this repository's own runs. The runs are in
  [`RESULTS.md`](../../../RESULTS.md); the command to re-derive is given next to the claim.
- **[earlier measurement]**: from research done before this repository existed (different code
  base, same data and same task). A later pass may re-derive it here; treat it as strong prior,
  not as a result of this repo.

## Headline numbers
| item | value | status |
|---|---|---|
| Champion, leave-one-month-out macro-F1, given labels | 0.5122 (fixed point 0.5064) | measured in this repo (`python -m ringtrace.build recipes/champion.json`) |
| Same, cleaned labels | 0.5665 (fixed point 0.5601) | measured in this repo |
| Same, generator-only target | 0.9010 | measured in this repo |
| October-like September, given labels, fixed point | 0.5093 (9 cells) against 0.5097 at full density | measured in this repo (`python -m ringtrace.octlike run ...`) |
| Context-only model (123 relational features) | 0.4671 given / 0.5192 cleaned | measured in this repo |
| Ceiling if every generator-made row were typed right | about 0.588 given / 0.611 cleaned | earlier measurement |
| Public score of the XGBoost champion (an earlier build with the same design) | 0.5470 | earlier measurement |

## Files
| file | covers |
|---|---|
| [features.md](features.md) | feature families, counter, hygiene against month leakage, ablations |
| [cascade.md](cascade.md) | screening score, type rows, dropped legit rows, tuning that tied |
| [decoding.md](decoding.md) | fixed point, fraud scale, why cut-offs fail, template rules, one type per event |
| [training-labels.md](training-labels.md) | given, cleaned (teacher) and event-cleaned labels; noise measurements |
| [validation.md](validation.md) | folds, three targets, October-like month, event bootstrap, nested selection |
| [upload-rule.md](upload-rule.md) | the "consistently better" rule and the candidates it rejected |
| [public-noise.md](public-noise.md) | leaderboard noise model and how to read differences |

## Ten things to know before touching anything
1. About 63% of fraud-labelled rows are ordinary-looking (they sit at the legit rate on every
   measured column). They cap the score near 0.55-0.60 and no model or decoder reaches them.
2. The evidence families and the device counter carry the score: the counter alone is worth
   +0.028 macro-F1 (measured in this repo, run 12 in `RESULTS.md`).
3. Learner, seeds, depth, rounds, blends, stacking, noise-robust losses, label repair variants all
   tie within about +-0.001 (earlier measurement; the repo's own ablations agree where they overlap).
4. Probabilities are bimodal, so fraud cut-offs between about 0.2 and 0.95 call the same rows.
5. The decoder is label-free and solved per month; tuned multipliers and cut-offs lose.
6. Never cut events across folds; random K-fold inflates macro-F1 by 0.006 (given) and 0.015
   (generator-only).
7. The October-like month must keep family events whole and take the counter from the unthinned
   frame; otherwise it is biased.
8. Compare with a paired bootstrap over events; a tie goes to the simpler option.
9. Public-slice noise is about 0.010-0.025 for one upload and 0.002-0.008 for two near-identical
   uploads; gaps inside that are ties.
10. Test rows are used only through code and aggregates; no outside data; no uploads from this
    repository.
