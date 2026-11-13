# Open ideas: what was never tried, ranked by expected value

**Summary.** Almost everything on the model side has been tried and tied. The remaining ideas are
small (expected gains +0.0001 to +0.0015), and the biggest unknown is not a model at all: it is how
the test labels treat ordinary rows (about 63% of every fraud class), which swings the score by
roughly 0.35-0.40 ([regime-hybrid.md](regime-hybrid.md), [ordinary-rows.md](ordinary-rows.md)). The
list below is ordered by expected value per hour. Expected gains are **guesses** anchored on the
measurements cited, labelled **[earlier]** unless re-derived here. The gate used in earlier
research: a candidate must beat the champion in each leave-one-month-out month on given, cleaned and
generator-only labels, win in all nine October-like cells with the counter held at its unthinned
values, and report the paired event bootstrap; the adoption bar on the October-like mean was +0.007.
Back to the [index](index.md).

## Ranked list

| # | idea | why it might help | expected gain | hours | risk |
|---|---|---|---|---|---|
| 1 | **Finish the five-member equal-weight blend in this repo** (LightGBM, CatBoost, cleaned-label and noise-floor-flip XGBoost members) | the best honest combination was positive in every month and every thinning on all three targets; adopts stability, not score | +0.0002 to +0.0005 | 3-5 (mostly compute) | low; CatBoost on GPU is not byte-reproducible, needs CPU refit |
| 2 | **Protect October with insurance tolerances** (mule inflow ceiling 6,000; chain hop 12 h; drain span 6 h; ring gap 72 h; repeat gap 21 d) | zero October rows change today, so expected score gain 0, but it removes the generator-drift cliff (see [stress-test.md](stress-test.md)) | 0 now; protects up to -0.012 to -0.046 | 1-2 plus a comparison run each | low |
| 3 | **Nested ensemble selection plus one fraud type per event** | two measured positives (three-family rank blend +0.0010 [+0.0001, +0.0018]; event-type unification +0.0009 [+0.0001, +0.0019]) were judged against a +0.007 bar and never combined; greedy selection with replacement keeps weights sparse | +0.0005 (0 to +0.0015) | 1.5-2 if out-of-fold probabilities exist | low if nested, high if selected on the scored month |
| 4 | **Disagreement-footprint screen** (count reachable rows on which a candidate's final calls differ from the champion; ceiling about half of them x 2e-5 LOMO, 6e-5 per month) | bounds any member or decoder before building it | 0 (saves the hours of 3, 5, 6, and later members) | 0.5 | none |
| 5 | **Tabular neural-net member (a TabM-style ensemble with piecewise-linear embeddings), rank-blended** | the only untried family with evidence at 1M+ rows; nets and trees share fewer true positives (25-35%) than two trees (38-51%) so its footprint can exceed a GBDT's | +0.0003 (0 to +0.0015) | 5-7 incl. GPU | moderate: count features at half density; run the screen first |
| 6 | **Event-level expected-test-utility decoder** | the plug-in decode is the large-sample limit; October holds only a few dozen events of the rarest classes; events share a label | +0.0001 (0 to +0.0003) | 2-3 | low-moderate, blind where the model is blind |
| 7 | **Pin the feature cache per run** (hash every cached feature table into the run fingerprint) | a rebuilt device-counter table silently changed three October rows by up to 0.81 probability in an earlier run | safety, not score | 1 | none |
| 8 | **Monitor col7 -> cashback and mf2/3 -> mule on October** (aggregates only) | col7's cashback share is only 0.37-0.44 with a generator-row margin of 0.0000 in one month; mf2 error rose 0.043 -> 0.066 -> 0.093 across the months; the rule's minority class is growing in both | risk control | 0.5 | none |
| 9 | **Drop the dead features**: the inflow-share column that is infinite on 93% of rows, the split-length "device seen once" count | hygiene; both carry about 0 gain | tie expected | 0.5 | none |
| 10 | **October sim_swap vs takeover edge audit** (compare the counter's sim|ato confidence on October drain-3 rows with train) | five to eight rows on which seeds disagree sit near the counter's cut | up to +0.0003 | 0.5 | none |
| 11 | **Re-run the October-shaped training copies on frames that protect label-free event families** (the clean rerun was never done) | the first forward-fold result (+0.004 to +0.009 given) inflated the fraud prior to about 7%; honest thinning gave +0.0014 / +0.0016 (a tie); a later +0.0065 came from cut merchant-ring events. Closed by argument only | about 0 to +0.002 | 3-4 | high (only the mixed raw+shaped arm is robust); see [more-data.md](more-data.md) |
| 12 | **A GNN on the ordinary-rows-as-legit target; temporal message passing; GNN without event edges** | listed as untested; the transaction GNN tied elsewhere | 0 to +0.001 | 6+ | high |
| 13 | **Specialists trained on event-cleaned labels; event-level specialists** | untested follow-ups of the specialist study | ~0 | 3 | n/a |
| 14 | **A col1 history rule if the victim pool recurs** | AUC 0.74 vs 0.62 from other-month cashback history, but only 1.9% of October col1 events are covered | <= +0.0005 | 1 | n/a (do not build unless coverage rises) |

## Things explicitly **not** worth trying [earlier]

- more seeds, other GBDT learners, plain blends, per-class selection, per-class weights, stacking
  ([ensembles.md](ensembles.md));
- tuned or bagged multipliers, cut-offs, prior-shift methods, census-anchored counts, F/2 rules,
  proportional pool splits ([decoders.md](decoders.md));
- fine-tuned LLMs, TabICL/TabPFN pickers, sequence models and GNNs for the look-alike pairs
  ([lookalikes.md](lookalikes.md));
- robust-loss and noise-correction objectives (class-conditional noise moves only thresholds;
  small-loss pruning measured -0.002 to -0.004), confident-learning pruning ([labels.md](labels.md));
- new rules for the 255 generator rows without evidence (whole pool at most +0.0033), more
  sibling features, a 3-month refit for the test month ([more-data.md](more-data.md));
- test-time adaptation and label-shift estimators (no shift loss is left: an oracle prior gains
  +0.00005; the October-like loss with the counter held is -0.0006);
- new rule families on top of the saturated ones ([rules.md](rules.md)): 38 request variants, 12
  drain variants, 11 money-flow variants and 12 merchant variants all tied;
- a train-taught prior for the counter's October segments (cut positions and per-segment
  occupancy). **Measured in this repo**
  ([counter.md](../../findings/counter.md)):
  - the segments improve at October's size (+0.030 agreement);
  - the counter rule built on them loses in September;
  - October's calls already follow the segment, and where the model and the segment disagree the
    model is right on 17 of 19 train rows.

## What actually moves the score [earlier]

- **The regime question** (are ordinary rows fraud in the test labels?): +0.40 macro-F1 of stake
  against a reachable +0.0035 from the model. Spend upload slots reading it, not on model variants;
  public-slice noise (sd 0.010-0.023 between public and private) swamps everything under +0.002
  ([regime-hybrid.md](regime-hybrid.md)).
- **Robustness**: byte-reproducible rebuild, hygiene, exact-constant monitoring.
- **Information that does not exist in the data**: ordinary rows (AUC 0.50 under five angles) and
  look-alike partner labels (AUC 0.46-0.68). A perfect separator of the six look-alike pairs
  would add +0.029 / +0.030; ordinary rows +0.40; nothing measured reaches either.
