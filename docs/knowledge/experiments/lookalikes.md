# Look-alike (sibling) separation

**Summary.** Five event shapes ("templates") are written by two different fraud classes that look
identical: ato/sim_swap drains, mule/chain hops, mule/bot bursts, cashback/phishing collect
sessions, refund/qr_swap merchant bursts (plus a sixth, a second collect shape). The device-counter
zone separates them when the event has a fraud-block device; otherwise nothing separates them. We
tried hand-built event features, label history, graph propagation, deeper and pretrained models
(GBDT, TabICL, TabPFN), a fine-tuned 7B language model, a transaction GNN and per-family
specialists: **none beats the cascade by more than 0.05 AUC, and none moves macro-F1.** A perfect
separator for all six pairs would add at most +0.029 (cleaned +0.030), but the best AUC measured is
0.55-0.75 and +0.003 needs AUC >= 0.80. The one lever is the **template rule** (every call in a
pair takes the pair's majority class), worth +0.0027 / +0.0032 and now in the champion.
Conclusion: **stop the look-alike work.**

Status key: everything here is **[earlier]** (no run of this repository measures separators; the
repository's own measurement is the template rule's contribution, see [rules.md](rules.md) and
[counter.md](counter.md)). Back to the [index](index.md). Related: [ensembles.md](ensembles.md),
[ordinary-rows.md](ordinary-rows.md).

## Overview

| experiment | best result | verdict |
|---|---|---|
| hand-built event features (about 60, LOMO) on five pairs | AUC 0.42-0.69, macro-F1 gain +0.000 (pair 4 +0.002) | no gain |
| deeper / longer GBDTs, 1,400 event aggregates, other-month history, LR, kNN | within 0.05 of the cascade; col1 only via history | no gain |
| TabICL v2 (rows and events) | 0.46-0.75; at or below the cascade except col1 with history | no gain |
| TabPFN v2 as picker inside pools | AUC 0.46-0.68; -0.0002 tied, -0.0108 forced to 0.5 | no gain |
| fine-tuned 7B LLM (QLoRA) | 0.50-0.59; below cascade on all three pairs | no gain |
| text-matched GBDT control | 0.54-0.63 | no extra information in text |
| transaction GNN (alone / as columns / blend) | -0.0173 / -0.0030 / tie | no gain |
| two-hop graph propagation of zone class | 0.48-0.52 for every pair | dead |
| family specialists (one model per family) | +0.0002 best with the rule; -0.007 to -0.010 without | tie |
| other-month label history for col1 | AUC 0.74 vs 0.62; +0.0002 to +0.0005 | real but not reachable in October |
| linkage, victim-selection, discrete-flag, structural tests | AUC <= 0.58 except noise | dead |
| template rule: one class per template | +0.0027 given / +0.0032 cleaned | **helps** |
| proportional split of the pool | -0.0001 to -0.016 | loses |
| one-bait fast-payback group to cashback | +0.002 train | folded into the rule |
| col1 history rule | +0.0005 / +0.0002 | do not build |
| October bots moved to another shape? | no | prior: bot about half of train |

## The pairs, and where the loss sits [earlier]

| pair (shape) | wrong rows (3 months) | within-shape model AUC | stake if perfect (cleaned macro) |
|---|---|---|---|
| ato -> sim_swap, drain 0.25 full signature | 1,567 rows, 114-151 errors per month | 0.63-0.64 | 0.0122 |
| cashback <-> phishing, col1 | 2,008 rows | 0.62 | 0.0074 |
| phishing -> cashback, col7 | 536 rows | 0.46-0.49 | 0.0065 |
| mule -> chain, mf1 | 2,633 rows | 0.64 | 0.0049 |
| bot -> mule, mf2/3 | 3,273 rows | 0.37-0.46 | 0.0028 |
| refund <-> qr_swap / collusion, rings | 3,095 rows | 0.72-0.90 | 0.0010 each |

Oracle total for type confusion (all generator rows typed right) is +0.060 given / +0.046 cleaned,
of which the reachable part with one class per template is +0.0027 / +0.0032. Within the
look-alike templates the model is at chance; it already separates templates the counter zones
separate (drain3/4 at AUC 0.986, collect shapes 4/5/6/9 at 0.999, rings 0.72-0.90). On fraud-block
devices the type is right 97.8% of the time (zone = truth 98.5%), elsewhere 91.3%.

## Hand-built features on event aggregates (sibling separation) [earlier]

About 60 label-free event aggregates (size, timing, amounts, ratios, SIM and device age, city,
login time, PIN retries, KYC, account-pool reuse, rows of 17 other templates touching the accounts,
other-month label history), HistGradientBoosting, leave-one-month-out, permutation null (50). Truth
for an event is its zone class if it has a fraud-block device, else the majority label.

| pair | events no-zone / zone | look-alike share | AUC mean (range) | null p | model gain | ceiling |
|---|---|---|---|---|---|---|
| ato vs sim_swap drains | 605 / 579 | 0.260 | 0.622 (0.61-0.63) | 0.02 | +0.000 | +0.012 |
| fast chains: chain vs mule | 378 / 50 | 0.212 | 0.487 | 0.55 | +0.000 | +0.005 |
| mule bursts: mule vs bot | 114 / 89 | 0.114 | 0.509 | 0.43 | +0.000 | +0.003 |
| one bait + payback: phishing vs cashback | 228 / 26 | 0.421 | 0.456 | 0.86 | +0.002 | +0.008 |
| round-amount merchant bursts: qr_swap vs refund | 148 / 15 | 0.088 | 0.419 | 0.75 | +0.000 | +0.001 |

- Ceilings total about +0.029. Propagating the zone class from the device row to the whole event adds
  about +0.003 (+0.0006 / +0.0004 / +0.0011 / +0.0005 / +0.0001), already in the base. Share of
  events with a zone: pair 1 49%, pair 3 44%, pair 2 12%, pair 4 10%, pair 5 9%.
- Pair 1: all 137 weakened no-zone drains are the template's own class; look-alikes are always
  full-signature. Among full-signature 0.25 drains the look-alike share is 41.7 / 36.9 / 16.0% for
  2 / 3 / 4 rows, never above 50%, so the majority rule is right. Pooled AUC 0.69 is login time,
  duration, SIM-age flag and length marking weakened variants; inside the full group it is 0.590.
  Predicting the sibling for every no-zone drain costs -0.027.
- Pair 4 (phishing/cashback fast payback): no separator (p = 0.86) but assigning the group to
  fake_cashback gains +0.0035 / +0.0014 / +0.0015 by month (+0.002 pooled) because cashback has the
  lower F1; it is a decoder choice and now sits in the template rule (col7 -> cashback).
- Pairs 2, 3, 5: nothing beats the null; the strong single features rest on 13 look-alike events
  and vanish out of fold (AUC 0.35 / 0.80 / 0.44 by month).

## Capacity: stronger learners on the no-zone events [earlier]

No-zone event AUC (mean over held-out months), events with a fraud-block device are already right at
0.99+:

| pair | cascade rows / events | best other method | best - cascade |
|---|---|---|---|
| mf1 mule vs chain | 0.641 / 0.641 | event LR 0.683 | +0.04 (1.2 SE) |
| drain1 ato vs sim | 0.630 / 0.606 | kNN rows 0.615 | <= 0 |
| mf2/3 bot vs mule | 0.389 / 0.370 | LR rows 0.530 | null (11 positive events) |
| col7 cashback vs phishing | 0.456 / 0.437 | event kNN 0.515 | null |
| col1 cashback vs phishing | 0.615 / 0.622 | **event GBDT with history 0.742**, TabICL-events 0.749 | +0.12, from label history only |
| rng1 refund vs qr_swap | 0.724 / 0.769 | TabICL rows 0.748 | +0.02 (13 positive events) |

Methods tried: deeper/longer GBDTs, about 1,400 event-level aggregates (sequences, windows, account
and device attributes, other-month history, same-month counts of other shapes), logistic regression,
kNN, TabICL v2 (rows, events; BSD-3 code, ungated checkpoint, 7-26 s per run). TabPFN's weights
needed a licence acceptance and token, so it was run separately (below). Positive control: on zone
events the same methods score 0.90-1.00. The mf2/3 and col7 AUCs below 0.5 for every method are the
known negative bias of leave-group-out CV with no signal.

**col1 is the exception, via other-month label history only.** The event model scores 0.738 with
history, 0.553 without; history-only 0.695. Features: the sender (AUC 0.66), device (0.66) or
receiver (0.65) has cashback-labelled rows in another train month (set on 34% of cashback col1 events
vs 1.8% of phishing ones): cashback victims recur across months. The same history does nothing for
the other pairs (0.49-0.62) or for ordinary rows. The col1 rule "cashback if sender, device or
receiver has cashback history in another train month, else phishing" gives +0.0005 given / +0.0002
cleaned over the template rule (flips 101 rows, 36 right), but the same override on the other five
pairs loses -0.0007 to -0.0027 each. **And it does not reach October:** share of col1 events whose
sender has cashback history in another train month is 5.8% (Jul), 3.7% (Aug), 0.4% (Sep), 1.9%
(Oct); the victim pool changed. It would touch about 4 test events. **Do not build.**

AUC-to-macro-F1 simulation (oracle thresholds, upper bound, over the template rule): AUC 0.65 -> +0.0016
for all six pairs; 0.70 -> +0.0025; 0.75 -> +0.0045; 0.80 -> +0.0066; 0.90 -> +0.0128; perfect ->
+0.0286 given / +0.0299 cleaned. Drain1 alone at AUC 0.80 is +0.0031, col7 +0.0024.

## Information search: features, rules, structure [earlier]

Pair-by-pair, label-free candidates all land at or below the cascade (event-level AUC):

| pair | linkage (OOF calls of linked rows) | same-month-label linkage (upper bound) | best honest discrete feature | victim selection | cascade + victim LR |
|---|---|---|---|---|---|
| mf1 | 0.553 | 0.552 | night share 0.574 | sender rows 0.580 | 0.586 |
| drain1 | 0.501 | 0.485 | login %10 0.569 | balance / median 0.608 | 0.599 |
| mf2/3 | 0.501 | 0.454 | (noise) | (noise) | 0.554 |
| col7 | 0.477 | 0.480 | day of month <= 7: 0.558 | receiver age 0.551 | 0.479 |
| col1 | 0.500 | 0.499 | amount %10 0.540 | log balance 0.540 | 0.564 |
| rng1 | 0.547 | 0.515 | device age whole number 0.681 (noise) | amount/median 0.673 | 0.574 |

Null SE of the 3-month mean: 0.035 / 0.028 / 0.102 / 0.038 / 0.031 / 0.085. With about 55 flags per
pair screened, the expected maximum is 0.5 + 2.2 SE. Twelve discrete flags by class on ordinary rows
all match legit within |z| <= 2.

Priors for generator artefacts mined from public competitions (instant-gratification-style magic
column, fake-row value pools, IEEE-style actor keys, row order as hidden time, RNG recovery,
adversarial real-vs-synthetic detectors, label propagation, positive-unlabelled learning,
AMLSim-style typology pools, taint tracking, identifiability theory): the counter is the one that
applied; the rest are tested in [counter.md](counter.md) (structural tests) and
[ordinary-rows.md](ordinary-rows.md) (selection and detectability). No public generator writes
these twelve classes.

**Missing-bot check [earlier].** October has 7 bot-pool bursts against 13.8 expected (269 rows
against 557; Poisson p = 0.035). Mule bursts rose by 114 rows (42 bursts against 33.8 expected, p
= 0.10); their new devices sit in the mule zone, not the bot zone (1 vs 0.8 expected), so they are
not rebranded bots. If the 288 missing rows had moved to a templated shape, about 8.6 more
template rows would sit in the bot zone (observed <= 1; p = 0.0007). Keep October's bot prior at
about half of train; do not move mule bursts to bot.

## Fine-tuned 7B language model [earlier]

QLoRA (4-bit, rank 16, 3 epochs, class-balanced loss, one fresh adapter per pair and held-out
month, nine fits; cost about US$0.82 of GPU rental) on event text (header, per-account and
per-device lines, one line per transaction; no labels, app version, risk score or ids; median 541
tokens). Event-level AUC on no-zone events (mean over months, [95% interval of gap to cascade]):

| pair | cascade | best baseline | text-matched GBDT / LR | zero-shot | **QLoRA** | LLM - cascade |
|---|---|---|---|---|---|---|
| mf1 | 0.641 | 0.683 | 0.630 / 0.590 | 0.522 | 0.517 | -0.124 [-0.197, -0.049] |
| drain1 | 0.606 | 0.625 | 0.538 / 0.580 | 0.524 | 0.588 | -0.018 [-0.074, +0.038] |
| col1 | 0.622 | 0.749 (with history) | 0.539 / 0.525 | 0.451 | 0.500 | -0.122 [-0.212, -0.037] |

Rank-blending it with the cascade lowers AUC on every pair; trees fitted to the same text reach the
cascade level or below, so the text holds no more signal than the cascade already uses and the LLM
extracts less than a GBDT. The in-sample fit was weak (class-weighted loss 0.52-0.76 against 0.693
for a constant), one recipe and one seed, no hyperparameter search, and the text carried offsets
rather than gaps (caveats that do not change the verdict). **Stop the LLM and sequence-model routes
for the pairs.**

## Graph and pretrained models [earlier]

- **Two-hop graph trace** (event -> account -> that account's other rows on fraud-block devices ->
  zone class): covers 52-60% of no-zone mf1 and drain1 events, scores 0.48-0.52 for every pair.
  One and two hops through accounts carry nothing; what is left untested is three or more hops and
  end-to-end embeddings. A GNN is "not worth the hours".
- **Transaction GNN** (two layers; accounts, devices, day nodes, event nodes as the graph;
  206 numeric and 19 categorical edge features, no relational windows; 120 epochs; an A100 fit takes
  about 2.5 min): on the October-like September, the champion scores 0.5396 cleaned; the champion
  plus 12 GNN columns 0.5366 (-0.0030, every replicate negative, generator-only -0.0046); the GNN
  alone 0.5223 (-0.0173, worse in every class, most on loan 0.360 vs 0.454). LOMO: columns -0.0003
  given and cleaned; alone 0.5023 given / 0.5366 cleaned (within 0.010 of the champion). Probability
  blends at 0.25 / 0.5 tie on given and cleaned (-0.0002 / -0.0000) and gain +0.0012 / +0.0034 on
  generator-only (positive in every replicate, below the bar). Reading: it learns the same edge
  features' information; as columns it only adds variance (replicate sd 0.0025 vs 0.0005). Without
  class-frequency re-weighting divided out, the fixed point fell from 0.505 to 0.386 in the pilot.
  Not tested: GNN on the ordinary-rows-as-legit target, temporal message passing, GNN without
  the event edge features.
- **TabPFN v2 as picker/stacker:** see [ensembles.md](ensembles.md).

## Family specialists [earlier]

One small model per event family (drains, money flow, collect, rings, repeat; bots has one class)
trained on that family's candidate rows. Mean of nine cells (three thinnings x three seeds),
event-cleaned labels; baseline 0.5352.

| variant | with template rule | without (specialist's own call stands) |
|---|---|---|
| XGBoost / LightGBM on family columns, type or full | -0.0002 to -0.0001 | -0.0073 to -0.0102 |
| stack on cascade probabilities | -0.0003 to +0.0000 | -0.0073 to -0.0098 |
| **XGBoost, one class per event** | **+0.0002** | -0.0076 |

The template rule alone is worth 0.0088 on this protocol (0.0061 LOMO) and no learned per-family
split replaces it; specialists' own calls are 1.4-2.9 points of row accuracy below the rule on drains,
money flow and collect. Thinning noise spreads the baseline by 0.009 (0.5309-0.5397), so a single
thinning cannot call a gain under 0.002. Untested: specialists on event-cleaned labels,
event-level specialists. **Lesson:** any future type model should run before the rule, not instead.

## Lessons

- The generator stores the class only in the phase, and writes the phase into the data only through
  the device counter; where it shows (zones) it is already used.
- A pair of identical class-conditional distributions has a posterior equal to its prior: the
  verdict is information, not capacity (see the information-limit argument in
  [ensembles.md](ensembles.md)).
- A cheap, correct decoder choice (majority class per template, with the counter anchors) is worth
  more than every learned separator tried.
- The most fragile rule is col7 -> cashback: cashback is only 37-44% of that pair, the rule's
  margin on generator rows was 0.0000 in one month, and the pair's error rate on generator rows
  rose 0.497 -> 0.544 -> 0.570 across the three months. In October the anchored rule keeps col7 at
  100% cashback (96 rows) because its two anchors have low confidence. Re-check on any new
  candidate ([open-ideas.md](open-ideas.md)).
