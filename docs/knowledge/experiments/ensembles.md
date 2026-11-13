# Ensembles: seeds, other learners, rank blends, a second implementation, stacking

**Summary.** Nothing in the ensemble family beats the single XGBoost champion by more than +0.0010
macro-F1. Seeds, LightGBM, CatBoost, two-family and three-family blends, fold-mean versus refit,
per-stage depth and round settings, per-class model choice, per-class blend weights, type-stage
stacking, a TabPFN stacker, a GNN stacker and an independent second implementation all tie
(within -0.0010 to +0.0010 of the champion), against a +0.007 adoption gate. The reason is
information, not capacity: the evidence features near-determine the decoded calls (a 5-round model
decodes to the same score as a 900-round one), and two seeds disagree on about 0.01% of rows. See
[the information-limit theory](#why-they-all-tie-the-information-limit).

Status key: **[repo]** = measured in this repository (see `RESULTS.md`); **[earlier]** = earlier
measurement, a later pass may re-derive it here. Vocabulary: "given" labels are as shipped,
"cleaned" labels are the repaired labels ([labels.md](labels.md)), "generator-only" scores only
rows made by the fraud generator. "LOMO" is leave-one-month-out (train two months, score the third;
mean of Jul/Aug/Sep). "October-like" is a September thinned to October's density ([more-data.md](more-data.md)).
Back to the [index](index.md).

## Overview

| experiment | LOMO effect (cleaned) | October-like effect | verdict |
|---|---|---|---|
| XGBoost seeds 3-5 instead of 0-2 | +0.0002 [-0.0000, +0.0004] | n/a | tie |
| six-seed XGBoost mean | +0.0001 [-0.0001, +0.0002] | 10-seed bag -0.0002 | tie |
| LightGBM cascade | +0.0002 [-0.0000, +0.0005] | +0.0005 | tie |
| CatBoost cascade | +0.0002 [-0.0001, +0.0005] | +0.0002 | tie |
| XGBoost + LightGBM (prob / rank) | +0.0002 | +0.0005 / +0.0008 | tie |
| four-model mean (2 XGB seed sets, LGBM, Cat) | +0.0002 [+0.0000, +0.0005] | n/a | tie |
| three-family rank blend | +0.0002 | **+0.0010** [+0.0001, +0.0018] | tie (best, still 1/7 of gate) |
| per-stage depth / rounds, mixed bags | n/a | -0.0002 to +0.0001 | tie |
| fold-mean (one model per month) vs refit | n/a | -0.0010 [-0.0018, +0.0004] | tie / slight loser |
| rank vs probability vs geometric averaging | n/a | +0.0001 to +0.0003 (rank) | tie |
| per-class model choice | +0.0002 [-0.0000, +0.0004], Sep negative | n/a | fails nested gate |
| per-class blend weights | -0.0001 [-0.0004, +0.0001] | n/a | loses |
| type-stage stacking (logistic regression) | -0.0003 [-0.0007, +0.0000] | n/a | loses |
| neighbour stage | -0.0003 [-0.0007, -0.0000] | n/a | loses |
| TabPFN stacker | +0.0003 | +0.0011 | tie |
| GNN columns / GNN blend | -0.0003 / n/a | -0.0030 / -0.0000 | loses / tie |
| independent second implementation, blended 3:1 | +0.0001 to +0.0004 | -0.00002 to +0.0003 cleaned per thinning, 95% lower bound -0.00036 | tie, does not transfer |
| equal-weight five-member blend | +0.0001 to +0.0003 | +0.00016 to +0.00085 on given, positive in every cell; 95% lower bound -0.00004 (cleaned) | tie (best honest combination; see below) |

## Baselines every ensemble is compared with

- **The champion** (XGBoost cascade, three seeds, fixed-point decoder, counter-anchored template
  rule) **[repo]**: LOMO given 0.5122 (Jul 0.5130 / Aug 0.5090 / Sep 0.5146), cleaned 0.5665;
  rebuilt from a clean checkout it reproduces the same predictions byte for byte
  (`python -m ringtrace.build recipes/champion.json`).
- **[earlier]** the original implementation scored 0.5126 given / 0.5662 cleaned / 0.9005
  generator-only; its public leaderboard score was 0.5470.
- **October-like September [repo]**: the champion's mean over nine cells (three thinning seeds x
  three seed sets) is 0.5093 given, with seed sets within +-0.0003 of each other
  (`python -m ringtrace.octlike run ...`).
- **[earlier]** the October-like September with event-cleaned labels scores 0.5398 for three seeds;
  the spread over ten disjoint three-seed bags is sd 0.0004, which is why the adoption gate
  (+0.007) is about twenty seed standard deviations.

## Seeds and bags [earlier]

- Seeds 3-5 against seeds 0-2: +0.0002 cleaned. The only class that consistently moves is
  phishing_collect (+0.0010 F1 every month), and it comes from the champion's own seed draw. The
  seed sets disagree on 54 train rows (18 a month) and 15 October rows; the net number of right
  calls changes by 0 in each month.
- October-like September seed spread: sd 0.0007 for one seed, 0.0004 for three, 0.0002 for five and
  ten. Two disjoint three-seed bags flip about 0.005% of test calls (about 13 of 260,582 rows).
- **Lesson:** the seed bag has no variance left to remove. If a refit is needed anyway, a six-seed
  mean is the stability form (it removes the champion-specific phishing rows and a handful of
  sim_swap/takeover disagreements in October), but it is a tie in score.

## Other learners [earlier]

| learner | LOMO given / cleaned / generator-only | notes |
|---|---|---|
| XGBoost seeds 0-2 (champion) | 0.5126 / 0.5662 / 0.9005 | |
| LightGBM cascade | 0.5128 / 0.5664 / 0.9009 | 4.5 min on CPU |
| CatBoost cascade | 0.5127 / 0.5664 / 0.9012 | 1.5 min on GPU; seed spread about double XGBoost's (sd 0.0007) |
| 339-feature variant with leaky id columns | 0.5127 / 0.5663 / 0.9010 | helps phishing and qr_swap by +0.0012 each, hurts loan by -0.0008; a tie |
| neighbour stage | 0.5123 / 0.5659 / 0.8996 | qr_swap -0.0035 every month; adds 41 extra qr_swap calls in October; **do not use** |

All were decoded with the champion's decoder, so only the probabilities differ. A 5-round
smoke model decodes to 0.5599 cleaned at the fixed point, level with the 900-round model before the
rule; the evidence layer carries the score.

## Blends [earlier]

October-like September, event-cleaned labels, three disjoint seed sets, deltas against the
three-seed XGBoost baseline (0.5398):

| blend | delta | 90% interval of first replicate |
|---|---|---|
| XGBoost + LightGBM (probability mean) | +0.0005 | [-0.0000, +0.0016] |
| XGBoost + LightGBM (rank) | +0.0008 | [+0.0000, +0.0016] |
| XGBoost + CatBoost (rank) | +0.0008 | [-0.0001, +0.0017] |
| XGBoost + LightGBM + CatBoost (probability mean) | +0.0008 | [-0.0000, +0.0015] |
| **XGBoost + LightGBM + CatBoost (rank)** | **+0.0010** | **[+0.0001, +0.0018]** |
| same, geometric mean | +0.0007 | [-0.0002, +0.0011] |

- Rank averaging beats probability averaging by +0.0001 to +0.0003 for every blend; the geometric
  mean is 0.0001 to 0.0003 below probability averaging.
- The three-family blend also gains +0.0010 to +0.0017 on generator-only and cleaned targets, at
  about six times the training cost (a full rebuild is about 1.9 compute units against 0.3).
- Its LOMO gains are given +0.0002, cleaned +0.0002, generator-only +0.0007; on the October file
  it changes only 17 of 260,582 calls (7 takeover -> sim_swap, 5 qr_swap -> legit, 5 others).
- Per-stage settings (binary depth 6 or 10, type depth 6 or 10, type depth 6 at 600 rounds, binary
  600 rounds, mixed-depth and mixed-type bags): all -0.0002 to +0.0001.
- Fold-mean (one model per training month, averaged) against a single refit on two months: the
  fold-mean analog is -0.0010; refit at twice the rounds is +0.0001. On real test rows the refit
  and the fold-mean disagree on 0.01% of calls (about 26 rows), so the choice does not matter for
  the file. See [more-data.md](more-data.md) for the refit question.
- **Lesson:** the three-family rank blend is real but a seventh of the gate. If a blend is ever
  wanted for stability, this is the form to use.

## Complementarity: per-class choice, per-class weights, stacking [earlier]

Nested, i.e. selected on two months and scored on the third; pool of six models.

| recipe | held-out cleaned delta (Jul / Aug / Sep) | pooled 90% interval | gate |
|---|---|---|---|
| per-class model choice | +.0005 / +.0001 / -.0001 (mean +.0002) | [-0.0000, +0.0004] | fail (September negative) |
| per-class blend weights (alt model, w in {0.5, 1}) | +.0001 / .0000 / -.0005 | [-0.0004, +0.0001] | fail |
| type-stage logistic-regression stacker | -.0002 / -.0003 / -.0005 | [-0.0007, +0.0000] | fail |

- Per-class strengths are real (every other evidence learner beats the champion's seeds on
  phishing_collect by +0.0009 to +0.0014; the neighbour stage loses qr_swap by 0.0035) but total at
  most +0.0003 macro-F1, and the chosen recipe changes from fold to fold. Only "layering_chain from
  CatBoost" recurs in all three folds, worth about 0.00002 macro.
- The stacker re-types 243-284 calls and the template rule undoes most of them (48 rows end up
  changed).
- A learned stacker over both codebases' probabilities plus event context (about 25 designs) also
  failed: the closest fails September and does as well with the champion alone.
- **Lesson:** do not spend more on blends, per-class selection, per-class weights or stacking.

## Pretrained and graph members [earlier]

- **TabPFN v2 as stacker** over the champion's out-of-fold probabilities on candidate rows
  (rows with a mark, an event id or a fraud-block device): +0.0011 on the October-like mean
  (positive in all three seed sets and all targets), +0.0003 LOMO. Used as a look-alike picker it
  ties when its threshold is fitted out of fold (-0.0002) and loses -0.0108 when forced to 0.5. It
  adds a dependency and licence duty; not adopted. Its picker AUCs (0.46-0.68 on the look-alike
  pairs) repeat the TabICL result in [lookalikes.md](lookalikes.md).
- **Transaction GNN** (two message-passing layers over accounts, devices, day nodes and event
  nodes): alone 0.5223 on October-like cleaned (-0.0173); as 12 extra cascade columns -0.0030
  (every replicate negative, -0.0046 generator-only); LOMO -0.0003. Probability blends at weights
  0.25 / 0.5 tie on given and cleaned (-0.0002 / -0.0000) and gain +0.0012 / +0.0034 on
  generator-only, below the bar. Detail in [lookalikes.md](lookalikes.md).
- **Combined challenger** (TabPFN stacker + one fraud type per event + guarded self-training +
  pool features + five-seed bag): every arm below +0.0005 against the +0.007 gate; the full bundle
  loses -0.0004 on generator-only LOMO, and TabPFN is the piece that turns it negative. The
  bundle without TabPFN is positive everywhere measured (+0.0003 cleaned October-like [-0.0002,
  +0.0007], +0.0005 LOMO) but not past the gate. The pieces do not add up.

## An independent second implementation and the equal-weight blend

- **The independent rebuild [repo]** reproduces the champion: LOMO 0.5122 given / 0.5665 cleaned
  against 0.5126 / 0.5662, same predictions after rebuild from a clean checkout. **[earlier]** its
  October file scored 0.5457 public against 0.5470 for the original; the difference is noise: the
  two files differ on 59 September rows, a public-size slice (78,175 rows) shows a difference with
  sd 0.0007, and one extra right row on that slice moves macro-F1 by 0.0002 (mule) to 0.0013 (loan).
  The same file's public and private scores differ with sd 0.0117.
- **[earlier]** The two implementations disagree on 150 LOMO rows; the champion is right on 67, the
  rebuild on 68. The largest cell is champion qr_swap calls the rebuild calls legit (56 rows; the
  rebuild is right on 34).
- **[earlier]** Blends with the rebuild, change against the champion (cleaned, October-like, per
  thinning seed): 3:1 blend +2.8 / +1.0 / -0.2 (x1e-4); with the cleaned-label rebuild +1.2 / +0.7 /
  +0.9; 1:1 +0.9 / -3.0 / -0.8; nine champion seeds against three -0.7 / -1.7 / +1.1. LOMO gains are
  +0.0002 to +0.0003 and do not transfer. 95% lower bounds are -3.6e-4 to -7.3e-4.
- **[earlier] The equal-weight five-member blend** (champion XGBoost, LightGBM, CatBoost, XGBoost
  on event-cleaned labels, XGBoost on event-cleaned labels with noise-floor flips; one fifth each,
  each a three-seed mean, decoded with the fixed point and the anchored rule) was the best
  honest combination: positive in every LOMO month and every October-like thinning seed on all
  three targets (in units of 1e-4: LOMO July +2 given / +3 cleaned / +2 generator-only, September
  +4 / +4 / +11; October-like given +3.7 / +2.9 / +2.3 per seed set for the first thinning, up to
  +8.5). Pooled LOMO 90% bootstrap [-0.2, +4.5] x1e-4 given and [+0.3, +4.9] cleaned; 95% lower
  bound on the October-like cells -1.1e-4 given / -0.4e-4 cleaned. It changes about 13 October
  calls against the champion, and the three-family rank blend differs from it by 4. Every honest
  route converges on the same handful of October rows: about 7 account_takeover -> sim_swap and
  about 4 qr_swap -> legit.
- Adding the cleaned-label rebuild too (six members) is worse (95% lower bound -3.0e-4 to -3.8e-4).
- **Lesson:** this five-member blend is the target of the current repository's rebuild plan
  (`PROGRESS.md`); the expected gain over the single champion is +0.0002 to +0.0005, and it is
  adopted for stability and sign-consistency, not for score. Members 2-5 are **not yet run in this
  repo** (only the champion and its cleaned-label variant are).
- Code review of the original found no bug changing the October file; two bit-identical speed
  fixes (about 110 s a run); one small LOMO-only leak (the type-row and legit-drop masks of a fold's
  training months came from score models that saw the held-out month).
- The independent harness matters for another reason: the original and the rebuild reproduce each
  other on every cell, so both can be trusted as the gate's instrument.

## Why they all tie: the information limit

**[earlier]** A literature-and-measurement note gives four reasons, each consistent with the
measurements:

1. **Disagreement caps blends.** For competent ensembles the error reduction is at most the
   members' disagreement rate. Two seeds disagree on 0.01% of rows against a 2.5% error rate
   (38,528 of 1,564,060 LOMO rows), so a vote can remove at most 0.4% of the errors. For macro-F1,
   one fixed reachable row is worth about 2e-5 of LOMO macro-F1, so a blend's ceiling is about half
   of its changed reachable rows times 2e-5. That explains +0.0002 on LOMO and +0.0010 October-like.
2. **Noise correction only moves thresholds.** Under class-conditional label noise the noisy
   posterior is a strictly monotone transform of the clean one, so corrections change thresholds,
   and the label-free fixed-point decoder already re-solves thresholds per month ([decoders.md](decoders.md),
   [labels.md](labels.md)). Eleven tied noise-handling variants changed 3-15 calls per 100,000.
3. **The metric's optimum calls neither ordinary rows nor the minority side of a pool.** For
   confusion-matrix metrics the Bayes rule thresholds the posterior; ordinary rows are an atom far
   below the threshold (about 0.2% per class against thresholds near 0.25) and a look-alike pool is
   an atom at its partner share (9-32%) well inside one class's region. Every proportional split of
   an inseparable pool lost (-0.0001 to -0.016).
4. **The generator's random part is out of reach.** The structure (templates, counter) is the
   deterministic part; the ordinary rows and partner labels are its noise floor
   ([ordinary-rows.md](ordinary-rows.md)).

Error-budget reading [earlier]: the 38,528 LOMO errors are mostly ordinary rows (about 98% of the
remaining gap), look-alike pools, and given-label-only false positives; outside the pools the
reachable room is at most about +0.006 LOMO, mostly label noise inside structures.

## Lessons

- Do not run more seeds, learners, blends or stackers expecting a score gain; at best they buy
  stability (+0.0002 to +0.0010).
- The unused instrument worth keeping: a **disagreement-footprint screen** (count the reachable
  rows on which a candidate's final calls differ from the champion's; skip any candidate whose
  implied ceiling is under +0.0003). It is untested but cheap; see [open-ideas.md](open-ideas.md).
- Caveat: CatBoost on GPU is not byte-reproducible, so a blend containing it needs a CPU refit.
- Measurement protocol: report the paired event bootstrap, use at least three thinning seeds
  (thinning moves the baseline by 0.009 and flips the sign of small deltas), and compare every
  candidate on the same decoder.
