# Baselines: the reference scores and what each layer adds

**Summary.** The reference is the XGBoost champion: 123 relational features + seven evidence
families + the device counter (230 features), a two-stage cascade (fraud score, then fraud type) with
three seeds x 300 rounds, a label-free fixed-point decoder (fraud scale 0.875) and a counter-anchored
template rule. It scores **0.5122 given / 0.5665 cleaned** leave-one-month-out, and **0.5093** on
October-like September (0.5097 for September at full density). A generator-only target scores
0.9010: a perfect detector of everything except ordinary rows would reach about 0.59-0.61. Every
number in this file is **measured in this repository** (`RESULTS.md`) unless marked otherwise; the
commands are given. Back to the [index](index.md).

## Layers, leave-one-month-out (Jul / Aug / Sep and mean)

Commands: `python -m ringtrace.build recipes/champion.json` (the champion),
`python -m ringtrace.run <recipe>` for the others in `recipes/`, `python -m ringtrace.compare` for
the paired event bootstrap, `python -m ringtrace.rescore` to rescore saved probabilities.

| layer | given | cleaned | what changed |
|---|---|---|---|
| context only (123 relational features) | 0.4660 / 0.4607 / 0.4748 (**0.4671**); argmax 0.4623 | 0.5149 / 0.5135 / 0.5292 (0.5192, rescored) | 3 seeds x 300 rounds, 6.6 min |
| context with hygiene (raw amounts, capped gaps, no calendar or whole-frame totals, receiver windows blanked on merchants) | 0.4563 / 0.4510 / 0.4654 (0.4576) | 0.5046 / 0.5021 / 0.5185 (0.5084) | -0.0095: refund (-0.064) and collusion (-0.037) lose the merchant-receiver windows; kept because the merchant and repeat families carry those classes |
| + seven evidence families, no counter, no rule | 0.4798 / 0.4736 / 0.4831 (0.4788) | 0.5299 / 0.5285 / 0.5381 (0.5322) | +0.0118 given [+0.0094, +0.0142] over context-only |
| + counter features (230 features), fixed point | 0.5130 / 0.5090 / 0.5146 (fixed point 0.5064, argmax 0.5066) | 0.5645 / 0.5639 / 0.5712 (fixed point 0.5601) | counter worth +0.028 |
| + anchored template rule (**the champion**) | **0.5122** mean | **0.5665** mean | +0.0058 given / +0.0064 cleaned over the fixed point; changed 773 train and 137 test rows |
| champion, plain one-class-per-template rule instead | 0.5109 / 0.5038 / 0.5135 (0.5094) | 0.5622 / 0.5589 / 0.5698 (0.5636) | anchoring adds +0.0028 / +0.0029 |
| champion, no rule, fraud scale 1.0 | 0.5066 / 0.5027 / 0.5096 (0.5063) | 0.5578 / 0.5575 / 0.5647 (0.5600) | scale 0.875 vs 1.0: +0.0001 (tie) |
| champion without counter features (rule on) | 0.4849 / 0.4809 / 0.4879 (0.4846) | 0.5358 / 0.5362 / 0.5445 (0.5388) | -0.028 |
| champion without money-flow features | 0.5121 / 0.5086 / 0.5141 (0.5116) | 0.5639 / 0.5636 / 0.5707 (0.5661) | tie, -0.0006 / -0.0004: other families and the rule cover chains and bursts |

- Paired event bootstrap (2,000 draws, 90%, mean of three months): the counter + template rule vs
  the no-counter model is +0.0333 [+0.0310, +0.0358] given and +0.0343 [+0.0319, +0.0369] cleaned;
  the seven families (with hygiene) vs context-only +0.0118 [+0.0094, +0.0142] given, +0.0130
  [+0.0104, +0.0155] cleaned.
- Ablations of drains, merchant, requests, marks, bursts, repeats and context were dropped at the
  end of the earlier session (queued, never run) in this repository: family-by-family removal is
  **not measured here** except the counter and money flow. See [rules.md](rules.md) for earlier
  measurements.
- Reproducibility: rebuilt from a clean checkout with an empty cache and fresh cleaned labels, the
  champion reproduces the same probabilities and the same submission file byte for byte (same
  silver statistics, 3,629 rows changed), in 11.1 min.
- Wall time: the champion is about 6 min alone on a GPU box, 10-13 min when jobs share the GPU.

## Per-class F1 of the champion (pooled over three months)

| class | given | cleaned | generator-only |
|---|---|---|---|
| legit | 0.987 | 0.988 | 1.000 |
| mule_account | 0.433 | 0.486 | 0.882 |
| velocity_bot | 0.519 | 0.583 | 0.935 |
| layering_chain | 0.551 | 0.611 | 0.925 |
| sim_swap | 0.499 | 0.553 | 0.868 |
| account_takeover | 0.434 | 0.485 | 0.836 |
| phishing_collect | 0.488 | 0.551 | 0.876 |
| qr_swap | 0.511 | 0.576 | 0.943 |
| refund_abuse | 0.532 | 0.598 | 0.952 |
| fake_cashback | 0.396 | 0.446 | 0.755 |
| loan_app_extortion | 0.414 | 0.474 | 0.905 |
| merchant_collusion | 0.393 | 0.461 | 0.941 |

Reading: recall is capped at 0.30-0.48 by the ordinary rows (about 63% of each class); precision
is 0.90-0.99 everywhere except fake_cashback (0.70-0.77, pulled by the phishing look-alike) and
sim_swap (about 0.80, pulled by takeover drains). **[earlier]** details in
[lookalikes.md](lookalikes.md).

## October-like September (transfer check)

`python -m ringtrace.octlike run <dir> champion:0,1,2 champion-cleantrain:0,1,2`: three thinned
Septembers (October's density, fraud events whole, counter columns joined from the unthinned build),
fitted on July + August, scored on September with the fixed-point decoder (the template rule is not
applied in this mode).

| frame | seed sets (0,1,2 / 3,4,5 / 6,7,8), fixed point |
|---|---|
| thinning 1 | 0.5060 / 0.5061 / 0.5058 |
| thinning 2 | 0.5094 / 0.5095 / 0.5098 |
| thinning 3 | 0.5126 / 0.5125 / 0.5124 |

Mean 0.5093; seed sets agree within +-0.0003, but the thinning draw moves the score by 0.0065
(frames 0.5060 / 0.5096 / 0.5125). Against 0.5097 for September at full density in the month folds:
no transfer loss. Cleaned-label training: mean 0.5091 (tie, see [labels.md](labels.md)). More on
protocol and pitfalls: [more-data.md](more-data.md).

## October file

The shipped October file from the champion is valid (260,582 rows, ids and order as in the sample
submission): legit 256,637; mule 789, bot 317, chain 556, sim_swap 356, takeover 242, phishing 497,
qr_swap 518, refund 280, cashback 207, loan 99, collusion 84 (3,945 fraud calls). **[earlier]** its
public score was 0.5470 and, after blends and re-seeding, public scores reached 0.5488; the
difference between near-identical files is noise ([regime-hybrid.md](regime-hybrid.md)).

## Other reference levels [earlier]

- Rules-only detector prototype: 0.476 raw September, +0.028 with counter zones; the first
  hybrid (detector features + relational GBDT + smoothing) +0.0224 over relational GBDT.
  ([rules.md](rules.md))
- Ceiling by oracle replacement on the first full model (0.5069 given / 0.5599 cleaned):
  - fix ordinary rows: +0.4290 / +0.4053;
  - fix type confusion inside generator rows: +0.0600 / +0.0460;
  - fix fraud calls on unreported rows: +0.0151 given / 0 cleaned;
  - fix generator rows called legit: +0.0028 / +0.0033;
  - perfect on generator rows: 0.5881 / 0.6105.
- Seed spread of the champion's October-like score: sd 0.0004 for three seeds; LOMO sd across
  three-seed bags 0.0001.
