# The experiment ledger: what was tried, what it did

**Summary.** About 150 distinct experiments were run on the October 2026 UPI fraud-ring task. Only
**three things paid**: the hand-built evidence families (+0.012), the device counter (+0.028 on its
own, +0.033 with the anchored template rule) and the one-class-per-look-alike-template rule. After
those, every model, ensemble, decoder, label-cleaning, data and rule variant tied or lost, with best
gains of +0.0001 to +0.0013 against an adoption bar of +0.007. The remaining gap is information
that the data does not contain: about 63% of every fraud class is ordinary-looking rows no feature
reaches (AUC 0.50 from every angle) and the choice between look-alike classes inside a template is at
chance (AUC 0.46-0.68). **Before trying anything new, find it in the table below.**

Status key: **[repo]** = measured in this repository (`RESULTS.md`; command in the file);
**[earlier]** = earlier measurement, a later pass may re-derive it. Scores are macro-F1 over 12
classes. "LOMO" = leave-one-month-out (train two months, score the third, mean of three).
"October-like" = September thinned to October's density, scored by a model fitted on July + August.
"Given" labels = as shipped, "cleaned" = repaired labels ([labels.md](labels.md)),
"generator-only" = only generator-made rows.

## Files

| file | contents |
|---|---|
| [baselines.md](baselines.md) | the reference scores, what each layer adds, how to reproduce them |
| [rules.md](rules.md) | evidence families and rule experiments: drains, money flow, requests, merchant, marked singles, pools, bots, repeats, templates, sweeps |
| [counter.md](counter.md) | the device counter, anchors, hard vs soft use, structural searches |
| [decoders.md](decoders.md) | decoder variants and prior-shift methods |
| [lookalikes.md](lookalikes.md) | sibling separation: features, LLM, GNN, TabICL / TabPFN, specialists |
| [ordinary-rows.md](ordinary-rows.md) | every angle on separating ordinary fraud rows from legit |
| [ensembles.md](ensembles.md) | seeds, learners, blends, stacking, a second fit of the method, five-member blend, information-limit theory |
| [labels.md](labels.md) | noise model, cleaned labels, cleaned-label training, noise-robust methods |
| [more-data.md](more-data.md) | learning curve, refit, self-training, transduction, October-shaped training, drift audits |
| [regime-hybrid.md](regime-hybrid.md) | the two label regimes, the hybrid, uploads, leaderboard noise |
| [stress-test.md](stress-test.md) | the generalisation stress test and the October drift audit |
| [open-ideas.md](open-ideas.md) | what was never tried, ranked |

## The summary table

Effect = change in macro-F1 against the reference named in the file. **helps** > +0.003 and robust;
**tie** within noise (|effect| < 0.002 or interval spans 0); **loses** negative beyond noise.

### Baselines and evidence layers

| experiment | verdict | effect | file |
|---|---|---|---|
| 123 relational features, cascade (context only) | baseline | 0.4671 given LOMO **[repo]** | [baselines.md](baselines.md) |
| hygiene on context features (month-identifying columns out) | loses slightly, kept for safety | -0.0095 given **[repo]** | [baselines.md](baselines.md) |
| seven evidence families (without counter) | **helps** | +0.0118 given [+0.0094, +0.0142] **[repo]** | [rules.md](rules.md) |
| counter features | **helps a lot** | +0.028 given and cleaned **[repo]** | [counter.md](counter.md) |
| counter + anchored template rule vs neither | **helps a lot** | +0.0333 given / +0.0343 cleaned **[repo]** | [counter.md](counter.md) |
| anchored vs plain template rule | helps | +0.0028 / +0.0029 **[repo]** | [rules.md](rules.md) |
| money-flow family removed | tie | -0.0006 / -0.0004 **[repo]** | [rules.md](rules.md) |
| 5-round model vs 900-round model | tie | same decoded score [earlier] | [ensembles.md](ensembles.md) |

### Device counter

| experiment | verdict | effect | file |
|---|---|---|---|
| hard-assign single-use fraud-block devices | loses | -0.0032 [earlier] | [counter.md](counter.md) |
| soft occupancy / burst features | helps | adopted [earlier] | [counter.md](counter.md) |
| richer anchors with learned emissions | tie | +0.0001 [-0.0001, +0.0003] [earlier] | [counter.md](counter.md) |
| device-number clock, file order, seed recovery, txn id, birth keys, balance | dead | no signal [earlier] | [counter.md](counter.md) |

### Rules

| experiment | verdict | effect | file |
|---|---|---|---|
| drain variants (12), cut-offs, cross-template model | tie / loses | 0.0000 to -0.0077 [earlier] | [rules.md](rules.md) |
| money-flow variants (11) | tie | best 0.0000 once the frame was fixed [earlier] | [rules.md](rules.md) |
| payment-request rules (38 variants) | tie | best +0.0002 [earlier] | [rules.md](rules.md) |
| merchant-ring variants (12) | tie | best +0.0007, 2 test rows [earlier] | [rules.md](rules.md) |
| marked-singles class model | loses | -0.0011 to -0.0073 [earlier] | [rules.md](rules.md) |
| entity pools: features / rules (9) | tie / loses | +0.0002 / best +0.0001 [earlier] | [rules.md](rules.md) |
| velocity-bot, repeat-payer features | tie | +0.006 / 0.000 on a probe [earlier] | [rules.md](rules.md) |
| parameter sweep, 37-run sweep, 160-trial cascade search | tie | +0.0002 to +0.0013 [earlier] | [rules.md](rules.md) |

### Decoders and prior shift

| experiment | verdict | effect | file |
|---|---|---|---|
| fixed point vs argmax; scale 0.875 vs 1.0 | tie | +0.0001 **[repo]** | [decoders.md](decoders.md) |
| prior shift to cleaned counts (a0-a4) | tie | -0.0002 to +0.0002 [earlier] | [decoders.md](decoders.md) |
| census-anchored counts | loses | -0.0040 to -0.1248 [earlier] | [decoders.md](decoders.md) |
| F/2 admission | tie / loses | -0.0003 to -0.0006 [earlier] | [decoders.md](decoders.md) |
| tuned or bagged multipliers / cut-offs | loses | -0.0004 to -0.0184 [earlier] | [decoders.md](decoders.md) |
| one class per event | tie | +0.0009 October-like, +0.0003 LOMO [earlier] | [decoders.md](decoders.md) |
| EM / BBSE / BCTS / census label shift | loses / tie | -0.0235 to +0.0005; oracle +0.00005 [earlier] | [decoders.md](decoders.md) |
| proportional split of inseparable pools | loses | -0.0001 to -0.016 [earlier] | [decoders.md](decoders.md) |

### Ordinary rows and look-alikes

| experiment | verdict | effect | file |
|---|---|---|---|
| ordinary rows: classifier, fingerprints, own-phone, other devices, sequence model, joint structure, pools, label history | dead | AUC 0.49-0.52 [earlier] | [ordinary-rows.md](ordinary-rows.md) |
| ordinary rows on freshly numbered phones | helps, tiny coverage | 2.8% of them [earlier] | [ordinary-rows.md](ordinary-rows.md) |
| look-alike features, GBDT, TabICL, TabPFN, kNN, LR | tie | AUC 0.42-0.75, gain 0.000 [earlier] | [lookalikes.md](lookalikes.md) |
| fine-tuned 7B LLM | loses | AUC 0.50-0.59 vs cascade 0.61-0.64 [earlier] | [lookalikes.md](lookalikes.md) |
| transaction GNN (alone / columns / blend) | loses / loses / tie | -0.0173 / -0.0030 / -0.0000 [earlier] | [lookalikes.md](lookalikes.md) |
| family specialists | tie (with rule), loses (without) | +0.0002 / -0.007 to -0.010 [earlier] | [lookalikes.md](lookalikes.md) |
| col1 other-month history rule | tie, does not reach October | +0.0005 / +0.0002 [earlier] | [lookalikes.md](lookalikes.md) |

### Ensembles

| experiment | verdict | effect | file |
|---|---|---|---|
| seeds, six-seed mean, 10-seed bag | tie | +0.0001 to -0.0002 [earlier] | [ensembles.md](ensembles.md) |
| LightGBM, CatBoost, four-model mean | tie | +0.0002 [earlier] | [ensembles.md](ensembles.md) |
| three-family rank blend | tie (best single blend) | +0.0010 [+0.0001, +0.0018] October-like [earlier] | [ensembles.md](ensembles.md) |
| per-class choice, per-class weights, type-stage stacking | fail nested gate | +0.0002 / -0.0001 / -0.0003 [earlier] | [ensembles.md](ensembles.md) |
| TabPFN stacker | tie | +0.0011 October-like [earlier] | [ensembles.md](ensembles.md) |
| second independent fit of the champion's method (reproduces it; blend 3:1) | tie | +0.0001 to +0.0004 LOMO, does not transfer [earlier] | [ensembles.md](ensembles.md) |
| equal-weight five-member blend | tie (most robust) | +0.0002 to +0.0005, 95% lower bound -0.0004 [earlier]; not yet run here | [ensembles.md](ensembles.md) |

### Labels

| experiment | verdict | effect | file |
|---|---|---|---|
| noise model, silver (cleaned) labels | measurement | 3,629 rows changed **[repo]** | [labels.md](labels.md) |
| champion trained on cleaned labels | tie | 0.5091 vs 0.5093, 9 cells **[repo]** | [labels.md](labels.md) |
| forward loss correction, soft labels, in-event relabelling | tie | -0.0005 to +0.0002 [earlier] | [labels.md](labels.md) |
| small-loss selection / co-teaching | loses | -0.0018 to -0.0039 [earlier] | [labels.md](labels.md) |
| privileged train-only columns as inputs | loses | -0.0294 [earlier] | [labels.md](labels.md) |

### More data, October shaping, regime, stress

| experiment | verdict | effect | file |
|---|---|---|---|
| learning curve: one month vs two | helps barely | +0.0006 [earlier] | [more-data.md](more-data.md) |
| refit vs fold-mean; three-month refit | tie | -0.0010 / +0.0001 [earlier] | [more-data.md](more-data.md) |
| guarded self-training; domain reweighting | tie | +0.0011 / -0.0001 [earlier] | [more-data.md](more-data.md) |
| October-shaped training copies | tie (first +0.004-0.009 was an inflated prior; later +0.0065 an artefact) | +0.0014 / +0.0016 honest [earlier] | [more-data.md](more-data.md) |
| October-like September (nine thinned cells) | transfer check | 0.5093 vs 0.5097 **[repo]** | [more-data.md](more-data.md) |
| all-legit upload to read the regime | decisive | 0.0816: ordinary rows count [earlier] | [regime-hybrid.md](regime-hybrid.md) |
| regime-hybrid gated detector | loses in the real regime | +0.020 / -0.015 [earlier] | [regime-hybrid.md](regime-hybrid.md) |
| generalisation stress test | fragile to slower delays, constants | x3 delays -0.046; October risk about -0.003 [earlier] | [stress-test.md](stress-test.md) |

## Ten rules of thumb the ledger supports

1. The learner no longer matters; the evidence features near-determine the calls. Two seeds disagree
   on about 0.01% of rows.
2. The score is set by the ordinary rows (about 98% of the remaining gap, +0.40 at stake) and the
   test labels' treatment of them, not by anything modelled.
3. Pick one class per look-alike template, anchored on the counter; do not try to separate them.
4. Label-free decoding beats any label-fitted decoder.
5. Cleaned labels are for scoring, not training.
6. Use a three-thinning, three-seed October-like check with the counter held at its unthinned
   values; one thinning moves the baseline by 0.009 and flips the sign of small deltas.
7. Public leaderboard slices are noisy (sd 0.0007 between near-identical files, 0.010-0.013 for one
   file between public and private); a +0.0017 public gain was 17 rows.
8. Prefer soft counter features to hard overrides; locate counter zones per split.
9. Keep split-length, density and file-order columns out of the model matrix.
10. The pipeline is a bet on exact generator constants; check constant-keyed event rates on any new
    month.
