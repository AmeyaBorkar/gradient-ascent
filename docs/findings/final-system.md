# The final system: one recipe, the evidence for each component, and the stress tests

**Question.** Which combination of members, training variant, weights, decoder and rules should
make the October file, chosen only on offline evidence, and does it generalise?

**Answer (measured in this repo).** `recipes/final.json`:
- **types:** the equal mean of the five tree members (XGBoost champion, LightGBM, CatBoost, XGBoost
  on the two event-cleaned label sets), fold-mean October;
- **fraud mass:** the legit probability is the mean of the five trees (2/15 each) and TabM (3
  seeds, 1/3); the fraud mass it leaves is shared out by the trees' type split (TabM's role
  `fraud`);
- **decoder:** the fixed point at 0.875 per month, then the anchored template rule;
- **base-rate rule:** `blank_merchant_qr`, capped at 40 October rows (it changes 15).

It is the only measurable component that passed the bar on top of the blend: never negative in any
month and target, positive on every thinning seed (given and cleaned), and above 0 on the 95 %
event-bootstrap lower bound of the mean, leave-one-month-out and October-like.

## Harness

`python -m ringtrace.final table <base> <challengers...>` scores recipes from saved arrays and
October-like cells only:
- per-month macro-F1 change, Jul / Aug / Sep, on the given, cleaned and gen-only targets;
- the paired Poisson bootstrap over label-free events of the mean change (2,000 draws, 95 %);
- the nine October-like cells (September thinned with seeds 13-15, members fitted on July + August
  with seed sets r0-2): the per-seed mean change and the bound of the nine-cell mean (1,000 draws;
  decision 17), with the earlier per-cell bound beside it;
- rows changed in train and October, and the October (from, to) pairs.

`ringtrace.pairboot` makes this fast. Weights are a pure function of (seed, unit, draw), so the
base's resampled counts are built once and each challenger adds only the units where a call
differs. It matches a dense reference exactly (`tests/test_pairboot.py`). One table costs about
15 min of base setup plus 8-10 min of October-like decoding, then 2-3 s per recipe.

`python -m ringtrace.final recipes/final.json` writes the October file, its sha256, the calls per
class and the base-rate rule's row counts. The build reproduces the blend's file byte for byte
(sha256 `c3af1418...`) and the blend + blank-QR file (`e7c59467...`).

## Each component's marginal evidence

Added greedily to the current best recipe, in order of evidence. Numbers are change per month
Jul / Aug / Sep; full tables in `RESULTS.md` (final-t2 to final-t7).

| component | given | cleaned | gen-only | October-like given / cleaned per seed | verdict |
|---|---|---|---|---|---|
| blank-merchant-QR rule (on the blend) | 0 / 0 / +0.00004 | 0 / 0 / +0.00003 | 0 / 0 / +0.00004 | 0 (train never has its October case) | base-rate entry, listed separately |
| TabM fraud mass 1/3 (on blend + QR) | +0.00073 / +0.00024 / +0.00023 [+0.00020, +0.00066] | +0.00068 / +0.00021 / +0.00033 [+0.00021, +0.00065] | +0.00089 / +0.00018 / +0.00007 | +0.00017 / +0.00024 / +0.00038, lb +0.00009 / +0.00025 / +0.00032 / +0.00033, lb +0.00011 | **adopted** |
| TabM fraud mass 1/2 | +0.00073 / +0.00025 / +0.00025 | +0.00068 / +0.00025 / +0.00039 | +0.00089 / +0.000014 / -0.0000022 | lb +0.00012 / +0.00012 | fails gen-only Sep by 2e-6; upload variant |
| TabM fraud mass 2/3, 4/5, 1 | | | gen-only Aug -0.00013 to -0.00070 | | rejected |
| TabM as a full member, 1/6, 1/3, 1/2 | positive | positive | Sep -0.00014 / -0.00010 / +0.000004 | | rejected or dominated |
| 6- and 9-seed TabM fraud mass (on TabM 1/3) | 0 | 0 | 0 | 0 | identical calls; 3 seeds kept |
| Mitra-2 in the fraud mass | -0.00009 / -0.00015 / -0.00018 | -0.00008 / -0.00013 / -0.00029 | | no cells | rejected |
| event types before the rule | +0.00033 / +0.00014 / 0 | +0.00047 / +0.00013 / 0 | +0.00071 / +0.00020 / 0 | 0 | tie, 0 October rows |
| anchor rescue | +0.00005 / -0.00002 / +0.00009 | | | lb -0.00009 | rejected |
| four-way (no CatBoost) | +0.00013 / +0.00008 / -0.00014 | | | -0.00010 / -0.00027 / -0.00012 (cleaned) | rejected |
| nested greedy weights | +0.00011 / +0.00005 / -0.00009 | | | lb -0.00020 | rejected |
| fraud_scale 0.85 / 0.90 | 0 / 0 / +0.00003; -0.00004 / 0 / -0.00004 | | | 0 | ties / rejected |
| refit / hedge October | identical on train | | | refit test fails the bar | hedge kept as an upload variant |
| TabPFN v2 in the fraud mass (1/6 or 1/15) | -0.00018 / -0.00014 / -0.00006; -0.00002 / -0.00001 / +0.00005 | | | no cells | rejected |
| TabDPT in the fraud mass | -0.00005 / -0.00012 / -0.00018 | -0.00005 / -0.00011 / -0.00029 | | no cells | rejected |
| event completion | | | | | changes no October row |
| TabM configurations (fraud-only role) | ties within ±0.00005 | | | | nothing to add |

The gain is additive by construction: the adopted recipe was re-scored as a whole against the
previous one (final-t6), not assembled from separate deltas.

## Base-rate component, listed separately

`blank_merchant_qr`: a QR payment to a merchant whose QR check is blank, called legit, takes its
most likely fraud class. Every such train row is fraud (678 of 678 on the cleaned labels); train
holds no legit-called case of it, so no offline score can measure it. The cap is 40 October rows;
it changes 20 on the blend and 15 on the final recipe (TabM already calls the other 5 fraud).

## Stress tests

- **Seed stability.**
  - A second seed set for every tree member (3-5) against the first: +0.00016 / +0.00011 /
    -0.00001 given, -0.00007 / +0.00011 / -0.00001 cleaned, 10 train and 2 October calls.
  - The 6- and 9-seed TabM fraud masses change 0 calls.
  - The recipe's gain over the blend (+0.0004 given on average, 5 October adds) is larger than the
    seed churn (2 October calls), so the recipe does not flip more than its gain warrants.
- **Generator-drift insurance** (`docs/knowledge/experiments/stress-test.md`, an earlier
  measurement; not re-run here, because it needs perturbed feature rebuilds and member refits).
  - The final recipe reads the same feature families and the same anchored rule as the champion,
    so it inherits the champion's profile: faster events harmless, slower events costly (x3 delays
    -0.046 cleaned), amount jitter with exact constants intact small (-0.003), broken constants
    catastrophic.
  - October's measured drift keeps every exact constant, and widening every fragile tolerance
    changes 0 October rows. The recipe's own rules have no tolerance of their own beyond the
    anchored rule's (0.8 confidence) and the blank-QR check (exact null).
  - TabM sees the same matrix (quantile-mapped). The expected private-set risk stays about -0.003.
- **Counter-free fallback** (`recipes/final-noctr.json`): the five trees retrained without the
  device-counter family, the plain template rule (the anchored rule reads the counter), blank-QR.
  - Against the blend: given -0.03013 / -0.03296 / -0.02721, cleaned -0.03120 / -0.03251 /
    -0.02761, 330 October calls.
  - TabM reads the counter family too and has no counter-free run, so the fallback has no TabM.
- **Reproducibility.** See the last section.

## Upload variants (ties in expectation)

- `final-hedge.json`: the trees' October from the hedge (mean of refit and fold mean).
- `final-s345.json`: the trees' second seed set.
- `final-half.json`: TabM's fraud mass at 1/2 (it misses only by 2e-6 in gen-only September).

Command: `python -m ringtrace.final table recipes/final.json recipes/final-*.json`.

Against `final.json` (final-t7):
- **final-hedge:** identical on train; 6 October calls.
- **final-s345:** every change within ±0.0002, intervals straddle 0; 6 train and 1 October call.
- **final-half:** the same; October-like +0.00000 to +0.00010 per seed; 13 train and 2 October
  calls.

## Reproducibility

- **Files.** A `git archive` of the branch was built twice into empty output folders. All five
  October files (`final`, three variants, the fallback) have identical sha256, the same as the
  first build. `final.csv`: `39cb4952d979ea3f1f7e2fecec552afd8b17925b3aa9f63da06d143415144db3`.
- **Members.**
  - The XGBoost champion and TabM are byte-reproducible from a clean checkout.
  - CatBoost on the GPU is not (two runs differ by at most 0.165 in one probability, and 2 of
    510,143 argmax calls). Its saved arrays are therefore the reference.
  - The four-way blend without CatBoost is not used as the reproducible form: it loses on every
    October-like seed on cleaned (-0.00010 / -0.00027 / -0.00012).
