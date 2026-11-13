# Ledger: every scored run, every October file, and what predicted the public score

One place for every scored run, every October candidate file and every upload, followed by an
analysis of which offline signals predicted the public-score moves.

**Answer in short.** The sign of the cleaned-label leave-one-month-out (LOMO) change agreed with the
public move on 6 of 7 upload steps, and the A-clean proxy on 4 of 5. The proxy tracks the cleaned
change to within 0.00005 per month on every upload pair where both exist. The October-like change
agreed on only 2 of 4. No offline signal predicted the size of a public move. Public moves of
0.0011-0.0017 came from files 6-49 rows apart, which is one or two rows of a small class. Only the
`col7` template flip (-0.0078, 94 rows) moved public by more than a few rows can. Details are in
[the analysis](#what-predicted-the-public-score).

## How to read the tables

- **Source labels.** A number with no mark is **measured in this repo**: it is in `RESULTS.md` or
  in a JSON under `$RINGTRACE_OUT`, and the command that produced it is next to it in `RESULTS.md`.
  **(e)** marks an **earlier measurement**, carried over from `docs/knowledge/`. **(d)** marks a
  number derived here by subtracting two recorded numbers; nothing was rescored.
- **Scores.** Macro-F1 over 12 classes. J / A / S are the held-out July / August / September of
  LOMO, with the three-month mean in brackets where it was recorded.
- **Targets.** given = labels as shipped; cleaned = the frozen silver labels; gen-only = cleaned
  with the ordinary-looking fraud rows set to legit.
- **A-clean.** The cleaned labels plus the expected 3,482 hidden ordinary fraud rows, scored as an
  expected confusion matrix (`analysis/closelook_core.py`). It is recorded only for the public files,
  the fraud-mass combiner runs and the decoder sweep; it is n/a everywhere else and was not
  computed here.
- **October-like.** September thinned to October's density and fitted on July + August. Changes
  are per thinning seed s13 / s14 / s15 (mean over the seed sets that exist), with the mean 95 %
  lower bound in brackets. The bound is the per-cell mean unless marked "pooled" (decision 17).
  g = given, c = cleaned.
- **Oct rows.** October calls that differ from the stated base. Candidate files also show the rows
  that differ from `candidates/blend-tabmF-bq.csv`, the best public file (0.5499354), with their
  class pairs as (that file's label -> this file's label).
- **Changes vs absolute values.** Each table states its base. A signed number is a change against
  that base; an unsigned one is an absolute score.
- **Uploads.** Uploads 1-5 keep their numbers. The ten later uploads are labelled L1-L10 in the
  order of the upload record (`docs/knowledge/leaderboard/uploads.md`); the record does not fix the
  position of L1.

Candidate-file aggregates (sha256, rows changed, class pairs) were computed for this ledger from
`$RINGTRACE_OUT/candidates/*.csv`, `$RINGTRACE_OUT/ga-blend.csv` and
`$RINGTRACE_OUT/runs/champion/submission.csv` (**measured in this repo**; aggregates only):

```python
import csv, hashlib, collections


def load(p):
    rows = dict(list(csv.reader(open(p)))[1:])  # id -> label
    return hashlib.sha256(open(p, "rb").read()).hexdigest(), rows


_, ref = load("candidates/blend-tabmF-bq.csv")
sha, cand = load("candidates/<file>.csv")
pairs = collections.Counter(f"{ref[i]} -> {cand[i]}" for i in ref if ref[i] != cand[i])
```

## 1. Members

The scores are absolute LOMO scores after the anchored template rule unless stated. October-like is
the change against the champion's cells. The base for the bootstraps is named in the verdict.

| run / file | what it is, what it changes | commit | given J / A / S (mean) | cleaned J / A / S (mean) | gen-only J / A / S (mean) | A-clean J / A / S (mean) | October-like s13 / s14 / s15 (lb) | Oct rows | public | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| run 1, context-only | 123 relational window features, two-stage cascade (3 seeds x 300 rounds) | 6fdef89 | 0.4660 / 0.4607 / 0.4748 (0.4671); argmax 0.4623 | n/a (see run 5) | n/a | n/a | n/a | n/a | n/a | baseline |
| run 2, context-hygiene | raw amounts, capped gaps, no calendar or whole-frame totals, receiver windows blanked on merchants | 1421970 | 0.4563 / 0.4510 / 0.4654 (0.4576); argmax 0.4572 | n/a (see run 6) | n/a | n/a | n/a | n/a | n/a | kept: -0.0095 (refund -0.064, collusion -0.037); the merchant and repeat families carry those classes |
| run 3, evidence without counter | first try | e954038 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | failed at fold 9 (code replaced mid-run); rerun as run 10 |
| run 4, evidence | context + 7 label-free event families + device-number counter + anchored template rule | 41deb00 | 0.5130 / 0.5090 / 0.5146 (0.5122); fixed point 0.5064 | 0.5645 / 0.5639 / 0.5712 (0.5665); fixed point 0.5601 | (0.9010) | n/a | n/a | rule changed 137 | n/a | the champion recipe; vs run 10: +0.0333 [+0.0310, +0.0358] given, +0.0343 [+0.0319, +0.0369] cleaned |
| run 5 | run 1 rescored on cleaned labels | 4bed3e0 | 0.4660 / 0.4608 / 0.4741 (0.4670) | 0.5149 / 0.5135 / 0.5292 (0.5192) | n/a | n/a | n/a | n/a | n/a | reference |
| run 6 | run 2 rescored on cleaned labels | 4bed3e0 | 0.4563 / 0.4510 / 0.4654 (0.4576) | 0.5046 / 0.5021 / 0.5185 (0.5084) | n/a | n/a | n/a | n/a | n/a | reference |
| run 9, **champion** | XGBoost two-stage cascade, run 4 recipe (`recipes/champion.json`) | c46528d | 0.5130 / 0.5090 / 0.5146 (0.5122) | 0.5645 / 0.5639 / 0.5712 (0.5665) | 0.9020 / 0.8996 / 0.9014 (0.9010) | 0.5453 / 0.5446 / 0.5520 (0.5473) | absolute, r0: g 0.5240 / 0.5092 / 0.5134, c 0.5822 / 0.5663 / 0.5707; nine-cell mean 0.5155 g / 0.5731 c / 0.8980 gen | file `a36833d5...`, 3,945 fraud calls | 0.5457 (e) for this build (L1); 0.5470 (e) for an earlier build (upload 2) | champion |
| run 10 | evidence without counter features and without the rule (rerun of run 3) | 4bed3e0 | 0.4798 / 0.4736 / 0.4831 (0.4788) | 0.5299 / 0.5285 / 0.5381 (0.5322) | n/a | n/a | n/a | n/a | n/a | counter worth +0.028 / +0.028 at the fixed point; vs run 1: +0.0118 [+0.0094, +0.0142] given, +0.0130 [+0.0104, +0.0155] cleaned |
| run 11 | champion from a clean `git archive`, empty cache, fresh silver labels | c46528d | as run 9 | as run 9 | as run 9 | n/a | n/a | same sha256 | n/a | reproducible |
| run 12 | ablation: champion without counter features | 4bed3e0 | 0.4849 / 0.4809 / 0.4879 (0.4846) | 0.5358 / 0.5362 / 0.5445 (0.5388) | n/a | n/a | n/a | n/a | n/a | counter worth -0.028 / -0.028 |
| run 13 | ablation: champion without money-flow features | 4bed3e0 | 0.5121 / 0.5086 / 0.5141 (0.5116) | 0.5639 / 0.5636 / 0.5707 (0.5661) | n/a | n/a | n/a | n/a | n/a | tie (-0.0006 / -0.0004) |
| run 14 | ablations without drains / merchant | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | stopped at fold 9; the other ablations never started |
| `blend-champion-m1`, `blend-champion-fmt` | champion reruns as blend member 1; and after merges and formatting | d02e04b; dbc304e | as run 9 | as run 9 | 0.9020 / 0.8996 / 0.9014 (0.9010) | n/a | n/a | same sha256 | n/a | byte-identical arrays; tries 1-6 of m1 failed (GPU out of memory) |
| `lgbm` | LightGBM cascade, champion features and folds | 363eab3 | 0.5130 / 0.5092 / 0.5146 (0.5122) | 0.5646 / 0.5642 / 0.5712 (0.5667) | 0.9021 / 0.9000 / 0.9012 (0.9011) | n/a | g -0.0003 / -0.0000 / +0.0005 (-0.0007); c -0.0001 / +0.0001 / +0.0005 (-0.0006) | rule changed 132; 3,951 fraud calls | n/a | ties the champion; gate FAIL; blend member |
| `catboost` | CatBoost cascade on the GPU | e8613ed | 0.5129 / 0.5090 / 0.5141 (0.5120) | 0.5644 / 0.5637 / 0.5706 (0.5662) | 0.9023 / 0.8995 / 0.9008 (0.9008) | n/a | g -0.0004 / -0.0006 / -0.0004 (-0.0012); c -0.0004 / -0.0006 / -0.0005 (-0.0014) | rule changed 88 | n/a | -0.0002 / -0.0003; gate FAIL; blend member. Not byte-reproducible (d83b3d7: max probability difference 0.165, 2 argmax calls) |
| `clean-event` | XGBoost cascade on event-cleaned training labels (2,337 legit -> fraud) | 62c5fdb | 0.5131 / 0.5091 / 0.5146 (0.5123) | 0.5646 / 0.5640 / 0.5713 (0.5666) | 0.9021 / 0.8997 / 0.9017 (0.9012) | n/a | g +0.0001 / +0.0002 / +0.0001 (-0.0002); c +0.0001 / +0.0004 / -0.0001 (-0.0003) | n/a | n/a | tie; gate FAIL; blend member (first try: GPU out of memory) |
| `clean-l2` | as `clean-event` plus 289 minority-type relabels | 62c5fdb | 0.5131 / 0.5091 / 0.5146 (0.5122) | 0.5645 / 0.5642 / 0.5713 (0.5666) | 0.9019 / 0.8999 / 0.9016 (0.9012) | n/a | g -0.0001 / +0.0004 / +0.0001 (-0.0003); c +0.0000 / +0.0006 / +0.0001 (-0.0002) | n/a | n/a | tie; gate FAIL; blend member |
| T1 | champion rerun on a second GPU | 25a768c | as run 9 | as run 9 | as run 9 | n/a | n/a | n/a | n/a | identical |
| T2 | TabM trial: fold 9 only, seed 0 | 56fc386 | - / - / 0.5150 | - / - / 0.5717 | - / - / 0.9014 | n/a | n/a | n/a | n/a | pass (+0.0004 / +0.0005 / 0.0000 on September) |
| T4, `tabm` | TabM: k = 16 weight-sharing MLP, 10 epochs, 12-class softmax, champion's 230 features; seeds 0-2 | 56fc386 | 0.5134 / 0.5084 / 0.5151 (0.5123) | 0.5649 / 0.5635 / 0.5718 (0.5667) | 0.9023 / 0.8984 / 0.9019 (0.9008) | n/a | r0: g +0.0002 / +0.0004 / +0.0006 (-0.0002); c +0.0006 / +0.0010 / +0.0005 (-0.0000). s13, all three sets: g +0.0001 (-0.0003), c +0.0005 (-0.0001) | 786 train argmax calls differ from the champion | n/a | gate FAIL (bounds); becomes the fraud-mass source |
| `tabm-repro` | T4 rerun | a423c96 | as T4 | as T4 | as T4 | n/a | n/a | n/a | n/a | byte-identical arrays |
| `tabm-s345` | TabM seeds 3-5 | n/a | 0.5136 / 0.5086 / 0.5151 (0.5124) | 0.5651 / 0.5636 / 0.5719 (0.5669) | 0.9025 / 0.8987 / 0.9017 (0.9010) | n/a | n/a | n/a | n/a | first attempt: GPU out of memory |
| six-seed TabM | `tabm` + `tabm-s345` | n/a | 0.5135 / 0.5086 / 0.5150 (0.5124) | 0.5649 / 0.5636 / 0.5718 (0.5668) | 0.9023 / 0.8986 / 0.9019 (0.9009) | n/a | n/a | n/a | n/a | seed variance does not limit TabM |
| TabM sweep, base config | one fit, seed 0, July + August; absolute | 3acbd77 | - / - / 0.5149 | - / - / 0.5716 | - / - / 0.9017 | n/a | absolute g 0.5236 / 0.5092 / 0.5140; c 0.5826 / 0.5670 / 0.5712 | n/a | n/a | reference for the sweep rows |
| TabM sweep `k32` | k = 32; change vs the sweep base | 3acbd77 | - / - / +0.0003 | - / - / +0.0003 | n/a | n/a | g +0.0007 / +0.0003 / +0.0003 (-0.0000); c +0.0002 / +0.0003 / +0.0003 (-0.0003). Champion + k32 1:1 vs champion: g +0.0004 / +0.0005 / +0.0006 (+0.000014), c +0.0007 / +0.0008 / +0.0006 (+0.000089), PASS on r0 | n/a | n/a | candidate on one seed; the 3-seed run (f862ff0) did not finish; not adopted |
| TabM sweep `short` | 6 epochs, lr 3e-3 | 3acbd77 | - / - / +0.0001 | - / - / +0.0001 | n/a | n/a | g +0.0007 / +0.0004 / +0.0002; c +0.0003 / +0.0004 / +0.0002 | n/a | n/a | not pursued |
| TabM sweep `small` | width 256, 2 blocks | 3acbd77 | - / - / +0.0002 | - / - / -0.0001 | n/a | n/a | g +0.0007 / +0.0004 / -0.0002; c +0.0002 / +0.0000 / -0.0001 | n/a | n/a | not pursued |
| TabM sweep `long` | 20 epochs, lr 1e-3 | 3acbd77 | - / - / -0.0003 | - / - / -0.0003 | n/a | n/a | g -0.0034 / - / -; c -0.0034 / - / - | n/a | n/a | stopped after s13 |
| TabM sweep `ple` | piecewise-linear embeddings | 3acbd77 | - / - / -0.0124 | - / - / -0.0142 | n/a | n/a | g -0.0099 / - / -; c -0.0113 / - / - | n/a | n/a | stopped after s13 |
| TabM sweep `two-stage` | TabM on screened rows | 3acbd77 | - / - / -0.0077 | - / - / -0.0082 | n/a | n/a | n/a | n/a | n/a | stopped after September |
| TabM sweep `l2`, `recent2` | event-cleaned labels; August twice | f862ff0 | - / - / -0.00003; +0.00000 | - / - / -0.00002; +0.00001 | n/a | n/a | n/a | n/a | n/a | ties |
| TabM sweep `k64`, `cw-sqrt`, `cw-full`, `large` | | f862ff0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | not finished |
| TabM sweep, fraud-only role (September) | TabM at 1/2 sets the fraud mass; change vs the five-way blend | 3acbd77 / f862ff0 | base +0.00030; k32 +0.00029; short +0.00027; long +0.00030; l2 +0.00030; recent2 +0.00025; ple +0.00029; two-stage +0.00007 (September) | base +0.00044; others +0.00040 to +0.00045; two-stage +0.00015 | base +0.00008; others +0.00001 to +0.00023 | n/a | n/a | n/a | n/a | every config but two-stage within ±0.00005 of the base in this role |
| `tabmwhy` | where TabM's gain comes from: error pockets on r0 cells and on LOMO | 161757a | generator rows called legit 285 -> 249 (Jul-Sep); 135 -> 107 (October-like, 3 cells) | wrong type 1722 -> 1731; legit called fraud 178 -> 194 | n/a | n/a | disagreements: 59 rows (TabM right 31, champion 23) | n/a | n/a | gain = legit -> fraud adds outside the templates. Champion smoothed toward its prior (e 0.01-0.05): at most ±0.0002, rejected as the mechanism |

## 2. Blends

The base is the five-way blend (`recipes/blend.json`) unless stated. The five-way blend row is
absolute.

| run / file | what it is, what it changes | commit | given J / A / S | cleaned J / A / S | gen-only J / A / S | A-clean J / A / S | October-like s13 / s14 / s15 (lb) | Oct rows | public | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **five-way blend** (`blend`) | equal-weight probability blend of the champion, LightGBM, CatBoost, clean-event, clean-l2; absolute | 3d9edb9 | 0.5130 / 0.5091 / 0.5147 (0.5123) | 0.5647 / 0.5641 / 0.5714 (0.5667) | 0.9022 / 0.9000 / 0.9018 (0.9013) | 0.5455 / 0.5448 / 0.5521 (0.5475) | vs champion: g +0.0001 / +0.0004 / +0.0002 (-0.0001); c +0.0003 / +0.0005 / +0.0002 (-0.0000) | 20 vs this build's champion file (13 takeover -> sim_swap, 7 singles); file `c3af1418...` | 0.5474 (L3); an earlier build of the recipe: 0.5485 (e) (L2), 40 rows apart (e) | gate FAIL on the bound only; becomes the base |
| four-way (no CatBoost), `blend4` | drops CatBoost | 15c0fe2; cdd53de | +0.00013 / +0.00008 / -0.00014 [-0.00006, +0.00011] | +0.00001 / +0.00007 / -0.00009 | +0.00005 / +0.00010 / -0.00011 | n/a | g +0.00003 / -0.00010 / -0.00013 (-0.00029); c -0.00010 / -0.00027 / -0.00012 (-0.00050) | 3 (legit -> refund 2, sim_swap -> takeover 1) | n/a | rejected: a tie in expectation, loses every October-like seed on cleaned |
| four-way + blank-QR rule, `blend4-bq` | absolute | built outside a commit (temporary `src` copy) | 0.5131 / 0.5092 / 0.5146 | 0.5647 / 0.5642 / 0.5713 | 0.9023 / 0.9000 / 0.9017 | n/a | n/a | 18 vs `blend4`; 2 vs the five-way + rule | n/a | not used |
| greedy selection, probability (nested) | member copies chosen per held month | c237d98; cdd53de | +0.00011 / +0.00005 / -0.00009 [-0.00008, +0.00014] | +0.00011 / +0.00004 / -0.00004 | +0.00017 / +0.00003 / -0.00003 | n/a | g +0.00002 / -0.00001 / -0.00011 (-0.00020); c -0.00005 / -0.00010 / -0.00010 (-0.00033) | 18 train / 5 | n/a | rejected |
| greedy selection, rank (nested) | | c237d98 | +0.00006 / +0.00006 / -0.00008 (bootstrap +0.00001 [-0.00012, +0.00015]) | +0.00006 / +0.00006 / -0.00003 | +0.00034 / +0.00012 / +0.00001 | n/a | not run | 16 train / 2 | n/a | rejected |
| equal-weight rank blend | ranks instead of probabilities | c237d98 | +0.00017 / +0.00008 / +0.00021 (bootstrap +0.00015 [+0.00003, +0.00030]) | +0.00016 / +0.00007 / +0.00020 | +0.00024 / +0.00010 / -0.00010 | n/a | not run | 8 train / 2 | n/a | rejected (gen-only September) |
| second seed set | all five members on seeds 3-5 | ea8cf74 | +0.00016 / +0.00011 / -0.00001 [-0.00000, +0.00019] | -0.00007 / +0.00011 / -0.00001 [-0.00016, +0.00015] | -0.00003 / +0.00015 / +0.00001 | n/a | n/a | 10 train / 2 | n/a | tie: seed noise is about 1e-4 per month and 2 October calls |
| counter-free fallback | five trees without the device-counter family, plain template rule, blank-QR rule (`final-noctr.csv`, sha256 `2dfe144b...`) | ea8cf74 | -0.03013 / -0.03296 / -0.02721 [-0.03259, -0.02782] | -0.03120 / -0.03251 / -0.02761 | -0.00667 / -0.01163 / -0.00497 | n/a | n/a | 1,857 train / 330 | n/a | fallback only |
| T3 | champion + TabM trial 1:1, September only; absolute | 56fc386 | - / - / 0.5152 | - / - / 0.5719 | - / - / 0.9018 | n/a | n/a | 325 September argmax calls differ | n/a | pass |
| T5, champion + TabM 1:1 | absolute | 967c773 | 0.5134 / 0.5092 / 0.5149 (0.5125) | 0.5652 / 0.5643 / 0.5716 (0.5670) | 0.9031 / 0.9000 / 0.9017 (0.9016) | n/a | vs champion, r0: g +0.0004 / +0.0005 / +0.0002 (-0.0001); c +0.0007 / +0.0008 / +0.0001 (-0.0000). s13, three sets: g +0.0003 (-0.0001), c +0.0005 (+0.0000) | n/a | n/a | gate FAIL (bounds) |
| champion + six-seed TabM 1:1 | absolute | n/a | 0.5135 / 0.5091 / 0.5149 (0.5125) | 0.5652 / 0.5642 / 0.5717 (0.5670) | 0.9031 / 0.8997 / 0.9017 (0.9015) | n/a | n/a | n/a | n/a | not pursued |
| TabM full member 1/6 (arm a) | TabM as a sixth equal member | ea8cf74 | +0.00058 / +0.00009 / +0.00024 [+0.00013, +0.00052] | +0.00054 / +0.00008 / +0.00022 | +0.00078 / +0.00008 / -0.00014 | n/a | n/a | 20 train / 2 | n/a | fails gen-only September |
| TabM full member 1/3 (arm b) | each tree 2/15 | ea8cf74 | +0.00055 / +0.00015 / +0.00021 [+0.00013, +0.00051] | +0.00067 / +0.00014 / +0.00027 | +0.00092 / +0.00013 / -0.00010 | n/a | n/a | 30 train / 5 | n/a | fails gen-only September |
| TabM full member 1/2 (arm c), `blend-tabm.csv` | each tree 1/10 | ea8cf74 | +0.00050 / +0.00016 / +0.00027 [+0.00011, +0.00055] | +0.00051 / +0.00018 / +0.00036 | +0.00065 / +0.00005 / +0.00005 | n/a | n/a | 42 train / 5 | n/a | never loses on LOMO |
| arm c + blank-QR rule, `blend-tabm-bq.csv` | base: five-way + blank-QR rule | 80ccae8; 8d4b5c8 | +0.00050 / +0.00016 / +0.00023 [+0.00010, +0.00054] | +0.00051 / +0.00018 / +0.00033 | +0.00065 / +0.00005 / +0.000004 | 0.5460 / 0.5450 / 0.5525 (0.5478); (d) +0.00050 / +0.00018 / +0.00033 | g +0.00013 / +0.00031 / +0.00049 (-0.00004); c +0.00021 / +0.00039 / +0.00043 (-0.00005) | 41 train / 7 (4 legit -> fraud, 3 type switches) | 0.5483 (L5) | dominated by the fraud-only design; TabM's type switches are a coin flip |

## 3. Decoder and rules

The base is the five-way blend unless stated.

| run / file | what it is, what it changes | commit | given J / A / S | cleaned J / A / S | gen-only J / A / S | A-clean J / A / S | October-like s13 / s14 / s15 (lb) | Oct rows | public | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| run 7 | run 4 with the plain one-class-per-template rule; absolute | 4bed3e0 | 0.5109 / 0.5038 / 0.5135 (0.5094) | 0.5622 / 0.5589 / 0.5698 (0.5636) | n/a | n/a | n/a | n/a | n/a | anchoring adds +0.0028 / +0.0029 |
| run 8 | run 4 without the rule, fraud scale 1.0; absolute | 4bed3e0 | 0.5066 / 0.5027 / 0.5096 (0.5063) | 0.5578 / 0.5575 / 0.5647 (0.5600) | n/a | n/a | n/a | n/a | n/a | scale 0.875 vs 1.0: tie (+0.0001) |
| event types, before the rule | one type per label-free event | c237d98; cdd53de | +0.00033 / +0.00014 / +0.00000 [+0.00001, +0.00036] | +0.00047 / +0.00013 / +0.00000 [+0.00005, +0.00039] | +0.00071 / +0.00020 / +0.00000 | n/a | g +0.0000 x3 (+0.0000); c +0.0000 x3 | 7 train / 0; file `squeeze-event_types.csv` = the blend's file | n/a | tie (0 October rows), rejected |
| event types, after the rule | | c237d98 | -0.0035 / -0.0040 / -0.0037 (bootstrap -0.0037 [-0.0057, -0.0017]) | -0.0037 / -0.0043 / -0.0054 | -0.0033 / -0.0053 / -0.0063 | n/a | g -0.0052 / -0.0008 / -0.0050 (-0.0089) | 639 train / 132 | n/a | rejected |
| anchor rescue (blind spot) | | c237d98; cdd53de | +0.00005 / -0.00002 / +0.00009 [-0.00003, +0.00012] | +0.00004 / -0.00002 / +0.00016 | -0.00001 / -0.00005 / +0.00015 | n/a | g -0.00001 / +0.00014 / -0.00004 (-0.00009); c +0.00008 / +0.00023 / -0.00005 (-0.00009) | 14 train / 1 | n/a | rejected |
| anchored rule min_conf 0.70 or 0.75 | | c237d98 | +0.0000 / +0.0000 / +0.0008 (bootstrap +0.0003 [+0.0000, +0.0009]) | +0.0000 / +0.0000 / +0.0009 | +0.0000 / +0.0000 / +0.0016 | n/a | not run | 24 train / 0 | n/a | no October change |
| anchored rule min_conf 0.85 / 0.90 | | c237d98 | +0.0000 / -0.0003 / +0.0000 | +0.0000 / -0.0003 / +0.0000 | +0.0000 / -0.0005 / +0.0000 | n/a | not run | 10 train / 1 and 12 | n/a | rejected |
| event-level expected-utility decoder | | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | not run |
| fraud scale 0.85 | | cdd53de | +0.00000 / +0.00000 / +0.00003 [+0.00000, +0.00003] | +0.00000 / +0.00000 / +0.00003 | +0.00000 / +0.00000 / -0.00003 | n/a | g +0.00000 / +0.00003 / +0.00000; c same | 1 train / 0 | n/a | tie, rejected |
| fraud scale 0.90 | | cdd53de | -0.00004 / +0.00000 / -0.00004 [-0.00007, +0.00000] | -0.00003 / +0.00000 / -0.00004 | -0.00004 / +0.00000 / +0.00004 | n/a | g +0.00000 / +0.00001 / +0.00000; c same | 2 train / 1 | n/a | rejected |
| **blank-check merchant QR rule**, `blend-blankqr.csv` | a QR payment to a merchant with a blank QR check is called fraud (678 of 678 train rows are fraud) | 07333a3; ffbdfb7; 92735ae; cdd53de | +0.00000 / +0.00000 / +0.00004 [+0.00000, +0.00004] | +0.00000 / +0.00000 / +0.00003 [+0.00000, +0.00004] | +0.00000 / +0.00000 / +0.00004 | 0.5455 / 0.5448 / 0.5521 (0.5475); (d) +0.00000 / +0.00000 / +0.00003 | +0.0000 on every seed and target (+0.0000): 0 rows change in the frames; FAIL (not positive) | 1 train / 20 (legit -> qr_swap 13, refund_abuse 7) | 0.5486269 (e) / 0.5486 (L4) | adopted on the base rate; not measurable offline |
| `col7` template -> phishing, `blend-col7-phishing.csv` | the `col7` template flipped from fake_cashback to phishing_collect; base: blank-QR file | 306885e | -0.0028 / -0.0007 / -0.0005 (abs 0.5102 / 0.5084 / 0.5143) | -0.0032 / -0.0014 / -0.0017 (abs 0.5615 / 0.5627 / 0.5697) | -0.0018 / +0.0019 / +0.0014 | 0.5424 / 0.5433 / 0.5504 (0.5454); (d) -0.00310 / -0.00153 / -0.00176 | not run | 94 (fake_cashback -> phishing_collect) | 0.5408 (L7) | probe; rejected |
| `mf2` + `mf3` templates -> velocity_bot, `blend-mf23-bot.csv` | base: blank-QR file | 306885e | -0.0174 / -0.0167 / -0.0199 | -0.0194 / -0.0194 / -0.0231 | -0.0344 / -0.0371 / -0.0433 | n/a | not run | 422 (mule_account -> velocity_bot) | n/a | rejected, not uploaded |
| non-QR-check signature rescue, `rescue-nonqr.csv` | a non-QR row with a QR-check value is called fraud; base: blank-QR file | 7f3ad0c | +0.00002 / +0.00000 / +0.00000 [-0.00001, +0.00004] | +0.00002 / +0.00000 / +0.00000 [-0.00001, +0.00003] | +0.00001 / +0.00000 / +0.00000 | n/a | +0.0000 on every seed and target (+0.0000): 0 rows change; FAIL (tie) | 2 train / 2 (legit -> phishing_collect) | n/a | never loses on LOMO; on the TabM file it became L10 |
| drain signature rescue (`drn:any`) | | 7f3ad0c | +0.00003 / -0.00002 / -0.00002 | +0.00002 / -0.00002 / -0.00002 | -0.00002 / -0.00005 / -0.00006 | n/a | n/a | 5 train / 1 (sim_swap) | n/a | rejected |
| non-QR check or drain | | 7f3ad0c | +0.00005 / -0.00002 / -0.00002 | +0.00004 / -0.00002 / -0.00002 | -0.00001 / -0.00005 / -0.00006 | n/a | g -0.00007 / -0.00004 / +0.00000 (-0.00014) | 7 train / 3 | n/a | rejected |
| event completion, bot bursts, q 0.5 | legit-called rows of a mostly-fraud event take its majority call | 9c9125c | +0.00000 / +0.00000 / +0.00004 [+0.00000, +0.00005] | +0 / +0 / +0.00012 | +0 / +0 / +0.00016 | n/a | not run | 4 train / 0 | n/a | nothing to build |
| event completion, drains, q 0.5 | | 9c9125c | +0.00007 / +0.00007 / -0.00002 | +0.00006 / +0.00007 / -0.00002 | +0.00008 / +0.00010 / -0.00006 | n/a | not run | 3 train / 1 on the blend, 0 on the TabM file | n/a | rejected |
| decoder sweep on the fused mass, best setting | hidden factor 2.7, fraud scale 0.8; base: TabM fraud-only file's decoder (hf 1, fs 0.875) | 85c2d30 | +0.00019 / -0.00016 / -0.00020 | +0.00010 / +0.00010 / -0.00002 | -0.00070 / -0.00053 / -0.00068 | +0.00012 / +0.00012 / +0.000004 | not run | 15 | n/a | rejected |
| decoder sweep, the other 14 settings | hidden factor 1-2.7 x fraud scale 0.8-1.0, and rare-only multipliers | 85c2d30 | -0.00032 to +0.00015 per month | -0.00033 to +0.00007 | n/a | -0.00031 to +0.00009; rare-only multipliers lose in every month | not run | 0-12 | n/a | rejected |
| rare-class bands below each cut-off | loan, collusion, cashback, bot, refund | 85c2d30 | n/a | n/a | n/a | every band loses or ties | not run | n/a | n/a | rejected |
| COLLECT + just-under-amount cell | adversarial cell check | b2d0684 | n/a | 14 of 216 called-legit train rows are cleaned fraud | n/a | n/a | n/a | 48 called legit vs 36 expected | n/a | a rule loses; rejected |
| October census | label-free census of October against the blank-QR file (95 units, 236 fine cells) | 868e9b6 | n/a | n/a | n/a | n/a | n/a | per class actual / expected 0.981-1.021 | n/a | no fix meets the 97 % precision bar; no file |

## 4. Fraud-mass work

TabM sets each row's total fraud probability, and the trees keep the split among the fraud types.
The base is the five-way blend + blank-QR rule (the base of `recipes/final.json`) unless stated.

| run / file | what it is, what it changes | commit | given J / A / S | cleaned J / A / S | gen-only J / A / S | A-clean J / A / S | October-like s13 / s14 / s15 (lb) | Oct rows | public | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| arm c decomposition | arm c's changes split by kind on train (`ringtrace.decompose`) | after 80ccae8 | adds +0.00035 / +0.00029 / +0.00027; type switches +0.00015 / -0.00013 / 0 | adds +0.00032 / +0.00031 / +0.00036; switches +0.00019 / -0.00013 / 0 | adds +0.00036 / +0.00027 / +0.00005 | n/a | n/a | 37 adds (precision 0.59 given / 0.70 cleaned); 5 switches (0.40 / 0.60) | n/a | the gain is in the fraud mass; type switches are a coin flip |
| **TabM fraud mass 1/2**, `blend-tabmF-bq.csv` | legit column from TabM 1/2 + trees 1/10 each; tree types rescaled | after 80ccae8; cd3da2b; 147b5a3 | +0.00073 / +0.00025 / +0.00025 [+0.00019, +0.00067] | +0.00068 / +0.00025 / +0.00039 [+0.00022, +0.00070] | +0.00089 / +0.00001 / -0.0000022 | 0.5462 / 0.5451 / 0.5525 (0.5479); (d) +0.00066 / +0.00025 / +0.00039 | g +0.00023 / +0.00033 / +0.00038 (pooled +0.00012, cell -0.00004); c +0.00030 / +0.00041 / +0.00032 (pooled +0.00012, cell -0.00004) | 51 train / 7 (legit -> phishing_collect 4, loan_app_extortion 1, merchant_collusion 1, sim_swap 1) | **0.5499354** (L6) | best public; fails gen-only September by one row (2e-6); kept as the `final-half` variant (same sha256) |
| TabM fraud mass 1/2, vs the five-way blend (no rule) | | after 80ccae8; ea8cf74 | +0.00073 / +0.00025 / +0.00029 [+0.00020, +0.00069] | +0.00068 / +0.00025 / +0.00043 [+0.00023, +0.00072] | +0.00089 / +0.00001 / +0.00004 | n/a | n/a | 52 train / 17 | n/a | never loses on LOMO (without the rule) |
| TabM fraud mass 1/2 vs arm c | base: arm c | after 80ccae8 | +0.00023 / +0.00009 / +0.00002 | +0.00017 / +0.00007 / +0.00006 | +0.00024 / -0.00004 / -0.00001 | (d) +0.00016 / +0.00007 / +0.00006 (both with the rule) | (d) g +0.00010 / +0.00002 / -0.00011; c +0.00009 / +0.00002 / -0.00011 | 7 between the two upload files | 0.5483 -> 0.5499354 | fraud-only better on given and cleaned |
| **TabM fraud mass 1/3**, `final.csv` | each tree 2/15; adopted as `recipes/final.json` | 147b5a3; 798d511 | +0.00073 / +0.00024 / +0.00023 [+0.00020, +0.00066] | +0.00068 / +0.00021 / +0.00033 [+0.00021, +0.00065] | +0.00089 / +0.00018 / +0.00007 [+0.00010, +0.00074] | n/a | g +0.00017 / +0.00024 / +0.00038 (pooled +0.00009, cell -0.00004); c +0.00025 / +0.00032 / +0.00033 (pooled +0.00011, cell -0.00003) | 38 train / 5; vs the blend 25; vs `blend-tabmF-bq` 2 | n/a (not uploaded) | **PASS, adopted** |
| TabM fraud mass 1/3, vs the five-way blend (no rule) | | ea8cf74 | +0.00073 / +0.00024 / +0.00027 [+0.00021, +0.00067] | +0.00068 / +0.00021 / +0.00037 [+0.00022, +0.00066] | +0.00089 / +0.00018 / +0.00011 | n/a | n/a | 39 train / 10 | n/a | LOMO passes |
| TabM fraud mass 2/3 | vs the five-way blend (no rule) | ea8cf74 | +0.00059 / +0.00020 / +0.00039 [+0.00016, +0.00066] | +0.00053 / +0.00020 / +0.00047 | +0.00042 / -0.00013 / +0.00007 | n/a | n/a | 64 train / 19 | n/a | fails gen-only August |
| TabM fraud mass 4/5 | | ea8cf74 | +0.00066 / -0.00002 / +0.00035 | +0.00059 / -0.00003 / +0.00042 | +0.00050 / -0.00070 / -0.00011 | n/a | n/a | 78 train / 17 | n/a | fails |
| TabM fraud mass 1 | | ea8cf74 | +0.00077 / -0.00002 / +0.00026 | +0.00069 / +0.00001 / +0.00034 | +0.00062 / -0.00068 / -0.00034 | n/a | n/a | 88 train / 16 | n/a | fails |
| TabM share 1/3, vs share 1/2 | base: share 1/2 with both rules | n/a | +0.00000 / -0.00001 / -0.00002 | +0.00000 / -0.00004 / -0.00006 | +0.00000 / +0.00017 / +0.00007 | n/a | n/a | n/a | n/a | no share beats 1/2 on given and cleaned in every month |
| TabM share 2/3, vs 1/2 | | n/a | -0.00014 / -0.00005 / +0.00010 | -0.00015 / -0.00005 / +0.00004 | -0.00047 / -0.00014 / +0.00003 | n/a | n/a | n/a | n/a | rejected |
| TabM share 4/5, vs 1/2 | | n/a | -0.00007 / -0.00027 / +0.00006 | -0.00008 / -0.00028 / -0.00001 | -0.00039 / -0.00071 / -0.00015 | n/a | n/a | n/a | n/a | rejected |
| TabM share 1, vs 1/2 | | n/a | +0.00004 / -0.00027 / -0.00002 | +0.00002 / -0.00024 / -0.00009 | -0.00027 / -0.00069 / -0.00038 | n/a | n/a | n/a | n/a | rejected |
| `final-half`, vs `final` | TabM share 1/2 instead of 1/3; base: `final.json` | 798d511 | +0.00000 / +0.00001 / +0.00002 [-0.00006, +0.00008] | +0.00000 / +0.00004 / +0.00006 [-0.00004, +0.00011] | +0.00000 / -0.00017 / -0.00007 | n/a | g +0.00006 / +0.00010 / +0.00000 (-0.00003); c +0.00005 / +0.00009 / -0.00001 (-0.00003) | 13 train / 2 | as L6 (same file) | tie |
| TabM 6 seeds (0-5), lens | base: `blend-tabmF-bq` | 63b98e1 | -0.00002 / +0.00004 / +0.00000 [-0.00004, +0.00005] | +0.00002 / +0.00004 / +0.00000 [-0.00002, +0.00007] | +0.00001 / +0.00011 / +0.00000 | +0.00002 / +0.00004 / +0.00000 | n/a | 6 train / 3 (legit -> phishing 1, phishing -> legit 2) | n/a | tie |
| TabM 9 seeds (0-8) | base: `blend-tabmF-bq` | 63b98e1; fee3fa8 | -0.00002 / +0.00007 / +0.00004 [-0.00001, +0.00008] | +0.00002 / +0.00006 / +0.00004 [+0.00000, +0.00009] | +0.00001 / +0.00012 / +0.00005 | +0.00002 / +0.00006 / +0.00004 | g +0.00001 / -0.00005 / +0.00000 (-0.00007); c +0.00001 / -0.00005 / -0.00000 (-0.00006) | 5 train / 3 (the same three rows) | n/a | tie (fails given July, October-like) |
| TabM 12 seeds (0-11) | base: `blend-tabmF-bq` | fee3fa8 | -0.00002 / +0.00004 / +0.00000 [-0.00002, +0.00004] | +0.00002 / +0.00004 / +0.00000 [-0.00001, +0.00006] | +0.00001 / +0.00005 / +0.00000 | +0.00002 / +0.00003 / +0.00000 | n/a | 3 train / 0 (October file identical) | n/a | tie |
| 6- and 9-seed TabM mass, vs `final` | base: TabM 1/3 recipe | 147b5a3 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | n/a | n/a | 0 / 0 | n/a | identical calls; 3 seeds stay |
| logit average | combiner of TabM and tree fraud masses; base: `blend-tabmF-bq` | ba61ab8 | -0.00034 / -0.00028 / -0.00029 [-0.00050, -0.00013] | -0.00032 / -0.00028 / -0.00035 [-0.00051, -0.00015] | -0.00039 / -0.00005 / -0.00001 | -0.00031 / -0.00028 / -0.00035 | g -0.00015 / -0.00028 / -0.00036 (-0.00074); c -0.00012 / -0.00025 / -0.00029 (-0.00067) | 44 train / 5 (phishing -> legit 3, loan -> legit 1, collusion -> legit 1) | n/a | rejected |
| geometric mean | | ba61ab8 | -0.00060 / -0.00029 / -0.00035 [-0.00064, -0.00022] | -0.00056 / -0.00029 / -0.00048 [-0.00067, -0.00024] | -0.00071 / -0.00002 / -0.00017 | -0.00054 / -0.00029 / -0.00047 | g -0.00023 / -0.00023 / -0.00045 (-0.00080); c -0.00030 / -0.00040 / -0.00038 (-0.00089) | 56 train / 7 | n/a | rejected |
| noisy-OR | | ba61ab8 | +0.00031 / -0.00012 / -0.00008 [-0.00017, +0.00025] | +0.00028 / +0.00013 / +0.00010 [-0.00007, +0.00041] | -0.00021 / -0.00049 / -0.00041 | +0.00029 / +0.00015 / +0.00011 | g -0.00057 / -0.00018 / -0.00044 (-0.00090); c +0.00014 / +0.00040 / -0.00041 (-0.00060) | 115 train / 14 | n/a | rejected |
| max | | ba61ab8 | +0.00022 / -0.00006 / -0.00006 [-0.00015, +0.00022] | +0.00021 / -0.00002 / -0.00005 [-0.00015, +0.00024] | -0.00017 / -0.00051 / -0.00045 | +0.00021 / +0.00000 / -0.00003 | g -0.00047 / -0.00018 / -0.00031 (-0.00073); c +0.00011 / +0.00029 / -0.00035 (-0.00053) | 74 train / 8 | n/a | rejected |
| nested stacker, C = 0.01 | L2 logistic regression on the fraud logits | ba61ab8 | +0.00027 / -0.00012 / +0.00007 [-0.00013, +0.00029] | +0.00023 / +0.00009 / +0.00018 [-0.00007, +0.00039] | -0.00044 / -0.00055 / -0.00051 | +0.00025 / +0.00011 / +0.00020 | g -0.00236 / -0.00116 / -0.00146 (-0.00241); c -0.00186 / -0.00033 / -0.00121 (-0.00206) | 133 train / 35 | n/a | rejected |
| nested stacker, C = 1 | | ba61ab8 | +0.00028 / -0.00011 / +0.00003 [-0.00015, +0.00028] | +0.00024 / +0.00010 / +0.00010 [-0.00008, +0.00038] | -0.00037 / -0.00051 / -0.00050 | +0.00025 / +0.00012 / +0.00012 | g -0.00240 / -0.00119 / -0.00149 (-0.00245); c -0.00189 / -0.00039 / -0.00134 (-0.00211) | 124 train / 33 | n/a | rejected |
| family gate | TabM mass only in families whose adds were >= 50 % right (nested) | 63b98e1 | -0.00068 / -0.00018 / -0.00023 [-0.00061, -0.00018] | -0.00064 / -0.00016 / -0.00030 [-0.00059, -0.00018] | -0.00086 / -0.00011 / +0.00004 | -0.00062 / -0.00016 / -0.00029 | not run | 34 train / 6 | n/a | rejected |
| class gate | adds of classes < 50 % right in both fit months back to legit (nested) | fee3fa8 | 0 | 0 | 0 | 0 | n/a | 0 / 0 | n/a | no class qualifies |
| TabM type share, nested classes | class k's share of the mass from TabM's types | 63b98e1 | -0.00013 / -0.00065 / +0.00001 [-0.00060, -0.00000] | -0.00013 / -0.00062 / +0.00001 [-0.00058, -0.00000] | -0.00004 / -0.00098 / +0.00004 | -0.00012 / -0.00060 / +0.00001 | not run | 10 train / 1 | n/a | rejected |
| TabM type share, layering_chain only | | 63b98e1 | +0.00000 / +0.00003 / +0.00001 [+0.00000, +0.00004] | +0.00000 / +0.00003 / +0.00001 | +0.00000 / +0.00008 / +0.00004 | +0.00000 / +0.00003 / +0.00001 | not run | 2 train / 0 | n/a | tie (0 October rows) |
| TabM type share, the other ten classes one at a time | per-class lines in `lens.md` | 63b98e1 | per class -0.00062 to +0.00019 in some month | see `lens.md` | see `lens.md` | n/a | not run | 1-17 train / 0-16 | n/a | each loses in some month except layering_chain |

## 5. Neural training techniques

A screen: one fit on July + August per technique (TabM seed 0), against the plain TabM with the same
seed. The cells are September (LOMO fold 9) and the r0 October-like frames. Entries are changes x
1e-4, given / cleaned / gen-only. Fraud-only = TabM sets the fraud mass at 1/2 and the trees the
types; alone = TabM decoded by itself. Commits 9403a2f-43730fd. July, August, A-clean, October rows
and public are n/a for every screen row.

| technique | role | September | s13 | s14 | s15 | never loses | verdict |
|---|---|---|---|---|---|---|---|
| SWA, last 3 epochs | fraud-only | +0.1 / +0.1 / +0.4 | +0.3 / +0.3 / +0.8 | +0.0 / +0.0 / +0.0 | +0.0 / +0.0 / +0.0 | yes | passed the screen; failed the full bar (below) |
| SWA | alone | +1.3 / +1.3 / +0.9 | +1.6 / +0.9 / +4.8 | +4.2 / +4.1 / +4.7 | +0.5 / +0.5 / +1.5 | yes | |
| EMA of weights | fraud-only | -1.3 / -1.4 / -4.2 | +0.0 / +0.0 / -0.0 | -0.8 / +3.0 / +4.4 | +0.0 / +0.0 / +0.0 | no | rejected |
| EMA | alone | +1.7 / +1.6 / +1.0 | +1.0 / +0.1 / +1.3 | +3.9 / +3.8 / +4.0 | +3.6 / +3.3 / +0.8 | yes | |
| QR check blanked on QR-to-person rows | fraud-only | +0.3 / +0.2 / +0.1 | +0.5 / +0.6 / +1.6 | +0.6 / +4.2 / +5.6 | +0.2 / +0.3 / +0.8 | yes | rejected (alone loses s14); no full run |
| QR blanking | alone | +2.4 / +2.3 / -1.3 | +1.4 / +1.6 / +5.8 | -1.1 / +1.1 / -12.6 | +2.4 / +2.2 / +0.2 | no | |
| distillation, alpha 0.3 | fraud-only | +0.1 / +0.1 / +0.4 | +0.5 / +0.6 / +1.6 | +0.8 / +0.8 / +1.0 | -0.8 / -0.7 / -1.0 | no | rejected |
| distillation 0.3 | alone | +1.4 / +1.4 / -2.5 | +0.5 / -0.2 / +10.3 | -3.1 / -2.9 / +0.6 | +2.5 / +2.7 / +8.5 | no | |
| distillation, alpha 0.7 | fraud-only | -0.4 / -0.4 / -0.6 | -1.2 / -1.0 / -0.4 | -0.3 / -0.3 / -0.8 | -1.5 / -1.2 / -0.2 | no | rejected |
| distillation 0.7 | alone | -0.1 / +0.8 / -1.0 | +2.6 / -2.0 / +13.3 | -4.5 / -8.9 / -5.9 | +2.1 / +1.6 / +8.6 | no | |
| mixup of embeddings | fraud-only | +0.1 / +0.1 / +0.4 | -0.3 / -0.2 / +0.6 | +0.8 / +0.8 / +1.0 | +0.2 / +0.3 / +0.8 | no | rejected |
| mixup | alone | +2.7 / +0.8 / -3.1 | +3.6 / +2.1 / +9.5 | +2.8 / +2.6 / -1.6 | +1.8 / +1.6 / -2.0 | no | |
| input noise | fraud-only | -0.6 / -0.6 / +0.2 | +0.5 / +0.6 / +1.6 | -0.8 / +3.0 / +4.4 | not run | no | rejected (partial) |
| input noise | alone | +0.2 / -0.1 / -4.4 | +3.7 / +2.2 / +10.5 | +0.7 / +0.6 / -3.2 | not run | no | |
| masked-feature pretraining | fraud-only | -0.4 / -0.4 / -0.5 | -0.2 / +0.7 / +0.9 | not run | not run | no | rejected (partial) |
| pretraining | alone | +2.4 / +2.3 / -1.2 | +6.9 / +2.0 / +12.2 | not run | not run | no | |
| label smoothing (legit / fraud) | fraud-only | -0.8 / -0.6 / -4.4 | -4.1 / +1.8 / -4.3 | -2.6 / +3.9 / -7.5 | -4.4 / -3.8 / -2.0 | no | rejected |
| label smoothing | alone | +2.3 / +2.1 / -1.7 | +0.0 / -0.7 / -8.0 | -3.6 / -0.9 / -21.8 | -3.6 / -4.0 / -5.4 | no | |
| logit adjustment | fraud-only | -3034.9 / -3387.9 / -6204.7 | -3290.0 / -3673.6 / -6473.4 | -3189.9 / -3570.9 / -6380.2 | -3369.5 / -3751.5 / -6607.9 | no | rejected (about -0.3) |
| logit adjustment | alone | -3696.7 / -4066.2 / -6677.8 | -3900.6 / -4297.8 / -7012.2 | -3336.9 / -3663.4 / -6106.0 | -3639.1 / -4005.8 / -6703.9 | no | |
| focal loss | fraud-only | -562.3 / -621.3 / -1256.6 | -572.5 / -621.1 / -1230.1 | -576.2 / -628.3 / -1244.9 | -489.6 / -539.3 / -1094.8 | no | rejected (about -0.05 to -0.06) |
| focal loss | alone | -582.2 / -640.7 / -1482.3 | -595.9 / -650.6 / -1463.2 | -505.2 / -549.5 / -1240.9 | -429.3 / -469.9 / -1084.8 | no | |
| RealMLP-style network | fraud-only | -2.8 / -1.0 / -12.9 | -0.3 / +2.8 / -3.0 | -0.6 / +5.5 / +3.4 | -7.1 / -5.5 / -5.5 | no | rejected |
| RealMLP-style network | alone | -6.7 / -7.4 / -20.9 | +0.0 / +0.4 / +8.9 | -5.7 / -5.4 / -15.8 | -6.8 / -8.9 / -15.8 | no | |

Full runs and October files of the screen winner, in the general column layout:

| run / file | what it is, what it changes | commit | given J / A / S | cleaned J / A / S | gen-only J / A / S | A-clean | October-like | Oct rows | public | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `nntech-swa`, alone | TabM with SWA over the last 3 epochs, seeds 0-2; absolute | 43730fd | 0.5134 / 0.5085 / 0.5151 (0.5124) | 0.5649 / 0.5637 / 0.5719 (0.5668) | 0.9023 / 0.8987 / 0.9017 (0.9009) | n/a | not fitted | n/a | n/a | plain TabM is 0.5123 / 0.5667 / 0.9008 |
| `nntech-swa-fraud-only`, `nntech-swa.csv` | SWA TabM in the fraud-only role; base: TabM fraud mass 1/2 | 43730fd | -0.000041 / +0.000013 / +0.000041 | -0.000044 / +0.000008 / +0.000037 | -0.000127 / -0.000025 / +0.000049 | n/a | not fitted (screen moved 0-1 calls per cell) | 1 (legit -> phishing_collect); same sha256 as L10 | as L10, 0.5499 (e) | FAIL: loses July on every target |
| SWA TabM replacing TabM (1/3) | base: `final.json` | b50cc5d | -0.00004 / -0.00001 / +0.00004 [-0.00004, +0.00003] | -0.00003 / -0.00001 / +0.00003 | -0.00004 / -0.00004 / -0.00004 | n/a | n/a | 3 train / 0 | n/a | rejected |
| SWA TabM sharing the fraud mass (1/6 + 1/6) | base: `final.json` | b50cc5d | -0.00004 / 0 / 0 | -0.00003 / 0 / 0 | -0.00004 / 0 / 0 | n/a | n/a | 1 train / 0 | n/a | rejected |

## 6. Weight-space methods

The base is the blend with TabM in the fraud-only role, using a re-fit of the given-label TabM
(seeds 0-2): given 0.51371 / 0.50940 / 0.51500, cleaned 0.56535 / 0.56440 / 0.57179, gen-only
0.90302 / 0.90005 / 0.90185. Every arm replaces only TabM's legit probability. October-like uses
seed set r0 only (three cells). The `ft`, `tv` and `rb` arms change September only. Commits: W1
2e4a5e1 (`ta`, `ti`), W2 / W3 ecb0bea (`ft`, `tv`, `rb`), W4 a27dfec (`head`), scoring W5 9894783.
A-clean and public are n/a for every arm.

| arm | what it is | given J / A / S | cleaned J / A / S | gen-only J / A / S | October-like g (lb); c (lb) | rows train / Oct | verdict |
|---|---|---|---|---|---|---|---|
| stored member | same recipe, earlier fit | +0.00000 / -0.00005 / +0.00000 [-0.00007, +0.00004] | +0.00001 / -0.00001 / +0.00000 [-0.00005, +0.00006] | +0.00009 / -0.00007 / +0.00000 [-0.00010, +0.00012] | 0 / 0 / 0 (0); 0 / 0 / 0 (0) | 8 / 1 | the noise floor |
| `ta0.5` | theta_g + 0.5 (theta_c - theta_g): halfway to the cleaned-label network | -0.00008 / -0.00014 / -0.00006 [-0.00022, +0.00003] | -0.00005 / -0.00012 / -0.00003 [-0.00021, +0.00007] | -0.00033 / -0.00052 / -0.00027 [-0.00074, -0.00006] | -0.00008 / -0.00003 / -0.00008 (-0.00017); +0.00001 / -0.00003 / -0.00009 (-0.00017) | 31 / 4 | fails |
| `ta1` | the cleaned-label network | +0.00006 / +0.00004 / +0.00001 [+0.00000, +0.00009] | +0.00006 / +0.00009 / +0.00001 [+0.00001, +0.00011] | +0.00014 / +0.00020 / +0.00004 [+0.00003, +0.00025] | 0 / 0 / -0.00006 (-0.00008); 0 / 0 / -0.00005 (-0.00007) | 8 / 1 | fails (October-like) |
| `ta1.5` | extrapolation | -0.00001 / -0.00003 / -0.00004 [-0.00013, +0.00007] | -0.00001 / +0.00003 / -0.00008 [-0.00013, +0.00008] | +0.00000 / +0.00006 / +0.00004 [-0.00015, +0.00022] | -0.04608 / -0.02006 / -0.00175 (-0.02427); -0.04963 / -0.02100 / -0.00176 (-0.02597) | 21 / 8 | fails |
| `ta2` | extrapolation | -0.13494 / -0.17025 / -0.16003 | -0.15100 / -0.19135 / -0.17925 | -0.27897 / -0.35131 / -0.31926 | -0.33730 / -0.32354 / -0.24945; -0.37673 / -0.36212 / -0.27863 | 257,041 / 73,631 | broken |
| `ti0.5` | time interpolation toward 2 epochs on the nearest month | -0.00003 / -0.00001 / +0.00001 [-0.00004, +0.00001] | -0.00003 / -0.00001 / +0.00001 [-0.00004, +0.00001] | -0.00009 / -0.00004 / +0.00004 [-0.00012, +0.00003] | 0 / 0 / 0 (0); 0 / 0 / 0 (0) | 3 / 0 | tie |
| `ti1` | | +0.00000 / -0.00001 / +0.00001 [-0.00001, +0.00001] | +0.00000 / -0.00001 / +0.00001 [-0.00001, +0.00001] | +0.00000 / -0.00004 / +0.00004 [-0.00003, +0.00004] | +0.00003 / 0 / 0 (0); +0.00003 / 0 / 0 (0) | 2 / 0 | tie |
| `ti1.5` | | +0.00000 / -0.00001 / +0.00003 [-0.00003, +0.00004] | +0.00001 / -0.00001 / +0.00003 [-0.00003, +0.00004] | +0.00009 / -0.00004 / +0.00008 [-0.00003, +0.00014] | +0.00003 / 0 / 0 (0); +0.00003 / 0 / 0 (0) | 6 / 0 | tie |
| `ti2` | | +0.00000 / +0.00003 / +0.00000 [-0.00003, +0.00005] | +0.00001 / +0.00003 / -0.00004 [-0.00005, +0.00005] | +0.00009 / +0.00008 / +0.00007 [-0.00002, +0.00019] | -0.00006 / 0 / 0 (-0.00009); -0.00005 / 0 / 0 (-0.00008) | 8 / 1 | tie |
| `ta1+ti1` | | +0.00005 / +0.00003 / +0.00001 [-0.00001, +0.00008] | +0.00004 / +0.00003 / +0.00001 [-0.00002, +0.00008] | +0.00009 / +0.00009 / +0.00004 [-0.00004, +0.00020] | 0 / 0 / -0.00006 (-0.00008); 0 / 0 / -0.00005 (-0.00007) | 10 / 1 | fails (October-like) |
| `ft_g` | 2-epoch given-label fine-tune | - / - / +0.00001 | - / - / +0.00001 | - / - / +0.00004 | 0 / 0 / 0 (0); 0 / 0 / 0 (0) | 1 / n/a | tie |
| `ft_c` | 2-epoch cleaned-label fine-tune | - / - / -0.00001 | - / - / -0.00001 | - / - / -0.00008 | 0 / -0.00008 / 0 (-0.00009); 0 / +0.00030 / 0 (0) | 3 / n/a | tie |
| `tv1` | fine-tuned label vector x 1 | - / - / -0.00001 | - / - / -0.00001 | - / - / -0.00008 | 0 / -0.00008 / 0 (-0.00009); 0 / +0.00030 / 0 (0) | 3 / n/a | tie |
| `tv2` | x 2 | - / - / +0.00006 | - / - / +0.00005 | - / - / -0.00001 | -0.00009 / -0.00008 / -0.00008 (-0.00027); +0.00026 / +0.00030 / -0.00009 (-0.00010) | 4 / n/a | fails |
| `tv4` | x 4 | - / - / +0.00002 | - / - / +0.00001 | - / - / -0.00012 | -0.02219 / -0.01224 / -0.02642 (-0.02214); -0.02326 / -0.01233 / -0.02854 (-0.02336) | 7 / n/a | fails |
| `tv8` | x 8 | - / - / -0.29559 | - / - / -0.32995 | - / - / -0.60860 | -0.32384 / -0.35350 / -0.38467; -0.36160 / -0.39569 / -0.42608 | 196,280 / n/a | broken |
| `rb_naive` | plain weight mean of three seeds | - / - / -0.24927 | - / - / -0.27806 | - / - / -0.52240 | -0.27200 / -0.26727 / -0.24442; -0.30310 / -0.29896 / -0.27316 | 163,306 / n/a | broken |
| `rb_wm` | weight-matched mean of three seeds | - / - / -0.24340 | - / - / -0.27150 | - / - / -0.51046 | -0.26757 / -0.26141 / -0.23700; -0.29813 / -0.29240 / -0.26487 | 159,997 / n/a | broken |
| `head` | legit-column rescale (a, c) chosen on two months; base: its own `head_base` | +0.00000 / -0.00010 / -0.00015 [-0.00024, +0.00008] | +0.00001 / -0.00010 / -0.00001 [-0.00021, +0.00014] | +0.00009 / -0.00064 / -0.00060 [-0.00081, -0.00002] | -0.00050 / -0.00021 / -0.00035 (-0.00083); +0.00022 / +0.00040 / -0.00033 (-0.00049) | 56 / 8 | fails |

## 7. In-context models

These models refit the champion's fraud stage on its candidate rows (about 5 % of train rows). The
base is named in each verdict. A-clean and public are n/a for every row.

| run / file | what it is, what it changes | commit | given J / A / S (mean) | cleaned J / A / S (mean) | gen-only J / A / S (mean) | October-like s13 / s14 / s15 (lb) | Oct rows | verdict |
|---|---|---|---|---|---|---|---|---|
| `icl-mitra` | Mitra alone; absolute | ea2c63d | 0.4962 / 0.4798 / 0.4837 (0.4866) | 0.5475 / 0.5342 / 0.5406 (0.5408) | 0.8960 / 0.9009 / 0.8987 (0.8985) | n/a | n/a | vs champion -0.0256 [-0.0273, -0.0241] given, -0.0258 [-0.0274, -0.0243] cleaned |
| five-way + Mitra 1/6 | absolute | 0b7fe9a | 0.5133 / 0.5091 / 0.5147 (0.5124) | 0.5649 / 0.5641 / 0.5714 (0.5668) | 0.9026 / 0.9001 / 0.9020 (0.9016) | not run | 3 | vs blend +0.0001 [+0.00001, +0.0002] given, +0.0001 [+0.00002, +0.0002] cleaned; October-like not run; not adopted |
| five-way + Mitra 1/3 | absolute | 0b7fe9a | 0.5133 / 0.5086 / 0.5145 (0.5121) | 0.5650 / 0.5637 / 0.5710 (0.5666) | 0.9028 / 0.8999 / 0.9016 (0.9014) | not run | 15 | tie |
| Mitra timing and determinism | fold 9, 100 features; repeats | ea2c63d; 060f6ce | n/a | n/a | n/a | n/a | n/a | timing run out of GPU memory; repeats bit-identical after seeding |
| Mitra in the fraud mass (TabM 1/6 + Mitra 1/6) | base: TabM 1/3 recipe | 147b5a3 | -0.00009 / -0.00015 / -0.00018 [-0.00030, -0.00002] | -0.00008 / -0.00013 / -0.00029 [-0.00032, -0.00004] | -0.00006 / -0.00011 / -0.00009 | no cells | 21 train / 3 | rejected |
| `icl-mitra-tabdpt` | TabDPT alone; absolute | 060f6ce | 0.4932 / 0.4781 / 0.4788 (0.4834) | 0.5441 / 0.5315 / 0.5340 (0.5365) | 0.8933 / 0.8985 / 0.8905 (0.8941) | n/a | n/a | vs champion -0.0288 [-0.0305, -0.0273] given, -0.0300 [-0.0317, -0.0285] cleaned |
| five-way + TabDPT 1/6 | absolute | 060f6ce | 0.5132 / 0.5091 / 0.5147 (0.5124) | 0.5649 / 0.5642 / 0.5714 (0.5668) | 0.9026 / 0.9001 / 0.9020 (0.9016) | not run | 2 | +0.0001 [+0.0000, +0.0002] given (bound not above 0); not adopted |
| five-way + TabDPT 1/3 | absolute | 060f6ce | 0.5133 / 0.5088 / 0.5146 (0.5122) | 0.5650 / 0.5638 / 0.5710 (0.5666) | 0.9029 / 0.9001 / 0.9015 (0.9015) | not run | 14 | tie |
| TabDPT in the fraud mass (1/6 + 1/6) | base: `final.json` | fca45b9 | -0.00005 / -0.00012 / -0.00018 [-0.00027, -0.00000] | -0.00005 / -0.00011 / -0.00029 [-0.00030, -0.00003] | -0.00002 / -0.00010 / -0.00009 | no cells | 20 train / 3 | rejected |
| `icl-tabicl-solo` | TabICL v2 alone; absolute | 1f9e787 | 0.5104 / 0.5072 / 0.4996 (0.5058) | 0.5621 / 0.5620 / 0.5571 (0.5604) | 0.9023 / 0.8992 / 0.9148 (0.9054) | vs champion: g -0.0229 / -0.0212 / -0.0185 (-0.0235); c -0.0235 / -0.0217 / -0.0194 (-0.0242) | n/a | vs champion -0.0064 [-0.0072, -0.0057] given, -0.0061 [-0.0069, -0.0054] cleaned; FAIL |
| `icl-tabicl-blend6` | five-way + TabICL 1/6; absolute | 1f9e787 | 0.5133 / 0.5092 / 0.5147 (0.5124) | 0.5649 / 0.5642 / 0.5714 (0.5668) | 0.9026 / 0.9000 / 0.9019 (0.9015) | champion 5/6 + TabICL 1/6: g +0.0000 / +0.0001 / +0.0000 (-0.0001); c +0.0001 / +0.0001 / +0.0000 (-0.0000) | n/a | September -4.0e-6 given; FAIL (tie) |
| `icl-tabicl-blend3` | five-way + TabICL 1/3; absolute | 1f9e787 | 0.5135 / 0.5092 / 0.5147 (0.5125) | 0.5651 / 0.5642 / 0.5713 (0.5669) | 0.9029 / 0.9001 / 0.9018 (0.9016) | champion 2/3 + TabICL 1/3: g +0.0000 / +0.0001 / +0.0000 (-0.0001); c +0.0002 / +0.0002 / +0.0000 (-0.0001) | n/a | September -2.2e-6 given, -4.2e-5 cleaned; FAIL (tie) |
| `icl-tabicl-base5` | five-way rescored as a reference | 1f9e787 | as the five-way blend | as the five-way blend | as the five-way blend | n/a | 0 | reference |
| `icl-tabpfn-v2` | TabPFN v2 alone; absolute | 139414a | 0.5019 / 0.5022 / 0.5135 (0.5059) | 0.5530 / 0.5556 / 0.5702 (0.5596) | 0.8850 / 0.8889 / 0.9008 (0.8916) | not run | 108 argmax calls vs champion | vs champion -0.0063 [-0.0072, -0.0053] given, -0.0069 [-0.0079, -0.0059] cleaned. First two tries failed (GPU taken; October context limit) |
| `icl-tabpfn-v2-blend6` | five-way + TabPFN v2 1/6; absolute | 139414a | 0.5131 / 0.5091 / 0.5147 (0.5123) | 0.5648 / 0.5642 / 0.5714 (0.5668) | 0.9024 / 0.9001 / 0.9017 (0.9014) | stopped | 4-11 vs the trees | +0.0001 [-0.0000, +0.0001]; gen-only September -0.0001; rejected |
| `icl-tabpfn-v2-blend3` | five-way + TabPFN v2 1/3; absolute | 139414a | 0.5130 / 0.5090 / 0.5147 (0.5122) | 0.5647 / 0.5640 / 0.5714 (0.5667) | 0.9023 / 0.8999 / 0.9017 (0.9013) | stopped | n/a | -0.0000 [-0.0001, +0.0001]; August -0.0001; rejected |
| TabPFN v2 sharing the fraud mass (TabM 1/6 + TabPFN 1/6) | base: `final.json` | fca45b9 | -0.00018 / -0.00014 / -0.00006 [-0.00025, -0.00001] | -0.00016 / -0.00012 / -0.00009 [-0.00025, -0.00002] | -0.00018 / -0.00007 / -0.00005 | no cells | 19 train / 2 | rejected |
| TabPFN v2 added (TabM 1/3, TabPFN 1/15) | base: `final.json` | fca45b9 | -0.00002 / -0.00001 / +0.00005 [-0.00004, +0.00005] | -0.00001 / -0.00001 / +0.00004 | +0.00001 / +0.00003 / +0.00008 | no cells | 6 train / 1 | rejected (July, August negative) |
| TabPFN v2 fraud-only role script | | 7329c93 | n/a | n/a | n/a | n/a | n/a | stopped after 45 min with no output; not scored |
| TabPFN-3, LimiX | | n/a | n/a | n/a | n/a | n/a | n/a | not run |

## 8. Refit and hedge

The refit fits each member on all train rows; the hedge is the mean of the refit and the fold
mean. Train scores of refit and hedge files equal their fold-mean base by construction, so LOMO
cannot judge them; the refit test on the October-like frames is the judge. A-clean is n/a for every
row.

| run / file | what it is, what it changes | commit | given J / A / S (mean) | cleaned J / A / S (mean) | gen-only J / A / S (mean) | October-like s13 / s14 / s15 (lb) | Oct rows | public | verdict |
|---|---|---|---|---|---|---|---|---|---|
| refit curve: one month | mean of both single-month fits, screen off; absolute | ec37c9d | 0.5130 / 0.5084 / 0.5140 (0.5118) | 0.5646 / 0.5633 / 0.5706 (0.5662) | 0.9020 / 0.8990 / 0.9007 (0.9006) | n/a | n/a | n/a | reference |
| refit curve: mean of two one-month models | fold-mean analogue | ec37c9d | 0.5136 / 0.5088 / 0.5143 (0.5122) | 0.5652 / 0.5639 / 0.5709 (0.5667) | 0.9029 / 0.8998 / 0.9010 (0.9012) | n/a | n/a | n/a | reference |
| refit curve: two months, 300 rounds | screen off | ec37c9d | 0.5136 / 0.5093 / 0.5145 (0.5125) | 0.5653 / 0.5643 / 0.5712 (0.5669) | 0.9031 / 0.9002 / 0.9011 (0.9015) | n/a | n/a | n/a | second month +0.0007 |
| refit curve: two months, 450 rounds | screen off | ec37c9d | 0.5138 / 0.5089 / 0.5147 (0.5124) | 0.5654 / 0.5640 / 0.5715 (0.5669) | 0.9032 / 0.8996 / 0.9014 (0.9014) | n/a | n/a | n/a | |
| refit test: refit vs fold mean | champion fitted on July + August vs the mean of July-only and August-only fits | 181574a | n/a | n/a | n/a | g +0.0006 / +0.0006 / -0.0000 (-0.0003); c +0.0006 / +0.0006 / -0.0000 (-0.0004); gen +0.0007 / +0.0012 / +0.0002 (-0.0003) | n/a | n/a | FAIL: keep the fold mean |
| refit test: July + August screen off vs fold mean | control | 181574a | n/a | n/a | n/a | g +0.0005 / +0.0003 / -0.0001 (-0.0003); c +0.0008 / +0.0004 / +0.0001 (-0.0001) | n/a | n/a | FAIL |
| refit test: screen on vs off | | 181574a | n/a | n/a | n/a | g +0.0000 / +0.0003 / +0.0001 (-0.0006); c -0.0002 / +0.0002 / -0.0001 (-0.0008) | n/a | n/a | FAIL (tie) |
| October refits of members | champion, clean-event, clean-l2 (031e194); lgbm, catboost (18b268b): 450 rounds on all train rows | 031e194; 18b268b | not scorable | not scorable | not scorable | n/a | argmax rows vs fold mean, refit / hedge: champion 52 / 25, clean-event 48 / 24, clean-l2 61 / 38, lgbm 50 / 19, catboost 24 / 11 | n/a | inputs to the files below; the first launch failed on shell quoting |
| `refit-blend.csv` | five refit members | 18b268b | as the five-way blend | as the five-way blend | as the five-way blend | not measurable | 11 vs the blend (legit -> qr 4, phishing 2, refund 2, cashback 1, collusion 1; cashback -> refund 1) | n/a | not uploaded |
| `hedge-blend.csv` | five hedge members | 18b268b | as the five-way blend | as the five-way blend | as the five-way blend | not measurable | 6 vs the blend (5 legit -> fraud, 1 cashback -> refund) | n/a | not uploaded |
| `refit-blend-bq.csv`, `hedge-blend-bq.csv` | the two files above with the blank-QR rule | n/a | n/a (no scored run recorded) | n/a | n/a | n/a | 11 / 7 vs the blank-QR file | n/a | not uploaded |
| `blend-tabmF-refit-bq.csv` | TabM fraud mass 1/2 with refit trees | n/a | as the TabM file | as the TabM file | as the TabM file | not measurable | 9 vs the TabM file | 0.5482 (e) (L8) | lost; keep fold-mean trees |
| `blend-tabmF-hedge-bq.csv` | TabM fraud mass 1/2 with hedge trees | n/a | as the TabM file | as the TabM file | as the TabM file | not measurable | 6 vs the TabM file | 0.5482 (e) (L9) | lost; keep fold-mean trees |
| hedge / refit, all five members, vs blend | final harness | ea8cf74 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 (the cells are fold models) | hedge 6, refit 11 | n/a | decided by the refit test |
| `final-hedge.csv` | the final recipe with hedge trees; base: `final.json` | 798d511 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 | 6 vs `final`; 28 vs the blend | n/a | tie; upload variant |
| `final-s345.csv` | the final recipe with the trees' second seed set; base: `final.json` | 798d511 | -0.00001 / -0.00004 / +0.00003 [-0.00005, +0.00003] | +0.00003 / -0.00004 / -0.00006 [-0.00008, +0.00003] | +0.00004 / -0.00005 / -0.00020 | n/a | 1 vs `final`; 25 vs the blend | n/a | tie; upload variant |

## 9. Validation frames and analyses (no candidate file)

| run | what it is | commit | result | verdict |
|---|---|---|---|---|
| October-like native mode, champion cells | September thinned by protected units to 0.494, fitted on July + August; 9 cells | 6b6b8e4 | nine-cell mean given 0.5155, cleaned 0.5731, gen-only 0.8980; self-check +0.0000 everywhere | FAIL as designed (the gate needs a positive change) |
| October-like, older thinned-frame mode | fixed-point decoder, no template rule; plus a cleaned-label-trained champion | a2b10b5 | champion mean 0.5093 (frames 0.5060 / 0.5096 / 0.5125) vs 0.5097 at full density; cleaned-train 0.5091 | tie; superseded by the native mode |
| October-shaped copies, two checks | train months reshaped toward October | 6fb100d; 0eeaa40 | mean gap to October, size-1 events: raw 2.74 vs copies 5.79 (first), 4.83 / 4.44 (second) | stopped before any fit |
| public-slice noise | 2,000 public-sized slices of September (`analysis/findings.py`) | 911b18d | gap sd 0.0002-0.0005 for files a few dozen rows apart; one file sd 0.0059; one correct row of a small class 0.00016-0.00095 | see `public-score-noise.md` |
| adversarial noise and census check | slice noise against other files; class rates per 100k | b2d0684 | against the champion (gap -0.0002) sd 0.0002; one file's slice sd 0.0094; no class short of the October estimate | analysis |
| error anatomy, October profiles | loss pockets of the blend | 2f71e67 | ordinary rows 0.3874, generator rows called legit 0.0025, wrong type 0.0206, legit called fraud 0.0018 | analysis; led to the blank-QR rule |
| signature survey, event-completion survey | generator signatures called legit; legit rows inside fraud events | 7f3ad0c; 9c9125c | 2 qualifying signatures; events already complete | analysis |
| member re-decode | each member decoded as its run decodes it | 911b18d | matches each run (champion = run 9) | check passed |
| LightGBM reproducibility | one cell refitted at 12 and 16 threads | n/a | byte-identical | reproducible |
| final reproducibility | two clean `git archive` builds of every October file | b24b0ea | all five sha256 identical | reproducible |
| final-t1 / t2 builds | `recipes/cand/base.json`, `blankqr.json` | 05cb804; cdd53de | byte-identical to the blend and blank-QR files | check passed |

## 10. Candidate October files

Every file under `$RINGTRACE_OUT/candidates/` has 260,582 rows with the sample's ids. Rows vs
`blend-tabmF-bq` and the class pairs are counted as described above, as (`blend-tabmF-bq` label ->
this file's label).

| file | what it is | sha256 | rows vs stated base | rows vs `blend-tabmF-bq` | class pairs vs `blend-tabmF-bq` | fraud calls | public |
|---|---|---|---|---|---|---|---|
| `ga-blend.csv` (in `$RINGTRACE_OUT`), `runs/blend/submission.csv` | five-way blend | `c3af1418ffab08f7843b1813359ad7353ee0ac6676f4293609d3cf530a696c05` | 20 vs the champion file | 27 | qr_swap -> legit 13, refund_abuse -> legit 7, phishing_collect -> legit 4, loan_app_extortion -> legit 1, merchant_collusion -> legit 1, sim_swap -> legit 1 | 3,946 | 0.5474 (L3) |
| `squeeze-combo.csv`, `squeeze-event_types.csv` | squeeze final combination; event types before the rule | `c3af1418...` (the blend's file) | 0 vs the blend | 27 | as the blend | 3,946 | as L3 |
| `runs/champion/submission.csv` | champion (run 9) | `a36833d5fb8c497ff3b6e7ea116226796ec8892ad661c17afa87bf92fb8e380e` | base | 41 | sim_swap -> account_takeover 13, qr_swap -> legit 12, refund_abuse -> legit 6, phishing_collect -> legit 5, sim_swap -> legit 2, loan_app_extortion -> legit 1, account_takeover -> legit 1, velocity_bot -> legit 1 | 3,945 | 0.5457 (e) (L1) |
| `blend4.csv` | four-way blend (no CatBoost) | `8df9b36aa3ad07e7fd31effe4410635e18c0c2063f5c83b85637153408220f48` | 3 vs the blend | 26 | qr_swap -> legit 13, refund_abuse -> legit 5, phishing_collect -> legit 4, loan_app_extortion -> legit 1, sim_swap -> account_takeover 1, merchant_collusion -> legit 1, sim_swap -> legit 1 | 3,948 | n/a |
| `blend4-bq.csv` | four-way + blank-QR rule | `ca13597b0fde6a69a7f4321f6cf750e928e057e021f56cbfa8e039babf2e1491` | 18 vs `blend4`; 2 vs the blank-QR file | 9 | phishing_collect -> legit 4, loan_app_extortion -> legit 1, sim_swap -> account_takeover 1, merchant_collusion -> legit 1, sim_swap -> legit 1, qr_swap -> phishing_collect 1 | 3,966 | n/a |
| `blend-blankqr.csv` | five-way + blank-QR rule | `e7c594672f0921e6def080f0c06347144cd848282b10c9fe6e384416eed043d0` | 20 vs the blend | 7 | phishing_collect -> legit 4, loan_app_extortion -> legit 1, merchant_collusion -> legit 1, sim_swap -> legit 1 | 3,966 | 0.5486269 (e) (L4) |
| `blend-col7-phishing.csv` | blank-QR file, `col7` template -> phishing | `91d1e767f0e8fe63643251fe7fd72579874a6772c9faa941a6cc1cca1b88fcd8` | 94 vs the blank-QR file | 101 | fake_cashback -> phishing_collect 94, phishing_collect -> legit 4, loan_app_extortion -> legit 1, merchant_collusion -> legit 1, sim_swap -> legit 1 | 3,966 | 0.5408 (L7) |
| `blend-mf23-bot.csv` | blank-QR file, `mf2` + `mf3` templates -> velocity_bot | `0cbf7e1b535567b3d6bbc32ef735cf70a7c7f45bc4e4310f69d82100ff635448` | 422 vs the blank-QR file | 429 | mule_account -> velocity_bot 422, phishing_collect -> legit 4, loan_app_extortion -> legit 1, merchant_collusion -> legit 1, sim_swap -> legit 1 | 3,966 | n/a |
| `blend-tabm.csv` | five-way + TabM full member 1/2 (arm c) | `397fd2a8ab3ca9f77e1ec50d27f096e8850b58fdafe3800cd5d953e9db83adc5` | 5 vs the blend | 25 | qr_swap -> legit 13, refund_abuse -> legit 7, phishing_collect -> legit 3, qr_swap -> merchant_collusion 1, loan_app_extortion -> fake_cashback 1 | 3,950 | n/a |
| `blend-tabm-bq.csv` | arm c + blank-QR rule | `07aee8f502fb7389570ce27ef0d124ccc03e40ff56a66f4b18cbcce0471f1759` | 7 vs the blank-QR file; 20 vs `blend-tabm` | 7 | phishing_collect -> legit 3, qr_swap -> merchant_collusion 1, loan_app_extortion -> fake_cashback 1, refund_abuse -> phishing_collect 1, qr_swap -> phishing_collect 1 | 3,970 | 0.5483 (L5) |
| `blend-tabmF-bq.csv` | five-way types + TabM fraud mass 1/2 + blank-QR rule | `8a46142660cde2af1576736289f822c1b5c6226204445bce88d4dc311fb926eb` | 7 vs the blank-QR file | 0 | (reference) | 3,973 | **0.5499354** (L6) |
| `final-half.csv` | final recipe with TabM share 1/2 | `8a46142660cde2af1576736289f822c1b5c6226204445bce88d4dc311fb926eb` | 2 vs `final` | 0 | (the same file as `blend-tabmF-bq`) | 3,973 | as L6 |
| `final.csv` | **final recipe**: five-way types + TabM fraud mass 1/3 + both rules | `39cb4952d979ea3f1f7e2fecec552afd8b17925b3aa9f63da06d143415144db3` | 25 vs the blend | 2 | phishing_collect -> legit 2 | 3,971 | n/a (not uploaded) |
| `final-hedge.csv` | final recipe, hedge trees | `2f38ee4d051b7c48c684fee4ef87ab5a44b2377d52d2eda84f12638095e01f1a` | 6 vs `final` | 8 | phishing_collect -> legit 2, loan_app_extortion -> fake_cashback 1, refund_abuse -> qr_swap 1, legit -> qr_swap 1, fake_cashback -> refund_abuse 1, legit -> phishing_collect 1, qr_swap -> refund_abuse 1 | 3,973 | n/a |
| `final-s345.csv` | final recipe, trees' second seed set | `0a53fd14c9623eba0df43a9293766e3d827b5ca768102c25f16f33720d6abddd` | 1 vs `final` | 3 | phishing_collect -> legit 2, loan_app_extortion -> fake_cashback 1 | 3,971 | n/a |
| `blend-tabmF-refit-bq.csv` | TabM file with refit trees | `7ecd86f7ec5bf1d625ee1269febb075c4859d5e839bc5a35084a32373b2083dd` | 9 vs the TabM file | 9 | refund_abuse -> qr_swap 2, refund_abuse -> mule_account 1, loan_app_extortion -> fake_cashback 1, refund_abuse -> loan_app_extortion 1, legit -> qr_swap 1, fake_cashback -> refund_abuse 1, legit -> phishing_collect 1, qr_swap -> refund_abuse 1 | 3,975 | 0.5482 (e) (L8) |
| `blend-tabmF-hedge-bq.csv` | TabM file with hedge trees | `30192d71ff9093f6bd331083fc05e1c4e56314c98e1ac2ffcf18fc2c9ebc37e9` | 6 vs the TabM file; 3 vs the refit file | 6 | loan_app_extortion -> fake_cashback 1, refund_abuse -> qr_swap 1, legit -> qr_swap 1, fake_cashback -> refund_abuse 1, legit -> phishing_collect 1, qr_swap -> refund_abuse 1 | 3,975 | 0.5482 (e) (L9) |
| `refit-blend.csv` | five refit members | `7557112ae26186658e6c9640f5181ddc2a86483c30fdb562b972f6880ec349a4` | 11 vs the blend | 23 | qr_swap -> legit 10, refund_abuse -> legit 5, phishing_collect -> legit 3, loan_app_extortion -> fake_cashback 1, legit -> qr_swap 1, fake_cashback -> refund_abuse 1, legit -> phishing_collect 1, sim_swap -> legit 1 | 3,956 | n/a |
| `hedge-blend.csv` | five hedge members | `28d9544465b36b028cb6c91069b4af26dc361d6978d5a4401e41f0d6ec7507ad` | 6 vs the blend | 26 | qr_swap -> legit 13, refund_abuse -> legit 6, phishing_collect -> legit 3, loan_app_extortion -> fake_cashback 1, legit -> qr_swap 1, fake_cashback -> refund_abuse 1, sim_swap -> legit 1 | 3,951 | n/a |
| `refit-blend-bq.csv` | refit members + blank-QR rule | `2cf9943c829e00e1027b50f5530929e1e7d1b3dd99c668d5474486eb2e1a613f` | 11 vs the blank-QR file | 13 | phishing_collect -> legit 3, refund_abuse -> qr_swap 2, refund_abuse -> mule_account 1, loan_app_extortion -> fake_cashback 1, refund_abuse -> loan_app_extortion 1, legit -> qr_swap 1, fake_cashback -> refund_abuse 1, legit -> phishing_collect 1, sim_swap -> legit 1, qr_swap -> refund_abuse 1 | 3,971 | n/a |
| `hedge-blend-bq.csv` | hedge members + blank-QR rule | `84fd9ab47721076f730ac88cdb9f4f6d341f9e959a16d9bb984bc6886ed79c2d` | 7 vs the blank-QR file | 9 | phishing_collect -> legit 3, loan_app_extortion -> fake_cashback 1, refund_abuse -> qr_swap 1, legit -> qr_swap 1, fake_cashback -> refund_abuse 1, sim_swap -> legit 1, qr_swap -> refund_abuse 1 | 3,970 | n/a |
| `rescue-nonqr.csv` | blank-QR file + non-QR-check rule | `0f0a60e8e86838cb3602f0a8453d14ee767c989cfbf00e5c2dbaa1cc35449206` | 2 vs the blank-QR file | 7 | phishing_collect -> legit 3, loan_app_extortion -> legit 1, merchant_collusion -> legit 1, legit -> phishing_collect 1, sim_swap -> legit 1 | 3,968 | n/a |
| `rescue-nonqr-on-tabmF.csv`, `nntech-swa.csv` | TabM file + non-QR-check rule; the SWA TabM fraud-only file gives the same bytes | `715138bd2d1d6bc48338f34873363c310a2493556f1769ec0cc9a94d2f05fc3a` | 1 vs the TabM file | 1 | legit -> phishing_collect 1 | 3,974 | 0.5499 (e) (L10) |

## 11. Uploads

The public slice is 30 % of October (78,175 rows).

| upload | file, what it is | public | change vs previous comparable | Oct rows vs that file | rows vs `blend-tabmF-bq` | LOMO at the time (cleaned mean) | A-clean (mean) |
|---|---|---|---|---|---|---|---|
| 1 | all-legit file (regime probe) | 0.0816 (e) | n/a | n/a | n/a | n/a | n/a |
| 2 | XGBoost champion, earlier build | 0.5470 (e) | +0.4654 (d) vs upload 1 | n/a | n/a | 0.5662 (e): 0.5641 / 0.5635 / 0.5709; given 0.5126 (e) | 0.5473 (this repo's build) |
| 3 | XGBoost + LightGBM + CatBoost rank blend | 0.5487 (e) | +0.0017 vs upload 2 | 17 (e) | n/a | +0.0002 vs upload 2 (e) | n/a |
| 4 | upload 3 + minimum type score 0.04 + one drifting counter dropped | 0.5488 (e) | +0.0001 vs upload 3 | 9 (e) | n/a | +0.00007 vs upload 3 (e) | n/a |
| 5 | pooled data, self-training, five-seed bag, one type per event | 0.5488 (e) | 0.0000 vs upload 4 | 15 (e) | n/a | +0.0005 vs the champion (e) | n/a |
| L1 | champion, this build, `a36833d5...` | 0.5457 (e) | -0.0013 vs upload 2 | 49 (e) | 41 | 0.5665 | 0.5473 |
| L2 | five-member equal-weight blend, earlier build | 0.5485 (e) | n/a | n/a | n/a | n/a | n/a |
| L3 | five-way blend, `c3af1418...` | 0.5474 | -0.0011 vs L2; +0.0017 (d) vs L1 | 40 vs L2 (e); 20 vs L1 | 27 | 0.5667 | 0.5475 |
| L4 | five-way + blank-QR rule, `e7c59467...` | 0.5486269 (e) | +0.0012 vs L3 | 20 | 7 | 0.5667 | 0.5475 |
| L5 | arm c (TabM full member 1/2) + blank-QR, `07aee8f5...` | 0.5483 | -0.0003 vs L4 | 7 | 7 | 0.5671 | 0.5478 |
| L6 | TabM fraud mass 1/2 + blank-QR, `8a461426...` | **0.5499354** | +0.0016 vs L5; +0.0013 vs L4 | 7 vs L5; 7 vs L4 | 0 | 0.5672 | 0.5479 |
| L7 | blank-QR file with `col7` -> phishing, `91d1e767...` | 0.5408 | -0.0078 vs L4 | 94 | 101 | 0.5646 | 0.5454 |
| L8 | L6 with refit trees, `7ecd86f7...` | 0.5482 (e) | -0.0017 vs L6 | 9 | 9 | as L6 | n/a |
| L9 | L6 with hedge trees, `30192d71...` | 0.5482 (e) | -0.0017 vs L6; 0.0000 vs L8 | 6; 3 vs L8 | 6 | as L6 | n/a |
| L10 | L6 + non-QR-check contradiction rule, `715138bd...` | 0.5499 (e) | 0.0000 vs L6 (four decimals) | 1 | 1 | as L6 (rule on the blend: +0.00002 / 0 / 0) | n/a |

Candidates of the earlier research that were never uploaded (all **earlier measurements**): the
regime-B twin of the champion (221 fewer fraud calls; LOMO 0.9209 gen-only, 0.5512 cleaned); a
rules-only detector (3,634 fraud calls, about 0.04 below the model); the champion with the
drifting counter dropped, six XGBoost seeds and LightGBM at 1/3 (cleaned 0.5664, pooled bound
-0.0000); upload 4's type-score change on the champion (October-like bound -0.0002, one seed
-0.0001); a five-model equal blend (+0.0002 to +0.0004 LOMO, 13 October rows vs the champion,
misses the 95 % bound); cleaned-label-trained blends (a tie on October-like).

## What predicted the public score

### The rule for noise

From `public-score-noise.md` (measured in this repo, September as a stand-in for October):

- One extra correct row on the public slice moves macro-F1 by 0.00016 (mule_account) to 0.00095
  (loan_app_extortion), that is 0.0002-0.001.
- Two files that differ on a few dozen rows have a public gap with sd 0.0002-0.0005. A chance
  error of 0.001 or more appears in at most 3 % of slices.
- A file that differs on n rows puts about 0.3 n of them in the public slice.

Reading used below:
- **Inside noise:** |public change| <= 0.001, which is one row of a small class or less.
- **One or two rows:** 0.001-0.002 from a file with up to about 50 changed rows. It is real for
  the few public rows involved, but it says little about the other 70 %. The two rebuilds of the
  same recipe (L1 vs upload 2, L3 vs L2) moved by this much with no intended change; a -0.0013
  gap or worse happens in 11 % of slices for the champion rebuild (e).
- **Outside noise:** larger than a few rows can produce. Only L7 qualifies.

### Upload pairs

Offline columns are changes Jul / Aug / Sep (mean), from the second file's point of view.
October-like is cleaned unless marked; bounds follow the ledger tables. Signs are given in
the order A-clean / cleaned / October-like: "+" agrees with public, "x" disagrees, "0" means the
signal predicts no change, "n/a" means it was not measured.

| pair | public change | Oct rows between them | A-clean change | cleaned LOMO change | October-like change | noise reading | sign agreement |
|---|---|---|---|---|---|---|---|
| P1: upload 2 -> upload 3 (e) | +0.0017 | 17 (takeover -> sim_swap 7, qr_swap -> legit 5, legit -> phishing 2, 3 singles) | n/a | +0.0002 (given +0.0002); stricter check: August -0.00003 | default mode +0.0010 [90 %: +0.0001, +0.0018]; whole-event mode pooled +0.00017 [-0.00013, +0.00046], one seed -0.0001 | one or two rows ("a row or two", upload record) | n/a / + / + |
| P2: upload 3 -> upload 4 (e) | +0.0001 | 9 (qr -> legit 4, phishing -> legit 2, refund -> legit 2, loan -> qr 1) | n/a | +0.00007 (given +0.00008) | not run | inside noise | n/a / + / n/a |
| P3: upload 4 -> upload 5 (e) | 0.0000 | 15 (legit -> qr 11, legit -> refund 2, qr -> legit 2) | n/a | vs the champion: upload 4 +0.00031, upload 5 +0.0005 | upload 5 vs champion +0.0003 [-0.0002, +0.0007]; upload 4 not run | inside noise (zero) | no public move |
| P4: upload 2 -> L1 (two builds of the champion) | -0.0013 (e) | 49 (e) | n/a | full September -0.0003 on 59 rows (e) | n/a | one or two rows; 11 % of slices (e) | n/a / + (September) / n/a |
| P5: L2 -> L3 (two builds of the five-way recipe) | -0.0011 (e) | 40 (e): 32 fewer fraud calls, 24 of them qr_swap -> legit | n/a | 0 expected (same recipe) | n/a | one or two rows | no prediction |
| P6: L1 -> L3 (champion -> five-way) | +0.0017 (d); +0.0004 (d) against upload 2 | 20 (takeover -> sim_swap 13; legit -> takeover, sim_swap, phishing, bot 1 each; qr, collusion, refund -> legit 1 each) | (d) +0.00020 / +0.00024 / +0.00015 (+0.00020) | (d) +0.00020 / +0.00025 / +0.00016 (+0.00021) | +0.0003 / +0.0005 / +0.0002 (-0.0000); g +0.0001 / +0.0004 / +0.0002 (-0.0001) | one or two rows; the champion's two builds alone are 0.0013 apart | + / + / + |
| P7: L3 -> L4 (blank-QR rule) | +0.0012 | 20 (legit -> qr_swap 13, legit -> refund_abuse 7) | (d) +0.00000 / +0.00000 / +0.00003 (+0.00001) | +0.00000 / +0.00000 / +0.00003 | +0.0000 x3: 0 rows change in the frames | one or two rows. 13 x 0.3 qr rows at 0.00022 plus 7 x 0.3 refund rows at 0.00041 give about +0.0017 (d) if every row is right | + / + / 0 |
| P8: L4 -> L5 (TabM full member) | -0.0003 | 7 (legit -> fraud 4: cashback, phishing, collusion, sim_swap; type switches 3: qr -> collusion, refund -> phishing, qr -> phishing) | (d) +0.00050 / +0.00018 / +0.00033 (+0.00033) | +0.00051 / +0.00018 / +0.00033 | +0.00021 / +0.00039 / +0.00043 (cell -0.00005) | inside noise | x / x / x |
| P9: L5 -> L6 (full member -> fraud mass) | +0.0016 | 7 (legit -> phishing 3; collusion -> qr, cashback -> loan, phishing -> refund, phishing -> qr 1 each) | (d) +0.00016 / +0.00007 / +0.00006 (+0.00010) | +0.00017 / +0.00007 / +0.00006 (without the rule) | (d) +0.00009 / +0.00002 / -0.00011 (mixed) | one or two rows; one loan_app_extortion row is 0.00095 | + / + / mixed |
| P10: L4 -> L6 (two steps) | +0.0013 (d) | 7 (legit -> phishing 4, loan 1, collusion 1, sim_swap 1) | (d) +0.00066 / +0.00025 / +0.00039 (+0.00043) | +0.00068 / +0.00025 / +0.00039 | +0.00030 / +0.00041 / +0.00032 (pooled +0.00012) | one or two rows | + / + / + |
| P11: L4 -> L7 (`col7` -> phishing) | -0.0078 | 94 (fake_cashback -> phishing_collect) | (d) -0.00310 / -0.00153 / -0.00176 (-0.00213) | -0.0032 / -0.0014 / -0.0017 | not run (gen-only -0.0018 / +0.0019 / +0.0014) | outside noise (about 28 public rows) | + / + / n/a |
| P12: L6 -> L8 (refit trees) | -0.0017 (e) | 9 (refund -> qr 2; refund -> mule, loan -> cashback, refund -> loan, legit -> qr, cashback -> refund, legit -> phishing, qr -> refund 1 each) | n/a | 0 / 0 / 0 (train-identical) | no cells; the refit test for the champion: +0.0006 / +0.0006 / -0.0000 (-0.0004) | one or two rows; loan -> cashback moves a loan_app_extortion call (0.00095 per public row) | n/a / 0 / x (indirect) |
| P13: L6 -> L9 (hedge trees) | -0.0017 (e) | 6, all also among L8's 9 changes | n/a | 0 / 0 / 0 | no cells | one or two rows; the same rows as P12 | n/a / 0 / n/a |
| P14: L8 -> L9 | 0.0000 (e) | 3 (three of L8's refund_abuse changes undone) | n/a | 0 | n/a | inside noise | ties agree |
| P15: L6 -> L10 (non-QR-check rule) | 0.0000 (e) at four decimals | 1 (legit -> phishing_collect) | n/a | (on the blend) +0.00002 / +0.00000 / +0.00000 | +0.0000, 0 rows change | inside noise; the row is outside the public slice (e) | ties agree |
| C1: L1 -> L6 (champion build -> best file) | +0.0042 (d); +0.0029 (d) against upload 2 | 41 (takeover -> sim_swap 13, legit -> qr 12, legit -> refund 6, legit -> phishing 5, legit -> sim_swap 2, legit -> loan, takeover, bot 1 each) | (d) +0.00086 / +0.00050 / +0.00058 (+0.00064) | (d) +0.00088 / +0.00050 / +0.00059 (+0.00066) | n/a (no single gate) | above the paired sd, made of steps that are each one or two rows | + / + / n/a |

The A-clean change equals the cleaned change to within 0.00005 in every month of every pair
where both exist (P6-P11, C1). On these files the proxy adds nothing to the cleaned labels.

### Sign agreement

Counted over the independent steps that have both a public move and a non-zero prediction. P10
and C1 are combinations of other steps and are listed separately. Ties (P3, P14, P15) are
excluded: there is no public move to predict.

| signal | steps counted | sign agrees | sign disagrees | blind (predicts 0 or not measured) | with P10 and C1 |
|---|---|---|---|---|---|
| A-clean proxy | P6, P7, P8, P9, P11 | 4 | 1 (P8, -0.0003, inside noise) | uploads 1-5, P12, P13: not recorded | 6 of 7 |
| cleaned LOMO | P1, P2, P6, P7, P8, P9, P11 | 6 | 1 (P8) | P12, P13 (0 by construction), P5 | 8 of 9 |
| October-like | P1, P6, P8, P12 | 2 (P1, P6) | 2 (P8; P12 through the champion's refit test) | P7 (0 rows in the frames), P9 (mixed), P11 (not run), P13 | 3 of 5 |
| gen-only target | P6, P7, P8, P9, P11 | 3 | 2 (P8; P11, mean (d) +0.00054: -0.0018 / +0.0019 / +0.0014) | | |

### Size agreement

Public change against the offline mean change, for the steps whose signs agree:

| step | public | cleaned LOMO mean | ratio | October-like cleaned mean (d) | ratio |
|---|---|---|---|---|---|
| P1 | +0.0017 (e) | +0.0002 (e) | about 8x | +0.0010 (default mode, e) | 1.7x |
| P6 | +0.0017 (or +0.0004 vs upload 2) | +0.00021 | 8x (or 2x) | +0.00033 | 5x (or 1.2x) |
| P7 | +0.0012 | +0.00001 | over 100x | 0 | blind |
| P9 | +0.0016 | +0.00010 | 16x | 0.00000 (mixed) | blind |
| P10 | +0.0013 | +0.00044 | 3x | +0.00034 | 4x |
| P11 | -0.0078 | -0.0021 | 3.7x | not run | |
| C1 | +0.0029 (vs upload 2) | +0.00066 | 4.4x | n/a | |

- No offline signal predicted the size of a public move. The public moves run 2-16 times the
  offline mean change, and more than 100 times for the blank-QR rule.
- The ratio is 3-4.4x for the 94-row `col7` flip (P11) and for the combined pairs (P10, C1). For
  single steps of 7-20 rows it is 8x or more (2x for P6 if upload 2 is the reference).
- Where a signal disagreed (P8, P12), it predicted +0.0003 to +0.0004 and public moved -0.0003 and
  -0.0017. Both are one or two rows.

### Answers

1. **Which changes moved public.** Only the `col7` template flip (L7, -0.0078 from 94 rows)
   moved public beyond what a few rows can do. Every other move of 0.0011-0.0017 came from files
   6-49 rows apart: the blank-QR rule (+0.0012), TabM fraud mass (+0.0013 over L4, +0.0016 over
   L5), refit and hedge trees (-0.0017 each, from shared rows), upload 3 (+0.0017), and the two
   rebuilds (-0.0013, -0.0011). Each is one or two rows of a small class. The rebuilds show that
   this size happens with no intended change at all. Moves of 0.0003 or less (uploads 4 and 5, L5,
   L9 vs L8, L10) are inside noise.
2. **A-clean proxy.** Its sign agreed on 4 of 5 steps (6 of 7 counting the combined pairs). The
   miss is L5's -0.0003, which is inside noise. It never predicted the size: its change was 2-120
   times smaller than the public move. It equals the cleaned change to within 0.00005 per month,
   so it is not a separate signal.
3. **October-like.** Its sign agreed on 2 of 4 steps (3 of 5 counting P10). It is blind to
   October-only rows (the blank-QR rule changes 0 rows in its frames) and to October-only arrays
   (refit and hedge trees have no frame cells). It was positive on both changes that lost on public
   (L5, and the refit through the champion's refit test).
4. **Cleaned LOMO.** Its sign agreed on 6 of 7 steps (8 of 9 counting P10 and C1), with the same
   single miss as the proxy. It is blind to changes that leave train untouched.
5. **Which signals can be trusted for choosing uploads.**
   - **Trust the cleaned LOMO sign** (the A-clean proxy is the same reading) for any change that
     touches train rows. Ask for "never loses in any month", as the bar does. Do not read the size
     of the public move from it.
   - **Keep October-like as a guard**, not as a forecast. It rejects gains that exist only at full
     density, but it called 2 of 4 public signs.
   - **Changes invisible to train get no offline signal.** These are October-only arrays (refit,
     hedge) and October-only rows (the blank-QR rule). Upload them only on a base rate the train
     months show: the blank-QR rule rests on 678 of 678 train rows and gained. The two without
     one, refit and hedge, both lost 0.0017.
   - **Do not use the gen-only target as a judge.** It ranked the `col7` flip up in August and
     September, and public fell 0.0078.
   - **Read public only above about 0.002, or for one-factor changes of many rows.** Below that, a
     public gap is one or two rows of a small class. `final.csv` and `blend-tabmF-bq.csv` differ on
     2 rows (phishing_collect -> legit). Each such row is worth about 0.00024 on the public slice,
     so public cannot separate the two; the choice between them rests on LOMO and the October-like
     bar (`final.csv` passes it, `blend-tabmF-bq` misses gen-only September by one row).
