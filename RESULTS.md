# Results

Every scored run, failures included. Leave-one-month-out (train Jul+Aug -> score Sep, etc.);
macro-F1 over 12 classes per held-out month; "given" = labels as shipped, "cleaned" = the silver
labels of `ringtrace.silver` (frozen once built: 3,629 train rows changed, 2,944 legit -> fraud,
685 wrong types; 62.9 % of fraud labels judged ordinary-looking), "gen" = cleaned labels with the
ordinary-looking fraud rows set to legit. Decoder = fixed point per month unless stated.

| # | run (recipe) | commit | given Jul / Aug / Sep (mean) | cleaned Jul / Aug / Sep (mean) | wall | notes |
|---|---|---|---|---|---|---|
| 1 | context-only | 6fdef89 | 0.4660 / 0.4607 / 0.4748 (**0.4671**); argmax 0.4623 | â€” (silver not built yet) | 6.6 min | 123 relational features, cascade 3 seeds x 300 rounds |
| 2 | context-hygiene | 1421970 | 0.4563 / 0.4510 / 0.4654 (**0.4576**); argmax 0.4572 | â€” | 4.0 min | raw amounts, capped gaps, no calendar/whole-frame totals, receiver windows blanked on merchants; ordinary rows legit in stage one. -0.0095: refund (-0.064) and collusion (-0.037) lose the merchant-receiver windows; kept, since the merchant/repeat families carry those classes |
| 3 | evidence-nophones | e954038 | â€” | â€” | â€” | died at fold 9 (its code dir was replaced by a later ship; the launcher now snapshots code per job); rerun as run 5 |
| 4 | **evidence** (context + 7 families + counter, anchored template rule) | 41deb00 | 0.5130 / 0.5090 / 0.5146 (**0.5122**); fixed point 0.5064, argmax 0.5066 | 0.5645 / 0.5639 / 0.5712 (**0.5665**); fixed point 0.5601, argmax 0.5602 | 5.8 min | 230 features; generator-only target 0.9010. Template rule changed 773 train / 137 test rows (+0.0058 given / +0.0064 cleaned over the fixed point) |
| 5 | rescore: context-only (run 1) on cleaned labels | 4bed3e0 | 0.4660 / 0.4608 / 0.4741 (0.4670) | 0.5149 / 0.5135 / 0.5292 (**0.5192**) | <1 min | |
| 6 | rescore: context-hygiene (run 2) | 4bed3e0 | 0.4563 / 0.4510 / 0.4654 (0.4576) | 0.5046 / 0.5021 / 0.5185 (0.5084) | <1 min | |
| 7 | rescore: run 4 with the plain one-class-per-template rule | 4bed3e0 | 0.5109 / 0.5038 / 0.5135 (0.5094) | 0.5622 / 0.5589 / 0.5698 (0.5636) | <1 min | anchoring adds +0.0028 / +0.0029 |
| 8 | rescore: run 4, no rule, fraud_scale 1.0 | 4bed3e0 | 0.5066 / 0.5027 / 0.5096 (0.5063) | 0.5578 / 0.5575 / 0.5647 (0.5600) | <1 min | scale 0.875 vs 1.0: tie (+0.0001), as expected |
| 9 | **champion** (`python -m ringtrace.build recipes/champion.json`) | c46528d | 0.5130 / 0.5090 / 0.5146 (**0.5122**) | 0.5645 / 0.5639 / 0.5712 (**0.5665**) | 12.8 min (3 jobs sharing the GPU) | same recipe as run 4; OOF and test probabilities byte-identical to run 4. Submission VALID, sha256 `a36833d5fb8c497ff3b6e7ea116226796ec8892ad661c17afa87bf92fb8e380e` |
| 10 | evidence-nophones (rerun of 3): no counter features, no rule | 4bed3e0 | 0.4798 / 0.4736 / 0.4831 (0.4788) | 0.5299 / 0.5285 / 0.5381 (0.5322) | 12.6 min (shared GPU) | vs run 4's fixed point (0.5064 / 0.5601): the counter is worth +0.028 / +0.028 |
| 11 | champion from a clean checkout (`git archive`) with an empty cache and fresh silver labels | c46528d | 0.5130 / 0.5090 / 0.5146 (0.5122) | 0.5645 / 0.5639 / 0.5712 (0.5665) | 11.1 min | reproducibility check: same sha256 `a36833d5...`, same silver stats (3,629 rows changed) |
| 12 | ablation: champion without the counter features (rule still on) | 4bed3e0 | 0.4849 / 0.4809 / 0.4879 (0.4846); fixed point 0.4788 | 0.5358 / 0.5362 / 0.5445 (0.5388); fixed point 0.5322 | 9.7 min | counter features: -0.028 / -0.028 at the fixed point, -0.028 / -0.028 after the rule |
| 13 | ablation: champion without the money-flow features (rule still on) | 4bed3e0 | 0.5121 / 0.5086 / 0.5141 (0.5116); fixed point 0.5065 | 0.5639 / 0.5636 / 0.5707 (0.5661); fixed point 0.5601 | 9.7 min | a tie (-0.0006 / -0.0004): the other families and the rule cover chains and bursts |
| 14 | ablations without drains / merchant | â€” | â€” | â€” | â€” | dropped at fold 9 when the ablation queue was stopped for wrap-up; requests, marks, bursts, repeats, context never started |

### Per-class F1 of run 4 (pooled over the three held-out months)

| class | given | cleaned | gen-only |
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

## Submission

`~/outputs/submission.csv` on the box, from run 9 (commit c46528d): 260,582 rows, ids and order
identical to `sample_submission.csv`, labels from the 12-class set, 4.46 MB.
**sha256 `a36833d5fb8c497ff3b6e7ea116226796ec8892ad661c17afa87bf92fb8e380e`.**
October calls: legit 256,637; mule 789, bot 317, chain 556, sim 356, takeover 242, phishing 497,
qr 518, refund 280, cashback 207, loan 99, collusion 84 (3,945 fraud).

## Paired event bootstrap (`python -m ringtrace.compare`, 2,000 draws, 90 % interval, mean of the three months)

| challenger vs base | given | cleaned | verdict |
|---|---|---|---|
| evidence (run 4) vs evidence-nophones (run 10): counter + template rule | +0.0333 [+0.0310, +0.0358] | +0.0343 [+0.0319, +0.0369] | challenger |
| evidence-nophones (run 10) vs context-only (run 1): seven families (with hygiene) | +0.0118 [+0.0094, +0.0142] | +0.0130 [+0.0104, +0.0155] | challenger |

## October-like September (member octlike, laptop: RTX 5070 Ti, 16 threads)

Native mode (`docs/decisions.md` item 13), commit 6b6b8e4:
`python -m ringtrace.octlike build`, `fit champion:0,1,2`, `score champion champion`.
September thinned by protected units to 0.494, fitted on July + August, decoded on its own
(fixed point 0.875 + anchored template rule). Probabilities in
`$RINGTRACE_OUT/octlike/champion_s<seed>_r<r>.npz`.

Frames (September 510,143 rows, 502,823 units, 2,546 of them multi-row, largest 72 rows;
with label units alone 191 of 3,008 September events would be cut, with both 0):

| frame | September rows kept | share | QR-to-person kept / blanked | frame rows |
|---|---|---|---|---|
| s13 | 252,462 | 0.4949 | 326 / 137 | 1,566,961 |
| s14 | 251,850 | 0.4937 | 350 / 161 | 1,566,349 |
| s15 | 252,111 | 0.4942 | 343 / 150 | 1,566,610 |

Champion, September macro-F1 per cell (given / cleaned / gen-only):

| frame | r0 (seeds 0-2) | r1 (3-5) | r2 (6-8) |
|---|---|---|---|
| s13 | 0.5240 / 0.5822 / 0.9054 | 0.5241 / 0.5824 / 0.9057 | 0.5240 / 0.5827 / 0.9060 |
| s14 | 0.5092 / 0.5663 / 0.8897 | 0.5089 / 0.5661 / 0.8893 | 0.5092 / 0.5664 / 0.8898 |
| s15 | 0.5134 / 0.5707 / 0.8986 | 0.5135 / 0.5708 / 0.8986 | 0.5134 / 0.5707 / 0.8984 |

Mean over the 9 cells: given 0.5155, cleaned 0.5731, gen-only 0.8980. Wall time: build 10 s for
the three frames (after a 2.2 min unthinned build with cleaned labels); fit about 110 s per cell
plus about 15 s of feature build per frame, about 20 min for all 9; gate scoring 4.8 min.
The first fit job was stopped by a 10 min shell limit after 4 cells and resumed (finished cells
are skipped), with identical outputs.

Self-check `score champion champion`: change +0.0000 on every thinning seed and target, averaged
lower bound +0.0000 (given, cleaned, gen-only) -> FAIL, as it should (the gate needs a positive
change).

## TabM member (member tabm, laptop: RTX 5070 Ti, 16 threads)

Model: `ringtrace.tabm` (`docs/decisions.md` item 14), recipe `recipes/tabm.json`: the champion's
230 features (337 inputs with 107 gap flags), k = 16 weight-sharing members, 10 epochs, direct
12-class softmax; decoded like the champion (fixed point 0.875 + anchored template rule).
torch 2.14.1+cu130. Two reruns on the GPU give bit-identical probabilities (synthetic check,
no nondeterminism warnings).

Macro-F1 after the rule, per held-out month Jul / Aug / Sep (mean):

| # | run | commit | given | cleaned | gen-only | wall | notes |
|---|---|---|---|---|---|---|---|
| T1 | champion, local rerun (`run recipes/champion.json`) | 25a768c | 0.5130 / 0.5090 / 0.5146 (0.5122) | 0.5645 / 0.5639 / 0.5712 (0.5665) | 0.9020 / 0.8996 / 0.9014 (0.9010) | 6.5 min | matches run 9 exactly; the base for everything below |
| T2 | go/no-go trial: tabm, fold 9 only (fit Jul + Aug), seed 0 (`run recipes/tabm-trial.json --folds 9 --no-test`) | 56fc386 | - / - / 0.5150 | - / - / 0.5717 | - / - / 0.9014 | 5.5 min | champion Sep 0.5146 / 0.5712 / 0.9014: +0.0004 / +0.0005 / 0.0000. **Pass** |
| T3 | trial blend champion + tabm 1:1, Sep (`blend champion+tabm-trial-fold-9 --months 9`) | 56fc386 | - / - / 0.5152 | - / - / 0.5719 | - / - / 0.9018 | <1 min | +0.0006 / +0.0007 / +0.0004 over the champion. **Pass**; 325 Sep argmax calls differ |
| T4 | **tabm**, seeds 0-2, leave-one-month-out + October (`run recipes/tabm.json`) | 56fc386 | 0.5134 / 0.5084 / 0.5151 (**0.5123**) | 0.5649 / 0.5635 / 0.5718 (**0.5667**) | 0.9023 / 0.8984 / 0.9019 (0.9008) | 51.5 min | fixed point alone 0.5070 / 0.5609 (champion 0.5064 / 0.5601); 786 argmax calls differ from the champion over 1.56 M train rows |
| T5 | **champion + tabm** 1:1 (`blend champion+tabm`) | 967c773 | 0.5134 / 0.5092 / 0.5149 (**0.5125**) | 0.5652 / 0.5643 / 0.5716 (**0.5670**) | 0.9031 / 0.9000 / 0.9017 (0.9016) | <1 min | +0.0003 / +0.0005 / +0.0006 over the champion; Aug and Sep given within ±0.0004 |

Probability check: the trial's decoder warning (`invalid value in divide`) came from the
1,053,917 July + August rows of the one-fold run (never predicted, all-zero, unscored), not from
TabM: September had 0 zero-sum and 0 non-finite rows, max |row sum - 1| = 1.5e-7. Fixed in
`run.py` (967c773): only held-out months are decoded, and every held-out and October row must be
finite and sum to 1 before decoding (`check_proba`). Re-decoding the trial gives the same
0.5150 / 0.5717 / 0.9014; the full run's `oof.npy` (1,564,060 rows) and `test.npy` (260,582)
pass the check.

### TabM in the five-way blend (leave-one-month-out, after the template rule)

Members' month-fold arrays copied read-only from the box runs `blend-champion-m1`, `lgbm`,
`catboost`, `clean-event`, `clean-l2`; scored with `python -m ringtrace.blend` (fixed point 0.875
per month, then the rules). The equal five-way blend re-decoded here reproduces the reference
October file exactly (0 differing rows). Change against the five-way blend, Jul / Aug / Sep:

| arm | weights | given | cleaned | gen-only | never loses |
|---|---|---|---|---|---|
| a | tabm 1/6, each member 1/6 | +0.00058 / +0.00009 / +0.00024 | +0.00054 / +0.00008 / +0.00022 | +0.00078 / +0.00008 / -0.00014 | no |
| b | tabm 1/3, each member 2/15 | +0.00055 / +0.00015 / +0.00021 | +0.00067 / +0.00014 / +0.00027 | +0.00092 / +0.00013 / -0.00010 | no |
| **c** | tabm 1/2, each member 1/10 | +0.00050 / +0.00016 / +0.00027 | +0.00051 / +0.00018 / +0.00036 | +0.00065 / +0.00005 / +0.00005 | **yes** |
| c + blank merchant QR rule (commit 80ccae8) | as c, rules `anchored_templates, blank_merchant_qr`, against the five-way blend with the same rules | +0.00050 / +0.00016 / +0.00023 | +0.00051 / +0.00018 / +0.00033 | +0.00065 / +0.00005 / +0.00000 | yes |

Arm c, absolute: given 0.5135 / 0.5093 / 0.5150 (0.5126), cleaned 0.5652 / 0.5643 / 0.5717
(0.5671), gen-only 0.9029 / 0.9000 / 0.9019 (0.9016). With the QR rule the five-way base is
given 0.5123, cleaned 0.5667 (mean) and arm c given 0.5126, cleaned 0.5671. October: arm c moves
5 calls against the five-way blend; with the QR rule, 7 calls against the five-way blend with
that rule. Wall time under 1 min per arm.

### Where arm c's changes come from, and a fraud-only variant (commit after 80ccae8)

Public feedback on the October file of arm c with the QR rule: 0.5483 against 0.5486 for the
five-way blend with the QR rule. Its 7 October changes were 4 legit->fraud adds and 3 fraud type
switches. Hypothesis: TabM's type calls are worse than the trees'. Tested on the train months
(`python -m ringtrace.decompose <five-way> <arm c>`). Calls after the template rule; the QR rule
changes one add in train and nothing else. Precision = share of changed rows where the new call
equals the label; contribution = macro-F1 change with only that category applied, Jul / Aug / Sep.

| category | n | precision given / cleaned | given contribution | cleaned contribution | gen-only contribution |
|---|---|---|---|---|---|
| legit -> fraud | 37 | 0.59 / 0.70 | +0.00035 / +0.00029 / +0.00027 | +0.00032 / +0.00031 / +0.00036 | +0.00036 / +0.00027 / +0.00005 |
| fraud -> legit | 0 | - | 0 | 0 | 0 |
| type switch | 5 | 0.40 / 0.60 | +0.00015 / -0.00013 / 0 | +0.00019 / -0.00013 / 0 | +0.00028 / -0.00022 / 0 |
| of which qr_swap -> refund_abuse | 3 | 0.67 / 1.00 | +0.00013 / 0 / 0 | +0.00028 / 0 / 0 | +0.00043 / 0 / 0 |
| of which refund_abuse -> qr_swap | 1 | 0.00 / 0.00 | +0.00002 / 0 / 0 | -0.00009 / 0 / 0 | -0.00014 / 0 / 0 |
| of which merchant_collusion -> refund_abuse | 1 | 0.00 / 0.00 | 0 / -0.00013 / 0 | 0 / -0.00013 / 0 | 0 / -0.00022 / 0 |

Finding: TabM's gain is in the fraud mass (adds that are right 59 % of the time on given labels,
70 % on cleaned, every month positive). Its type switches are a coin flip: 2 of 5 wrong on
cleaned labels, and the wrong ones cost as much as a right add gains. This agrees with the public
result.

**tabm-fraud-only** (`blend.save_fraud_split`): legit column from arm c; the 11 fraud types are
the five-way blend's distribution rescaled to arm c's fraud mass. Leave-one-month-out after the
template rule: given 0.5137 / 0.5093 / 0.5150 (0.5127), cleaned 0.5654 / 0.5644 / 0.5718
(0.5672), gen-only 0.9031 / 0.9000 / 0.9019 (0.9016). Change:

| comparison | given | cleaned | gen-only | never loses |
|---|---|---|---|---|
| vs five-way blend | +0.00073 / +0.00025 / +0.00029 | +0.00068 / +0.00025 / +0.00043 | +0.00089 / +0.00001 / +0.00004 | yes |
| vs five-way, both with the QR rule | +0.00073 / +0.00025 / +0.00025 | +0.00068 / +0.00025 / +0.00039 | +0.00089 / +0.00001 / -0.000002 | gen-only Sep -2.3e-6 (one row) |
| vs arm c | +0.00023 / +0.00009 / +0.00002 | +0.00017 / +0.00007 / +0.00006 | +0.00024 / -0.00004 / -0.00001 | no (gen-only only) |

Its changes against the five-way blend: 49 adds (precision 0.55 given / 0.65 cleaned) and 3 type
switches, all qr_swap -> refund_abuse and all right. October file with the QR rule last:
`candidates/blend-tabmF-bq.csv`, VALID, sha256 `8a46142660cde2af1576736289f822c1b5c6226204445bce88d4dc311fb926eb`;
7 changes against the five-way blend with the QR rule, all legit -> fraud (phishing_collect 4,
loan_app_extortion 1, merchant_collusion 1, sim_swap 1), no type switches.

### October-like gate (seed set 0 only: 3 cells, cascade seeds 0-2 / TabM seeds 0-2)

`octlike fit tabm:0` (about 15 min per cell on the laptop GPU), then
`octlike score champion <challenger> --sets 0`. TabM per cell, given / cleaned / gen-only:
s13 0.5242 / 0.5828 / 0.9057, s14 0.5096 / 0.5674 / 0.8902, s15 0.5140 / 0.5712 / 0.8986
(champion r0: 0.5240 / 0.5822 / 0.9054, 0.5092 / 0.5663 / 0.8897, 0.5134 / 0.5707 / 0.8986).

| challenger vs champion | given per seed s13 / s14 / s15 (lower bound) | cleaned (lower bound) | gen-only (lower bound) | verdict |
|---|---|---|---|---|
| tabm | +0.0002 / +0.0004 / +0.0006 (-0.0002) | +0.0006 / +0.0010 / +0.0005 (-0.0000) | +0.0002 / +0.0005 / +0.0000 (-0.0009) | FAIL (bounds) |
| champion+tabm | +0.0004 / +0.0005 / +0.0002 (-0.0001) | +0.0007 / +0.0008 / +0.0001 (-0.0000) | +0.0010 / +0.0005 / -0.0002 (-0.0003) | FAIL (bounds) |

Positive on every thinning seed for given and cleaned, but the bootstrap lower bounds do not
clear 0, so the gate fails. Only 3 of the 9 cells were fitted (time box); s13 r1-r2 are being
fitted now. The five-way + tabm October-like gate is not run: the other members' cells do not
exist locally.

### Upload: blend-tabmF-bq (public 0.5499354, 13 Nov 23:00 IST)

`candidates/blend-tabmF-bq.csv` scored **0.5499354** public, +0.0013 over the five-way blend with
the QR rule (0.5486). It now builds with one command from the committed recipe
`recipes/blend-tabmF-bq.json`:

    python -m ringtrace.blend recipes/blend-tabmF-bq.json --write <out.csv>

Rebuilt from a clean `git archive` of `cd3da2b`, using the existing member runs (no refits):
sha256 `8a46142660cde2af1576736289f822c1b5c6226204445bce88d4dc311fb926eb`, byte-identical to the
uploaded file. The first attempt from the clean checkout failed: `blend.py` had the
`__main__` guard above `fraud_split_from`, so `python -m` stopped with a NameError. Fixed in
cd3da2b, with a test.

October-only variants (same recipe, trees' October arrays swapped via `test_run`; TabM's October
array and leave-one-month-out unchanged):

| file | recipe | sha256 | changes vs blend-tabmF-bq |
|---|---|---|---|
| blend-tabmF-refit-bq.csv | recipes/blend-tabmF-refit-bq.json (`*-refit` test arrays) | `7ecd86f7ec5bf1d625ee1269febb075c4859d5e839bc5a35084a32373b2083dd` | 9: refund_abuse->qr_swap 2, refund_abuse->loan_app_extortion 1, refund_abuse->mule_account 1, qr_swap->refund_abuse 1, fake_cashback->refund_abuse 1, loan_app_extortion->fake_cashback 1, legit->qr_swap 1, legit->phishing_collect 1 |
| blend-tabmF-hedge-bq.csv | recipes/blend-tabmF-hedge-bq.json (`*-hedge` test arrays) | `30192d71ff9093f6bd331083fc05e1c4e56314c98e1ac2ffcf18fc2c9ebc37e9` | 6: refund_abuse->qr_swap 1, qr_swap->refund_abuse 1, fake_cashback->refund_abuse 1, loan_app_extortion->fake_cashback 1, legit->qr_swap 1, legit->phishing_collect 1 |

**Rebuild order from scratch** (each step one command; data in `$RINGTRACE_DATA`):
1. Members, each `python -m ringtrace.run <recipe>` (seeds 0, 1, 2; leave-one-month-out; October
   = fold mean): the champion (run name `blend-champion-m1` on the box), `lgbm`, `catboost`,
   `clean-event` and `clean-l2`. Their recipes are on the other members' branches.
2. TabM: `python -m ringtrace.run recipes/tabm.json` (run `tabm`).
3. Blend: `python -m ringtrace.blend recipes/blend-tabmF-bq.json --write submission.csv`.

**TabM from scratch:** torch 2.14.1+cu130 (`pip install torch==2.14.1 --index-url
https://download.pytorch.org/whl/cu130`, the `nn` extra), laptop RTX 5070 Ti, 16 threads.
`recipes/tabm.json` = TabMSpec defaults with seeds 0, 1, 2 and 10 epochs (k 16, width 512,
3 blocks, dropout 0.1, periodic embedding 24 frequencies to 8 dims, batch 2,048, AdamW lr 2e-3,
weight decay 3e-4, 5 % warm-up then linear decay). Determinism: `torch.manual_seed(seed)`,
a seeded `torch.Generator` for the batch order, `torch.use_deterministic_algorithms(True)`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, cudnn benchmark off, TF32 off; preprocessing quantiles use
a numpy generator seeded 0. Verified: two GPU runs on a synthetic problem (40,000 rows) are
bit-identical with no nondeterminism warnings, and the CPU test reruns are identical. **Not yet
verified:** a byte-identical rerun of the full `tabm` member (51 min). Another GPU or driver may
change float rounding, so a rebuilt `test.npy` can differ slightly and move a few October calls.
Wall time 51.5 min (3 folds x 3 seeds, about 17 min per fold).

October-like s13, seed sets r1 and r2 (TabM seeds 3-5 and 6-8): given / cleaned / gen-only
0.5243 / 0.5829 / 0.9062 and 0.5241 / 0.5830 / 0.9063, 15.5-16 min each. The champion's r1 and
r2 in the same frame: 0.5241 / 0.5824 / 0.9057 and 0.5240 / 0.5827 / 0.9060.

### October-like gate on s13, all three seed sets (`score champion <c> --seeds 13 --sets 0,1,2`)

| challenger vs champion | given change (lower bound) | cleaned | gen-only | verdict |
|---|---|---|---|---|
| tabm | +0.0001 (-0.0003) | +0.0005 (-0.0001) | +0.0004 (-0.0006) | FAIL |
| champion+tabm | +0.0003 (-0.0001) | +0.0005 (+0.0000) | +0.0006 (-0.0001) | FAIL |

The s14 and s15 r1-r2 cells are not on the laptop, so the nine-cell gate is pending.

### TabM share of the fraud mass (fraud-only split, both rules), change vs w = 1/2

| w | given Jul / Aug / Sep | cleaned | gen-only |
|---|---|---|---|
| 1/3 | +0.00000 / -0.00001 / -0.00002 | +0.00000 / -0.00004 / -0.00006 | +0.00000 / +0.00017 / +0.00007 |
| 2/3 | -0.00014 / -0.00005 / +0.00010 | -0.00015 / -0.00005 / +0.00004 | -0.00047 / -0.00014 / +0.00003 |
| 4/5 | -0.00007 / -0.00027 / +0.00006 | -0.00008 / -0.00028 / -0.00001 | -0.00039 / -0.00071 / -0.00015 |
| 1 | +0.00004 / -0.00027 / -0.00002 | +0.00002 / -0.00024 / -0.00009 | -0.00027 / -0.00069 / -0.00038 |

No share beats 1/2 on given and cleaned in every month; no file written.

### TabM reproducibility: full member rerun (`tabm-repro`, same recipe under another name)

Rerun of `recipes/tabm.json` on the same laptop GPU (commit a423c96 code, same seeds): `oof.npy`
and `test.npy` are **byte-identical** to run T4. Scores are identical too: given 0.5123, cleaned
0.5667, gen-only 0.9008. Wall 47.3 min. So the TabM member, and with it `blend-tabmF-bq`,
rebuilds exactly on this hardware and software. Another GPU model or driver is still unverified.

`tabm-s345` (seeds 3-5): the first attempt died with CUDA out of memory in fold 7, seed 3, after
epoch 7 (log `tabm-s345-run-oom.log`). No other job of this member was on the GPU at the time.
It has been relaunched.
