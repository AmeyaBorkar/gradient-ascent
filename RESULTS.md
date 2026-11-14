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

## Neural training techniques (commits 9403a2f-43730fd)

### Screen: one fit on July + August per technique, TabM seed 0, against the plain TabM, same seed

`python -m ringtrace.nnexp fit --seeds 0 <variant>`, then `nnexp score <variant>-sd0 --base base-sd0`.
Cells: `sep` (leave-one-month-out fold 9) and the October-like frames s13 / s14 / s15 (r0).
Roles:
- **fraud-only:** legit column from TabM at 1/2 plus the trees at 1/2; fraud types from the
  trees. On `sep` the trees are the five-way blend's month-fold probabilities. On the October-like
  cells only champion, lgbm, catboost and clean-event have r0 cells; clean-l2 has none.
- **alone:** TabM decoded by itself.

Calls are the fixed point, then `anchored_templates` and `blank_merchant_qr`. Each entry is the
change in macro-F1 x 1e-4 on given / cleaned / gen-only. "Never loses" means no target in any
cell is below 0.

Wall time per cell: about 4.5 min on the 4090 for TabM techniques (13-28 min on the shared A6000
for kd30 / kd70), under 1 min for RealMLP. Each technique's settings are in
`nnexp.VARIANTS`.

| technique | role | sep | s13 | s14 | s15 | never loses |
|---|---|---|---|---|---|---|
| swa | fraud-only | +0.1 / +0.1 / +0.4 | +0.3 / +0.3 / +0.8 | +0.0 / +0.0 / +0.0 | +0.0 / +0.0 / +0.0 | yes |
| swa | alone | +1.3 / +1.3 / +0.9 | +1.6 / +0.9 / +4.8 | +4.2 / +4.1 / +4.7 | +0.5 / +0.5 / +1.5 | yes |
| ema | fraud-only | -1.3 / -1.4 / -4.2 | +0.0 / +0.0 / -0.0 | -0.8 / +3.0 / +4.4 | +0.0 / +0.0 / +0.0 | no |
| ema | alone | +1.7 / +1.6 / +1.0 | +1.0 / +0.1 / +1.3 | +3.9 / +3.8 / +4.0 | +3.6 / +3.3 / +0.8 | yes |
| qraug | fraud-only | +0.3 / +0.2 / +0.1 | +0.5 / +0.6 / +1.6 | +0.6 / +4.2 / +5.6 | +0.2 / +0.3 / +0.8 | yes |
| qraug | alone | +2.4 / +2.3 / -1.3 | +1.4 / +1.6 / +5.8 | -1.1 / +1.1 / -12.6 | +2.4 / +2.2 / +0.2 | no |
| kd30 | fraud-only | +0.1 / +0.1 / +0.4 | +0.5 / +0.6 / +1.6 | +0.8 / +0.8 / +1.0 | -0.8 / -0.7 / -1.0 | no |
| kd30 | alone | +1.4 / +1.4 / -2.5 | +0.5 / -0.2 / +10.3 | -3.1 / -2.9 / +0.6 | +2.5 / +2.7 / +8.5 | no |
| kd70 | fraud-only | -0.4 / -0.4 / -0.6 | -1.2 / -1.0 / -0.4 | -0.3 / -0.3 / -0.8 | -1.5 / -1.2 / -0.2 | no |
| kd70 | alone | -0.1 / +0.8 / -1.0 | +2.6 / -2.0 / +13.3 | -4.5 / -8.9 / -5.9 | +2.1 / +1.6 / +8.6 | no |
| mixup | fraud-only | +0.1 / +0.1 / +0.4 | -0.3 / -0.2 / +0.6 | +0.8 / +0.8 / +1.0 | +0.2 / +0.3 / +0.8 | no |
| mixup | alone | +2.7 / +0.8 / -3.1 | +3.6 / +2.1 / +9.5 | +2.8 / +2.6 / -1.6 | +1.8 / +1.6 / -2.0 | no |
| noise | fraud-only | -0.6 / -0.6 / +0.2 | +0.5 / +0.6 / +1.6 | -0.8 / +3.0 / +4.4 | - | no |
| noise | alone | +0.2 / -0.1 / -4.4 | +3.7 / +2.2 / +10.5 | +0.7 / +0.6 / -3.2 | - | no |
| pre | fraud-only | -0.4 / -0.4 / -0.5 | -0.2 / +0.7 / +0.9 | - | - | no |
| pre | alone | +2.4 / +2.3 / -1.2 | +6.9 / +2.0 / +12.2 | - | - | no |
| smooth | fraud-only | -0.8 / -0.6 / -4.4 | -4.1 / +1.8 / -4.3 | -2.6 / +3.9 / -7.5 | -4.4 / -3.8 / -2.0 | no |
| smooth | alone | +2.3 / +2.1 / -1.7 | +0.0 / -0.7 / -8.0 | -3.6 / -0.9 / -21.8 | -3.6 / -4.0 / -5.4 | no |
| logitadj | fraud-only | -3034.9 / -3387.9 / -6204.7 | -3290.0 / -3673.6 / -6473.4 | -3189.9 / -3570.9 / -6380.2 | -3369.5 / -3751.5 / -6607.9 | no |
| logitadj | alone | -3696.7 / -4066.2 / -6677.8 | -3900.6 / -4297.8 / -7012.2 | -3336.9 / -3663.4 / -6106.0 | -3639.1 / -4005.8 / -6703.9 | no |
| focal | fraud-only | -562.3 / -621.3 / -1256.6 | -572.5 / -621.1 / -1230.1 | -576.2 / -628.3 / -1244.9 | -489.6 / -539.3 / -1094.8 | no |
| focal | alone | -582.2 / -640.7 / -1482.3 | -595.9 / -650.6 / -1463.2 | -505.2 / -549.5 / -1240.9 | -429.3 / -469.9 / -1084.8 | no |
| realmlp | fraud-only | -2.8 / -1.0 / -12.9 | -0.3 / +2.8 / -3.0 | -0.6 / +5.5 / +3.4 | -7.1 / -5.5 / -5.5 | no |
| realmlp | alone | -6.7 / -7.4 / -20.9 | +0.0 / +0.4 / +8.9 | -5.7 / -5.4 / -15.8 | -6.8 / -8.9 / -15.8 | no |

Outputs: `$RINGTRACE_OUT/nntech/<variant>-sd0_<cell>.npz` (rows and probabilities of the held-out
September) and `score_<variant>-sd0_vs_base-sd0.json`. Noise and pretraining are partial
(noise: sep, s13, s14; pretraining: sep, s13), because the machine running them was shut down.
Both already lose in the fraud-only role on `sep`. Failures:
- the first champion-cascade teacher crashed: the cascade needs two training months;
- the 3-seed first launch was abandoned for time;
- the A6000 was contended at 85 s per epoch.

### Full member: SWA (last 3 epochs), seeds 0-2, leave-one-month-out + October

`python -m ringtrace.run recipes/nntech-swa.json` (commit 43730fd), 51 min on the A6000.

- **Alone, after the rule:** given 0.5134 / 0.5085 / 0.5151 (0.5124), cleaned 0.5649 / 0.5637 /
  0.5719 (0.5668), gen-only 0.9023 / 0.8987 / 0.9017 (0.9009). Plain TabM (T4) scored 0.5123 /
  0.5667 / 0.9008.
- **Fraud-only** (`blend.save_fraud_split` as for `tabm-fraud-only`, run `nntech-swa-fraud-only`),
  with rules `anchored_templates,blank_merchant_qr`, against `tabm-fraud-only`, Jul / Aug / Sep:

| target | change |
|---|---|
| given | -0.000041 / +0.000013 / +0.000041 |
| cleaned | -0.000044 / +0.000008 / +0.000037 |
| gen-only | -0.000127 / -0.000025 / +0.000049 |

**FAIL**: it loses July on every target. The October-like cells of the 3-seed member were not
fitted, because no machine was left. In the screen's fraud-only role SWA moved 0-1 calls per
October-like cell (s14 and s15 exactly 0), so it cannot be positive on every thinning seed.

October file (template rule, then `blank_merchant_qr`): `candidates/nntech-swa.csv`, VALID,
sha256 `715138bd2d1d6bc48338f34873363c310a2493556f1769ec0cc9a94d2f05fc3a`. Against
`blend-tabmF-bq.csv` (sha256 `8a461426...fb926eb`) it changes 1 call: legit -> phishing_collect.
Not a submission candidate.

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

## Neural techniques, second screen (commits f2c3526-98b546d)

### Screen: one fit on July + August per technique, seed 0, against the plain TabM on the same GPU

`python -m ringtrace.nnexp fit <variant> --seeds 0 --job <job>`, then `nnexp score <job>-<variant>-sd0
--base <job>-base-sd0` and `nnexp table`. Cells, roles, rules and units are those of the first
screen above: change in macro-F1 x 1e-4 on given / cleaned / gen-only, calls changed in brackets.
**Fraud-only** puts the variant in TabM's half of the legit column; the trees keep the type
split.

The plain TabM was refitted on each machine (`wildrun-base-sd0` on the laptop RTX 5070 Ti,
`wildrun-box-base-sd0` on the box RTX 5060 Ti), and every variant is judged against the base of
its own machine. The laptop base and the earlier screen's base (fitted on another GPU) disagree on
70 September argmax calls, max probability gap 0.52. Hardware alone moves the fit as much as a
seed change. All scoring ran on the box CPU, with the laptop's tree cells copied over. The box's
own `champion_s1x_r0` cells hold different rows and were not used.

| technique | where | role | sep | s13 | s14 | s15 | never loses |
|---|---|---|---|---|---|---|---|
| selftrain | laptop | fraud-only | -0.1 / -0.1 / +0.2 (2) | -1.2 / +2.3 / +2.7 (2) | +0.0 / +0.0 / +0.0 (0) | +0.0 / +0.0 / +0.0 (0) | no |
| selftrain | laptop | alone | -0.1 / -0.0 / +1.7 (10) | -3.7 / -3.6 / -3.1 (12) | +1.9 / +3.0 / +8.1 (8) | -2.1 / -1.3 / -5.1 (16) | no |
| entity | box | fraud-only | -4.2 / -6.1 / -13.1 (46) | -4.5 / -5.1 / -17.6 (37) | -4.6 / -6.0 / -16.3 (25) | -4.7 / -4.7 / -11.2 (23) | no |
| entity | box | alone | -62.4 / -70.5 / -121.1 (169) | -25.3 / -34.9 / -62.4 (106) | -48.8 / -57.7 / -138.1 (112) | -45.5 / -53.2 / -96.5 (95) | no |
| tabr | laptop | fraud-only | +0.1 / +1.5 / +1.9 (16) | +0.6 / +2.5 / +3.8 (11) | -1.0 / -3.7 / -5.4 (6) | -3.4 / -1.1 / -0.2 (13) | no |
| tabr | laptop | alone | -88.6 / -103.2 / -139.7 (180) | -85.5 / -105.4 / -135.9 (110) | -81.9 / -104.1 / -143.5 (91) | -86.9 / -101.6 / -136.9 (99) | no |
| tabm + tabr, equal mean (no new fit) | laptop | fraud-only | +0.9 / +1.8 / +2.8 (6) | +1.3 / +3.1 / +4.3 (5) | +0.7 / -3.0 / -4.5 (3) | +0.4 / +0.5 / +1.4 (6) | no |
| tabm + tabr, equal mean (no new fit) | laptop | alone | -7.1 / -6.1 / -17.0 (43) | -10.3 / -14.4 / -22.7 (44) | -4.3 / -3.4 / -4.2 (23) | -20.1 / -20.1 / -25.7 (36) | no |
| shift | box | fraud-only | -0.2 / -0.1 / +0.3 (3) | +0.4 / +0.5 / +2.5 (5) | +0.3 / +0.3 / +0.7 (1) | +0.2 / +0.3 / +2.1 (5) | no |
| shift | box | alone | +0.6 / +0.7 / +2.0 (7) | +1.6 / -2.0 / +12.0 (27) | -3.5 / -4.5 / +0.3 (6) | -5.3 / -5.9 / +11.6 (16) | no |
| metric | laptop | fraud-only | -0.4 / -0.4 / -0.5 (1) | -1.9 / +1.4 / +0.3 (5) | +0.6 / +0.5 / +0.2 (2) | +0.0 / +0.0 / +0.0 (0) | no |
| metric | laptop | alone | +2.0 / +1.9 / +2.0 (6) | -2.4 / -2.6 / -6.6 (8) | +1.4 / +2.2 / +1.7 (7) | +0.0 / +1.0 / +1.7 (4) | no |
| gentarget | box | fraud-only | -20.3 / -18.5 / +29.5 (45) | -21.7 / -18.9 / +25.5 (22) | -11.9 / -10.8 / +15.2 (12) | -21.0 / -24.5 / -4.9 (61) | no |
| gentarget | box | alone | -155.3 / -141.9 / +198.6 (404) | -156.6 / -146.1 / +195.2 (218) | -163.1 / -153.1 / +180.8 (216) | -523.2 / -553.1 / -283.3 (746) | no |
| entity ablation: pooled history alone | box | fraud-only | -0.8 / -0.8 / -0.9 (5) | +0.5 / +0.6 / +1.6 (2) | +0.0 / +3.7 / +5.4 (2) | +0.5 / +0.5 / +1.5 (2) | no |
| entity ablation: pooled history alone | box | alone | +1.7 / +1.7 / +3.0 (12) | +3.8 / +3.2 / +10.2 (15) | +1.3 / +0.8 / -3.7 (10) | +0.7 / +1.1 / +9.0 (11) | no |
| entity ablation: embeddings alone | laptop | fraud-only | -16.7 / -18.0 / -48.0 (136) | -11.4 / -12.4 / -34.3 (48) | -8.0 / -12.4 / -30.6 (34) | -2.6 / -2.6 / -6.7 (18) | no |
| entity ablation: embeddings alone | laptop | alone | -46.4 / -52.3 / -102.1 (161) | -46.2 / -56.6 / -117.4 (133) | -61.3 / -70.6 / -164.7 (142) | -38.4 / -44.3 / -96.5 (88) | no |

**Nothing passes; nothing was promoted, and no October file was written.** The bar for
promotion was a fraud-only change that is non-negative in every cell and target and moves more
than a handful of calls. The closest is `shift`: positive on every target in all three
October-like cells, but on 1-5 calls, and -0.2 / -0.1 on September.

Settings (all in `nnexp`; each technique's finding in `docs/findings/nn-*.md`):
- **selftrain**: teacher = the plain seed-0 TabM's probabilities of the scored rows; every scored
  row kept as a soft target at weight 0.3; fraud classes capped at the floor of the teacher's
  mass (never binds); legit uncapped.
- **entity**: sender / receiver / device hashed into 2^18 x 8 embeddings with id dropout 0.2.
  Ids not seen in the fit rows go to the unknown row. Means of up to 8 strictly earlier rows per
  sender and per receiver over 8 raw columns.
- **tabr**: `TabRSpec` defaults (d 256, k 32, memory = all fit fraud rows + 200,000 legit).
- **shift**: LightGBM domain classifier (cross-fitted halves, balanced, 200 rounds) on 212 of 230
  columns; weights clipped to [0.2, 5], mean 1.
- **metric**: cross-entropy + 0.5 x REINFORCE batch macro-F1 with an EMA baseline.
- **gentarget**: generator-only labels rebuilt per cell from a July <-> August teacher, with every
  `silver.build` estimate on the fit rows.

Wall time per cell: plain TabM 5.4-5.7 min (laptop) / 5.0-5.1 min (box); selftrain 7.0-8.5 min;
entity 6.1-6.2 min; tabr 8.7-9.1 min; shift 5.4-5.7 min including the classifier; metric
5.6-5.9 min; gentarget 5.7-5.9 min including the teacher. Scoring ran at 2 threads on the box
CPU.

Failures and fixes:
- **shift, first fit (a6d347d), stopped.** On all 230 columns the domain AUC was 1.0000 on
  `sep`, and 99.99 % of the weights sat on the clip floor. The 1-day and 7-day window counts are
  divided by a per-month density factor (`context.density`), so their values name the month.
  665e150 leaves out the 18 columns whose own AUC between July + August and the unthinned
  September is more than 0.05 from 0.5. With them out, the `sep` domain AUC is 0.594 and the
  frames' AUC is 0.786-0.788.
- **gentarget, s15.** `silver.build`'s old-app likelihood ratio has a near-zero denominator when
  fit on two months: 2.77 / 5.11 / 10.47 / inf across `sep` / s13 / s14 / s15. In s15 it labels
  86 % of fraud ordinary, against 63 % elsewhere. The cell was kept as specified; the shared
  label code is unchanged.
- **Colab:** the helper script was not present, so no runtime was used.
- **Laptop memory:** a preliminary score running beside a fit pushed free RAM to 4.8 of 31 GB.
  From then on the laptop ran one heavy process at a time, and all scoring moved to the box.
