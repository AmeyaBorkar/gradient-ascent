# Results

Every scored run, failures included. Leave-one-month-out (train Jul+Aug -> score Sep, etc.);
macro-F1 over 12 classes per held-out month; "given" = labels as shipped, "cleaned" = the silver
labels of `ringtrace.silver` (frozen once built: 3,629 train rows changed, 2,944 legit -> fraud,
685 wrong types; 62.9 % of fraud labels judged ordinary-looking), "gen" = cleaned labels with the
ordinary-looking fraud rows set to legit. Decoder = fixed point per month unless stated.

| # | run (recipe) | commit | given Jul / Aug / Sep (mean) | cleaned Jul / Aug / Sep (mean) | wall | notes |
|---|---|---|---|---|---|---|
| 1 | context-only | 6fdef89 | 0.4660 / 0.4607 / 0.4748 (**0.4671**); argmax 0.4623 | — (silver not built yet) | 6.6 min | 123 relational features, cascade 3 seeds x 300 rounds; team's relational-v1 was 0.4663 given |
| 2 | context-hygiene | 1421970 | 0.4563 / 0.4510 / 0.4654 (**0.4576**); argmax 0.4572 | — | 4.0 min | raw amounts, capped gaps, no calendar/whole-frame totals, receiver windows blanked on merchants; ordinary rows legit in stage one. -0.0095: refund (-0.064) and collusion (-0.037) lose the merchant-receiver windows; kept, since the merchant/repeat families carry those classes |
| 3 | evidence-nophones | e954038 | — | — | — | died at fold 9 (its code dir was replaced by a later ship; the launcher now snapshots code per job); rerun as run 5 |
| 4 | **evidence** (context + 7 families + counter, anchored template rule) | 41deb00 | 0.5130 / 0.5090 / 0.5146 (**0.5122**); fixed point 0.5064, argmax 0.5066 | 0.5645 / 0.5639 / 0.5712 (**0.5665**); fixed point 0.5601, argmax 0.5602 | 5.8 min | 230 features; generator-only target 0.9010. Target 0.5126 / 0.5662. Template rule changed 773 train / 137 test rows (+0.0058 given / +0.0064 cleaned over the fixed point) |
| 5 | rescore: context-only (run 1) on cleaned labels | 4bed3e0 | 0.4660 / 0.4608 / 0.4741 (0.4670) | 0.5149 / 0.5135 / 0.5292 (**0.5192**) | <1 min | pack's relational-v1: 0.4663 / 0.5175 |
| 6 | rescore: context-hygiene (run 2) | 4bed3e0 | 0.4563 / 0.4510 / 0.4654 (0.4576) | 0.5046 / 0.5021 / 0.5185 (0.5084) | <1 min | |
| 7 | rescore: run 4 with the plain one-class-per-template rule | 4bed3e0 | 0.5109 / 0.5038 / 0.5135 (0.5094) | 0.5622 / 0.5589 / 0.5698 (0.5636) | <1 min | the notes' plain template-rule model: 0.5095 / 0.5630; anchoring adds +0.0028 / +0.0029 |
| 8 | rescore: run 4, no rule, fraud_scale 1.0 | 4bed3e0 | 0.5066 / 0.5027 / 0.5096 (0.5063) | 0.5578 / 0.5575 / 0.5647 (0.5600) | <1 min | scale 0.875 vs 1.0: tie (+0.0001), as the research notes says |
| 9 | **champion** (`python -m ringtrace.build recipes/champion.json`) | c46528d | 0.5130 / 0.5090 / 0.5146 (**0.5122**) | 0.5645 / 0.5639 / 0.5712 (**0.5665**) | 12.8 min (3 jobs sharing the GPU) | same recipe as run 4; OOF and test probabilities byte-identical to run 4. Submission VALID, sha256 `a36833d5fb8c497ff3b6e7ea116226796ec8892ad661c17afa87bf92fb8e380e` |
| 10 | evidence-nophones (rerun of 3): no counter features, no rule | 4bed3e0 | 0.4798 / 0.4736 / 0.4831 (0.4788) | 0.5299 / 0.5285 / 0.5381 (0.5322) | 12.6 min (shared GPU) | vs run 4's fixed point (0.5064 / 0.5601): the counter is worth +0.028 / +0.028 (pack: +0.025 to +0.028) |
| 11 | champion from a clean checkout (`git archive`) with an empty cache and fresh silver labels | c46528d | 0.5130 / 0.5090 / 0.5146 (0.5122) | 0.5645 / 0.5639 / 0.5712 (0.5665) | 11.1 min | reproducibility check: same sha256 `a36833d5...`, same silver stats (3,629 rows changed) |
| 12 | ablation: champion without the counter features (rule still on) | 4bed3e0 | 0.4849 / 0.4809 / 0.4879 (0.4846); fixed point 0.4788 | 0.5358 / 0.5362 / 0.5445 (0.5388); fixed point 0.5322 | 9.7 min | counter features: -0.028 / -0.028 at the fixed point, -0.028 / -0.028 after the rule |
| 13 | ablation: champion without the money-flow features (rule still on) | 4bed3e0 | 0.5121 / 0.5086 / 0.5141 (0.5116); fixed point 0.5065 | 0.5639 / 0.5636 / 0.5707 (0.5661); fixed point 0.5601 | 9.7 min | a tie (-0.0006 / -0.0004): the other families and the rule cover chains and bursts |
| 14 | ablations without drains / merchant | — | — | — | — | dropped at fold 9 when the ablation queue was stopped for wrap-up; requests, marks, bursts, repeats, context never started |

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
qr 518, refund 280, cashback 207, loan 99, collusion 84 (3,945 fraud; the research notes' champion called
3,982 with a very similar mix).

## Paired event bootstrap (`python -m ringtrace.compare`, 2,000 draws, 90 % interval, mean of the three months)

| challenger vs base | given | cleaned | verdict |
|---|---|---|---|
| evidence (run 4) vs evidence-nophones (run 10): counter + template rule | +0.0333 [+0.0310, +0.0358] | +0.0343 [+0.0319, +0.0369] | challenger |
| evidence-nophones (run 10) vs context-only (run 1): seven families (with hygiene) | +0.0118 [+0.0094, +0.0142] | +0.0130 [+0.0104, +0.0155] | challenger |

## October-like September (phase 2: probabilities for the blend test)

`python -m ringtrace.octlike run ~/data-oct/s<seed> champion:0,1,2 champion-cleantrain:0,1,2`
(commit a2b10b5). Fit on July + August of each thinned frame, September predicted; counter
columns from the unthinned build joined by stacked row. Score = September given-label macro-F1
after the fixed-point decoder (no template rule in this mode), aggregates only. Files:
`~/outputs/octlike/<variant>_s<seed>_r<r>.npz`.

| frame | seed set (cascade seeds) | champion, fixed point | argmax |
|---|---|---|---|
| s13 | r0 (0, 1, 2) | 0.5060 | 0.5056 |
| s13 | r1 (3, 4, 5) | 0.5061 | 0.5056 |
| s13 | r2 (6, 7, 8) | 0.5058 | 0.5056 |
| s14 | r0 | 0.5094 | 0.5098 |
| s14 | r1 | 0.5095 | 0.5097 |
| s14 | r2 | 0.5098 | 0.5095 |
| s15 | r0 | 0.5126 | 0.5124 |
| s15 | r1 | 0.5125 | 0.5126 |
| s15 | r2 | 0.5124 | 0.5130 |

Champion mean over the 9 cells 0.5093 (frames 0.5060 / 0.5096 / 0.5125; seed sets within
+-0.0003), against 0.5097 for September at full density in the month folds (run 4, fixed point).
Wall time: 4.5-5 min per cell with three frames sharing the GPU and 4 CPU threads each.

Cleaned-label variant (`recipes/champion-cleantrain.json`; cleaned labels built on each frame
from July + August only, the teacher fitted on the other month), same scoring:

| frame | r0 (0, 1, 2) | r1 (3, 4, 5) | r2 (6, 7, 8) | rows changed by cleaning (of 1,053,917) |
|---|---|---|---|---|
| s13 | 0.5063 | 0.5061 | 0.5064 | 2,419 (1,954 legit -> fraud, 465 wrong types) |
| s14 | 0.5080 | 0.5094 | 0.5089 | 2,421 (1,950 legit -> fraud, 471 wrong types) |
| s15 | 0.5125 | 0.5120 | 0.5121 | 2,408 (1,935 legit -> fraud, 473 wrong types) |

Cleantrain mean over the 9 cells 0.5091 against 0.5093 for the given-label champion: a tie on the
given labels (the cleaned target cannot be scored on September here, which has no cleaned labels
by protocol). All 18 cells ran in 28 min (16:39-17:07 IST), three frames in parallel.

## Member `blend` (the blend and the one-command build)

Leave-one-month-out macro-F1 per held-out month Jul / Aug / Sep (mean), after the fixed point
(scale 0.875) and the anchored template rule.

| run (output dir) | commit | given | cleaned | gen-only | wall | notes |
|---|---|---|---|---|---|---|
| blend-champion (`~/outputs/runs/blend-champion-m1`), tries 1-6 | b893b74 / 50c44a6 | — | — | — | — | failed: CUDA out of memory in fold 7 (GPU shared with another member's job; the retry gate first watched the wrong device). Logs `~/logs/blend-champion-oom{1,2}.log` |
| **blend-champion-m1**: `python -m ringtrace.run recipes/champion.json --name blend-champion-m1` | d02e04b | 0.5130 / 0.5090 / 0.5146 (**0.5122**) | 0.5645 / 0.5639 / 0.5712 (**0.5665**) | 0.9020 / 0.8996 / 0.9014 (**0.9010**) | 10.7 min (4 threads, shared GPU) | member 1 from this branch: identical to `champion` (run 9). Fixed point 0.5064 / 0.5601 / 0.8921. Submission VALID against the sample, sha256 `a36833d5fb8c497ff3b6e7ea116226796ec8892ad661c17afa87bf92fb8e380e` (same as run 9) |
