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

## October-like September (laptop: RTX 5070 Ti, 16 threads)

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

Earlier thinned-frame mode (before the native rewrite): Cleantrain mean over the 9 cells 0.5091 against 0.5093 for the given-label champion: a tie on the
given labels (the cleaned target cannot be scored on September here, which has no cleaned labels
by protocol). All 18 cells ran in 28 min (16:39-17:07 IST), three frames in parallel.

## Blend member 2: LightGBM cascade (`lgbm`)

`python -m ringtrace.run recipes/lgbm.json` with `RINGTRACE_THREADS=10`. This is the champion recipe (families,
drop list, XGBoost screening score and cut-offs, month folds, seeds 0/1/2, decoder, template rule)
with both cascade stages on CPU LightGBM: lr 0.05, 63 leaves, min_data_in_leaf 30, feature/bagging
fraction 0.5/0.8 (freq 1), lambda_l2 2.0, max_bin 255, 300 rounds, deterministic, row-wise. Outputs:
`~/outputs/runs/lgbm/` (oof.npy, test.npy, summary.json, submission.csv).

| run | commit | stage | given Jul / Aug / Sep (mean) | cleaned Jul / Aug / Sep (mean) | gen-only Jul / Aug / Sep (mean) | wall |
|---|---|---|---|---|---|---|
| lgbm | 363eab3 | argmax | 0.5077 / 0.5039 / 0.5108 (0.5075) | 0.5588 / 0.5588 / 0.5661 (0.5612) | 0.8949 / 0.8925 / 0.8954 (0.8942) | 14.9 min |
| lgbm | 363eab3 | fixed point | 0.5076 / 0.5034 / 0.5110 (0.5073) | 0.5586 / 0.5583 / 0.5664 (0.5611) | 0.8945 / 0.8914 / 0.8958 (0.8939) | |
| lgbm | 363eab3 | **rules (final)** | 0.5130 / 0.5092 / 0.5146 (**0.5122**) | 0.5646 / 0.5642 / 0.5712 (**0.5667**) | 0.9021 / 0.9000 / 0.9012 (0.9011) | |
| champion (run 9) | c46528d | rules (final) | 0.5130 / 0.5090 / 0.5146 (0.5122) | 0.5645 / 0.5639 / 0.5712 (0.5665) | — / — / — (0.9010) | 12.8 min |

On its own the LightGBM member ties the champion: +0.0000 given and +0.0002 cleaned after the rule, and
+0.0009 / +0.0010 at the fixed point. CPU only, at 892 % CPU (10 threads), with a peak RSS of 8.0 GB.
The template rule changed 742 train / 132 test rows. October calls: 3,951 fraud (champion 3,945).

## Blend member 3: CatBoost cascade

Champion features, screening and folds; both stages CatBoost on the GPU (decision 14), 4 CPU threads.
Output: `~/outputs/runs/catboost/` (`oof.npy`, `test.npy`, `summary.json`, `submission.csv`).

| run | commit | given Jul / Aug / Sep (mean) | cleaned Jul / Aug / Sep (mean) | gen-only Jul / Aug / Sep (mean) | wall | notes |
|---|---|---|---|---|---|---|
| catboost (rules) | e8613ed | 0.5129 / 0.5090 / 0.5141 (**0.5120**) | 0.5644 / 0.5637 / 0.5706 (**0.5662**) | 0.9023 / 0.8995 / 0.9008 (0.9008) | 3.6 min | vs champion 0.5122 / 0.5665: -0.0002 / -0.0003. Rule changed 459 train / 88 test rows |
| catboost (fixed point) | e8613ed | 0.5098 / 0.5048 / 0.5123 (0.5090) | 0.5611 / 0.5594 / 0.5680 (0.5628) | 0.8994 / 0.8930 / 0.8988 (0.8971) | — | champion fixed point 0.5064 / 0.5601: +0.0026 / +0.0027 |
| catboost repro (fold Sep, seed 0, twice) | d83b3d7 | — | — | — | 2 x 1 min | max abs prob diff 0.165, mean 4.3e-6, 2 argmax changes |

## Blend members 4 and 5: event-cleaned training labels (`clean`)

`python -m ringtrace.run recipes/clean-<v>.json` (champion recipe plus `train_on`), seeds 0/1/2,
leave-one-month-out, October = fold mean; 3 CPU threads each, two jobs sharing the GPU. Labels are
recleaned inside each fold from the fit months' labels only (`ringtrace.eventclean`). Scores after
the template rule (fixed point in brackets); given / cleaned / gen-only, Jul / Aug / Sep (mean).

| run | commit | rows changed (all train; per fold 7/8/9) | given | cleaned | gen-only | wall |
|---|---|---|---|---|---|---|
| champion (run 9, reference) | c46528d | 0 | 0.5130 / 0.5090 / 0.5146 (**0.5122**); fp 0.5064 | 0.5645 / 0.5639 / 0.5712 (**0.5665**); fp 0.5601 | 0.9010 | 12.8 min |
| clean-event, first try | 62c5fdb | — | — | — | — | failed in fold 7: CUDA out of memory (shared GPU); relaunched |
| clean-event | 62c5fdb | 2,337 (all legit -> fraud); 1,509 / 1,519 / 1,530 | 0.5131 / 0.5091 / 0.5146 (**0.5123**); fp 0.5068 | 0.5646 / 0.5640 / 0.5713 (**0.5666**); fp 0.5604 | 0.9012 | 12.4 min |
| clean-l2 | 62c5fdb | 2,626 (2,337 legit -> fraud, 289 minority types); 1,697 / 1,727 / 1,710 | 0.5131 / 0.5091 / 0.5146 (**0.5122**); fp 0.5060 | 0.5645 / 0.5642 / 0.5713 (**0.5666**); fp 0.5594 | 0.9012 | 12.0 min |

Both tie the champion after the rule (+0.0001 / +0.0001 and 0.0000 / +0.0001); at the fixed point
`clean` is +0.0004 / +0.0003 and `l2` -0.0004 / -0.0007. Their value is as blend members.
Outputs: `~/outputs/runs/clean-event/` and `~/outputs/runs/clean-l2/` (`oof.npy`, `test.npy`,
`summary.json`, `submission.csv`).

## The blend and the one-command build

Leave-one-month-out macro-F1 per held-out month Jul / Aug / Sep (mean), after the fixed point
(scale 0.875) and the anchored template rule.

| run (output dir) | commit | given | cleaned | gen-only | wall | notes |
|---|---|---|---|---|---|---|
| blend-champion (`~/outputs/runs/blend-champion-m1`), tries 1-6 | b893b74 / 50c44a6 | — | — | — | — | failed: CUDA out of memory in fold 7 (GPU shared with another job; the retry gate first watched the wrong device). Logs `~/logs/blend-champion-oom{1,2}.log` |
| **blend-champion-m1**: `python -m ringtrace.run recipes/champion.json --name blend-champion-m1` | d02e04b | 0.5130 / 0.5090 / 0.5146 (**0.5122**) | 0.5645 / 0.5639 / 0.5712 (**0.5665**) | 0.9020 / 0.8996 / 0.9014 (**0.9010**) | 10.7 min (4 threads, shared GPU) | member 1 rerun: identical to `champion` (run 9). Fixed point 0.5064 / 0.5601 / 0.8921. Submission VALID against the sample, sha256 `a36833d5fb8c497ff3b6e7ea116226796ec8892ad661c17afa87bf92fb8e380e` (same as run 9) |
| **blend-champion-fmt**: champion from the merged and `ruff format`-ed tree (`~/outputs/runs/blend-champion-fmt`) | dbc304e | 0.5130 / 0.5090 / 0.5146 (**0.5122**) | 0.5645 / 0.5639 / 0.5712 (**0.5665**) | 0.9020 / 0.8996 / 0.9014 (**0.9010**) | 11.8 min (incl. 38 s feature rebuild) | formatting changed the source hash of every family, so all nine were rebuilt under new cache keys (38 s in total). `oof.npy` and `test.npy` byte-identical to `blend-champion-m1`; submission sha256 `a36833d5...` unchanged: the merges and the formatting leave the champion recipe's output unchanged |
| **blend** (`python -m ringtrace.blend recipes/blend.json`, `~/outputs/runs/blend`): members `blend-champion-m1`, `lgbm`, `catboost`, `clean-event`, `clean-l2`, weights 1/5 | 3d9edb9 | 0.5130 / 0.5091 / 0.5147 (**0.5123**) | 0.5647 / 0.5641 / 0.5714 (**0.5667**) | 0.9022 / 0.9000 / 0.9018 (**0.9013**) | 22 s (4 threads, CPU) | every member passed the alignment check. Submission VALID (260,582 rows, sample order), sha256 `c3af1418ffab08f7843b1813359ad7353ee0ac6676f4293609d3cf530a696c05`. October calls: legit 256,636; mule 789, bot 318, chain 556, sim 370, takeover 230, phishing 498, qr 517, refund 279, cashback 207, loan 99, collusion 83 (3,946 fraud). October rows changed vs each member's own decode: champion 20, lgbm 11, catboost 10, clean-event 5, clean-l2 4 (train rows: 39 / 37 / 69 / 44 / 46) |

### The members as re-decoded by the blend (fixed point + template rule, from their saved arrays)

| member | given Jul / Aug / Sep (mean) | cleaned Jul / Aug / Sep (mean) | gen-only Jul / Aug / Sep (mean) |
|---|---|---|---|
| champion (`blend-champion-m1`) | 0.5130 / 0.5090 / 0.5146 (0.5122) | 0.5645 / 0.5639 / 0.5712 (0.5665) | 0.9020 / 0.8996 / 0.9014 (0.9010) |
| lgbm | 0.5130 / 0.5092 / 0.5146 (0.5122) | 0.5646 / 0.5642 / 0.5712 (0.5667) | 0.9021 / 0.9000 / 0.9012 (0.9011) |
| catboost | 0.5129 / 0.5090 / 0.5141 (0.5120) | 0.5644 / 0.5637 / 0.5706 (0.5662) | 0.9023 / 0.8995 / 0.9008 (0.9008) |
| clean-event | 0.5131 / 0.5091 / 0.5146 (0.5123) | 0.5646 / 0.5640 / 0.5713 (0.5666) | 0.9021 / 0.8997 / 0.9017 (0.9012) |
| clean-l2 | 0.5131 / 0.5091 / 0.5146 (0.5122) | 0.5645 / 0.5642 / 0.5713 (0.5666) | 0.9019 / 0.8999 / 0.9016 (0.9012) |
| **blend** | 0.5130 / 0.5091 / 0.5147 (**0.5123**) | 0.5647 / 0.5641 / 0.5714 (**0.5667**) | 0.9022 / 0.9000 / 0.9018 (**0.9013**) |

The blend ties its best member (+0.0001 given, +0.0000 to +0.0002 cleaned): the five members agree
on all but a few dozen October rows.

## Findings (analysis of the member runs, no refits)

Each member is decoded as the run decodes it (fixed point 0.875 per month, then
`anchored_templates`) by `analysis/findings.py`. The script ran on the box with 4 CPU threads in
43 s and wrote `~/outputs/findings-analysis/findings.json`. The write-ups are in
`docs/findings/`.

| member run | commit | given Jul / Aug / Sep (mean) | cleaned Jul / Aug / Sep (mean) | gen-only Jul / Aug / Sep (mean) | wall |
|---|---|---|---|---|---|
| champion | 911b18d | 0.5130 / 0.5090 / 0.5146 (0.5122) | 0.5645 / 0.5639 / 0.5712 (0.5665) | 0.9020 / 0.8996 / 0.9014 (0.9010) | 43 s (whole script) |
| catboost | 911b18d | 0.5129 / 0.5090 / 0.5141 (0.5120) | 0.5644 / 0.5637 / 0.5706 (0.5662) | 0.9023 / 0.8995 / 0.9008 (0.9008) | (same) |
| clean-event | 911b18d | 0.5131 / 0.5091 / 0.5146 (0.5123) | 0.5646 / 0.5640 / 0.5713 (0.5666) | 0.9021 / 0.8997 / 0.9017 (0.9012) | (same) |
| clean-l2 | 911b18d | 0.5131 / 0.5091 / 0.5146 (0.5122) | 0.5645 / 0.5642 / 0.5713 (0.5666) | 0.9019 / 0.8999 / 0.9016 (0.9012) | (same) |
| lgbm | 911b18d | 0.5130 / 0.5092 / 0.5146 (0.5122) | 0.5646 / 0.5642 / 0.5712 (0.5667) | 0.9021 / 0.9000 / 0.9012 (0.9011) | (same) |

The champion row matches run 9, which checks the decoding. The members' spread on cleaned labels
(0.5662-0.5667) is below the 0.0002-0.0005 public-slice sd of a pairwise gap
(`docs/findings/public-score-noise.md`).

## Refit (more data)

Champion XGBoost cascade, seeds 0, 1, 2; decoded with the fixed point and the anchored template
rule. Details and per-class tables: `docs/findings/more-data.md`.

| run | commit | given Jul / Aug / Sep (mean) | cleaned Jul / Aug / Sep (mean) | gen-only Jul / Aug / Sep (mean) | wall | notes |
|---|---|---|---|---|---|---|
| refit-curve: one month (mean of both single-month fits), screen off | ec37c9d | 0.5130 / 0.5084 / 0.5140 (0.5118) | 0.5646 / 0.5633 / 0.5706 (0.5662) | 0.9020 / 0.8990 / 0.9007 (0.9006) | 29.5 min (whole curve job, shared GPU) | |
| refit-curve: probability mean of the two one-month models | ec37c9d | 0.5136 / 0.5088 / 0.5143 (0.5122) | 0.5652 / 0.5639 / 0.5709 (0.5667) | 0.9029 / 0.8998 / 0.9010 (0.9012) | (same job) | fold-mean analogue |
| refit-curve: two months, 300 rounds, screen off | ec37c9d | 0.5136 / 0.5093 / 0.5145 (0.5125) | 0.5653 / 0.5643 / 0.5712 (0.5669) | 0.9031 / 0.9002 / 0.9011 (0.9015) | (same job) | second month: +0.0007 |
| refit-curve: two months, 450 rounds, screen off | ec37c9d | 0.5138 / 0.5089 / 0.5147 (0.5124) | 0.5654 / 0.5640 / 0.5715 (0.5669) | 0.9032 / 0.8996 / 0.9014 (0.9014) | (same job) | |
| champion (reference, run 9) | c46528d | 0.5130 / 0.5090 / 0.5146 (0.5122) | 0.5645 / 0.5639 / 0.5712 (0.5665) | — | 12.8 min | |
| champion-refit / champion-hedge (October only, unscored) | 031e194 | — | — | — | 5.3 min | 450 rounds on all train rows; 52 / 25 argmax rows changed vs fold mean |
| clean-event-refit / clean-event-hedge | 031e194 | — | — | — | 5.0 min | 48 / 24 argmax rows changed |
| clean-l2-refit / clean-l2-hedge | 031e194 | — | — | — | 7.1 min | 61 / 38 argmax rows changed |
| refit-october, first launch | ec37c9d | — | — | — | <1 min | failed: shell quoting in the recipe loop; relaunched via `scripts/refit_october.sh` |

## Squeeze: label-free candidates on top of the blend

`python -m ringtrace.squeeze lomo | select | audit | final` (commit c237d98), CPU only on the box,
6-8 threads per job. The base is the blend as `ringtrace.blend` decodes it, recomputed here at
0.5123 / 0.5667 / 0.9013 (given / cleaned / gen-only, as recorded). Changes are against the
blend, Jul / Aug / Sep. The bootstrap is the paired event bootstrap, 2,000 draws, 95 %.
October-like ran on the champion's cells (`scripts/squeeze_gate.sh`), because the other members'
cells did not exist yet. Box outputs: `~/outputs/squeeze-out/squeeze/{lomo,select,audit}.json`,
gate reports in `~/outputs/squeeze-out/octlike/`, logs in `~/logs/squeeze-*.log`.

| candidate | given | cleaned | gen-only | bootstrap, given | October-like given per seed (lower bound) | rows, train / Oct | wall | verdict |
|---|---|---|---|---|---|---|---|---|
| event types, before the rule | +0.0003 / +0.0001 / +0.0000 | +0.0005 / +0.0001 / +0.0000 | +0.0007 / +0.0002 / +0.0000 | +0.0002 [+0.0000, +0.0004] | +0.0001 / +0.0001 / +0.0000 (+0.0000) FAIL | 7 / 0 | 20 min (whole lomo job) | tie, rejected |
| event types, after the rule | -0.0035 / -0.0040 / -0.0037 | -0.0037 / -0.0043 / -0.0054 | -0.0033 / -0.0053 / -0.0063 | -0.0037 [-0.0057, -0.0017] | -0.0052 / -0.0008 / -0.0050 (-0.0089) FAIL | 639 / 132 | (same) | rejected |
| anchor rescue (blind spot) | +0.00005 / -0.00002 / +0.00009 | +0.00004 / -0.00002 / +0.00016 | -0.00001 / -0.00005 / +0.00015 | +0.00004 [-0.00003, +0.00012] | +0.0000 / +0.0001 / -0.0000 (-0.0001) FAIL | 14 / 1 | (same) | rejected |
| greedy selection, probability (nested) | +0.00011 / +0.00005 / -0.00009 | +0.00011 / +0.00004 / -0.00004 | +0.00017 / +0.00003 / -0.00003 | +0.00002 [-0.00008, +0.00015] | not run (no member cells) | 18 / 5 | 26 min (whole select job) | rejected |
| greedy selection, rank (nested) | +0.00006 / +0.00006 / -0.00008 | +0.00006 / +0.00006 / -0.00003 | +0.00034 / +0.00012 / +0.00001 | +0.00001 [-0.00012, +0.00015] | not run | 16 / 2 | (same) | rejected |
| equal-weight rank blend | +0.00017 / +0.00008 / +0.00021 | +0.00016 / +0.00007 / +0.00020 | +0.00024 / +0.00010 / -0.00010 | +0.00015 [+0.00003, +0.00030] | not run | 8 / 2 | (same) | rejected (gen-only Sep) |
| anchored rule min_conf 0.70 or 0.75 (edge audit) | +0.0000 / +0.0000 / +0.0008 | +0.0000 / +0.0000 / +0.0009 | +0.0000 / +0.0000 / +0.0016 | +0.0003 [+0.0000, +0.0009] | not run | 24 / 0 | 12 min (audit job) | no October change |
| anchored rule min_conf 0.85 / 0.90 | +0.0000 / -0.0003 / +0.0000 | +0.0000 / -0.0003 / +0.0000 | +0.0000 / -0.0005 / +0.0000 | -0.0001 [-0.0003, +0.0000] | not run | 10 / 1 and 12 | (same) | rejected |
| event-level expected-utility decoder | — | — | — | — | — | — | — | not run |

Nothing is adopted. The adopted combination is therefore the blend itself:
- `~/outputs/runs/squeeze-final/submission.csv` is VALID against the sample, with sha256
  `c3af1418ffab08f7843b1813359ad7353ee0ac6676f4293609d3cf530a696c05`, the blend's own file;
- 0 October rows change against the blend;
- the event-types file (`~/outputs/runs/squeeze-event-types/`) has the same sha256.

Findings: `docs/findings/squeeze-*.md`.

## Fresh look: error anatomy, October profiles, one rule

Analysis of the saved blend run (`blend`), no refits. Box CPU, 16 threads. Findings:
`docs/findings/fresh-*.md`.

| run | commit | result | wall |
|---|---|---|---|
| fresh-anatomy (`python analysis/fresh.py anatomy`) | 2f71e67 | blend given 0.5130 / 0.5091 / 0.5147, cleaned 0.5647 / 0.5641 / 0.5714; loss pockets (cleaned): ordinary 0.3874, generator rows called legit 0.0025, wrong type 0.0206, legit called fraud 0.0018 | about 4 min |
| fresh-october (`python analysis/fresh.py october`) | 2f71e67 | no class departs from out-of-fold calls in a way the model gets wrong; sim-like evidence at 22:00-24:00 is called sim_swap 22 / 24 in October (98.2 % at 00:00-05:30) | about 4 min |
| fresh-anatomy, first launch | bde7602 | failed: new files were not shipped (untracked); relaunched after commit | <1 min |
| fresh-rule-bq: blend + `blank_merchant_qr` vs blend, leave-one-month-out | 07333a3 | given +0.00000 / +0.00000 / +0.00004 (mean +0.00001, 95 % [+0.00000, +0.00004]); cleaned +0.00000 / +0.00000 / +0.00003 ([+0.00000, +0.00004]); gen-only +0.00000 / +0.00000 / +0.00004; 1 train row and 20 October rows changed | about 10 min |
| fresh-octlike-bq: champion + `blank_merchant_qr` vs champion, 9 cells | ffbdfb7 | +0.0000 in every seed on given, cleaned and gen-only; lower bound +0.0000; 0 rows changed; **FAIL** (not positive) | about 10 min |
| fresh-octlike-bq, first two launches | ffbdfb7 | failed: missing frames in the shared cache, then cells from an older frame build (251,860 vs 252,462 September rows); scored on current-frame champion cells | <2 min |
| fresh-blend-blankqr (`python -m ringtrace.blend recipes/blend-blankqr.json`), upload candidate | 92735ae | given 0.5130 / 0.5091 / 0.5148 (0.5123); cleaned 0.5647 / 0.5641 / 0.5714 (0.5667); gen-only 0.9022 / 0.9000 / 0.9019 (0.9013); valid; sha256 `e7c594672f0921e6def080f0c06347144cd848282b10c9fe6e384416eed043d0`; against the blend's file: 20 October rows legit -> fraud (13 qr_swap, 7 refund_abuse) | about 5 min |

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

| fresh-blend-col7-phishing (`python -m ringtrace.blend recipes/blend-col7-phishing.json`), upload candidate: col7 template -> phishing_collect | 306885e | given 0.5102 / 0.5084 / 0.5143 (0.5110, vs blend-blankqr -0.0028 / -0.0007 / -0.0005); cleaned 0.5615 / 0.5627 / 0.5697 (0.5646; -0.0032 / -0.0014 / -0.0017); gen-only 0.9004 / 0.9019 / 0.9033 (0.9019; -0.0018 / +0.0019 / +0.0014); valid; sha256 `91d1e767f0e8fe63643251fe7fd72579874a6772c9faa941a6cc1cca1b88fcd8`; 94 October rows fake_cashback -> phishing_collect | about 6 min |
| fresh-blend-mf23-bot (`python -m ringtrace.blend recipes/blend-mf23-bot.json`), upload candidate: mf2+mf3 template -> velocity_bot | 306885e | given 0.4956 / 0.4924 / 0.4949 (0.4943; -0.0174 / -0.0167 / -0.0199); cleaned 0.5453 / 0.5447 / 0.5483 (0.5461; -0.0194 / -0.0194 / -0.0231); gen-only 0.8678 / 0.8629 / 0.8586 (0.8631; -0.0344 / -0.0371 / -0.0433); valid; sha256 `0cbf7e1b535567b3d6bbc32ef735cf70a7c7f45bc4e4310f69d82100ff635448`; 422 October rows mule_account -> velocity_bot | about 6 min |

## Member: refit, October refit blends

Every member is refitted on all train rows (450 rounds or CatBoost iterations, the screen
cross-fitted over the three months, seeds 0, 1, 2), and each blend is decoded like `recipes/blend.json`.
The train scores are the members' unchanged out-of-fold files, so they match the five-way blend and do
not validate the refit. The change tables compare against the five-way blend (`runs/blend`, sha256
c3af1418...).

| run | commit | given Jul / Aug / Sep (mean) | cleaned (mean) | gen-only (mean) | wall | notes |
|---|---|---|---|---|---|---|
| lgbm-refit / lgbm-hedge | 18b268b | — | — | — | 7.2 min | 450 rounds; 50 / 19 argmax rows and 14 / 7 decoded calls changed vs its fold mean |
| catboost-refit / catboost-hedge | 18b268b | — | — | — | 1.8 min | 450 iterations, gpu_ram_part 0.45; 24 / 11 argmax rows and 9 / 6 decoded calls changed |
| refit-blend (five -refit members) | 18b268b | 0.5130 / 0.5091 / 0.5147 (0.5123) | 0.5667 | 0.9013 | <2 min | VALID, sha256 `7557112ae26186658e6c9640f5181ddc2a86483c30fdb562b972f6880ec349a4`; 11 October calls changed vs blend (10 legit -> fraud: qr 4, phishing 2, refund 2, cashback 1, collusion 1; 1 cashback -> refund) |
| hedge-blend (five -hedge members) | 18b268b | 0.5130 / 0.5091 / 0.5147 (0.5123) | 0.5667 | 0.9013 | <2 min | VALID, sha256 `28d9544465b36b028cb6c91069b4af26dc361d6978d5a4401e41f0d6ec7507ad`; 6 October calls changed vs blend (5 legit -> fraud, one each: cashback, phishing, qr, collusion, refund; 1 cashback -> refund) |

## October-like gate

Box (one RTX 5060 Ti, 24 CPU threads in total), commit 181574a for the fits and the scoring.
`python -m ringtrace.octlike fit <member>` on frames s13-s15 (the same frame files as the
champion's cells above), then `python -m ringtrace.octlike score <base> <challenger>` (1,000 draws,
95 % event-bootstrap bounds). The champion's nine cells are the ones above, not refitted. The
event-cleaned members clean only the July + August labels (September never votes). Before 0de8df6
the fit trained them on the given labels without warning. Blends average the members'
probabilities per cell and decode as the champion does (fixed point 0.875, then anchored
templates). Details: `docs/findings/october-like-gate.md`.

Member cells, September macro-F1, given / cleaned / gen-only:

| member | frame | r0 | r1 | r2 | wall per cell |
|---|---|---|---|---|---|
| lgbm | s13 | 0.5237 / 0.5823 / 0.9060 | 0.5237 / 0.5823 / 0.9060 | 0.5238 / 0.5824 / 0.9063 | 180-199 s |
| lgbm | s14 | 0.5091 / 0.5664 / 0.8900 | 0.5089 / 0.5662 / 0.8901 | 0.5092 / 0.5666 / 0.8903 | 172-187 s |
| lgbm | s15 | 0.5139 / 0.5712 / 0.8982 | 0.5140 / 0.5712 / 0.8982 | 0.5141 / 0.5713 / 0.8984 | 165-220 s |
| catboost | s13 | 0.5236 / 0.5820 / 0.9053 | 0.5233 / 0.5817 / 0.9048 | 0.5239 / 0.5823 / 0.9056 | 47-55 s |
| catboost | s14 | 0.5083 / 0.5655 / 0.8881 | 0.5086 / 0.5658 / 0.8886 | 0.5086 / 0.5657 / 0.8884 | 47-72 s |
| catboost | s15 | 0.5132 / 0.5703 / 0.8979 | 0.5131 / 0.5702 / 0.8978 | 0.5130 / 0.5703 / 0.8980 | 47-48 s |
| clean-event | s13 | 0.5241 / 0.5823 / 0.9058 | 0.5242 / 0.5826 / 0.9060 | 0.5241 / 0.5826 / 0.9062 | 148-155 s |
| clean-event | s14 | 0.5092 / 0.5667 / 0.8901 | 0.5092 / 0.5664 / 0.8897 | 0.5095 / 0.5670 / 0.8902 | 95-102 s |
| clean-event | s15 | 0.5135 / 0.5706 / 0.8985 | 0.5136 / 0.5706 / 0.8986 | 0.5135 / 0.5706 / 0.8983 | 94-104 s |
| clean-l2 | s13 | 0.5240 / 0.5824 / 0.9059 | 0.5239 / 0.5823 / 0.9055 | 0.5239 / 0.5827 / 0.9062 | 94-98 s |
| clean-l2 | s14 | 0.5096 / 0.5670 / 0.8901 | 0.5092 / 0.5666 / 0.8900 | 0.5095 / 0.5670 / 0.8902 | 140-151 s |
| clean-l2 | s15 | 0.5135 / 0.5707 / 0.8986 | 0.5135 / 0.5708 / 0.8988 | 0.5136 / 0.5708 / 0.8989 | 94-97 s |

Two fit jobs shared the GPU, 12 threads each: lgbm then clean-event (45 min), and catboost
(`gpu_ram_part` 0.45) then clean-l2 (25 min). A warm-up job built the frames' feature caches first
(about 3 min).

Gate: the change against the base per thinning seed (mean of r0-r2), and the averaged 95 % lower
bound (LB). Base mean over the 9 cells: given 0.5155, cleaned 0.5731, gen-only 0.8980.

| challenger vs base | given s13 / s14 / s15, LB | cleaned s13 / s14 / s15, LB | gen-only s13 / s14 / s15, LB | verdict |
|---|---|---|---|---|
| lgbm vs champion | -0.0003 / -0.0000 / +0.0005, -0.0007 | -0.0001 / +0.0001 / +0.0005, -0.0006 | +0.0004 / +0.0005 / -0.0003, -0.0007 | FAIL |
| catboost vs champion | -0.0004 / -0.0006 / -0.0004, -0.0012 | -0.0004 / -0.0006 / -0.0005, -0.0014 | -0.0005 / -0.0013 / -0.0006, -0.0019 | FAIL |
| clean-event vs champion | +0.0001 / +0.0002 / +0.0001, -0.0002 | +0.0001 / +0.0004 / -0.0001, -0.0003 | +0.0003 / +0.0004 / -0.0000, -0.0004 | FAIL |
| clean-l2 vs champion | -0.0001 / +0.0004 / +0.0001, -0.0003 | +0.0000 / +0.0006 / +0.0001, -0.0002 | +0.0002 / +0.0005 / +0.0002, -0.0004 | FAIL |
| five-way blend vs champion | +0.0001 / +0.0004 / +0.0002, -0.0001 | +0.0003 / +0.0005 / +0.0002, -0.0000 | +0.0004 / +0.0003 / +0.0002, -0.0002 | FAIL (lower bound only) |
| four-way blend (no CatBoost) vs five-way | +0.0000 / -0.0001 / -0.0001, -0.0003 | -0.0001 / -0.0003 / -0.0001, -0.0005 | -0.0001 / +0.0000 / +0.0001, -0.0003 | FAIL |

Scoring took 25 min for the six comparisons at 8 threads. Gate files:
`~/outputs/gate/octlike/gate_<challenger>_vs_<base>.json`.

Four-way against five-way on leave-one-month-out (`python -m ringtrace.gate lomo`, the members'
`oof.npy`, 1,000 draws per month, 95 % event-bootstrap bounds). Laptop, commit 15c0fe2, about 1 min.
Each cell is five-way -> four-way, change [95 % bounds]:

| target | Jul | Aug | Sep |
|---|---|---|---|
| given | 0.5130 -> 0.5131, +0.0001 [-0.0000, +0.0003] | 0.5091 -> 0.5092, +0.0001 [-0.0001, +0.0002] | 0.5147 -> 0.5146, -0.0001 [-0.0003, -0.0000] |
| cleaned | 0.5647 -> 0.5647, +0.0000 [-0.0002, +0.0002] | 0.5641 -> 0.5642, +0.0001 [-0.0001, +0.0002] | 0.5714 -> 0.5713, -0.0001 [-0.0003, +0.0000] |
| gen-only | 0.9022 -> 0.9023, +0.0001 [-0.0003, +0.0003] | 0.9000 -> 0.9001, +0.0001 [-0.0001, +0.0003] | 0.9018 -> 0.9017, -0.0001 [-0.0004, +0.0001] |

September is negative on all three targets, so the four-way fails the leave-one-month-out bar as
well.

October files, built on the laptop from the box runs' arrays with `python -m ringtrace.blend`. Every
file is VALID against the sample submission:

| file | sha256 | October calls changed | notes |
|---|---|---|---|
| `runs/blend` (five-way, rebuilt as a check) | c3af1418ffab08f7843b1813359ad7353ee0ac6676f4293609d3cf530a696c05 | 0 | byte-identical to the box build |
| `runs/blend4` (`recipes/blend4.json`) | 8df9b36aa3ad07e7fd31effe4410635e18c0c2063f5c83b85637153408220f48 | vs five-way: 3 (legit->refund_abuse 2, sim_swap->account_takeover 1) | |
| `runs/blend4-bq` (four-way, then `blank_merchant_qr` after the template rule) | ca13597b0fde6a69a7f4321f6cf750e928e057e021f56cbfa8e039babf2e1491 | vs blend4: 18 (legit->qr_swap 12, legit->refund_abuse 5, legit->phishing_collect 1); vs the five-way with the same rule (e7c59467...): 2 (qr_swap->phishing_collect 1, sim_swap->account_takeover 1) | the rule is not in this repository's code yet: built from a temporary copy of `src`, with an uncommitted recipe. LOMO: given 0.5131 / 0.5092 / 0.5146, cleaned 0.5647 / 0.5642 / 0.5713, gen-only 0.9023 / 0.9000 / 0.9017 |

October calls of `blend4`: legit 256,634, mule 789, bot 318, chain 556, sim 369, takeover 231,
phishing 498, qr 517, refund 281, cashback 207, loan 99, collusion 83. `blend4-bq` changes these
by: legit 256,616, phishing 499, qr 529, refund 286.

LightGBM reproducibility: `lgbm_s13_r0` refitted at 12 and at 16 threads (`lgbm-rep12`,
`lgbm-rep16`) is byte-identical to the original cell (max abs difference 0.0, 0 argmax changes).
Each refit took about 3.5 min.

## Refit test

The October analogue on the October-like frames: champion recipe, commit 181574a. In each cell,
the champion is fitted on July alone (`champion-m7`) and on August alone (`champion-m8`) with the
screen off, since one month cannot cross-fit it. Their averaged probabilities (the fold mean) are
compared with the champion's July + August cell (the refit). Control: July + August with the screen
off (`champion-noscreen`). Details: `docs/findings/refit-test.md`.

| cells | s13 r0, r1, r2 (given/cleaned/gen-only) | s14 | s15 | wall per cell |
|---|---|---|---|---|
| champion-m7 | 0.5234/0.5817/0.9046, 0.5231/0.5813/0.9042, 0.5238/0.5821/0.9054 | 0.5084/0.5655/0.8880, 0.5079/0.5651/0.8872, 0.5086/0.5657/0.8886 | 0.5139/0.5712/0.8983, 0.5138/0.5711/0.8983, 0.5136/0.5709/0.8977 | 96-119 s |
| champion-m8 | 0.5230/0.5816/0.9052, 0.5222/0.5806/0.9045, 0.5231/0.5817/0.9055 | 0.5070/0.5643/0.8873, 0.5068/0.5641/0.8869, 0.5074/0.5648/0.8879 | 0.5125/0.5699/0.8975, 0.5125/0.5699/0.8982, 0.5130/0.5703/0.8981 | 102-117 s |
| champion-noscreen | 0.5242/0.5828/0.9062, 0.5235/0.5820/0.9048, 0.5243/0.5831/0.9065 | 0.5089/0.5661/0.8889, 0.5085/0.5659/0.8887, 0.5089/0.5663/0.8889 | 0.5135/0.5709/0.8988, 0.5135/0.5710/0.8988, 0.5132/0.5706/0.8980 | 90-167 s |

Two GPU fit jobs at 8 threads each took about 35 min for the 27 cells. Gate scoring took 18 min at
8 threads. Fold-mean base: 0.5151 given, 0.5728 cleaned, 0.8972 gen-only.

| challenger vs base | given s13 / s14 / s15, LB | cleaned s13 / s14 / s15, LB | gen-only s13 / s14 / s15, LB | verdict |
|---|---|---|---|---|
| refit (champion) vs fold mean | +0.0006 / +0.0006 / -0.0000, -0.0003 | +0.0006 / +0.0006 / -0.0000, -0.0004 | +0.0007 / +0.0012 / +0.0002, -0.0003 | FAIL |
| July + August, screen off vs fold mean | +0.0005 / +0.0003 / -0.0001, -0.0003 | +0.0008 / +0.0004 / +0.0001, -0.0001 | +0.0009 / +0.0004 / +0.0003, -0.0002 | FAIL |
| screen on vs screen off (July + August) | +0.0000 / +0.0003 / +0.0001, -0.0006 | -0.0002 / +0.0002 / -0.0001, -0.0008 | -0.0002 / +0.0008 / -0.0000, -0.0009 | FAIL (tie) |

The refit fails, so no October refit of lgbm or catboost was run, and no refit blend was built.

Failed launches (no output): the first warm-up and the first two fit jobs stopped at once on shell
quoting. The first refit-cell fits stopped on `--tag -m7`, which argparse reads as an option; they
were relaunched with `--tag=-m7`.

## Final system

`python -m ringtrace.final` (recipe format, one-command build and harness; `recipes/final.json`,
candidates in `recipes/cand/`). The harness scores any recipe from saved `oof.npy` / `test.npy` and
October-like cells, no refits: per-month change on given / cleaned / gen-only (Jul / Aug / Sep), the
paired event bootstrap of the mean change (2,000 draws, 95 %, `ringtrace.pairboot`), the nine
October-like cells (per-seed mean change, mean 95 % lower bound; 1,000 draws), and rows changed
(train / October). Box CPU, 8 threads. Cells: `~/outputs/gate/octlike/` (current frames).

### Builds

| run | commit | result | wall |
|---|---|---|---|
| final-t1 build `recipes/cand/base.json` | 05cb804 | sha256 `c3af1418...6c05`, byte-identical to the blend's file; 0.5123 / 0.5667 / 0.9013 | 4 s from arrays |
| final-t1 build `recipes/cand/blankqr.json` | 05cb804 | sha256 `e7c59467...43d0`, byte-identical to the blend + blank-QR file; 20 October rows changed (cap 40), 1 train row | 3 s |
| final-t1 table | 05cb804 | stopped after 50 min in the October-like and bootstrap stage (one core); replaced by the shared-count bootstrap | 50 min |
| final-t2 table, 17 challengers | cdd53de | October-like 663 s; base bootstrap counts about 15 min once, then 2-3 s per challenger; stopped at the hedge recipe (the champion's hedge run has no `oof.npy`, fixed with `variant_run`) | 27 min |

### Components against the blend (final-t2, base = five-way blend, fold mean, anchored rule)

Change per month Jul / Aug / Sep, [95 % bootstrap interval of the mean]; October-like per seed
s13 / s14 / s15 (mean lower bound); rows train / October.

| candidate | given | cleaned | gen-only | October-like given | October-like cleaned | rows | verdict |
|---|---|---|---|---|---|---|---|
| blank-merchant-QR rule (base-rate) | +0.00000 / +0.00000 / +0.00004 [+0.00000, +0.00004] | +0.00000 / +0.00000 / +0.00003 [+0.00000, +0.00004] | +0.00000 / +0.00000 / +0.00004 | +0.0000 x3 (lb +0.0000) | +0.0000 x3 (lb +0.0000) | 1 / 20 | not measurable offline; enters on base rate |
| event types before the rule | +0.00033 / +0.00014 / +0.00000 [+0.00001, +0.00036] | +0.00047 / +0.00013 / +0.00000 [+0.00005, +0.00039] | +0.00071 / +0.00020 / +0.00000 | +0.0000 x3 (lb +0.0000) | +0.0000 x3 (lb +0.0000) | 7 / 0 | tie (0 October rows), rejected |
| anchor rescue | +0.00005 / -0.00002 / +0.00009 [-0.00003, +0.00012] | +0.00004 / -0.00002 / +0.00016 | -0.00001 / -0.00005 / +0.00015 | -0.00001 / +0.00014 / -0.00004 (lb -0.00009) | +0.00008 / +0.00023 / -0.00005 (lb -0.00009) | 14 / 1 | rejected |
| four-way (no CatBoost) | +0.00013 / +0.00008 / -0.00014 [-0.00006, +0.00011] | +0.00001 / +0.00007 / -0.00009 | +0.00005 / +0.00010 / -0.00011 | +0.00003 / -0.00010 / -0.00013 (lb -0.00029) | -0.00010 / -0.00027 / -0.00012 (lb -0.00050) | 20 / 3 | rejected; a tie with the five-way in expectation |
| fraud_scale 0.85 | +0.00000 / +0.00000 / +0.00003 [+0.00000, +0.00003] | +0.00000 / +0.00000 / +0.00003 | +0.00000 / +0.00000 / -0.00003 | +0.00000 / +0.00003 / +0.00000 | +0.00000 / +0.00003 / +0.00000 | 1 / 0 | tie, rejected |
| fraud_scale 0.90 | -0.00004 / +0.00000 / -0.00004 [-0.00007, +0.00000] | -0.00003 / +0.00000 / -0.00004 | -0.00004 / +0.00000 / +0.00004 | +0.00000 / +0.00001 / +0.00000 | +0.00000 / +0.00001 / +0.00000 | 2 / 1 | rejected |
| nested greedy selection (prob) | +0.00011 / +0.00005 / -0.00009 [-0.00008, +0.00014] | +0.00011 / +0.00004 / -0.00004 | +0.00017 / +0.00003 / -0.00003 | +0.00002 / -0.00001 / -0.00011 (lb -0.00020) | -0.00005 / -0.00010 / -0.00010 (lb -0.00033) | 18 / 5 | rejected |

Nested weights (copies of champion, lgbm, catboost, clean-event, clean-l2): held Jul 1,1,1,0,0;
Aug 0,1,0,1,0; Sep 0,1,0,2,0; October 0,1,0,1,0. Anchor rescue and nested selection reproduce the
earlier squeeze measurements to the fifth decimal.

### final-t3 (commit ea8cf74, 24 min; October-like 561 s): variants, second seed, counter-free, TabM

Same base and columns as above. October-like is n/a where a recipe's members lack cells (the
second-seed and counter-free members have none; the TabM cells reached the box after the job
started).

| candidate | given | cleaned | gen-only | October-like | rows | verdict |
|---|---|---|---|---|---|---|
| hedge (all five members) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 (cells are fold models) | 0 / 6 | train-identical by construction; decided by the refit test, which fails the bar |
| refit (all five members) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 / 11 | as hedge |
| second seed set (seeds 3-5, all five) | +0.00016 / +0.00011 / -0.00001 [-0.00000, +0.00019] | -0.00007 / +0.00011 / -0.00001 [-0.00016, +0.00015] | -0.00003 / +0.00015 / +0.00001 | n/a | 10 / 2 | tie: seed noise is about 1e-4 per month and 2 October calls |
| counter-free (no device-counter family, plain template rule) | -0.03013 / -0.03296 / -0.02721 [-0.03259, -0.02782] | -0.03120 / -0.03251 / -0.02761 | -0.00667 / -0.01163 / -0.00497 | n/a | 1,857 / 330 | fallback only |
| TabM full member, 1/6 | +0.00058 / +0.00009 / +0.00024 [+0.00013, +0.00052] | +0.00054 / +0.00008 / +0.00022 | +0.00078 / +0.00008 / -0.00014 | n/a | 20 / 2 | fails gen-only Sep |
| TabM full member, 1/3 | +0.00055 / +0.00015 / +0.00021 [+0.00013, +0.00051] | +0.00067 / +0.00014 / +0.00027 | +0.00092 / +0.00013 / -0.00010 | n/a | 30 / 5 | fails gen-only Sep |
| TabM full member, 1/2 | +0.00050 / +0.00016 / +0.00027 [+0.00011, +0.00055] | +0.00051 / +0.00018 / +0.00036 | +0.00065 / +0.00005 / +0.00005 | n/a | 42 / 5 | LOMO non-negative; October-like pending |
| TabM fraud mass 1/3 (types from the trees) | +0.00073 / +0.00024 / +0.00027 [+0.00021, +0.00067] | +0.00068 / +0.00021 / +0.00037 [+0.00022, +0.00066] | +0.00089 / +0.00018 / +0.00011 | n/a | 39 / 10 | LOMO passes; October-like pending |
| TabM fraud mass 1/2 | +0.00073 / +0.00025 / +0.00029 [+0.00020, +0.00069] | +0.00068 / +0.00025 / +0.00043 [+0.00023, +0.00072] | +0.00089 / +0.00001 / +0.00004 | n/a | 52 / 17 | LOMO passes; October-like pending |
| TabM fraud mass 2/3 | +0.00059 / +0.00020 / +0.00039 [+0.00016, +0.00066] | +0.00053 / +0.00020 / +0.00047 | +0.00042 / -0.00013 / +0.00007 | n/a | 64 / 19 | fails gen-only Aug |
| TabM fraud mass 4/5 | +0.00066 / -0.00002 / +0.00035 | +0.00059 / -0.00003 / +0.00042 | +0.00050 / -0.00070 / -0.00011 | n/a | 78 / 17 | fails |
| TabM fraud mass 1 | +0.00077 / -0.00002 / +0.00026 | +0.00069 / +0.00001 / +0.00034 | +0.00062 / -0.00068 / -0.00034 | n/a | 88 / 16 | fails |

- 2026-11-14 00:40 IST: first `recipes/final.json` = the five-way blend + `blank_merchant_qr`

### final-t5 (commit 8d4b5c8, 21 min; October-like 482 s): the nine-cell TabM gate

Base `recipes/final.json` (five-way blend + blank-QR rule); every challenger carries the same rule.
All nine TabM cells (s13-15 x r0-2). October-like lower bound here is the mean of the nine per-cell
2.5 % points (the gate's earlier convention).

| candidate | given | cleaned | gen-only | October-like given (lb) | October-like cleaned (lb) | rows | verdict |
|---|---|---|---|---|---|---|---|
| TabM fraud mass 1/2 | +0.00073 / +0.00025 / +0.00025 [+0.00019, +0.00067] | +0.00068 / +0.00025 / +0.00039 [+0.00022, +0.00070] | +0.00089 / +0.000014 / **-0.0000022** [-0.00002, +0.00068] | +0.00023 / +0.00033 / +0.00038 (-0.00004) | +0.00030 / +0.00041 / +0.00032 (-0.00004) | 51 / 7 | **fails**: gen-only September is negative |
| TabM fraud mass 1/3 | +0.00073 / +0.00024 / +0.00023 [+0.00020, +0.00066] | +0.00068 / +0.00021 / +0.00033 [+0.00021, +0.00065] | +0.00089 / +0.00018 / +0.00007 [+0.00010, +0.00074] | +0.00017 / +0.00024 / +0.00038 (-0.00004) | +0.00025 / +0.00032 / +0.00033 (-0.00003) | 38 / 5 | every month and seed positive; see the pooled bound (final-t6) |
| TabM full member 1/2 | +0.00050 / +0.00016 / +0.00023 [+0.00010, +0.00054] | +0.00051 / +0.00018 / +0.00033 | +0.00065 / +0.00005 / +0.000004 | +0.00013 / +0.00031 / +0.00049 (-0.00004) | +0.00021 / +0.00039 / +0.00043 (-0.00005) | 41 / 7 | dominated by the fraud-only 1/3 arm; type switches in October |

October changes against the final recipe: fraud mass 1/3 adds 5 fraud calls (legit -> phishing 2,
sim_swap 1, loan 1, collusion 1); 1/2 adds 7; the full member also switches 3 fraud types.

- 2026-11-14 01:30 IST: the October-like bound is now pooled (decision 17); final-t6 re-gates.

### final-t6 (commit 147b5a3, 22 + 21 min): pooled bound, seed-averaged TabM, Mitra

Against the blend + blank-QR recipe; October-like lb = pooled bound of the nine-cell mean
(decision 17), cell lb = the earlier per-cell mean.

| candidate | given | cleaned | gen-only | October-like given | October-like cleaned | rows | verdict |
|---|---|---|---|---|---|---|---|
| TabM fraud mass 1/3 | +0.00073 / +0.00024 / +0.00023 [+0.00020, +0.00066] | +0.00068 / +0.00021 / +0.00033 [+0.00021, +0.00065] | +0.00089 / +0.00018 / +0.00007 [+0.00010, +0.00074] | +0.00017 / +0.00024 / +0.00038 (lb +0.00009, cell lb -0.00004) | +0.00025 / +0.00032 / +0.00033 (lb +0.00011, cell lb -0.00003) | 38 / 5 | **PASS, adopted** |
| TabM fraud mass 1/2 | +0.00073 / +0.00025 / +0.00025 [+0.00019, +0.00067] | +0.00068 / +0.00025 / +0.00039 [+0.00022, +0.00070] | +0.00089 / +0.00001 / -0.000002 | +0.00023 / +0.00033 / +0.00038 (lb +0.00012) | +0.00030 / +0.00041 / +0.00032 (lb +0.00012) | 51 / 7 | fails gen-only Sep by 2e-6; kept as an upload variant |

Against the TabM 1/3 recipe:

| candidate | change, every month and target | rows | verdict |
|---|---|---|---|
| 6-seed TabM fraud mass (seeds 0-2 + 6-8) | 0 | 0 / 0 | identical calls: a tie, the 3-seed mass stays (simpler) |
| 9-seed TabM fraud mass (+ seeds 3-5) | 0 | 0 / 0 | identical calls; the mixed legit probability moves at most 0.041 (372 train rows by more than 0.01, 46 rows in the 0.3-0.6 band) |
| Mitra in the fraud mass (TabM 1/6 + Mitra 1/6) | given -0.00009 / -0.00015 / -0.00018 [-0.00030, -0.00002]; cleaned -0.00008 / -0.00013 / -0.00029 [-0.00032, -0.00004]; gen -0.00006 / -0.00011 / -0.00009 | 21 / 3 | **rejected** (no Mitra cells either) |

Other components judged from their own branches: event completion (`feat/rescue`) changes no October
row of the TabM file, nothing to build.

- 2026-11-14 02:15 IST: `recipes/final.json` = five-way blend types + TabM (3 seeds) fraud mass at
  1/3 (each tree 2/15) + anchored rule + blank-QR rule. Passes the bar against the previous recipe.

### final-b2, final-repro, final-t7 (commit 798d511 / b24b0ea): October files, reproducibility, variants

Files in `~/outputs/final/build2/` on the box, copied to the local candidates folder. All valid
against the sample submission. Changes are against `ga-blend.csv` (the blend) and against the
TabM upload file (`blend-tabmF-bq.csv`).

| file | sha256 | blank-QR rows | vs blend | vs TabM upload file |
|---|---|---|---|---|
| `final.csv` | `39cb4952d979ea3f1f7e2fecec552afd8b17925b3aa9f63da06d143415144db3` | 15 | 25: legit -> qr 13, refund 7, phishing 2, loan 1, collusion 1, sim 1 | 2: phishing -> legit |
| `final-hedge.csv` | `2f38ee4d051b7c48c684fee4ef87ab5a44b2377d52d2eda84f12638095e01f1a` | 10 | 28 | 8 |
| `final-s345.csv` | `0a53fd14c9623eba0df43a9293766e3d827b5ca768102c25f16f33720d6abddd` | 9 | 25 | 3 |
| `final-half.csv` | `8a46142660cde2af1576736289f822c1b5c6226204445bce88d4dc311fb926eb` | 10 | 27 | 0 (the same calls) |
| `final-noctr.csv` (fallback) | `2dfe144b656b95e2d08d0fdf14ac19cece5c9b3c6b6eb852ca697df4fc96ea3e` | | | |

Calls per class, `final.csv`: legit 256,611; mule 789; bot 318; chain 556; sim 371; takeover 230;
phishing 500; qr 530; refund 286; cashback 207; loan 100; collusion 84.

Reproducibility (final-repro): `git archive` of b24b0ea into an empty directory, every file built
twice (`~/outputs/final/repro1`, `repro2`): all five sha256 identical to each other and to
final-b2. The October files are byte-reproducible from the saved member arrays. The members
themselves: the XGBoost champion is byte-reproducible from a clean checkout (run 11), and so is
TabM (`feat/tabm`, byte-identical rerun). CatBoost on the GPU is not, so its saved arrays are the
reference copy. The four-way blend without CatBoost is a tie in expectation but loses on every
October-like seed on cleaned, so the five-way stays.

Upload variants against `final.json` (final-t7, 20 min):

| variant | given | cleaned | gen-only | October-like given / cleaned (lb) | rows | verdict |
|---|---|---|---|---|---|---|
| final-hedge | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 | 0 / 6 | tie |
| final-s345 | -0.00001 / -0.00004 / +0.00003 [-0.00005, +0.00003] | +0.00003 / -0.00004 / -0.00006 [-0.00008, +0.00003] | +0.00004 / -0.00005 / -0.00020 | n/a | 6 / 1 | tie |
| final-half | +0.00000 / +0.00001 / +0.00002 [-0.00006, +0.00008] | +0.00000 / +0.00004 / +0.00006 [-0.00004, +0.00011] | +0.00000 / -0.00017 / -0.00007 | +0.00006 / +0.00010 / +0.00000 (-0.00003); +0.00005 / +0.00009 / -0.00001 (-0.00003) | 13 / 2 | tie |

- 2026-11-14 03:00 IST: final recipe unchanged; variants `final-hedge`, `final-s345`,
  `final-half` built for the upload slots; reproducibility confirmed.

### final-t8 (commit fca45b9, 13 min): in-context members in the fraud mass of the final recipe

Against `final.json`; no in-context cells, so October-like is n/a and none could pass.

| candidate | given | cleaned | gen-only | rows | verdict |
|---|---|---|---|---|---|
| TabPFN v2 sharing the fraud mass (TabM 1/6 + TabPFN 1/6) | -0.00018 / -0.00014 / -0.00006 [-0.00025, -0.00001] | -0.00016 / -0.00012 / -0.00009 [-0.00025, -0.00002] | -0.00018 / -0.00007 / -0.00005 | 19 / 2 | rejected |
| TabPFN v2 added (TabM 1/3, TabPFN 1/15) | -0.00002 / -0.00001 / +0.00005 [-0.00004, +0.00005] | -0.00001 / -0.00001 / +0.00004 | +0.00001 / +0.00003 / +0.00008 | 6 / 1 | rejected (Jul, Aug negative) |
| TabDPT sharing the fraud mass (1/6 + 1/6) | -0.00005 / -0.00012 / -0.00018 [-0.00027, -0.00000] | -0.00005 / -0.00011 / -0.00029 [-0.00030, -0.00003] | -0.00002 / -0.00010 / -0.00009 | 20 / 3 | rejected |

Also judged from its own branch: the TabM configuration sweep (`feat/tabm2`) ties the base within
±0.00005 in the fraud-only role, so nothing to add. TabPFN-3 and LimiX were not run on their branch.

### final-t9 (commit b50cc5d, 13 min): SWA TabM (`feat/nntech`) in the fraud mass

| candidate | given | cleaned | gen-only | rows | verdict |
|---|---|---|---|---|---|
| SWA TabM replacing TabM (1/3) | -0.00004 / -0.00001 / +0.00004 [-0.00004, +0.00003] | -0.00003 / -0.00001 / +0.00003 | -0.00004 / -0.00004 / -0.00004 | 3 / 0 | rejected |
| SWA TabM sharing the fraud mass (1/6 + 1/6) | -0.00004 / 0 / 0 | -0.00003 / 0 / 0 | -0.00004 / 0 / 0 | 1 / 0 | rejected (0 October rows either way) |

The other techniques on `feat/nntech` (EMA, QR augmentation, distillation, mixup, noise,
pretraining, label smoothing, logit adjustment, focal loss, RealMLP) have single-seed screens
only. Every one except QR augmentation loses in some cell of its fraud-only screen. QR
augmentation never loses there, but it has no full run, so it has no arrays to gate here.

- 2026-11-14 03:20 IST: every arrived component judged; `recipes/final.json` unchanged since
  02:15 (`final.csv` sha256 `39cb4952...4db3`).
  (cap 40 October rows). No measurable component has passed the bar yet.

## Look-alike pairs (`analysis/pairs.py`; finding `docs/findings/pairs-partner-labels.md`)

Against the best file's out-of-fold calls (`blend-tabmF-bq`, from `prep.npz`), per held-out month
Jul / Aug / Sep; rows changed train / October. Wall time under 10 s per command on the laptop.

| change | given | cleaned | gen-only | rows | verdict |
|---|---|---|---|---|---|
| own-phone drain: takeover score >= 0.57 (C = 1.0, sim_swap-zone events kept) -> account_takeover (`pairs-tmplA-flip`) | +0.0007 / +0.0002 / +0.0007; event bootstrap +0.00055 [+0.00015, +0.00106] | +0.0007 / +0.0002 / +0.0006; [+0.00009, +0.00102] | +0.0012 / +0.0004 / +0.0010 | 40 / 6 | passes the train-month bar; small |
| the same, threshold nested over the other two months (no zone guard) | +0.0005 / +0.0002 / +0.0004 | +0.0006 / +0.0002 / +0.0003 | | | confirms |
| the same at threshold 0.50 | +0.0005 / -0.0006 / -0.0006 | +0.0006 / -0.0005 / -0.0010 | | 112 / - | loses |
| col7: bait other than Rs 5 -> phishing_collect | -0.0004 / +0.0001 / +0.0001 | -0.0002 / -0.0005 / -0.0006 | +0.0022 / +0.0023 / +0.0020 | 344 / 66 | rejected |
| col7 -> phishing_collect (all) | -0.0032 / -0.0013 / -0.0019 | -0.0035 / -0.0022 / -0.0036 | -0.0024 / +0.0009 / -0.0018 | 506 / 94 | rejected |
| col7: event model (AUC 0.48 to 0.57), top-scored -> phishing_collect | -0.0015 to +0.0005 | -0.0014 to +0.0002 | | | rejected |
| generic event aggregates, all six templates, nested threshold | template 2 Aug -0.0018; others +0.0003 to -0.0013 | | | | rejected |

The partner labels inside the templates are the writing routine, not noise: every own-phone drain
event with a fraud-block phone sits in the zone of its label (10 takeover, 36 sim_swap), as do the
new-phone drains (356 / 156) and the collect rows (col7 14 / 9, col1 44 / 1). So the clean October
labels keep them, and retyping by template (the rejected `tmpl-simato`) should lose in October as
it does in train.

## Unanswered baits and operator roles (`analysis/bait.py`; finding `docs/findings/bait-operator-roles.md`)

Commit 8bdbb3d (rules in `ringtrace.baitroles`, 94a2b67). Base: the best file's out-of-fold calls
(`blend-tabmF-bq`, `prep.npz`). Per held-out month Jul / Aug / Sep; event bootstrap of the mean
change, 2,000 draws, 95 %; precision = moved rows that are phishing_collect on the cleaned labels
against the break-even (half of phishing's F1: 0.286 / 0.275 / 0.268). Laptop, 4 threads:
`validate` 4.8 min, `octlike` 8 min (nine cells), `thinned` 15 min (20 seeds), `build` 10 s.

| change | given | cleaned | gen-only | cleaned precision | rows train / Oct | October-like cells, cleaned mean (pooled lb) | fresh seeds, cleaned mean (lb) | verdict |
|---|---|---|---|---|---|---|---|---|
| `roles`: lone unanswered Rs 10, sender an operator in another month -> phishing_collect | +0.00011 / +0.00001 / +0.00010; [+0.00000, +0.00015] | +0.00009 / +0.00000 / +0.00008; [-0.00001, +0.00014] | +0.00007 / -0.00017 / -0.00001 | 4/7, 2/7, 3/5 | 19 / 4 | +0.00011 (+0.00001); s13 -0.00005 | +0.00008 (+0.00004), 14 of 20 | passes, narrowly |
| `roles-band` (Rs 1 to 10, any amount) | same as `roles` | same | same | same | 19 / 4 | | | no extra row |
| `pay`: first-time P2P PAY >= Rs 1,000, whole, not a multiple of ten | +0.00025 / +0.00007 / +0.00001 | +0.00020 / +0.00004 / -0.00002 | -0.00007 / -0.00019 / -0.00025 | 12/27, 6/19, 4/16 | 62 / 7 | -0.00001 (-0.00015) | -0.00001 (-0.00007) | fails |
| `pay-roles`: `pay` with the receiver an operator in another month | +0.00022 / +0.00018 / +0.00008; [+0.00008, +0.00026] | +0.00020 / +0.00016 / +0.00007; [+0.00007, +0.00024] | +0.00018 / +0.00019 / +0.00010 | 6/6, 5/6, 2/2 | 14 / 3 | +0.00007 (-0.000006), every cell positive | +0.00008 (+0.00005), 18 of 20 | passes the months; cell bound misses by 0.000006 |
| **`combo`**: `roles` + `pay-roles` | +0.00033 / +0.00019 / +0.00018; [+0.00012, +0.00037] | +0.00029 / +0.00017 / +0.00016; [+0.00010, +0.00033] | +0.00024 / +0.00002 / +0.00008 | 10/13, 7/13, 5/7 | 33 / 7 | +0.00018 (+0.00005); seeds +0.00003 / +0.00030 / +0.00022 | +0.00016 (+0.00011), 19 of 20 | **passes** |
| `roles-all4` (every month sees July to October, own month too) | +0.00015 / +0.00012 / +0.00014 | +0.00013 / +0.00010 / +0.00012 | +0.00003 / -0.00007 / +0.00003 | 5/8, 5/11, 4/6 | 25 / 4 (same rows) | +0.00014 (+0.00002) | | passes; same October file |
| `pay-roles-all4` | +0.00029 / +0.00018 / +0.00012 | +0.00027 / +0.00016 / +0.00011 | +0.00026 / +0.00019 / +0.00015 | 8/8, 5/6, 3/3 | 17 / 3 (same rows) | +0.00009 (+0.000002) | | passes; same October file |
| control `history`: sender sent a fraud-labelled row in another train month | -0.00090 / -0.00063 / -0.00064 | -0.00101 / -0.00065 / -0.00072 | -0.00277 / -0.00196 / -0.00222 | 6/98, 4/64, 4/68 | 230 / 29 | | | fails |

Candidates on top of `blend-tabmF-bq.csv` (all legit -> phishing_collect): `bait-combo.csv` 7 rows,
sha256 `3b25869e1cd7ac3e390dd33568033d25d3b765b23f0e1f35dbbf3f2dff5e1974` (passes);
`bait-roles.csv` 4 rows, `a6b55dd09914a2f18f098bd7644942959ddeaf574609de844ee790d54a7a8954`;
`bait-pay-roles.csv` 3 rows, `6171470da676dec2f009913120f4aa038c66dd9c6f438aa4f35ea3fe1d13f2ee`.
A-clean change (Jul / Aug / Sep): `roles` +0.00009 / +0.00001 / +0.00008, `pay` +0.00020 / +0.00004 /
-0.00001, `pay-roles` +0.00020 / +0.00016 / +0.00007, `combo` +0.00029 / +0.00017 / +0.00016.
Expected October change of `bait-combo`: about +0.0003.

## Recovery stage on legit calls (operator roles in other months): rejected

`recipes/recovery.json` = `bait-combo` plus the base-rate rule `recovery` (threshold 0.5,
`ringtrace.recovery`): a LightGBM multiclass model on the rows the calls leave legit, with the blend
probabilities, the sender's and person receiver's operator roles in other train months, and a few
row fields; train months out of fold, October from a fit on all three months. Against `bait-combo`
(`python -m ringtrace.final table recipes/bait-combo.json recipes/recovery.json`, 2,000 draws):

| target | Jul | Aug | Sep | 95 % interval |
|---|---|---|---|---|
| given | +0.00007 | -0.000001 | -0.000001 | [-0.00002, +0.00008] |
| cleaned | +0.00006 | +0.00004 | +0.00009 | [+0.00001, +0.00013] |
| gen-only | +0.00008 | +0.00001 | +0.00009 | [-0.00002, +0.00015] |

10 train rows and 3 October rows change (legit -> phishing_collect 2, legit -> qr_swap 1). The
October-like cells cannot fit the stage (no aligned labels on a thinned frame), so it is skipped
there. It fails the bar on the given target in August and September: a tie worth about one
October row. Not used.

## The mule/bot counter edge as a rule (`mule_edge`), on top of the bait rules

`recipes/bait-edge.json` = `bait-combo` plus the base-rate rule `mule_edge`: zone rows (a counter
segment on a fraud-phase phone, no template, no row anchor) that the counter puts in the mule
segment but that are called velocity_bot take mule_account. `python -m ringtrace.final build
recipes/bait-edge.json` writes the file with sha256 `b571af3e...2f9dfc`: 10 October rows
velocity_bot -> mule_account against `bait-combo`, 6 train rows. The train months share one fraud
phase, so they cannot test this edge (they lose -0.00004 / -0.00008 / -0.00012, all six rows at
train's exact edge); the October evidence is the per-split counter zones (October's mule zone 1.21
times wider, matching mule volume) and the anchored-device density in the disputed band (about 40
to 1 for mule). A judgement call with about +0.0006 expected, uploaded as a file that becomes the
final pick only if its public score beats `bait-combo`'s.

## Three October rules on top of `bait-edge`: far segments, legit top-ups, just-under first payments

`recipes/research-combo-1.json` and `recipes/research-combo-2.json` are `bait-edge` with three
base-rate rules appended, in this order, each capped at the October rows it changes:

* `far_segment`: a fraud-phase row whose counter segment is confident (`ctr_conf >= 0.9`) and
  whose fraud call is neither the segment's class, a neighbour of it in label-list order, nor a
  class the segment's zone is cross-written with (`phones.MARKED` weight >= 0.04) takes the
  segment's class. All eleven segments count.
* `legit_topup`: a row called legit after the rules and by the plain decode (the fixed point per
  month, no rules) whose most likely fraud class under the decoder's scaling (fraud columns / 0.875,
  renormalised) is qr_swap, and in combo-1 also phishing_collect, with a scaled probability of at
  least 0.125 takes that class.
* `just_under_pay`: a first-time P2P PAY of a just-under-limit multiple of ten (1,950 / 1,990 /
  4,950 / 4,990 / 9,950 / 9,990 / 24,950 / 24,990), called legit, to a receiver that holds an
  operator role (`ringtrace.baitroles`) in another month of all four (October's label-free acts
  count), becomes phishing_collect. The other four just-under amounts (1,999 / 4,999 / 9,999 /
  24,999) are `first_pay`'s; the analysis mask used all twelve, and here the four add no row.

```bash
python -m ringtrace.final build recipes/research-combo-1.json --out <dir> --against <dir>/bait-edge.csv
python -m ringtrace.final build recipes/research-combo-2.json --out <dir> --against <dir>/bait-edge.csv
```

| | `research-combo-1` | `research-combo-2` |
|---|---|---|
| top-up classes (q >= 0.125) | qr_swap, phishing_collect | qr_swap |
| caps: far / top-up / just-under | 1 / 6 / 1 | 1 / 2 / 1 |
| October rows changed vs `bait-edge` | 8: legit -> phishing_collect 5, legit -> qr_swap 2, qr_swap -> merchant_collusion 1 | 4: legit -> qr_swap 2, legit -> phishing_collect 1, qr_swap -> merchant_collusion 1 |
| train rows changed, Jul / Aug / Sep | 28 / 31 / 14 | 12 / 19 / 10 |
| given, change Jul / Aug / Sep | +0.00033 / +0.00010 / +0.00013 | +0.00024 / +0.00010 / +0.00013 |
| cleaned | +0.00032 / +0.00029 / +0.00015 | +0.00021 / +0.00026 / +0.00015 |
| gen-only | +0.00017 / +0.00007 / +0.00005 | +0.00018 / +0.00017 / +0.00011 |
| sha256 | `35e4a48466c61c1b5bc817634c1498b47c2345d845f4c4e66a36f350644d1169` | `214926a627984cbfd9a4f3b2b18ff6e2dca8b3a34e14a9b35dc3d0eb48986344` |

Rows per rule, in recipe order: `far_segment` 0 train / 1 October; `legit_topup` 65 / 6 (combo-1)
or 31 / 2 (combo-2); `just_under_pay` 8 / 1 (combo-1, where the phishing top-up takes 2 of its 10
train rows first, to the same class) or 10 / 1 (combo-2). The train changes come from the build
summary (out-of-fold calls against `bait-edge`); they include `just_under_pay`'s four-month view,
which reads October's label-free acts.

### `far_segment`: a judgement call

The train months share one fraud phase and hold no row that meets the rule, so out-of-fold
scoring cannot test it. The October case is one row.

| evidence | value |
|---|---|
| train rows Jul / Aug / Sep | 0 / 0 / 0 (given and cleaned change 0) |
| October rows | 1: qr_swap -> merchant_collusion |
| at `ctr_conf >= 0.7` and `>= 0.5` | the same single row, 0 train |
| expected October macro-F1 change, P(merchant_collusion) / P(qr_swap) = 0.95 / 0.03, 0.80 / 0.15, 0.50 / 0.50, 0.30 / 0.70 | +0.00036, +0.00028, +0.00011, -0.00001 (train class sizes); +0.00028, +0.00022, +0.00009, +0.00001 (sizes implied by the October F1 estimates) |

It is positive whenever the row is merchant_collusion with probability above about 0.3; the case
for it is that a confident last-segment phone called qr_swap fits none of the ways the collusion
zone is written.

### `legit_topup`: qr_swap measured, the phishing half a judgement call

Change in macro-F1 against `bait-edge`, in units of 0.0001, per held-out month Jul / Aug / Sep;
the interval is the paired event bootstrap (1,000 draws) of the mean change.

| evidence | qr_swap (combo-2) | qr_swap + phishing_collect (combo-1) |
|---|---|---|
| train rows Jul / Aug / Sep | 8 / 15 / 8 | 25 / 28 / 12 |
| October rows | 2 (legit -> qr_swap) | 6 (also legit -> phishing_collect 4) |
| given | +1.40 / +0.51 / +0.98 | +2.69 / +0.41 / +1.00 |
| cleaned | +1.22 / +1.16 / +1.30 | +2.66 / +1.84 / +1.26 |
| bootstrap, given | +0.96 [+0.07, +1.91] | +1.37 [+0.03, +2.64] |
| bootstrap, cleaned | +1.23 [+0.36, +2.20] | +1.92 [+0.64, +3.22] |
| precision of the moved train rows (cleaned) | 0.55 | qr_swap 0.55, phishing_collect 0.41 |

qr_swap is non-negative in every month and target at thresholds 0.10, 0.125 and 0.15. The
phishing half alone is not: given +1.3 / -0.1 / +0.0, bootstrap given +0.41 [-0.45, +1.32] and
cleaned +0.69 [-0.19, +1.59], though a leave-one-month-out threshold choice picks 0.125 in all
three folds. It enters combo-1 on October's volume: on the calibrated October class sizes the best
phishing_collect call count is 4 above the decoder's (about +0.00008 expected), and the top-up adds
4. For qr_swap the same view prefers 11 fewer calls, so its 2 October rows rest on the train
measurement alone.

### `just_under_pay`: a judgement call on the role view

Change in macro-F1 against `bait-edge`, in units of 0.0001, Jul / Aug / Sep. The rule uses the
four-month role view; the train-month view (roles from July to September only) is the control.
On October rows the two views are the same.

| evidence | four-month view (the rule) | train-month view |
|---|---|---|
| train rows Jul / Aug / Sep (phishing_collect on the cleaned labels) | 4 / 4 / 2 (3 / 4 / 1) | 4 / 3 / 1 (3 / 3 / 0) |
| October rows | 1 (legit -> phishing_collect) | 1 |
| given | +0.97 / +0.51 / +0.28 | +0.97 / +0.13 / -0.13 |
| cleaned | +0.86 / +1.40 / +0.24 | +0.86 / +1.05 / -0.14 |
| gen-only | +0.91 / +1.80 / +0.11 | +0.91 / +1.35 / -0.38 |

Eight of the ten train rows are phishing_collect on the cleaned labels, but two of the ten find
their operator only through October's acts, and without them September loses. The case for the
rule is the operator pool shared by all four months (`docs/findings/bait-operator-roles.md`); the
October row's receiver holds a role in a train month under either view.

### The October rules on the full bar: rejected at October's density

`python -m ringtrace.final table recipes/bait-edge.json recipes/proven-qr-jup.json
recipes/proven-qrphish-jup.json recipes/research-combo-1.json --cells <outputs>/lens-cells`
(2,000 event-bootstrap draws, nine October-like cells). Changes against `bait-edge`:

| recipe | given Jul / Aug / Sep [95 %] | cleaned [95 %] | October-like given per seed (lb) | October-like cleaned per seed (lb) | October rows |
|---|---|---|---|---|---|
| proven-qr-jup (`legit_topup` qr_swap + `just_under_pay`) | +0.00024 / +0.00010 / +0.00013 [+0.00005, +0.00027] | +0.00021 / +0.00026 / +0.00015 [+0.00010, +0.00032] | -0.000004 / -0.00007 / +0.000003 (-0.00010) | +0.00013 / -0.00001 / +0.000003 (-0.00005) | 3 |
| proven-qrphish-jup (top-up qr_swap + phishing_collect + `just_under_pay`) | +0.00033 / +0.00010 / +0.00013 [+0.00005, +0.00034] | +0.00032 / +0.00029 / +0.00015 [+0.00011, +0.00040] | -0.00010 / -0.00017 / -0.00006 (-0.00021) | +0.00006 / -0.00012 / -0.00008 (-0.00017) | 7 |
| research-combo-1 (adds `far_segment`) | same as above | same as above | same as above | same as above | 8 |

The month-fold gains do not survive at October's density: both combinations are negative on at least
one thinning seed and their October-like lower bounds are below zero. The top-up cut (scaled
probability >= 0.125) is calibrated on full-density months. None of these recipes is uploaded;
`bait-edge` stays the final file. `far_segment` changes no train or cell rows, so it cannot be judged.

## Pair templates (`research/boundary-events`, commit `94618d6`): no call changes

`pair_templates` reads each template event's fraud-phase phones through the refined counter's
posterior restricted to the template's two classes (`docs/findings/pair-templates.md`). Against
`counter-news`, 24 recipes (`recipes/research/pt-*.json`: pair share 0.6 to 0.9, pair mass 0.5 or
0.2, all templates or one) change 0 train rows and 0 October rows, so every month-fold and
October-like number equals the base. Of 354 train and 62 October phones inside template events,
350 and 61 already carry a confident segment of the pair. Rejected; `counter-news` stays the file.

## October border and event rules on top of `counter-news` (`feat/october-events`)

Each file is `counter-news` plus the base-rate rules named, built with
`python -m ringtrace.final build recipes/stack/<name>.json --out <dir>` from the same six member
runs. Public scores are read off the leaderboard. Findings: `docs/findings/october-events.md`,
`docs/findings/october-borders.md`.

| file | rules added | October rows against `counter-news` | public | sha256 |
|---|---|---|---|---|
| `counter-news` | - | - | 0.5516 | `b8c3183ab76d9c0730cd8d60dba60c36f8647633937b2a639a560accb0967799` |
| `stack-burst-b8` | `fresh_bursts` (20 rows, 50 h), `border_shift` (border 8, rank 0) | 29: mule_account -> velocity_bot 28, loan_app_extortion -> fake_cashback 1 | 0.5531 | `31251e77f9607c417cfa4c14375e7501908d5c6a091423e5d18a061076eb63d3` |
| `stack-burst12-b8` | `fresh_bursts` (12 rows, 50 h), `border_shift` | 56: mule_account -> velocity_bot 55, loan_app_extortion -> fake_cashback 1 | 0.5532 | `a9a3e35d0baca7fc2f92e1ddcd206b51883b3c105dffe3d706756b79226ed20b` |
| `stack-burst12-sim45` | `fresh_bursts` (12 rows, SIM 45 to 50 h) | 40: mule_account -> velocity_bot 40 | **0.5556** (best) | `c278490094706e156dc4f4d573989fafda24de896f3a7ec147fb5fc886eb2efc` |
| `stack-burst13-b8` | `fresh_bursts` (13 rows, 50 h), `border_shift` | 44: mule_account -> velocity_bot 43, 1 border row | 0.5520 | `f51fe71b96c574931d088556ab8b28ce24a66ea4d906c2ae8a69837790ec3659` |
| `stack-burst20-12-b8` | `fresh_bursts` twice (20 rows; 12 to 14 rows), `border_shift` | 41: mule_account -> velocity_bot 40, 1 border row | not uploaded | `0ff63433a1339be6225b5a0b220efa8c3e010643700746f1da143fda78027e1f` |
| `stack-burst12-b8-rings8` | as the best, plus `capped_rings` (size 8) | 96: the best's 56, refund_abuse -> merchant_collusion 40 | not uploaded | `37d327ed4848d20eb53c88a19b5faaa7659bca95e97df8481a7e2e330767475b` |
| `stack-burst13-b8-rings8` | as `stack-burst13-b8`, plus `capped_rings` | 84 | not uploaded | `27a85b3d7d6bcd478963d64483e6bd261a01f11f29cabf47a7045e4515649ff0` |
| `stack-burst` | `fresh_bursts` (20 rows, 50 h) | 28 | not uploaded | `303570004c83439811ac0dbbf0317cdd20e9d2e6bea64297ecb1a0676dc2ce57` |

- The compute box rebuilt `counter-news` and `stack-burst12-b8` from the saved member runs with the
  sha256 above (the second at commit `a0b6ed1`).
- Read of the public scores: the 28-row burst and the 12-row burst are velocity_bot, the 15-row
  burst is mule_account. `stack-burst12-sim45` removes the 15-row burst and the border row from
  `stack-burst12-b8` (16 rows) and gains 0.0024, in line with the border row being public and
  wrong (about 0.0013) and the 15-row burst being mule_account (about 0.0010). The two removals
  were not uploaded apart.
- The compute box built `stack-burst12-sim45` from the saved member runs with the sha256 above.
- `fresh_bursts` changes 199 train rows at 20 rows and 50 hours, on bursts of which 4 of 9 are
  velocity_bot: it would lose on month folds and is a base-rate rule, not a gated one.
- Border files alone (`recipes/borders/`, 21 files that change a row): none uploaded.
- `pair_templates` (24 recipes): 0 rows changed, rejected (section above).
