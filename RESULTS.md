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

## In-context tabular member: TabPFN v2 (evaluation only)

`python -m ringtrace.icl lomo --name icl-tabpfn-v2 --context 20000` (`ringtrace.icl`, licence
record in `docs/decisions.md` item 16). The champion everywhere except its candidate rows
(champion fraud probability >= tau, 5 % of train rows), where TabPFN v2 gives p(fraud) from a
context of candidate rows of the other months; types split as the champion's. Top 100 features
by |AUC - 0.5| on the context. Context limit 20,000 rows: every fraud row, legit rows sampled
(seed 0) and the odds corrected for the sampling rate; October's 23,617 fraud rows raise its
limit to 29,522. One RTX A6000 48 GB, shared with other jobs from the second fold on; 8 threads.
TabPFN-3 (gated by account and licence acceptance) and LimiX-2 (no installable package) were not
run.

- tau 2.0e-5; candidates Jul 26,977 / Aug 22,901 / Sep 28,418 / Oct 15,013; they hold 41.0 % of
  fraud-labelled train rows (41.6 / 40.2 / 41.1 % per month).
- Determinism: two runs of fold 9 on 5,000 queries (context 20,000) gave identical probabilities
  (max difference 0.0); 104 s and 101 s.

| run | commit | given Jul / Aug / Sep (mean) | cleaned Jul / Aug / Sep (mean) | gen-only Jul / Aug / Sep (mean) | wall | notes |
|---|---|---|---|---|---|---|
| icl-tabpfn-v2, first try | 9f5174a | — | — | — | ~22 min | died in fold 8 without a log (GPU memory taken by a CatBoost job on the same card; the launcher logged only the last command of a chain, fixed in 440998d) |
| icl-tabpfn-v2, second try | 440998d | — | — | — | 22 min | failed in the October fold: 23,617 fraud rows exceed the 20,000 limit; fixed in 139414a |
| **icl-tabpfn-v2** (`~/outputs/runs/icl-tabpfn-v2`) | 139414a | 0.5019 / 0.5022 / 0.5135 (**0.5059**) | 0.5530 / 0.5556 / 0.5702 (**0.5596**) | 0.8850 / 0.8889 / 0.9008 (**0.8916**) | 77 min (folds 23 / 20 / 18 / 16 min, shared GPU) | fixed point 0.5003 / 0.5533 / 0.8830 (champion 0.5064 / 0.5601 / 0.8921); 108 October argmax calls differ from the champion |
| champion (reference) | — | 0.5130 / 0.5090 / 0.5146 (0.5122) | 0.5645 / 0.5639 / 0.5712 (0.5665) | 0.9020 / 0.8996 / 0.9014 (0.9010) | | |
| blend, five-way (rerun, `~/outputs/runs/blend`) | 139414a | 0.5130 / 0.5091 / 0.5147 (**0.5123**) | 0.5647 / 0.5641 / 0.5714 (**0.5667**) | 0.9022 / 0.9000 / 0.9018 (**0.9013**) | <1 min | sha256 `c3af1418...` as before |
| **icl-tabpfn-v2-blend6**: member 1/6, others 1/6 | 139414a | 0.5131 / 0.5091 / 0.5147 (**0.5123**) | 0.5648 / 0.5642 / 0.5714 (**0.5668**) | 0.9024 / 0.9001 / 0.9017 (**0.9014**) | <1 min | gen-only September -0.0001; 4-11 October rows changed vs the trees |
| **icl-tabpfn-v2-blend3**: member 1/3, others 2/15 | 139414a | 0.5130 / 0.5090 / 0.5147 (**0.5122**) | 0.5647 / 0.5640 / 0.5714 (**0.5667**) | 0.9023 / 0.8999 / 0.9017 (**0.9013**) | <1 min | August -0.0001 given and cleaned |

Paired event bootstrap (`python -m ringtrace.compare`, 2,000 draws, 5-95 %):

| challenger vs base | given | cleaned |
|---|---|---|
| icl-tabpfn-v2 vs champion | -0.0063 [-0.0072, -0.0053] | -0.0069 [-0.0079, -0.0059] |
| blend6 vs blend | +0.0001 [-0.0000, +0.0001] | +0.0001 [-0.0000, +0.0001] |
| blend3 vs blend | -0.0000 [-0.0001, +0.0001] | -0.0000 [-0.0001, +0.0001] |

October-like: not run to the end. The four tree members' cells were completed (s13 fitted here,
the rest copied from the gate run); the member's cells were stopped before the first one finished,
because every variant already fails the leave-one-month-out part of the bar.

Fraud-only role (`python analysis/icl_fraud_role.py`, laptop, 8 threads, commit 7329c93):
stopped after about 45 min with no output, on a laptop busy with other work. Not scored. It needs
a rerun on an idle machine with the five trees' and TabM's arrays; the role's
`blank_merchant_qr` rule is not in this tree, so the script applies the template rule only.
