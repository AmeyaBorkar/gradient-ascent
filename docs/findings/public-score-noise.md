# Public-score noise

**Question.** How much does the 30 % public slice (78,175 of 260,583 October rows) move a score,
and how big must a gap between two near-identical files be before it means anything?

**Answer.** September stands in for October: cleaned labels, and 2,000 random public slices of
30.0 % of September's 510,143 rows. Each member is decoded as the run decodes it.

| member vs champion | rows that differ | full-month gap | sd of public gap | P(\|public gap - true gap\| >= 0.0005) | >= 0.001 | public and private disagree on sign |
|---|---|---|---|---|---|---|
| catboost | 33 | -0.00062 | 0.00047 | 29.7 % | 3.1 % | 9.8 % |
| clean-event | 18 | +0.00014 | 0.00021 | 1.7 % | 0 % | 32.9 % |
| clean-l2 | 20 | +0.00005 | 0.00025 | 3.8 % | 0 % | 72.2 % |
| lgbm | 14 | -0.00000 | 0.00045 | 30.5 % | 1.4 % | 98.6 % |

- **Two files that differ on a few dozen rows:** the public gap has an sd of **0.0002-0.0005**. A
  chance error of 0.001 or more shows up in at most 3 % of slices. An error of 0.002 never did.
- **One file:** the public score has an sd of **0.0059**, and public minus private has an sd of
  **0.0084**. A single file's public score does not predict its private score to the third
  decimal.
- **One extra correct row on the public slice:** the expected true counts come from September's
  class mix scaled to 78,175 rows. The call counts are the champion's October calls x 0.30. TP
  comes from the September F1.

| class | public true | public calls | Sep F1 | +1 correct row moves macro-F1 by |
|---|---|---|---|---|
| legit | 75,145 | 76,991 | 0.988 | 0.000001 |
| mule_account | 548 | 237 | 0.485 | 0.00016 |
| velocity_bot | 431 | 95 | 0.566 | 0.00023 |
| layering_chain | 355 | 167 | 0.602 | 0.00022 |
| sim_swap | 201 | 107 | 0.538 | 0.00039 |
| account_takeover | 205 | 73 | 0.485 | 0.00045 |
| phishing_collect | 353 | 149 | 0.536 | 0.00024 |
| qr_swap | 380 | 155 | 0.565 | 0.00022 |
| refund_abuse | 191 | 84 | 0.643 | 0.00041 |
| fake_cashback | 132 | 62 | 0.456 | 0.00066 |
| loan_app_extortion | 102 | 30 | 0.487 | 0.00095 |
| merchant_collusion | 133 | 25 | 0.505 | 0.00078 |

So a public gap of 0.001 equals two to six correct rows of a small class.

**Method.** `analysis/core.py:slice_scores` draws public index sets with
`numpy.random.default_rng(0)`. It scores each member on the slice and on the rest of the month.
`one_row_gain` gives (2(TP+1)/(N+P+1) - F1)/12 per class. It ignores the loss on the class the
row had before; in practice that class is legit, worth about 1e-6.

**Command.**
`RINGTRACE_THREADS=4 scripts/box_in.sh $RINGTRACE_BOX_DIR "export RINGTRACE_OUT=~/outputs; python analysis/findings.py"`
writes `~/outputs/findings-analysis/findings.json` (key `noise`). It runs in 43 s.

**What it changes.**
- The five members are within 0.0006 of each other on September (cleaned labels). That is no
  more than the noise of one public slice.
- Do not rank members or pick blend weights from public scores.
- A public difference under about 0.001 between files that differ on tens of rows is noise.
