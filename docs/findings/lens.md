# How the TabM and tree fraud masses are combined, and where TabM's adds are right

**Verdict: every candidate is rejected.** The fixed probability average in `blend-tabmF-bq` stays.
It is the legit column from TabM 1/2 and each tree 1/10, with the tree types rescaled to that mass.
No October candidate file was written.

All numbers are **measured in this repo**. Code: `src/ringtrace/lens.py`. Commands are in each
section. Raw numbers are in `RESULTS.md` ("Lens: fraud-mass combiners, seeds, add anatomy").

## Setup

- **Base:** `blend-tabmF-bq`, rebuilt from the saved arrays. Its October calls equal the
  `blend-tabmF-bq.csv` file on every row (0 differ). Scores: given 0.5137 / 0.5094 / 0.5150,
  cleaned 0.5654 / 0.5644 / 0.5718, gen-only 0.9031 / 0.9000 / 0.9019, A-clean proxy
  0.5462 / 0.5451 / 0.5525.
- **Shared pipeline:** every challenger changes only the fraud mass, or one class's share of it.
  The decoder (fixed point 0.875), the anchored template rule and `blank_merchant_qr` are the same
  for all of them.
- **Scoring:**
  - per-month change Jul / Aug / Sep on given, cleaned, gen-only and the A-clean proxy;
  - the paired event bootstrap (2,000 draws, 95 %) of the mean change;
  - the nine October-like cells (1,000 draws each). Tree cells come from the gate's frames and
    TabM cells from the TabM gate.
- **"Adds":** train rows that the trees-only five-way blend (same pipeline, trees' mean mass)
  calls legit and the challenger calls fraud.
- **Pocket b:** an add whose gen-only label is fraud, that is, a generator row.
- **False alarm:** an add whose cleaned label is legit.

## 1. How the fraud mass is combined (`python -m ringtrace.lens combine`)

Two definitions:
- **logit and geometric mean:** the same weights (TabM 1/2, each tree 1/10), on logits and on log
  masses.
- **noisy-OR and max:** of TabM's mass and the trees' mean mass.

The **nested stacker** is an L2 logistic regression on [TabM fraud logit, five tree fraud logits,
std of the tree logits]:
- the target is cleaned fraud;
- it is fitted on two months and scores the third;
- October uses all three months;
- the October-like cells use the July + August fit.

| combiner | adds | pocket b | false alarms | given change Jul / Aug / Sep | gen-only change | October-like given per seed |
|---|---|---|---|---|---|---|
| probability average (base) | 48 | 29 | 13 | - | - | - |
| logit average | 13 | 9 | 2 | -0.00034 / -0.00028 / -0.00029 | -0.00039 / -0.00005 / -0.00001 | -0.00015 / -0.00028 / -0.00036 |
| geometric mean | 6 | 4 | 1 | -0.00060 / -0.00029 / -0.00035 | -0.00071 / -0.00002 / -0.00017 | -0.00023 / -0.00023 / -0.00045 |
| noisy-OR | 157 | 83 | 68 | +0.00031 / -0.00012 / -0.00008 | -0.00021 / -0.00049 / -0.00041 | -0.00057 / -0.00018 / -0.00044 |
| max | 116 | 66 | 44 | +0.00022 / -0.00006 / -0.00006 | -0.00017 / -0.00051 / -0.00045 | -0.00047 / -0.00018 / -0.00031 |
| stacker, C = 0.01 | 176 | 86 | 83 | +0.00027 / -0.00012 / +0.00007 | -0.00044 / -0.00055 / -0.00051 | -0.00236 / -0.00116 / -0.00146 |
| stacker, C = 1 | 166 | 83 | 76 | +0.00028 / -0.00011 / +0.00003 | -0.00037 / -0.00051 / -0.00050 | -0.00240 / -0.00119 / -0.00149 |

**Answer.** The combiners fall into two groups, and neither beats the plain average.
- **Fewer adds (logit, geometric mean).** These pull the mass toward the trees, whose fraud masses
  sit near 0 on these rows. They give up most of TabM's adds, and they lose in every month.
- **More adds (noisy-OR, max, stacker).** These recover up to about 57 more pocket-b rows than the
  base, but the extra adds are a coin flip:
  - noisy-OR: 109 more adds, 54 of them pocket b and 55 more false alarms;
  - stacker: 128 more adds, about 50 % right.

  They gain on cleaned and the A-clean proxy in some months. Every one loses on gen-only in every
  month, and badly on October-like.
- **The stacker's October-like loss** (-0.0012 to -0.0024 given) shows its calibration does not
  carry over to fold models fitted on two months.

The base's 48 adds are right 73 % of the time on cleaned labels. Every rule that adds more rows
adds them at about 50 %, which macro-F1 does not reward.

## 2. Seed stability (`seeds`, `seedcells`)

TabM is the mean of k seed-set runs: seeds 0-2, 3-5, 6-8. The trees and everything else are
unchanged.

| TabM seeds | October adds vs trees | October rows vs the 3-seed base | train adds (right, cleaned) | LOMO given / cleaned / gen-only change vs 3 seeds |
|---|---|---|---|---|
| 3 | 7 | 0 | 48 (35) | 0 |
| 6 | 6 | 3 (1 add, 2 removals, all phishing_collect) | 48 (35) | -0.00002 / +0.00004 / 0; +0.00002 / +0.00004 / 0; +0.00001 / +0.00011 / 0 |
| 9 | 6 | 3 (the same three rows) | 51 (38) | -0.00002 / +0.00007 / +0.00004; +0.00002 / +0.00006 / +0.00004; +0.00001 / +0.00012 / +0.00005 |

- **Single seed sets** add 7, 6 and 8 October rows. Their pairwise Jaccard overlap is 0.62, 0.88
  and 0.75, so 1-2 of about 7 October adds churn with the seed.
- **More seeds:**
  - from 6 to 9 seeds the add set does not move (Jaccard 1.0);
  - every row that churns is `phishing_collect`, the class where TabM's adds are a coin flip
    (section 3).
- **October-like, 9 seeds against 3:** TabM is the mean of the three seed-set cells for each
  thinning seed. It changes 5 rows over nine cells: given +0.00001 / -0.00005 / +0.00000 (lb
  -0.00007), cleaned +0.00001 / -0.00005 / -0.00000.
- **Verdict:** a tie. The 1/2 share is already stable at 3 seeds. More seeds swap one or two
  `phishing_collect` calls and fail the bar (given Jul -0.00002).
- **12 seeds:** the seed-set 9-11 run had not landed by the end of the time box, so 12 seeds were
  not measured.

## 3. Where the adds are right (`adds`)

The base's 48 train adds, by month (adds, right on cleaned labels):
- 31 are given-labelled fraud, 4 given-legit / cleaned-fraud, 13 false alarms;
- 29 are pocket-b generator rows.

| called class | adds | right (cleaned) | Jul | Aug | Sep |
|---|---|---|---|---|---|
| phishing_collect | 26 | 19 | 7, 7 | 12, 9 | 7, 3 |
| refund_abuse | 7 | 3 | 1, 1 | 5, 1 | 1, 1 |
| velocity_bot | 4 | 4 | 0 | 0 | 4, 4 |
| fake_cashback | 3 | 3 | 1, 1 | 0 | 2, 2 |
| qr_swap | 3 | 1 | 1, 0 | 0 | 2, 1 |
| account_takeover / layering_chain / mule_account / sim_swap | 5 | 5 | - | - | - |

| evidence family (first family whose shape holds the row) | adds | right | Jul | Aug | Sep |
|---|---|---|---|---|---|
| none | 27 | 17 | 6, 5 | 11, 5 | 10, 7 |
| col | 13 | 10 | 4, 4 | 6, 5 | 3, 1 |
| bot | 4 | 4 | 0 | 0 | 4, 4 |
| drn | 3 | 3 | 2, 2 | 1, 1 | 0 |
| mark only | 1 | 1 | 0 | 0 | 1, 1 |

43 of the 48 adds are outside every template.

**Structure.**
- Adds with bot, drain and other generator-family evidence are always right, but rare: 8 rows in
  three months.
- The bulk is `phishing_collect` and rows with no family. Their precision swings from 100 % in
  one month to 43 % in another.
- The merchant pair (refund_abuse, qr_swap) is 40 % right.

**Two label-free gates, chosen per held month on the other two months:**
- **Family gate.** TabM's mass is used only in families whose adds were at least 50 % right in
  both fit months; the trees' mass is used elsewhere. It keeps bot, drn, mf, rep and rng, and
  drops "none" in most folds. It loses everywhere: given -0.00068 / -0.00018 / -0.00023
  [-0.00061, -0.00018], cleaned -0.00064 / -0.00016 / -0.00030, A-clean
  -0.00062 / -0.00016 / -0.00029. The "none" adds are the bulk of the gain, despite their noise.
- **Class gate.** An add goes back to legit if its called class was below 50 % in both fit months.
  No class qualifies in any fold: refund_abuse is 1/1 in July and 1/5 in August. The gate changes
  0 rows.

**Answer.** No label-free condition separates the good adds from the bad ones with enough rows to
matter.

## 4. TabM's type share for single classes (`types`)

For each fraud class k, class k's share of the fraud mass comes from TabM's type distribution, and
the other classes keep their tree proportions. The test is the change against the base, Jul / Aug /
Sep, given.

| class | rows train / Oct | given | cleaned | gen-only |
|---|---|---|---|---|
| layering_chain | 2 / 0 | 0 / +0.00003 / +0.00001 | 0 / +0.00003 / +0.00001 | 0 / +0.00008 / +0.00004 |
| loan_app_extortion | 2 / 2 | 0 / 0 / +0.00019 | 0 / 0 / +0.00019 | 0 / 0 / -0.00006 |
| account_takeover | 3 / 1 | -0.00002 / +0.00003 / +0.00001 | same | -0.00005 / +0.00008 / +0.00004 |
| velocity_bot | 3 / 0 | 0 / -0.00001 / +0.00001 | same | 0 / +0.00003 / +0.00004 |
| mule_account | 6 / 1 | 0 / -0.00004 / +0.00004 | 0 / -0.00003 / -0.00001 | - |
| sim_swap | 4 / 1 | -0.00002 / +0.00003 / -0.00004 | same | - |
| fake_cashback | 2 / 12 | -0.00011 / +0.00003 / 0 | same | - |
| phishing_collect | 13 / 11 | -0.00014 / -0.00002 / +0.00003 | - | - |
| qr_swap | 15 / 16 | -0.00009 / -0.00008 / +0.00004 | - | - |
| refund_abuse | 17 / 7 | -0.00023 / -0.00038 / 0 | - | - |
| merchant_collusion | 7 / 2 | 0 / -0.00062 / +0.00001 | - | - |

**Nested choice.** A class is kept if it is non-negative on given and cleaned in both fit months
and positive in one.
- Held July keeps layering_chain, account_takeover, fake_cashback and loan_app_extortion.
- Held August keeps velocity_bot, layering_chain, loan_app_extortion and merchant_collusion.
- Held September keeps layering_chain.
- October keeps layering_chain and loan_app_extortion.

Result: given -0.00013 / -0.00065 / +0.00001 [-0.00060, -0.00000], cleaned
-0.00013 / -0.00062 / +0.00001, A-clean -0.00012 / -0.00060 / +0.00001. **Rejected.**
- merchant_collusion is chosen from two months where it changes nothing, then costs 0.0006 in
  August.
- layering_chain is the only class that never loses, but it moves 2 train rows and 0 October rows.
  That is a tie.

As a full model, TabM's per-class F1 gains do not survive into the type split once the trees set
the mass.

## 5. What the earlier analyses missed

1. **Most of TabM's gain is not hidden-label recovery.** Of the 48 adds, 31 are already fraud on
   the given labels, and only 4 are given-legit rows that the cleaned labels call fraud. The adds
   are ordinary fraud rows the trees miss. 29 of them are generator rows, which is the pocket-b
   count. So the gain shows on given labels too (+0.0007 / +0.0003 / +0.0003), and it does not
   depend on the label cleaning.
2. **The marginal add is a coin flip, whatever combiner produces it.** The base's adds are right
   73 % of the time. The next 70-130 adds, from noisy-OR, max or the stacker, are right about
   50 %, and they cost on gen-only in every month. The average sits where precision drops.
3. **October churn is one class.** All seed-to-seed October churn is `phishing_collect` calls. This
   is the class where TabM's adds are 100 % right in one month and 43 % in another. Of the 7
   October adds in `blend-tabmF-bq`, 4 are phishing_collect, so about half of the file's lead on
   the public board rests on that class's coin flip.
4. **A calibrated stacker transfers worst.** It ties or gains on cleaned in the train months but
   loses 0.001-0.002 on every October-like cell. Its learned weights rely on the OOF calibration of
   three-fold models, which two-month fold models do not share.
