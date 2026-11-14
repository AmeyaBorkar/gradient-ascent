# October's festival amounts and class mix: neither costs the model anything worth correcting

**Question.** October differs from the training months in two measured ways. Its ordinary payments
are about 1.166 times larger, at every quantile and payment type, and its balances about 1.02
times larger, while the shares of round amounts stay the same. Its class mix also moves; the
estimate from the counter segment volumes (whole class volume, ordinary rows included), October
over train, is: mule 1.34, refund 1.26, phishing 1.14, collusion 1.12, chain 1.09, qr 1.06,
bot 1.04, sim 1.01, cashback 0.96, takeover 0.95, loan 0.77. Do these shifts cost the current
model anything, and does making training look like October win it back?

**Answer.** No on both counts.

- **Amounts (known).** A September rescaled to October's amounts costs the champion nothing:
  +0.00003 / -0.00006 / -0.00010 on fold 9 (given / cleaned / gen-only) and -0.00016 / -0.00009 /
  +0.00011 on fold 8, against a stop line of 0.0005. Training on rescaled months has nothing to
  recover, so it was not run.
- **Class mix (known).** Correcting the decoded probabilities with an EM prior estimate loses on
  July (-0.0006 given, -0.0009 cleaned). The EM estimate of each month's prior is noise, at 0.5 to
  2.3 times the truth on months whose true ratio is 1. A correction by the mix itself (an oracle
  on months resampled to that mix) gains only +0.0001 to +0.0002, moves about 8 rows per month and
  6 October rows, and costs up to -0.00004 when the month does not carry that mix. Class-mix
  training weights could at most reproduce that shift through one tenth of the fraud mass, so they
  were not run.

## 1. A September that looks like October

`python -m ringtrace.festival` rescales the ordinary-routine rows (structured silver label 0) of
chosen train months and runs a recipe on that ledger under its own cache key.
`python analysis/festival_check.py --calls <best file>` compares the scaled months with October's
rows called legit by the best file. All the numbers below are aggregates.

| ordinary rows | mean amount | whole rupee | x10 | x100 | median balance | amount/balance > 0.3 / 0.5 / 0.9 / 1.0 |
|---|---|---|---|---|---|---|
| train, unscaled | 1,052 | 0.2590 | 0.2522 | 0.0242 | 6,271 | 0.145 / 0.0361 / 0.0010 / 0 |
| train, scaled | 1,227 | 0.2590 | 0.2522 | 0.0242 | 6,442 | 0.170 / 0.0361 / 0.0010 / 0 |
| October, called legit | 1,242 | 0.2594 | 0.2528 | 0.0251 | 6,380 | 0.174 / 0.0452 / 0.0012 / 0 |

October over scaled train, quantiles 10 / 25 / 50 / 75 / 90 %: all rows 1.011 / 1.019 / 1.007 /
1.003 / 1.004; per payment type within 0.96 to 1.04. Mean amount 1.012, median balance 0.990.

The scaling had to keep three properties, and two early versions broke them:

1. **Round-amount classes.** Rounding a scaled multiple of 10 to the nearest 10 puts one in ten on
   a multiple of 100. The first version doubled the multiple-of-100 share (0.024 to 0.048). Each
   amount now takes a neighbouring multiple of its own unit that is not a multiple of a coarser
   one, so every class keeps its share exactly. The choice between the two neighbours is
   stochastic, so small multiples of 100 (100, 200, 300) scale on average instead of staying put.
2. **Equal amounts stay equal.** A stream-drawn rounding split equal amounts. The draw is now a
   hash of the amount itself.
3. **The balance ceiling.** No ordinary row pays more than its balance, in train or in October.
   Amounts times 1.166 against balances times 1.02 put 718 of September's ordinary rows (0.15 %)
   above their balance. In train, paying more than the balance is a fraud signature: 6.3 % of
   refund_abuse rows, 9.5 % of sim_swap and 7.6 % of account_takeover do it. A row whose scaled
   amount-to-balance ratio would pass both 0.5 and its own ratio now has its balance scaled with
   its amount (28,577 of September's 502,206 rows). Its ratio stays where it was. That keeps the
   ceiling, matches October above 0.3 (0.170 against 0.174), and leaves the tail above 0.5 at the
   train share, a little under October's.

## 2. What the festival shift costs the champion

Fold 9 is fitted on July + August and scores September; fold 8 is fitted on July + September and
scores August. The baseline and the challenger are refitted on the same GPU from the same commit.
They differ only in the held-out month's money columns, plus the cross-month window features they
feed. Macro-F1 after the fixed point and the template rule:

| run | given | cleaned | gen-only |
|---|---|---|---|
| fold 9, unscaled | 0.51455 | 0.57120 | 0.90144 |
| fold 9, September scaled | 0.51458 (+0.00003) | 0.57114 (-0.00006) | 0.90134 (-0.00010) |
| fold 9, scaled without the balance ceiling | 0.51196 (-0.00259) | 0.56832 (-0.00288) | 0.89437 (-0.00707) |
| fold 8, unscaled | 0.50897 | 0.56389 | 0.89960 |
| fold 8, August scaled | 0.50881 (-0.00016) | 0.56379 (-0.00009) | 0.89971 (+0.00011) |

The run without the balance ceiling is the one to learn from. Its whole loss is on rows October
does not have: the champion calls ordinary rows that were pushed past their balance fraud. In that
frame, September's refund_abuse calls rise from 606 to 690 (refund F1 -0.026 given, -0.063
gen-only) and phishing_collect calls from 950 to 992 (F1 -0.006 given). With the ceiling kept,
refund calls stay at 606 and phishing calls move by 3. The cost is then inside the run-to-run
noise in both months, well under the 0.0005 stop line. The festival line stops here: training on scaled months
(step 3) has no loss to recover.

## 3. Prior correction of the best file's probabilities

`python analysis/augment_em.py --draws 8` uses the best file's out-of-fold probabilities
(`closelook/prep.npz`), with no refits. The plain decode reproduces the best file's calls on
every train row. Each held-out month is also resampled to October's mix: whole units (an event,
or a row on its own) of fraud class k appear floor(r_k) times, plus once more with probability
r_k - floor(r_k). The realised ratios match the targets to within 0.01 to 0.02. The probabilities
are then corrected, decoded with the fixed point at 0.875, passed through the two row rules and
scored. The change against no correction:

| correction | plain Jul / Aug / Sep, given | plain, cleaned | resampled (mean of 8 draws), given | resampled, cleaned | October rows |
|---|---|---|---|---|---|
| EM prior (`mix.em_priors`), other months' mean OOF as reference | -0.00061 / -0.00007 / +0.00005 | -0.00088 / -0.00013 / +0.00005 | -0.00029 / +0.00015 / +0.00025 | -0.00048 / +0.00021 / +0.00025 | 20 |
| EM prior on the type split only | -0.00058 / -0.00013 / +0.00004 | -0.00084 / -0.00019 / +0.00003 | -0.00019 / +0.00017 / +0.00015 | -0.00039 / +0.00024 / +0.00010 | 10 |
| October's ratios themselves | -0.00004 / +0.00000 / +0.00008 | -0.00001 / -0.00001 / +0.00007 | +0.00007 / +0.00015 / +0.00025 | +0.00013 / +0.00021 / +0.00024 | 6 |
| October's ratios, type split only | -0.00003 / 0 / +0.00004 | -0.00003 / 0 / +0.00004 | +0.00011 / +0.00016 / +0.00014 | +0.00014 / +0.00023 / +0.00015 | 7 |

On the plain months the last two rows apply October's ratios to a month that does not carry them.
That is the cost of the correction when the estimate is wrong.

- **EM fails** the bar on the plain months (July loses on given and cleaned). It also loses on
  resampled July. The cause is the estimate: on the plain months, where the true ratio is 1 for
  every class, EM's estimated prior over the reference ranges from 0.51 to 1.50 in July, 0.72 to
  1.37 in August and 0.66 to 2.28 in September. For October it says loan 2.49, refund 1.97 and bot
  0.51, where the counter estimate says 0.77, 1.26 and 1.04. Month-to-month drift in the
  probabilities' calibration is larger than the mix shift EM is meant to find.
- **The known mix barely matters.** Even with the exact ratios, the gain on a resampled month is
  +0.0001 to +0.0002, from about 8 rows per month. On October the same correction moves 6 rows
  (legit -> phishing 2, qr -> refund 3, loan -> cashback 1), and it loses a little on July when the
  mix is wrong. It does not pass the bar (non-negative on given and cleaned in every month). Its
  October value is a few rows' worth, below the public slice's noise.

## 4. Class-mix training weights

Not run. For a calibrated model, weighting class k by r_k in training multiplies its posterior
odds by r_k. That is the known-mix correction above, applied through the champion alone (one tenth
of the fraud mass and one fifth of the type split) instead of the whole blend. Its effect on
October is bounded by the 6 rows above.

## Verdicts

| component | bar | verdict |
|---|---|---|
| festival-scaled training (steps 2-3) | cost of the shift < 0.0005 on given and cleaned in both folds | stopped: nothing to recover |
| EM prior correction (`em`, `em_types`) | loses on plain July | rejected |
| known-mix prior shift (`known`, `known_types`) | -0.00004 / -0.00003 on plain July; 6-7 October rows | rejected (too small, and not non-negative) |
| class-mix weights | bounded by the known-mix shift | not run |

**Belief.** October's amounts would only matter to a feature that compares amounts with fixed
thresholds learned in train. The trees' thresholds on amount-derived columns evidently carry over;
the ordinary amount distribution moves, but the fraud signatures (exact amounts, balance ratios,
round classes) do not.

**Guess.** The scaled ledger keeps the fraud-routine rows unscaled. October's fraud-called amounts
sit at 0.95 to 1.1 times train's at the median for most classes (sim_swap and takeover about
1.17), so this looks right, but a fraud routine that scales with the festival would not be
covered here.
