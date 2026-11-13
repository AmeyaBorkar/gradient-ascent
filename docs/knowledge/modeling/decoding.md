# Decoding for macro-F1

**Summary.** Probabilities are turned into hard labels in three label-free steps, solved on the
predicted month itself: (1) a fixed-point rule for macro-F1 with fraud probabilities divided by
0.875; (2) a template rule that gives each look-alike event shape one class, anchored by the device
counter; (3) optionally one fraud type per event. The rules add +0.0058 (given) / +0.0064 (cleaned)
over the fixed point in this repo. Everything that fits cut-offs or multipliers to labels loses or
ties, because the probabilities are bimodal.

## 1. The fixed point
- Macro-F1 is linear-fractional in the confusion matrix, so the optimal classifier is a
  cost-weighted argmax with costs from the gradient of the metric at the optimum (the plug-in
  result of the multiclass-F-measure literature).
- Stationary rule (derived): predict `argmax_k (2 p_k - F_k) / D_k` with `D_k = pi_k + P_k`, where
  `F_k` is the class F1, `pi_k` its true share and `P_k` its predicted share. Class k wins only if
  `p_k > F_k / 2`, the multiclass form of "the F1-optimal threshold is half the optimal F1". It is a
  per-class scale and offset, so plain multipliers are a restricted family.
- Solve it by iteration: decode, recompute `F_k` and `P_k` from the decode using the probabilities
  themselves as the expected confusion, damp, repeat. No free parameters, no labels, one solve per
  month (the "fixed point per month").
- Earlier measurement, synthetic (6 classes, 95.5% majority, calibrated posteriors): held-out
  macro-F1 argmax 0.503, fixed point 0.602, coordinate-ascent multipliers 0.600.
- Measured in this repo (run 4): fixed point 0.5064 given / 0.5601 cleaned against argmax 0.5066 /
  0.5602, i.e. a tie here: the probabilities are already sharp. Its value is as a safe default that
  also absorbs a prior shift.

## 2. The fraud scale (0.875)
- Fraud probabilities are divided by 0.875, undoing the 12.2% of generator fraud filed as legit
  (1 - 0.122 = 0.878).
- Measured in this repo (runs 4 and 8): scale 0.875 against 1.0 is a tie, +0.0001 given. Earlier
  measurement: from 0.70 to 1.00 the calls move by 11-22 per 100,000 rows and the cleaned
  macro-F1 by at most 0.0004, so the exact value does not matter.
- Earlier measurement (relational features only): the fixed point solved on the month itself gained
  +0.0053 / +0.0056 / +0.0016 on a plain 12-class model, and +0.0047 / +0.0057 / +0.0052 on the
  cascade with the fraud scale of 0.875 (sd 0.001-0.002).

## 3. Why cut-offs do nothing on bimodal probabilities
Histogram of P(fraud) on leave-one-month-out months (earlier measurement):
- 95.1% of rows sit at 0.01-0.05 (2.4% of them are fraud-labelled: the ordinary-looking rows);
- 1.6% sit at >= 0.95 (97.2% fraud on cleaned labels, 87.9% on given);
- only 17 rows per 100,000 lie between 0.2 and 0.95; the 0.05-0.2 band holds 291 per 100,000 at
  5.3% fraud, under any class's F/2.

So every fraud cut between about 0.2 and 0.95 calls the same rows. The rows the train labels miss
are already scored >= 0.95 (the cascade removed them from the legit side). What is left for a
decoder is the fraud type inside the called rows, which the template rule mostly settles.

## 4. Decoder challengers (earlier measurement)
Three-seed mean on an October-like September (event-cleaned labels); baseline = fixed point at
0.875 plus the anchored template rule = 0.5396 cleaned / 0.8629 generator-only. Gate: +0.007.

| idea | result vs baseline (cleaned) |
|---|---|
| (a) per-class or uniform prior shift to the cleaned label shares | -0.0002 to +0.0002; calls move 1,670-1,683 per 100,000 against 1,681 |
| (b) per-class counts anchored to the label-free event census | -0.0040 to -0.1248: loses (asking for the ordinary rows too fills the gap with legit rows) |
| (c) F/2 admission rule (three variants) | -0.0006 to -0.0003; same cut as the fixed point |
| (d) multipliers / cut-offs fitted on given or cleaned labels | -0.0184 to -0.0022 (one thinning seed fits a cut-off that loses 0.051) |
| (e) bagged multipliers or cut-offs (25 event bootstraps) | -0.0024 to -0.0004; steadier, not better |
| (f) one class per event (five variants) | +0.0008 to +0.0009 [+0.0001, +0.0019]; +0.0003 leave-one-month-out; 0.0000 in the families mode |
| fraud scale 0.70 / 0.80 / 0.90 / 1.00 | -0.0004 to 0.0000 |

Thresholds or multipliers fitted on one month and applied to the next gain only about half of what
they gain in-sample (+0.0028 / +0.0035 on multipliers against +0.0055-0.0081 on the fit month) and
lose on another model. Decoders fitted for different targets score within 0.002 of each other.
The "no thresholds, multipliers or EM priors fitted on train months" rule follows. EM prior shift
on raw probabilities is off by 16-21% per class on a rehearsal (mule x1.3-1.5, ato x0.66-0.85): not
used. The census of a month (events, rows by score) is kept as a diagnostic, never a count guard.

## 5. The template rule
Some event shapes are shared by two or more fraud types (generator subroutines are reused), so
within a shape the row evidence cannot tell the types apart (in-shape AUC 0.46-0.65). The rule
assigns **one class per template** to every row of that shape.
- **Plain rule:** each template gets its majority class. Measured in this repo (run 7): 0.5094 given
  / 0.5636 cleaned, which is +0.0030 / +0.0035 over the fixed point.
- **Counter-anchored rule:** a row whose device-counter segment places it confidently in the
  partner class keeps that class (anchors with confidence >= 0.8); the rest take the template class.
  Measured in this repo (run 4): 0.5122 given / 0.5665 cleaned. It changes 773 train rows and 137
  test rows. Anchoring adds +0.0028 given / +0.0029 cleaned over the plain rule (run 7 against run 4).
  Rule over fixed point: +0.0058 / +0.0064.
- Earlier measurement (a build with the same design): anchored rule vs none +0.0059 cleaned /
  +0.0053 given; plain rule is -0.0032 below the anchored one.
- **Groups** (class chosen; share of the pair that is the kept class; earlier measurement):
  | template | class | kept share | gain over rule without it (cleaned, Jul/Aug/Sep) |
  |---|---|---|---|
  | merchant-burst 1 (mule/chain look-alike) | chain | .85/.83/.87 | +.0010/+.0008/+.0010 |
  | drain 0.25 full signature (ato/sim) | sim | .72/.70/.74 | +.0019/+.0005/+.0022 |
  | multi-sender bursts 2/3 (bot/mule) | mule | .96/.88/.89 | +.0007/+.0029/+.0015 |
  | collect col7 (phishing/cashback) | cashback | .44/.37/.42 | +.0018/+.0008/+.0019 |
  | collect col1 | phishing | .93 | +.0004/.0000/+.0004 |
  | ring shape 1 | qr_swap | .96 | +.0001/.0000/+.0001 |
- No group is negative in any month. The rule trades phishing and ato F1 (about -0.009 and -0.008)
  for cashback, sim, bot, mule and chain (+0.007 to +0.030).
- **Fragile cells:** collect col7 is a minority-class call (cashback is 37-44% of the pair; it wins
  only because cashback is the smaller class under macro-F1) and its error rate on generator rows is
  rising (0.497, 0.544, 0.570). The bot share of the 2/3 bursts rose 4% to 11-12%. In October the
  anchored rule keeps col7 at 100% cashback because its 2 anchors have confidence 0.42. Re-check
  the break-even on any new candidate.
- The rule reads every family and the counter from the store, whatever the recipe's feature
  families are, so ablations that drop a family still apply it with full evidence; the no-counter
  ablation turns the rule off explicitly (measured in this repo, decision 9).

## 6. One type per event and the consistency stages
- Earlier measurement: giving a whole event one label (including legit) hurts (-0.0125 on the noisy
  target for sender-day groups). Unifying only the fraud type among rows already called fraud gains
  +0.0019 to +0.0037 on a plain model (device burst +0.0025, sender-day +0.0019, receiver burst
  +0.0025, chained +0.0037, sd 0.0011), and rows with P(legit) < 0.9 joining a receiver burst
  that has two fraud calls gained +0.0044 / +0.0050 / +0.0076. Smoothing over connected components
  loses 0.004-0.008.
- After the template rule the same idea is almost redundant: before the rule the event vote
  changes 50-60 rows, after it only 4 events (28 rows) remain split. Gain +0.0009 (October-like),
  +0.0003 (leave-one-month-out), 0.0000 (families mode): a tie, not worth an upload alone.
- The burst-consistency stage of an earlier build (255 rows changed) tied within +-0.0003 and is off.
  If wanted, the simplest form unifies the type over template event ids before the template rule.

## 7. A gate that was tested and rejected
If the test labels had *not* counted the ordinary-looking rows, a gate "call fraud only on rows with
a mark or a label-free event id" would have gained +0.020 on the generator-only target (0.8913 to
0.9113, positive in all three months) and fraud calls on those rows fall from 1,265 to 85. Under the
opposite reading it loses -0.015. The first upload read the test labels as counting ordinary rows,
so the gate is not used (earlier measurement; see [public-noise.md](public-noise.md)).
