# A wide look for a +0.0025 lever: where the headroom sits and what can reach it

**Question.** The pipeline sits on a plateau where every measured change moves macro-F1 by about
0.0001. Is there a lever worth +0.0025 on October, anywhere from the generator to the upload rule?

**Answer (measured in this repo unless marked).** No single lever of that size was found.

- The whole cleaned headroom (0.030) is the typing of events inside the six look-alike templates:
  perfect typing there is worth **+0.032 to +0.034** a month. Nothing label-free reaches it. The
  class the template rule assigns is already the best split of every pool in every month.
- Every error outside those pools adds up to 0.0043 (generator rows called legit 0.0025, legit
  rows called fraud 0.0018, `fresh-error-anatomy.md`). A +0.0025 gain from better detection would
  need 60 % of all of it.
- Finding ordinary-looking fraud rows cannot pay even with a perfect detector, because their type
  is unknown (section 3).
- One structural October weakness was found and sized: the counter's segments are fitted from one
  sixth of train's phones. It costs about 9 October rows, worth **+0.0005 to +0.001** if recovered
  (lever 1).

No candidate file was built: no lever was cheap enough to build and gate inside the time box.

## Levers, ranked by expected October gain x probability / cost

| # | lever | mechanism | evidence | gain | cost |
|---|---|---|---|---|---|
| 1 | Counter segments at October's size | add what train's block teaches (segment lengths on the number line, each segment's occupancy) to the per-split fit | thinning train's block to October's size loses 8.6 ± 5.1 unanchored rows of 171 (agreement 0.897 against 0.942); the last four segments fall from 0.896 to 0.800 | +0.0005 expected, 0 to +0.002 | half a day, CPU only |
| 2 | Generator rows called legit | a member that finds them with the right type | the pocket holds 0.0025 in total; the TabM fraud mass took +0.0004 of it | +0.0003 to +0.001 per member that works | already under test |
| 3 | Pair typing inside the templates | a signal written in class order, as the phone counter is | +0.032 to +0.034 available; every check below is negative | 0 unless a new signal appears | closed |
| 4 | Chain / mule account history | accounts shared by two events keep the class slightly more often than chance | same class 0.62 against 0.57 ± 0.02 shuffled | about 0 (the pool's oracle is +0.005, the signal is weak) | 2 h; not recommended |

The experiment to run first is lever 1 (section 5). It cannot reach +0.0025 alone.

## 1. Where the headroom sits

`python analysis/farlook.py pairs` decodes the blend as the submission is decoded and, for each
template, takes the pool of rows called the template's class. Shares are of the cleaned labels.

| template (class -> partner) | pool rows Jul / Aug / Sep | partner share | slope of macro-F1 at the current split | best share to move | whole pool to the partner | perfect typing |
|---|---|---|---|---|---|---|
| drain1 (sim_swap -> account_takeover) | 555 / 504 / 474 | 0.268 / 0.274 / 0.243 | -0.017 / -0.015 / -0.019 | 0 / 0 / 0 | -0.023 / -0.021 / -0.026 | **+0.0118 / +0.0121 / +0.0105** |
| col7 (fake_cashback -> phishing_collect) | 154 / 153 / 198 | 0.539 / 0.595 / 0.551 | -0.003 / -0.002 / -0.003 | 0 / 0 / 0 | -0.003 / -0.002 / -0.004 | +0.0053 / +0.0061 / +0.0075 |
| col1 (phishing_collect -> fake_cashback) | 716 / 673 / 614 | 0.085 / 0.079 / 0.086 | -0.033 / -0.033 / -0.031 | 0 / 0 / 0 | -0.033 / -0.033 / -0.032 | +0.0067 / +0.0063 / +0.0059 |
| mf1 (layering_chain -> mule_account) | 842 / 914 / 842 | 0.147 / 0.152 / 0.129 | -0.023 / -0.025 / -0.026 | 0 / 0 / 0 | -0.029 / -0.031 / -0.033 | +0.0048 / +0.0055 / +0.0044 |
| mf2+mf3 (mule_account -> velocity_bot) | 1,201 / 1,151 / 1,260 | 0.044 / 0.078 / 0.104 | -0.041 / -0.037 / -0.038 | 0 / 0 / 0 | -0.048 / -0.043 / -0.045 | +0.0018 / +0.0035 / +0.0050 |
| rng1 (qr_swap -> refund_abuse) | 325 / 333 / 310 | 0.046 / 0.027 / 0.052 | -0.018 / -0.019 / -0.017 | 0 / 0 / 0 | -0.018 / -0.019 / -0.017 | +0.0014 / +0.0007 / +0.0012 |
| **sum of perfect typing** | | | | | | **+0.0319 / +0.0343 / +0.0344** |

- **The headroom is pair typing.** The six pools account for the whole 0.030 gap of
  `headroom-per-class.md`. account_takeover (gap 0.092) is drain1; fake_cashback (0.095) and
  phishing_collect (0.053) are col7 and col1.
- **The decoder question is settled.** Moving a random share q of a pool to the partner changes
  the two classes' F1 along a concave curve. The slope at q = 0 is negative in all 18 pool-months
  and the best q is 0 everywhere. In col7 the partner is the majority (0.54 to 0.60) and the
  template's class still wins, because fake_cashback is the smaller class. The margin there is
  thin (0.002 to 0.004), so col7 is the only pool where a change in October's mix could flip the
  answer.
- **The typing is not a function of anything observable.** Inside each pool, with the cleaned
  partner label as the target:

  | pool | rows / events | partner share | blend's own type ratio: AUC, partner share in its top tenth | row + event features, boosted trees, 5 folds by event: AUC, top tenth |
  |---|---|---|---|---|
  | drain1 | 1,527 / 537 | 0.263 | 0.610, 0.382 | 0.600, 0.368 |
  | col7 | 499 / 253 | 0.567 | 0.493, 0.490 | 0.503, 0.571 |
  | col1 | 1,929 / 1,269 | 0.087 | 0.561, 0.104 | 0.487, 0.083 |
  | mf1 | 2,596 / 570 | 0.143 | 0.627, 0.120 | 0.559, 0.069 |
  | mf2+mf3 | 3,610 / 200 | 0.076 | 0.516, 0.183 | 0.345, 0.030 |
  | rng1 | 966 / 490 | 0.041 | 0.637, 0.135 | 0.574, 0.031 |

  A sub-group is worth calling the partner only when the partner's share in it passes the
  break-even of the two marginal gains: about 0.48 in drain1 and about 0.33 for fake_cashback in
  col1. The best tenth reaches 0.38 and 0.10. An AUC of 0.6 is not enough.

## 2. Rings and entities, for typing

`python analysis/farlook.py loyalty`. Events are the label-free events; an event's class is the
majority of its cleaned labels. For events of one look-alike pair that share an account, the
table gives how often the two events carry the same class, against the same count with the event
labels shuffled (200 draws).

| pair, shared account | event pairs | same class | shuffled |
|---|---|---|---|
| sim_swap / account_takeover, receiver | 941 | 0.498 | 0.499 ± 0.015 |
| fake_cashback / phishing_collect, receiver | 601 | 0.699 | 0.692 ± 0.015 |
| fake_cashback / phishing_collect, sender | 84 | 0.667 | 0.590 ± 0.046 |
| layering_chain / mule_account, sender | 402 | 0.619 | 0.572 ± 0.022 |
| layering_chain / mule_account, receiver | 511 | 0.620 | 0.565 ± 0.020 |
| qr_swap / refund_abuse, receiver | 754 | 0.882 | 0.847 ± 0.008 |
| mule_account / velocity_bot, sender | 568 | 0.790 | 0.522 ± 0.028 |
| mule_account / velocity_bot, receiver | 70 | 0.971 | 0.897 ± 0.020 |

- Drain beneficiaries serve sim_swap and account_takeover alike, so train history cannot type an
  October drain. The same holds for the collect pair.
- The mule / bot figure is bot bursts sharing the bot pool's accounts with each other. It does not
  reach the pool that matters: of the 19 bot-written events inside the mule-burst template, 1 has
  a sender from the bot pool (2,044 accounts), against 10 of 185 mule-written ones. October has
  42 such events and none has two bot-pool senders.
- Chain / mule and qr / refund show a few points of loyalty. That is lever 4: too weak to move a
  pool whose break-even is far from its base rate.

## 3. Ordinary-looking fraud: detection cannot pay without the type

Earlier work shows these rows cannot be found (AUC 0.50). The stronger statement is that finding
them would not help. A class is worth calling on a group of rows only when its share of the group
exceeds half the class's F1, which is 0.22 to 0.30 for every fraud class. Among ordinary rows the
largest class share is mule_account at 19.3 % (446 of 2,307 per 100,000 rows, earlier
measurement); the others are 3.7 to 13.5 %. Calling every ordinary October row mule_account, with
a perfect detector, would move mule F1 from about 0.50 to about 0.42 (arithmetic on the earlier
volume estimates, not a run).

So only signals that carry the **type** matter. The phone counter is the one known: it types the
3 % of ordinary rows on a new phone. Work aimed at ordinary rows (self-training on the held-out
month, entity embeddings, a retrieval member) can pay only through generator-made rows, where the
whole remaining stake is 0.0043.

Cells checked among the blend's legit calls (`python analysis/farlook.py cells`; base rate 2.31 %
cleaned fraud, 0.02 % generator-made, 1,538,790 rows):

| cell | train rows | called legit | cleaned fraud among those | October rows called legit |
|---|---|---|---|---|
| another account's home phone (listed as untested) | 75,518 | 74,567 | 2.28 % | 12,722 |
| own phone away from home | 97,634 | 96,374 | 2.44 % | 16,018 |
| phone first used today | 294 | 226 | 2.21 % | 46 |
| login within 1 s; SIM younger than 1 h | 267; 30 | 0; 0 | all already called fraud | 0; 0 |
| merchant not yet onboarded; account not yet open; sender pays itself; phone older than the account | 0; 0; 2; 3 | | no such rows to use | 0 |

## 4. Generator signatures

`python analysis/farlook.py generator`.

- **Transaction ids.** 1,824,642 ids, no duplicate, 17,084 distinct gaps between sorted ids, gap
  cv 1.000. A counter pushed through an affine map would leave at most three distinct gaps. The
  ids are a uniform draw without replacement and carry no generation order.
- **Time order inside the counter's segments.** If a class routine wrote its events in time
  order, a phone's number would date it and unused numbers would bracket hidden events. The rank
  correlation of phone number and first timestamp inside each train segment is between -0.17 and
  +0.09 (62 to 454 phones each). There is no order.
- **The two dead columns in October.** October's fraud-block rows are fraud-pass rows by
  construction, and legit-block rows are legit-pass rows.

  | October rows | rows | mean IP risk | IP risk > 0.3 | IP risk > 0.5 | mean app version |
  |---|---|---|---|---|---|
  | fraud block | 564 | 0.230 | 20.7 % | 0.7 % | 5.102 |
  | bot pool | 269 | 0.235 | 25.3 % | 1.5 % | 5.100 |
  | legit block | 7,462 | 0.221 | 19.1 % | 0.6 % | 5.099 |
  | home phones | 252,287 | 0.220 | 19.0 % | 0.5 % | 5.100 |

  In train the same split reads 26.5 % and 40.5 % against 1.8 % above 0.5. October keeps a
  likelihood ratio of at most 1.3, and by section 3 a pass detector without the type is worth
  nothing. The columns stay dropped.

## 5. Lever 1: the counter's segments at October's size

`python analysis/farlook.py thin`. The counter fits its eleven segments on each split's fraud
block from anchors alone. October's block has one sixth of train's phones, so its cuts rest on one
sixth of the anchors. We thinned train's block to that size (each phone kept with probability
1/6, 30 seeds), refitted the segments without labels, and compared with the full block's segments
on the same rows. The rows are fraud rows with no anchor of their own, whose class comes from the
segment.

| | full-block fit | thinned fit |
|---|---|---|
| agreement with the cleaned label, unanchored fraud rows (171 ± 9 per draw) | 0.942 ± 0.015 | **0.897 ± 0.028** |
| the same in the last four segments (30 ± 5 rows) | 0.896 ± 0.056 | **0.800 ± 0.094** |
| rows the thinned fit gets wrong that the full fit gets right | | 8.6 ± 5.1 (it wins back 0.9 ± 1.1) |
| unanchored rows with segment confidence below 0.9 | | 27 ± 6 |

October, counted only: 564 fraud-block rows, 213 without an anchor, 35 of those with confidence
below 0.9; the last three segments hold 31 rows and 13 are below 0.9. The blend's calls equal the
segment on 545 of the 564 rows.

**Size of the prize.** Scaled to October's 213 unanchored rows, the thinning loss is about 9 rows,
3 of them in the last four segments. From `public-score-noise.md`, a corrected October row is
worth 0.00012 to 0.00029 in refund_abuse, fake_cashback, loan_app_extortion and
merchant_collusion, and 0.00005 to 0.00007 in mule_account, velocity_bot and layering_chain. That
gives about +0.001 if all of the loss is recovered and +0.0005 as an expectation; 19 lost rows in
the worst draw gives the +0.002 upper end.

**The experiment.**
- *Inputs.* The ledger, `ringtrace.evidence.phones`, the anchors; nothing else.
- *Code change, in `phones.decode_phase`.* Add two terms to each phone's log-likelihood:
  1. a prior on where each cut falls on the number line: the ten cut positions of train's block as
     fractions of its length, each with a normal spread of 0.03 (October's markers sit about 0.016
     later, earlier measurement). Train's cuts are the ones `decode_phase` finds on the full
     block, so the prior is built by code from the train split and applied to the test split.
     It must stay soft: cutting October at train's positions alone misplaces 19 % of its phones
     (earlier measurement);
  2. a geometric emission for the count of unused numbers before each phone, with each segment's
     own rate, measured on train's block (occupancy about 0.53, 0.37, 0.43, 0.69, 0.26, 0.55, 0.33,
     0.51, 0.52, 0.52, 0.46 in segment order).
- *First gate, 20 s a run.* `python analysis/farlook.py thin` with the new fit, the prior built
  from the phones outside each draw. Pass: thinned agreement on unanchored rows rises by at least
  0.02 with the 30-seed mean difference more than two standard errors above 0, the last four
  segments do not fall, and the full block's agreement does not fall.
- *Second gate.* Rebuild the `phones` family and re-decode the final recipe with the rule
  `{"name": "counter", "trusted": 11, "min_conf": 0.9}` after `anchored_templates`, limited to
  rows with no anchor of their own (a new flag on the rule); no member is refitted. Run
  `python -m ringtrace.final table`. Leave-one-month-out must not lose in any month or target.
  The October-like cells hold the counter from the unthinned frame, so they cannot see this
  change: it is a base-rate component like `blank_merchant_qr`, and the first gate is its
  evidence.
- *Candidate.* Expect 5 to 25 October rows to change, all on fraud-block phones. Stop if more
  than 25 change or if any changed row has confidence 0.9 or more under the current fit.

## 6. The label regime, the operating point and probes

- **Operating point.** Not re-run here. `final-system.md` has fraud_scale 0.85 and 0.90 as ties,
  and an oracle prior gains 0.00005 (earlier measurement). Section 1 adds that the type split
  inside the pools is at its optimum too.
- **Why October scores 0.55 against 0.567 on the train months** (arithmetic on earlier volume
  estimates, not a run). With perfect detection a class's F1 is 2S / (2S + O) for S structured
  and O ordinary rows per 100,000. October's volumes move three classes: velocity_bot S 238 ->
  126 (F1 0.574 -> 0.417), merchant_collusion 55 -> 36 (0.455 -> 0.353), mule_account 243 -> 280
  (0.489 -> 0.525). That is -0.019 macro-F1, and 0.567 - 0.019 = 0.548 against 0.5499 public. The
  gap is October's class volumes, not a hidden failure of the model.
- **Probes.** None is worth a slot. The quantities a probe could read (a class's prevalence, a
  pool's partner share on the public rows) either do not change a decision or are known from
  train six times more precisely than 30 % of one October pool can give. The exception would be
  col7, whose margin is thin, but its flip already scores -0.002 to -0.004 a month on train.
- **The upload rule.** The final ranking scores the best public upload on the private rows. A
  better file counts only if it also beats 0.5499354 on the public rows. For files that differ on
  tens of rows the public gap has an sd of 0.0002 to 0.0005, so a true +0.0005 wins the public
  comparison about nine times in ten and a true +0.0001 about six in ten. A file that is worse in
  expectation should not be uploaded as a probe: when it wins by chance it becomes the final file.

**Commands.** `RINGTRACE_THREADS=6 python analysis/farlook.py all` (21 s on 6 threads; reads run
`blend`; writes `$RINGTRACE_OUT/farlook/*.json`). Helpers in `analysis/farlook_core.py`, tested in
`tests/test_farlook_core.py`.
