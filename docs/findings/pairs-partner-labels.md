# Look-alike pairs: the partner label is the routine, and a small takeover flip

**Question.** Inside a look-alike template, are the minority labels "filed under the wrong type"
noise (so the clean October label is the template's own class), or events that the partner
routine wrote with the template (so the clean label keeps them)? If they are real, can anything
in the event reveal them?

**Answer (train rows; October as counts only).** The partner labels are real. Three independent
checks agree:

1. **The device counter.** A phone from the fraud block is minted in the turn of the routine that
   wrote the row, before any label noise. In every template, the events that carry such a phone
   sit in the zone of their own label:

   | template | label | events (rows for collect) with a fraud-block phone | in the zone of |
   |---|---|---|---|
   | own-phone drain (shape 1: 0.25 steps, fresh SIM) | account_takeover | 10 | account_takeover (10 of 10) |
   | | sim_swap | 36 | sim_swap (36 of 36) |
   | new-phone drain (shape 3: 0.35 steps) | account_takeover | 356 | account_takeover (356 of 356) |
   | | sim_swap | 156 | sim_swap (156 of 156) |
   | one bait answered within 4 h (collect shape 7) | phishing_collect | 14 rows | phishing_collect (14 of 14) |
   | | fake_cashback | 9 rows | fake_cashback (9 of 9) |
   | just-under collect (collect shape 1) | phishing_collect | 44 rows | phishing_collect (44 of 44) |
   | | fake_cashback | 1 row | fake_cashback |

   If the minority labels were wrong-type noise, the 10 takeover-labelled own-phone events would
   sit in the sim_swap zone.
2. **Parameters set at generation time.** In the own-phone drain, the takeover-labelled events
   are shorter than the sim_swap-labelled ones, and they always fit the template:

   | label | events | 2 / 3 / 4 rows | events with a login above 25 s or an old payee |
   |---|---|---|---|
   | sim_swap | 375 | 130 / 125 / 120 | 34 |
   | account_takeover | 166 | 83 / 62 / 21 | 0 |

   The partner side of the new-phone drain is short in the same way (sim_swap 63 / 67 / 26
   against takeover 129 / 105 / 122). A label flip applied after generation cannot change the
   length or the payee of an event.
3. **The noise is per row.** In collect shape 7, the bait and its answer carry the same fraud
   type in 184 of 186 events where both are phishing_collect or fake_cashback. The 12 %
   unreported and the 2 to 3 % random wrong-type labels are independent row flips.

So the clean October labels keep the partner types. The template-clean target, on which the
"retype every sim / takeover call by template" file gained, is not the October truth. That file
loses about 0.015 on the given and cleaned targets in every train month, and the same loss is the
expected October effect. It stays rejected.

**The generator's apparent rule per pair.**

| pair | own template, own routine | partner share of the template's events | where the partner shows |
|---|---|---|---|
| sim_swap / account_takeover | own-phone drain (sim_swap); new-phone drain (takeover) | 30.7 % takeover in the own-phone drain (166 of 541); 30.5 % sim_swap in the new-phone drain (156 of 512) | shorter events, strict template |
| phishing_collect / fake_cashback | just-under collect, unanswered baits of Rs 1 / 2 / 5 (phishing); one or two baits with a payback (cashback) | 9 % cashback and 4 % loan_app_extortion among col1 fraud rows (155 and 70 of 1,743); phishing writes 151 of 258 one-bait fast-answer events (col7) | cashback writes almost no unanswered bait (2 of 561 rows); phishing writes no answer after 4 h (0 of 90 rows) and almost no two-bait answer (3 of 492 rows) |
| loan_app_extortion | three or more evening payments to lenders | partner rows in col1 (70, about ten times the uniform wrong-type rate) | none found |

**Do our calls follow the rule?** Yes. The best file calls 98 % of own-phone drain rows sim_swap
(October: 248 sim_swap, 3 account_takeover), types the new-phone drain by the counter zone (it
agrees with the given label event by event), calls col1 phishing_collect and col7 fake_cashback
(the template rule; col7 is about 59 % phishing_collect, but cashback has the lower F1).

**Separability inside the templates.** An event-level logistic model, out of fold by month:
- own-phone drain, 13 label-free features (size, login and payee breaks, phone, PIN share, login,
  step gap, night share, balance, account-age error, SIM age): AUC 0.639 / 0.642 / 0.681;
- col7, 12 features (bait amount, payback type and size, 1.1 x balance, delay, phones, login,
  PIN): AUC 0.49 to 0.57, nothing. cashback's col7 baits are Rs 5 on 42 % of events against
  25 % for phishing, but "col7 bait other than Rs 5 -> phishing_collect" gives
  -0.0002 / -0.0005 / -0.0006 on cleaned (+0.0020 to +0.0023 on gen-only): rejected;
- generic event aggregates on all six templates: only the own-phone drain shows anything;
- account attributes inside the own-phone drain (sender bank, OS, app, KYC), target-encoded out of
  fold: AUC 0.46 to 0.55. The bank-by-bank agreement of our calls in the new-phone drain comes
  from the counter, not the bank (bank AUC 0.50 to 0.52 there);
- chain / mule hops (mf1): the zone agrees with the label again (chain 59 of 59 events, mule 8 of
  8); mule-labelled chains are faster but have the same sizes, so nothing to flip.

**Does the October mix differ?** Not where the counter can see it: in the new-phone drain, 31.1 %
of October rows are called sim_swap by the zone (76 of 244) against 28.9 % of train rows
labelled sim_swap (431 of 1,492).

**The flip (`pairs-tmplA-flip`).** In the own-phone drain, the sim_swap calls of an event whose
takeover score is at least 0.57 (C = 1.0) become account_takeover, unless the event has a phone
in the counter's sim_swap zone. Score and threshold were fixed before the October build.

| target | Jul | Aug | Sep |
|---|---|---|---|
| given | +0.0007 | +0.0002 | +0.0007 |
| cleaned | +0.0007 | +0.0002 | +0.0006 |
| gen-only | +0.0012 | +0.0004 | +0.0010 |

- Train: 21 events flipped (75 % takeover-labelled), 40 rows.
- Event bootstrap (400 Poisson draws over event groups, mean of the months): given +0.00055
  [+0.00015, +0.00106], cleaned +0.00054 [+0.00009, +0.00102].
- Threshold chosen on the other two months for each held-out month (nested, thresholds 0.45 to
  0.65, without the zone guard): given +0.0005 / +0.0002 / +0.0004, cleaned +0.0006 / +0.0002 / +0.0003.
- Every threshold from 0.54 to 0.58 passes for C = 0.1, 0.3 and 1.0. Below 0.52 it loses in
  August or September.
- October: 3 events, 6 rows, sim_swap -> account_takeover. sha256
  `34700878cb399b19cf9a04b4065253a3b060ac8c21a6b21c0c72ff28e60da64f`.

**Verdict.**
- The pairs do not explain any score above our ceiling. The 0.598 ceiling scores the gen-only
  labels, which already type every look-alike event right, so pair typing cannot lift a file
  above it. The gap to a higher score must come from ordinary-looking rows or from October's
  class mix.
- `pairs-tmplA-flip` passes the train-month bar, but it is small: about +0.0005 a month, 6
  October rows. Expected public change: about +0.0002, inside the noise of one file. Fold it into
  a combined upload; it touches only sim_swap / account_takeover rows of the own-phone drain.
- Stop work on col7, col1 and loan_app_extortion inside col1: at chance.

**Commands.**
`python analysis/pairs.py evidence` (zone and size tables),
`python analysis/pairs.py validate --prep $RINGTRACE_OUT/closelook/prep.npz` (train months; the
prep file holds the best file's out-of-fold calls in ledger order),
`python analysis/pairs.py build --best $RINGTRACE_OUT/candidates/blend-tabmF-bq.csv`.
