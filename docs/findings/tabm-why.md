# Why TabM beats the champion: generator rows the trees call legit

**Question.** TabM alone beats the XGBoost champion on every October-like seed (r0: given
+0.0002 / +0.0004 / +0.0006, cleaned +0.0006 / +0.0010 / +0.0005). Which rows carry that gain?

**Answer.** Recovered generator rows: TabM calls fewer generator rows legit (pocket b). Typing
inside the look-alike templates (pocket c) is unchanged, so that pocket stays closed. Smoothing
the champion's probabilities does not reproduce the gain, so it is not a decoder or calibration
side effect.

## Setup

- `python -m ringtrace.tabmwhy` (commit 161757a; laptop CPU, about 25 min) compares the champion
  with `tabm`, writing `$RINGTRACE_OUT/tabmwhy/tabm_vs_champion.json`.
- **Inputs:** the r0 October-like cells (s13, s14, s15; both models use seeds 0-2) and the
  leave-one-month-out runs `runs/champion` and `runs/tabm`.
- **Decoding:** as in the gate (fixed point at 0.875, then the anchored template rule).
- **Pocket definitions** (as in the blend's error anatomy):
  - generator row = gen-only label is fraud;
  - (b) generator row called legit;
  - (c) generator row called another fraud type than its cleaned label;
  - (d) label legit, called fraud.

## Per-class F1 change, TabM minus champion

October-like is the mean of the three r0 cells; leave-one-month-out is Jul / Aug / Sep.

| class | given oct-like | given Jul / Aug / Sep | cleaned oct-like | cleaned Jul / Aug / Sep |
|---|---|---|---|---|
| legit | 0.0000 | 0 / 0 / 0 | 0.0000 | 0 / 0 / 0 |
| mule_account | +0.0003 | +0.0001 / +0.0003 / +0.0003 | +0.0002 | +0.0001 / +0.0003 / +0.0003 |
| velocity_bot | +0.0005 | +0.0003 / 0 / +0.0009 | +0.0019 | +0.0003 / 0 / +0.0018 |
| layering_chain | +0.0003 | 0 / 0 / +0.0004 | +0.0003 | 0 / 0 / +0.0004 |
| sim_swap | +0.0001 | +0.0009 / +0.0005 / +0.0008 | +0.0014 | +0.0007 / +0.0004 / +0.0007 |
| account_takeover | 0.0000 | +0.0010 / +0.0009 / 0 | 0.0000 | +0.0010 / +0.0008 / 0 |
| phishing_collect | +0.0010 | +0.0034 / +0.0008 / +0.0007 | +0.0010 | +0.0025 / +0.0011 / +0.0011 |
| qr_swap | +0.0003 | -0.0013 / -0.0007 / +0.0001 | +0.0003 | -0.0006 / -0.0006 / +0.0002 |
| refund_abuse | +0.0012 | +0.0004 / -0.0033 / +0.0016 | +0.0010 | +0.0002 / -0.0035 / +0.0015 |
| fake_cashback | +0.0014 | +0.0006 / 0 / +0.0013 | +0.0013 | +0.0005 / 0 / +0.0012 |
| loan_app_extortion | -0.0006 | 0 / 0 / 0 | +0.0008 | 0 / 0 / 0 |
| merchant_collusion | 0.0000 | 0 / -0.0055 / 0 | 0.0000 | 0 / -0.0036 / 0 |

The gain is small and spread over the bot, collect, cashback and SIM classes. The losses sit in
the merchant pair (qr_swap / refund_abuse / merchant_collusion), and in August only.

## Error pockets (rows, champion -> TabM)

| pocket | oct-like (3 cells) given | oct-like cleaned | Jul-Sep given | Jul-Sep cleaned |
|---|---|---|---|---|
| (b) generator rows called legit | 135 -> 107 | 135 -> 107 | 285 -> 249 | 285 -> 249 |
| (c) generator rows, wrong type | 1104 -> 1108 | 906 -> 908 | 2112 -> 2122 | 1722 -> 1731 |
| (d) legit called fraud | 1478 -> 1505 | 68 -> 89 | 3066 -> 3086 | 178 -> 194 |

Pocket (c) by template, cleaned, October-like (champion / TabM):

| template pair | champion | TabM |
|---|---|---|
| drain1 account_takeover -> sim_swap | 183 | 185 |
| col7 phishing_collect -> fake_cashback | 171 | 171 |
| mf2+mf3 velocity_bot -> mule_account | 188 | 188 |
| mf1 mule_account -> layering_chain | 177 | 177 |
| col1 fake_cashback -> phishing_collect | 73 | 73 |
| rng1 refund_abuse -> qr_swap | 29 | 29 |

After the template rule, the template pockets are identical: the rule overwrites both models'
type calls there. TabM's raw top type inside templates is better on col7 (0.53 vs 0.50), mf1
(0.85 vs 0.82) and drain1 on Jul-Sep (0.72 vs 0.69), but worse on mf2+mf3 (0.77 vs 0.87). None of
that reaches the decoded calls.

## Disagreements (decoded calls differ; who matches the cleaned label)

| (champion call -> TabM call) | oct-like n | champion right | TabM right | Jul-Sep n | champion right | TabM right |
|---|---|---|---|---|---|---|
| legit -> phishing_collect | 33 | 21 | 11 | 40 | 14 | 23 |
| legit -> velocity_bot | 7 | 0 | 7 | 5 | 0 | 5 |
| legit -> sim_swap | 5 | 0 | 3 | 7 | 0 | 3 |
| legit -> fake_cashback | 3 | 0 | 2 | 3 | 0 | 2 |
| legit -> refund_abuse | 3 | 1 | 2 | 11 | 6 | 4 |
| phishing_collect -> legit | 1 | 0 | 1 | 9 | 6 | 2 |
| qr_swap -> refund_abuse | 2 | 0 | 1 | 5 | 1 | 4 |
| merchant_collusion -> refund_abuse | 0 | - | - | 4 | 4 | 0 |
| refund_abuse -> qr_swap | 0 | - | - | 3 | 3 | 0 |
| **all** | **59** | **23** | **31** | **106** | **41** | **55** |

By template, 52 of the 59 October-like disagreements are on rows outside every template.

- **Legit -> fraud adds, outside the collect class:** TabM is almost always right (bot 7 of 7,
  SIM 3 of 5). These rows carry generator marks that the champion's trees score as legit.
- **legit -> phishing_collect:** a coin flip that flips sign. On October-like the champion is
  right 21 to 11; on Jul-Sep TabM is right 23 to 14.
- **Merchant-pair type switches:** the losing part (Aug only).

## Calibration of P(fraud) = 1 - P(legit), cleaned labels

| bin | champion n (oct-like) | mean p | fraud rate | TabM n | mean p | fraud rate |
|---|---|---|---|---|---|---|
| [0, 0.01) | 743,921 | 0.000 | 0.023 | 22,222 | 0.001 | 0.001 |
| [0.01, 0.05) | 259 | 0.021 | 0.120 | 710,451 | 0.033 | 0.023 |
| [0.05, 0.1) | 64 | 0.068 | 0.250 | 11,140 | 0.062 | 0.031 |
| [0.1, 0.5) | 67 | - | 0.26-0.59 | 484 | - | 0.08-0.57 |
| [0.5, 0.95) | 191 | - | 0.83-0.97 | 11,399 | - | 0.87-0.995 |
| [0.95, 1] | 11,921 | 0.998 | 0.995 | 727 | 0.962 | 0.990 |

- **TabM is smooth.**
  - It puts about 0.03 on every ordinary row: the base rate of the hidden fraud the labels
    hold, which no feature separates.
  - It caps confident rows near 0.89.
  - TabM's mass between 0.05 and 0.95: 23,023 rows and 10,912 of its 35,301 fraud mass
    (October-like, 3 cells). For the champion it is 322 rows and 184 of 12,098.
- **The champion is sharp:** 0 or 1, and under-confident in its thin middle (rate above mean p).
- **Brier:** TabM 0.02237 vs 0.02261 (October-like), 0.02244 vs 0.02278 (Jul-Sep).

**The calibration difference is not the mechanism.** Smoothing the champion toward its own class
prior, p' = (1 - e) p + e * mean(p) for e in {0.01, 0.02, 0.03, 0.05}, changes macro-F1 by at most
±0.0002 in all 9 champion cells (given, cleaned and gen-only); most cells do not move at all.
The decoder is indifferent to a uniform mass shift, so TabM's gain comes from its ranking:
which rows sit at the margin.

## Conclusion

- **Mechanism:** TabM's gain is legit -> fraud adds of generator rows that the trees miss.
  - Pocket (b) shrinks by 21 % on October-like (135 -> 107) and 13 % on Jul-Sep (285 -> 249).
  - The adds land on the bot, SIM, cashback and collect classes, mostly outside templates.
- **Typing inside look-alike templates is not the mechanism.** Pocket (c) is flat
  (906 -> 908 cleaned), because the template rule decides those rows for both models.
- **Costs:**
  - a few more legit rows called fraud: pocket (d) 68 -> 89 cleaned on October-like;
  - wrong merchant-pair type switches in August.
- **Consequence:** this agrees with the earlier decomposition (adds right, type switches a coin
  flip). TabM's useful signal is P(fraud), not P(type | fraud).
