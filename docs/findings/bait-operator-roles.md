# Unanswered baits and first-time payments to operators: two small phishing rules that pass

**Question.** The phishing routine sends unanswered baits of Rs 1, 2, 5 and 10. The best file
(`blend-tabmF-bq`) calls every unanswered Rs 1 / 2 / 5 bait, but a lone unanswered Rs 10 is legit
on about 93 % of train rows. Can a label-free view of the sender, "is this account an operator in
another month", recover the phishing ones? And does a first-time P2P PAY of Rs 1,000 or more in
whole rupees, not a multiple of ten, become phishing_collect?

**Answer.** Yes for the bait, by a small margin; no for the PAY rule on its own; yes, with a wide
margin, for the PAY rule restricted to receivers who are operators in another month. Together
they move 7 October rows (legit -> phishing_collect), worth about +0.0003 on October.

## 1. What the best file does with small P2P baits

`python analysis/bait.py measure`. Bait sessions of the requests family (same sender and
receiver, P2P PAY of exactly Rs 1 / 2 / 5 / 10, gaps of at most 10 minutes; the answer is the
first P2P row back within 72 h). Train columns are summed over July to September.

| session | train rows | cleaned phishing | called fraud | called legit and cleaned phishing | October rows | October called fraud |
|---|---|---|---|---|---|---|
| N1: one unanswered Rs 1 / 2 / 5 | 245 | 245 | 245 | 0 | 33 | 33 |
| N1X: one unanswered Rs 10 | 1,149 | 67 (+ 19 other fraud) | 44 | 31 | 154 | 12 |
| N2: two or more unanswered | 316 | 310 | 316 | 0 | 50 | 50 |
| A1S: one bait answered within 4 h | 268 | 155 (111 cashback) | 268 | 0 | 48 | 48 |
| A1L: answered in 4 to 72 h | 45 | 0 (44 cashback) | 45 | 0 | 10 | 10 |
| A2: two or more, answered | 328 | 2 (325 cashback) | 328 | 0 | 56 | 56 |

- The only bait rows left on the table are lone unanswered Rs 10s: 10 / 8 / 13 cleaned
  phishing rows called legit in July / August / September, among 389 / 349 / 367 legit calls.
- Small P2P PAYs of Rs 1 to 10 that are not exact bait amounts (157 train rows, 4 fraud on the
  cleaned labels; 18 October rows) are never called and never carry a role, so the Rs 1 to 10
  band adds nothing to the Rs 10 rule (`roles-band` is identical to `roles`).

## 2. Operator roles

`ringtrace.baitroles.operator_roles` lists, per account and month, six label-free acts, each
fraud on more than 97 % of train rows (cleaned): receives a QR payment addressed to a person;
receives a just-under-limit COLLECT; sends a bait in a recognised session (every shape but the
lone Rs 10); receives a session's answer; sends from a bot-pool phone; pays a merchant a ring
amount inside a merchant ring. The fraud side draws its operators from one pool for all months,
so an operator in one month is likely an operator in another.

Senders with a role in another train month (train rows):

| senders of | rows | with a role | share |
|---|---|---|---|
| recognised baits (N1, N2, A1S, A1L, A2) | 1,202 | 732 | 0.61 |
| lone Rs 10, cleaned phishing | 67 | 36 | 0.54 |
| lone Rs 10, cleaned legit | 1,063 | 13 | 0.012 |

A role multiplies the odds of phishing by about 44. Per role, the lone-Rs-10 legit senders carry
qr 4, collect 5, bait 0, answer 0, bot 7, round 1 roles.

**Asymmetry.** A held-out train month takes its roles from the other two train months, October
from all three; so October has about 1.5 times the history. With only August and September,
October's bait rule would move 3 rows instead of 4, and the PAY rule the same 3. When every month
sees the same four months (July to October, own month included: `roles-all4`,
`pay-roles-all4`), both rules still pass and the October rows are exactly the same (same sha256).

## 3. The rules on the train months

`python analysis/bait.py validate`. Base: the best file's out-of-fold calls. Change of macro-F1
per held-out month (Jul / Aug / Sep); moved rows right / moved against phishing_collect's
break-even (half its base F1 in that month); paired event bootstrap of the mean change, 2,000
draws, 95 %.

| rule | given | cleaned | gen-only | cleaned right / moved (break-even) | bootstrap given; cleaned | rows train / Oct | verdict |
|---|---|---|---|---|---|---|---|
| `roles`: lone Rs 10, sender is an operator in another month | +0.00011 / +0.00001 / +0.00010 | +0.00009 / +0.00000 / +0.00008 | +0.00007 / -0.00017 / -0.00001 | 4/7 (0.286), 2/7 (0.275), 3/5 (0.268) | +0.00007 [+0.00000, +0.00015]; +0.00006 [-0.00001, +0.00014] | 19 / 4 | passes the months, narrowly |
| `pay`: first-time P2P PAY >= Rs 1,000, whole, not a multiple of ten | +0.00025 / +0.00007 / +0.00001 | +0.00020 / +0.00004 / -0.00002 | -0.00007 / -0.00019 / -0.00025 | 12/27 (0.286), 6/19 (0.275), 4/16 (0.268) | +0.00011 [-0.00002, +0.00025]; +0.00007 [-0.00005, +0.00020] | 62 / 7 | fails (September) |
| `pay-roles`: the same, receiver is an operator in another month | +0.00022 / +0.00018 / +0.00008 | +0.00020 / +0.00016 / +0.00007 | +0.00018 / +0.00019 / +0.00010 | 6/6, 5/6, 2/2 | +0.00016 [+0.00008, +0.00026]; +0.00015 [+0.00007, +0.00024] | 14 / 3 | passes |
| `combo`: `roles` + `pay-roles` | +0.00033 / +0.00019 / +0.00018 | +0.00029 / +0.00017 / +0.00016 | +0.00024 / +0.00002 / +0.00008 | 10/13, 7/13, 5/7 | +0.00023 [+0.00012, +0.00037]; +0.00021 [+0.00010, +0.00033] | 33 / 7 | passes |
| `roles-all4` (four-month view) | +0.00015 / +0.00012 / +0.00014 | +0.00013 / +0.00010 / +0.00012 | +0.00003 / -0.00007 / +0.00003 | 5/8, 5/11, 4/6 | +0.00013 [+0.00004, +0.00023]; +0.00011 [+0.00003, +0.00020] | 25 / 4 | passes |
| `pay-roles-all4` | +0.00029 / +0.00018 / +0.00012 | +0.00027 / +0.00016 / +0.00011 | +0.00026 / +0.00019 / +0.00015 | 8/8, 5/6, 3/3 | +0.00020 [+0.00011, +0.00031]; +0.00018 [+0.00010, +0.00028] | 17 / 3 | passes |
| control `history`: sender sent a fraud-labelled row in another train month | -0.00090 / -0.00063 / -0.00064 | -0.00101 / -0.00065 / -0.00072 | | 6/98, 4/64, 4/68 | -0.00072; -0.00079 | 230 / 29 | fails |

- The label-history control fails at a precision of 6 %: fraud labels mark victims and
  ordinary-looking rows as much as operators. The structural roles are what carry the signal.
- Per role (bait rule, train moved / right): qr 8 / 5, collect 6 / 2, bait 1 / 1, answer 1 / 1,
  bot 6 / 1, round 5 / 4. PAY rule (receiver): qr 5 / 5, collect 4 / 4, bait 2 / 1, answer 1 / 1,
  bot 3 / 3, round 7 / 7. Not tuned: every rule uses all six roles.
- The PAY rule at Rs 500 moves the same rows as at Rs 1,000; at Rs 100 it adds 4 wrong rows.
- The A-clean proxy (cleaned labels with the expected unreported rows spread over the cleaned
  legit rows) agrees with cleaned: `roles` +0.00009 / +0.00001 / +0.00008, `pay` +0.00020 /
  +0.00004 / -0.00001, `pay-roles` +0.00020 / +0.00016 / +0.00007, `combo` +0.00029 / +0.00017 /
  +0.00016.
- The bait rule's August cleaned change is +0.0000034 (2 of 7 right against 0.275): it passes the
  months by one row. Its gen-only target loses in August and September: some of the cleaned
  phishing lone Rs 10s are judged ordinary-looking there.

## 4. October density

`python analysis/bait.py octlike`: the best recipe re-decoded in the nine October-like cells
(September thinned to October's density by protected units, every member fitted on July +
August; seeds 13 to 15 x seed sets r0 to r2), each rule applied on top. September takes its
roles from July and August. Mean change over the cells, the pooled 95 % lower bound of the
paired unit bootstrap (2,000 draws), the mean per thinning seed, and the moved rows right /
moved summed over the cells (break-even per cell 0.262 to 0.273 on cleaned).

| rule | given mean (lb) | cleaned mean (lb) | cleaned per seed s13 / s14 / s15 | cleaned right / moved | verdict |
|---|---|---|---|---|---|
| `roles` | +0.00013 (+0.00001) | +0.00011 (+0.00001) | -0.00005 / +0.00022 / +0.00017 | 17 / 26 | passes (s13 keeps only its two wrong rows) |
| `pay` | +0.00002 (-0.00012) | -0.00001 (-0.00015) | +0.00004 / +0.00001 / -0.00008 | 18 / 71 | fails |
| `pay-roles` | +0.00008 (-0.00001) | +0.00007 (-0.00001) | +0.00008 / +0.00008 / +0.00006 | 9 / 11 | positive in every cell; the bound misses by 0.000006 (one or two rows per cell) |
| **`combo`** | **+0.00020 (+0.00006)** | **+0.00018 (+0.00005)** | +0.00003 / +0.00030 / +0.00022 | 26 / 37 | **passes** (every seed positive on given and cleaned) |
| `roles-all4` | +0.00015 (+0.00003) | +0.00014 (+0.00002) | -0.00005 / +0.00022 / +0.00024 | 20 / 29 | passes |
| `pay-roles-all4` | +0.00010 (+0.00001) | +0.00009 (+0.00000) | +0.00008 / +0.00008 / +0.00013 | 12 / 14 | passes |

Per cell, `combo` moves 2 to 6 rows; its precision is 1/2, 1/3 and 1/3 in the s13 cells and
0.67 to 1.0 elsewhere, above the break-even in every cell.

`python analysis/bait.py thinned --seeds 101-120` repeats the thinning with 20 fresh seeds on the
out-of-fold calls (no refit; same units, same keep rate):

| rule | given mean (lb) | cleaned mean (lb) | seeds positive (cleaned) | cleaned right / moved |
|---|---|---|---|---|
| `roles` | +0.00009 (+0.00005) | +0.00008 (+0.00004) | 14 of 20 | 28 / 47 |
| `pay` | +0.00001 (-0.00004) | -0.00001 (-0.00007) | 9 of 20 | 40 / 160 |
| `pay-roles` | +0.00009 (+0.00005) | +0.00008 (+0.00005) | 18 of 20 (the other two keep no moved row) | 21 / 21 |
| **`combo`** | **+0.00018 (+0.00012)** | **+0.00016 (+0.00011)** | 19 of 20 | 49 / 68 |

## 5. Candidates (on top of `blend-tabmF-bq.csv`; October as counts only)

`python analysis/bait.py build <variant>`; every file has 260,582 rows in the sample's id order
with labels from the 12 classes.

| file | sha256 | rows changed | moves | bar |
|---|---|---|---|---|
| **`bait-combo.csv`** | `3b25869e1cd7ac3e390dd33568033d25d3b765b23f0e1f35dbbf3f2dff5e1974` | 7 | legit -> phishing_collect 7 | passes: months (given, cleaned, gen), train bootstrap, nine cells, fresh seeds |
| `bait-roles.csv` | `a6b55dd09914a2f18f098bd7644942959ddeaf574609de844ee790d54a7a8954` | 4 | legit -> phishing_collect 4 | passes the months by one row in August and the cells' bound by +0.00001 |
| `bait-pay-roles.csv` | `6171470da676dec2f009913120f4aa038c66dd9c6f438aa4f35ea3fe1d13f2ee` | 3 | legit -> phishing_collect 3 | passes the months; nine-cell bound -0.000006, fresh-seed bound +0.00005 |

`bait-combo` is the union of the other two (4 + 3 rows, disjoint). The four-month variants give
the same files byte for byte. The plain PAY rule fails and has no file.

The rules compose with other components: on `pairs-tmplA-flip.csv` the same 7 rows move; on
`final.csv` and `blend-tabm-bq.csv` 8 (one more legit call qualifies there). Rebuild on any base
with `python analysis/bait.py build combo --best <file> --out <file>`.

**Expected effect.** In the cells, `combo` moves 4.1 rows per cell for +0.00018 (cleaned); October
moves 7, so about +0.0003 on October, and the same on the private 70 % in expectation. About two
of the seven rows fall in the public 30 %, so the public score cannot confirm it: fold it into a
combined upload rather than spend an upload on it. It touches only legit calls of lone Rs 10
baits and first-time whole-rupee PAYs, which no other component changes.

**Verdict.** `bait-combo` passes the bar. Its precision in train (22 of 33 rows, against a
break-even of 0.27 to 0.29) and in the cells (26 of 37) carries it. The label-history control
and the plain PAY rule fail. A scan of every legit call with an operator on either side (22,260
train rows, cleaned fraud 2.7 %, the base rate; cells by type, receiver kind, payee, amount
grid and size) finds no third pattern. Only two cells with five or more train rows exceed 0.25:
the first-time PAY to an operator (13 of 14) and QR payments to merchants by operator senders
(3 of 5, no October row).

**Commands.**
`python analysis/bait.py measure` (5 s), `validate [variants]` (5 min, mostly the bootstrap),
`octlike [variants]` (8 min, nine cells), `thinned [variants] --seeds 101-120`,
`build combo` (10 s; writes `candidates/bait-combo.csv` and its JSON report). The rules are
`bait_roles` and `first_pay` in `ringtrace.rules`; `recipes/bait-combo.json` is the best recipe
with both added after the anchored template rule.
