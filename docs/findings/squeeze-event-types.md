# One fraud type per label-free event, on top of the blend

**Question.** Does giving every fraud-called row of a label-free event the event's
probability-weighted type add anything on top of the five-member blend?

**Method.** Rule `event_types` (`src/ringtrace/rules.py`). An event is the first evidence family
(bursts, drains, flows, requests, merchant, repeats) that places the row in an event, with no
labels used. For each event, the fraud-type probabilities are summed over its fraud-called rows;
every one of those rows takes the argmax. Legit calls are never touched. The knowledge base
describes the simplest form as unifying over events before the template rule, so the primary
variant runs between the fixed point and `anchored_templates`. A variant after the rule was also
measured.

**Answer.** Leave-one-month-out against the blend:

| variant | given Jul / Aug / Sep | cleaned Jul / Aug / Sep | gen-only Jul / Aug / Sep | 95 % event bootstrap, given / cleaned | rows changed, train / October |
|---|---|---|---|---|---|
| before the rule | +0.0003 / +0.0001 / +0.0000 | +0.0005 / +0.0001 / +0.0000 | +0.0007 / +0.0002 / +0.0000 | +0.0002 [+0.0000, +0.0004] / +0.0002 [+0.0000, +0.0004] | 7 / 0 |
| after the rule | -0.0035 / -0.0040 / -0.0037 | -0.0037 / -0.0043 / -0.0054 | -0.0033 / -0.0053 / -0.0063 | -0.0037 [-0.0057, -0.0017] / -0.0045 [-0.0066, -0.0023] | 639 / 132 |

October-like, on the champion's cells (the other members' cells did not exist yet):
- before the rule: per seed given +0.0001 / +0.0001 / +0.0000, cleaned +0.0001 / +0.0001 /
  +0.0000; lower bound +0.0000. FAIL.
- after the rule: given -0.0052 / -0.0008 / -0.0050, cleaned -0.0076 / -0.0032 / -0.0058. FAIL.

**Verdict.** Rejected: a tie.
- Before the rule, it never loses on leave-one-month-out, but it is not positive on every thinning
  seed, and its lower bound is not above 0. It also changes no October row, so it cannot move the
  file.
- After the rule, it undoes the anchored template classes (fake_cashback -> phishing_collect on 89
  October rows) and loses everywhere.

The earlier +0.0009 was measured on a weaker base, before the template rule. As the knowledge
base already noted, the template rule now does this job.
