# The blind-spot rule: anchored rows called legit

**Question.** `validation-blind-spot.md` shows that the rows the champion calls legit despite
generator evidence look generator-made. Does the smallest label-free rule, which calls them fraud
with the class their evidence implies, gain on top of the blend?

**Method.** Rule `anchor_rescue` (`src/ringtrace/rules.py`), run after `anchored_templates`. A row
called legit whose family shape or one-class mark anchors a class (`anchors.row_anchor`) takes
that class. Fraud-block phones were left out: their fraud-pass share was 21 % (0-100) in the
finding.

**Answer.** It changes 14 train rows and 1 October row (legit -> sim_swap).

| target | Jul / Aug / Sep | 95 % event bootstrap |
|---|---|---|
| given | +0.00005 / -0.00002 / +0.00009 | +0.00004 [-0.00003, +0.00012] |
| cleaned | +0.00004 / -0.00002 / +0.00016 | +0.00006 [-0.00003, +0.00018] |
| gen-only | -0.00001 / -0.00005 / +0.00015 | +0.00003 [-0.00012, +0.00021] |

October-like, on the champion's cells, per seed:
- given +0.0000 / +0.0001 / -0.0000;
- cleaned +0.0002 / +0.0004 / -0.0000;
- lower bound -0.0001.

FAIL.

**Verdict.** Rejected. It loses in August on every target and on thinning seed 15. On the blend
the blind spot is only about 5 rows a month, and they are not all fraud.
