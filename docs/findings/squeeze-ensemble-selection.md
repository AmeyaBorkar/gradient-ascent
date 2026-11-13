# Nested ensemble selection over the five members

**Question.** Does greedy forward selection with replacement over the members (Caruana et al.,
2004), or rank averaging, beat the equal-weight probability blend?

**Method.** `python -m ringtrace.squeeze select`.
- In each fold, the selection is scored on the two training months only: given labels, decoded as
  the blend is (fixed point 0.875 per month, then the anchored template rule). The held-out month
  is then scored.
- Selection stops when no added copy improves, and ties go to the earlier member.
- October's weights are selected on all three months.
- Rank averaging takes per-month, per-class ranks and maps them back onto the probability mean's
  own values.

Member order: champion, lgbm, catboost, clean-event, clean-l2.

**Answer.** Copies chosen:

| form | fold Jul | fold Aug | fold Sep | October |
|---|---|---|---|---|
| greedy, probability | 1,1,1,0,0 | 0,1,0,1,0 | 0,1,0,2,0 | 0,1,0,1,0 |
| greedy, rank | 0,2,1,0,1 | 0,0,1,1,0 | 0,1,0,1,0 | 0,0,1,1,0 |

Changes against the blend, Jul / Aug / Sep:

| form | given | cleaned | gen-only | bootstrap, given | rows, train / October |
|---|---|---|---|---|---|
| greedy, probability | +0.00011 / +0.00005 / -0.00009 | +0.00011 / +0.00004 / -0.00004 | +0.00017 / +0.00003 / -0.00003 | +0.00002 [-0.00008, +0.00015] | 18 / 5 |
| greedy, rank | +0.00006 / +0.00006 / -0.00008 | +0.00006 / +0.00006 / -0.00003 | +0.00034 / +0.00012 / +0.00001 | +0.00001 [-0.00012, +0.00015] | 16 / 2 |
| equal-weight rank, no selection | +0.00017 / +0.00008 / +0.00021 | +0.00016 / +0.00007 / +0.00020 | +0.00024 / +0.00010 / -0.00010 | +0.00015 [+0.00003, +0.00030] | 8 / 2 |

October-like was not run: the other four members' cells did not exist during this job, and the
gate's cell blends average probabilities only.

**Verdict.** All three are rejected.
- The selected weights change from fold to fold, which is the signature of noise, and both nested
  forms lose in September.
- The equal-weight rank blend comes closest: it is positive on given and cleaned in every month,
  but it loses -0.0001 on gen-only in September. It changes 2 October rows. Re-check it on
  October-like when the members' cells exist.
