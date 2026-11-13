# The sim_swap / takeover edge in October (aggregates only)

**Question.** Do the October drain rows near the counter's cut call for any change?

**Method.** `python -m ringtrace.squeeze audit`. Drain shapes 1-4, train against October:
- the blend's calls;
- the counter's segment confidence on fraud-block rows in the sim / takeover segments;
- the rows where members disagree on the sim share by more than 0.1.

The anchored rule's `min_conf` (0.8) was then moved to 0.7, 0.75, 0.85 and 0.9.

**Answer.**

| drain shape | train rows (calls sim / ato) | October rows (calls sim / ato) | pair-segment rows with confidence < 0.9, train / October | members disagree, train / October |
|---|---|---|---|---|
| 1 (template) | 1,567 (1,533 / 28) | 252 (248 / 3) | 0 / 1 (in 0.85-0.9) | 201 / 9 |
| 2 | 175 (166 / 0) | 36 (34 / 0) | 0 / 0 | 4 / 4 |
| 3 (sim / takeover mix) | 1,515 (443 / 1,072) | 244 (76 / 168) | 0 / 0 | 10 / 19 |
| 4 | 206 (0 / 205) | 46 (0 / 46) | 0 / 0 | 0 / 0 |

Every drain-3 row, in train and in October, is on a fraud-block phone in the sim or takeover
segment with confidence >= 0.9. None sits near the 0.8 cut. Moving the cut:

| min_conf | given Jul / Aug / Sep | October rows changed |
|---|---|---|
| 0.70 or 0.75 | +0.0000 / +0.0000 / +0.0008 | 0 |
| 0.85 | +0.0000 / -0.0003 / +0.0000 | 1 (fake_cashback -> phishing_collect) |
| 0.90 | +0.0000 / -0.0003 / +0.0000 | 12 (11 velocity_bot -> mule_account, 1 fake_cashback -> phishing_collect) |

**Verdict.** No change.
- The October drain rows are far from the cut, so this audit finds no edge.
- Lowering the cut changes no October row, so it cannot move the file.
- Raising the cut loses in August.
- The 19 drain-3 rows on which members disagree are already settled by the counter.
