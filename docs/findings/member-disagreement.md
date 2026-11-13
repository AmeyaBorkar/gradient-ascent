# Where the members disagree with the champion

**Question.** On which rows do the other four members' calls differ from the champion's, and
which side do the cleaned labels take?

**Answer.** Very few rows differ, both on the held-out months (1,564,060 train rows) and on
October (260,583 rows). Each member is decoded as the run decodes it.

| member | held-out rows that differ | cleaned labels side with member / champion / neither | given labels: member / champion / neither | October rows that differ |
|---|---|---|---|---|
| catboost | 95 | 33 / 53 / 9 | 37 / 44 / 14 | 26 |
| clean-event | 64 | 33 / 26 / 5 | 30 / 26 / 8 | 21 |
| clean-l2 | 68 | 35 / 26 / 7 | 31 / 28 / 9 | 22 |
| lgbm | 48 | 27 / 14 / 7 | 25 / 13 / 10 | 24 |

Largest held-out cells, written as champion call -> member call: rows (cleaned labels side with
member / champion).

| member | 1st cell | 2nd cell | 3rd cell | 4th cell |
|---|---|---|---|---|
| catboost | phishing_collect -> legit: 32 (4 / 25) | qr_swap -> legit: 11 (3 / 8) | legit -> phishing_collect: 8 (6 / 1) | qr_swap -> refund_abuse: 5 (5 / 0) |
| clean-event | legit -> phishing_collect: 18 (12 / 5) | phishing_collect -> legit: 10 (3 / 7) | legit -> qr_swap: 8 (5 / 3) | qr_swap -> legit: 7 (3 / 4) |
| clean-l2 | legit -> phishing_collect: 19 (11 / 6) | legit -> qr_swap: 9 (6 / 3) | phishing_collect -> legit: 9 (3 / 6) | qr_swap -> legit: 6 (3 / 3) |
| lgbm | phishing_collect -> legit: 8 (2 / 6) | legit -> sim_swap: 7 (3 / 0) | legit -> phishing_collect: 5 (5 / 0) | qr_swap -> legit: 3 (1 / 1) |

On October (calls only), every member's largest cell is the same: **13 rows that the champion
calls account_takeover and the member calls sim_swap**. The other October cells hold 1-3 rows
each, between legit and a fraud type.

**Method.** `analysis/core.py:disagreement` compares the decoded calls (fixed point at 0.875 per
month, then `anchored_templates`).

**Command.** The same as in `public-score-noise.md` (key `disagree`).

**What it changes.**
- The members are near copies of the champion: at most 95 of 1.56 M held-out rows differ, and at
  most 26 October rows. The blend can move only a few dozen rows, and its value comes from those
  rows, not from a broad average.
- The cleaned-label members and lgbm win their disagreements: the cleaned labels side with them
  33:26 (clean-event), 35:26 (clean-l2) and 27:14 (lgbm).
- catboost loses its disagreements (33:53), mostly because it moves phishing_collect calls to
  legit.
- The 13-row account_takeover/sim_swap cell is the one place where every member disagrees with
  the champion in the same way. One correct sim_swap or account_takeover row is worth about
  0.0004 on the public slice, so this cell can move the score by a few thousandths either way.
