# October-shaped training copies, protected mode: stopped at the artefact check (twice)

**Measured in this repository** (`python -m ringtrace.octshape mix`, commit 6fb100d; numbers in
`RESULTS.md`, "October-shaped copies").

## What was built

`ringtrace.octshape` makes a copy of a train month m with a copy seed (101, 102; distinct from the
validator's 13-15). Only month m changes, as octlike changes September:

- protected units (generator events and label units together) are kept whole with probability 0.494;
- legit-labelled kept rows have their amount scaled x1.166 (rounded to the paisa);
- the QR check is blanked on 44% of kept QR payments to a person;
- every feature family is rebuilt on the copy's frame, with the device counter held at its
  unthinned values.

Copy statistics: kept share 0.492-0.495 per month, QR blank share 0.39-0.48 among QR-to-person
rows (October 0.439), legit mean amount 1,240-1,261 against 1,069-1,082 raw.

## Artefact check (before any model)

Label-free merchant ring events (shapes 1-3) per 100k rows:

- **size 1:** October 9.98, raw July/August 13.28/12.15, copies 14.18-16.98;
- **size 2-3:** October 23.79, raw 19.73/20.31, copies 13.03-20.00.

Mean absolute gap to October over the July and August copies:

| size | raw train | copies |
|---|---|---|
| 1 | 2.74 | 5.79 |
| 2-3 | 3.77 | 6.97 |
| 4-8 | 0.10 | 0.87 |
| 9+ | 1.04 | 1.25 |

So the copies move the ring mix away from October: more single-row rings and fewer small ones. This
is the same direction as the row-thinning artefact (more size-1 rings). Flow events also shrink:
5.5-6.4 rows per event against 7.05-7.17 raw and 7.82 in October. Only the request events move towards
October: 177-189 per 100k against 252-270 raw and 212 in October.

The octlike judge frames, built the same way but without the amount scale, sit at size-1 9.5-10.7,
which is close to October. The likely cause is therefore the amount scale. Scaling legit rows moves
those that carry a ring amount off it, so ring chains that ran through them break into smaller
pieces. This cause is **not measured**: no copy without the scale was built.

## Verdict

**Stopped, as the job rule requires.** The copies depart from October more than raw train does, so
no model was fitted. There are no October-like cells, no leave-one-month-out run and no per-class
effect on `qr_swap`, `merchant_collusion` and `refund_abuse`.

## Second check: scaling restricted to rows outside generator structures

**Measured in this repository** (commit 0eeaa40; table in `RESULTS.md`).

About 12% of real fraud rows carry a legit label in train, ring rows among them, so the first build
also scaled generator rows off their round amounts. The fix scales only legit-labelled rows with no
event id in any evidence module and no row mark (`scale="outside"`). A control arm turns the scale
off (`scale="off"`). Thinning and QR blanking are unchanged.

Mean absolute gap to real October over the July and August copies (per 100k rows):

| ring size | raw train | first build | outside | off |
|---|---|---|---|---|
| 1 | 2.74 | 5.79 | 4.83 | 4.44 |
| 2-3 | 3.77 | 6.97 | 5.14 | 5.14 |
| 4-8 | 0.10 | 0.87 | 1.45 | 1.45 |
| 9+ | 1.04 | 1.25 | 1.54 | 1.54 |

- The fix helps where it can: size-1 rings fall from 5.79 to 4.83, and size 2-3 from 6.97 to 5.14.
  Flow events are no longer systematically smaller: 6.60-7.70 rows per event against 7.05-7.17 raw.
- The control shows that the remaining departure comes from the thinning itself, not from the
  scale. With the scale off, the 2-3, 4-8 and 9+ bins are identical and size 1 is only 0.4 closer.
- The copies keep their source month's ring mix, and July and August already carry more size-1
  rings than October (13.3 and 12.2 against 10.0). Whole-unit thinning then adds sampling noise to
  a ratio that is already off: 4-8 rises to 5.0-6.9 against 4.2, for example.
- The judge frames thin September, whose raw mix (8.6) is already near October's, which is why they
  match. This part is an explanation, **not a measured cause**.

**Verdict: stopped again.** The copies sit further from October than raw train in all four ring
bins, so no model was fitted: no October-like cells, no LOMO and no per-class numbers.

## Next step, if it is reopened

Both scale variants are measured above and fail. A copy can only pass if its ring mix moves
towards October, for example by taking September as the source month: its copies, s101 and s102,
sit at size 1 8.75 / 6.74 and size 2-3 21.07 / 21.00. The October-like judge cannot score copies of
September, but leave-one-month-out can. Then fit (octlike r0 on s13-15, then LOMO), with `python -m ringtrace.octshape octlike` and
`python -m ringtrace.octshape lomo`.
