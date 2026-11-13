# October-shaped training copies, protected mode: stopped at the artefact check

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

## Next step, if it is reopened

Rebuild the copies with the amount scale off, or scale only rows outside every ring chain. Repeat
the check, then fit (octlike r0 on s13-15, then LOMO), with `python -m ringtrace.octshape octlike` and
`python -m ringtrace.octshape lomo`.
