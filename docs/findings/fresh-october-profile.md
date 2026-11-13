# October call profiles against train, and the sim_swap night window

**Question.** Per fraud class, does October's call profile depart from train? We compare
October's calls with the train labels and the out-of-fold calls across hour, day of month, amount
band, app version, device age and receiver type. Does it depart in a way the model could be
getting wrong? Also, are sim-swap-like evidence rows at 22:00-24:00 called sim_swap less often in
October than those at 00:00-05:30?

**Answer (measured in this repo; October as aggregates only).** We use the total-variation
distance (TV) between a class's bin mix in October calls and in out-of-fold calls.

- **App version.** All October calls fall in the top band (>= 6), while every train call is
  below 6. The whole of October moved, so this is a known dead column and not a class problem.
  It is not a feature.
- **Hour and day.**
  - The velocity_bot calls (TV 0.46 hour, 0.36 day) and merchant_collusion calls (0.41 day) are
    lumpy because October holds only a few of their events. Bursts land on a few days and hours.
    This is the expected shape of a handful of events, not a shift in what is called.
  - Every other class is under TV 0.18.
- **Amount, device age and receiver type.** Every class is under TV 0.15. October moves towards
  the out-of-fold profile, not away from it.
- **sim_swap night window.**

  | rows with sim-like evidence | train rows | called sim_swap | October rows | called sim_swap |
  |---|---|---|---|---|
  | 00:00-05:30 | 1,211 | 97.8 % | 223 | 98.2 % |
  | 05:30-06:00 | 34 | 91.2 % | 7 | 100 % |
  | 06:00-22:00 | 459 | 97.2 % | 34 | 100 % |
  | 22:00-24:00 | 38 | 100 % | 24 | 91.7 % |

  Sim-like evidence is a sim_swap anchor, the drain-1 template or the 0.25 balance-step mark.
  All of these October rows are called fraud. At 22:00-24:00, 22 of 24 are called sim_swap; the
  other 2 went to the look-alike class.

  October's share of sim_swap calls at 22:00-24:00 is 8.9 %, against 3.9 % out of fold and 6.1 %
  in train labels. So October does carry more late-evening sim-swap events, and the model calls
  them. The evidence already fires at every hour (459 train rows at 06:00-22:00), so the
  continuous hour feature does not gate it.

**Verdict.** No class departs in a way the model gets wrong. The sim_swap lead is no action:
2 rows differ, and they are a sim_swap / takeover edge question, which is measured separately.

**Command.** `python analysis/fresh.py october` (box, run `blend`; writes
`$RINGTRACE_OUT/fresh/october-blend.json`).
