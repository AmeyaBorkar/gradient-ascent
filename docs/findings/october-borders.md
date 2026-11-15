# October's class borders: a posterior for each border, and how far it can be trusted

**Question.** October's fraud phase is a sixth of train's: 920 numbers and 410 phones against 5,455
and 2,240. Each class still owns one consecutive zone, but the borders are not marked. How uncertain
is each border, are the probabilities right at that size, and which October calls sit on the
unlikely side?

**Answer (measured in this repo, `feat/october-events`).** Every border is uncertain: its most
likely position holds 0.21 to 0.98 of the probability, and about 55 phone-border pairs per block
are undecided. The probabilities are right when pooled, and wrong at two borders. Only 7 October
calls sit on the unlikely side. One border move (1 row) is in the best upload; the rest are built
and not uploaded.

## The model (`ringtrace.evidence.borders`)

The phase is read on the number line, used and unused numbers alike, with three pieces of evidence
fixed from the train phase:

- **density:** how often a class uses its numbers (`refined.GAP_MU`); an unused number is charged
  to the zone it lies in;
- **tags and rows:** the anchors a class's phones carry and their row counts
  (`phones.emission_table`, `refined.ROW_PMF`);
- **zone length:** each class's share of the phase (`LENGTH_SHARE`: 0.1206, 0.0911, 0.0867,
  0.0753, 0.3171, 0.0726, 0.1248, 0.0409, 0.0242, 0.0189, 0.0277, from a monotone fit to the
  cleaned labels that agrees with 0.975 of 2,113 labelled phones), as a log-normal prior of
  log-sd 0.8.

`segment` sums over every placement of the ten borders and returns each border's posterior.

## Replay at October's size

The train phase is thinned to October's size (numbers dropped and closed up) with the class mix
shifted by a log-normal factor per class. Phones in the wrong zone per block, 240 blocks:

| class-mix log-sd | refined counter | border model, no length prior | length sd 0.3 | 0.5 | 0.8 |
|---|---|---|---|---|---|
| 0 | 11.73 | 11.35 | 9.53 | 10.08 | 10.50 |
| 0.25 | 11.50 | 11.14 | 10.55 | 10.30 | 10.48 |
| 0.4 | 11.26 | 10.82 | 12.03 | 10.68 | 10.52 |

A tight length prior helps only while the mix stays close to train's; at log-sd 0.8 it never
hurts, which is why 0.8 is fixed in the code. The gain over the refined counter is about one phone
per block.

## Calibration at October's size

For every border and every phone near it the model gives the probability that the phone lies above
the border. 300 thinned blocks, pooled over the ten borders:

| said | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| happened, no mix shift | 0.07 | 0.15 | 0.25 | 0.30 | 0.40 | 0.56 | 0.68 | 0.79 | 0.90 | 0.98 |
| happened, mix log-sd 0.25 | 0.07 | 0.18 | 0.29 | 0.39 | 0.49 | 0.59 | 0.69 | 0.76 | 0.86 | 0.95 |

Two borders do not hold:

| border | said | happened | pairs |
|---|---|---|---|
| layering_chain \| sim_swap | 0.13 to 0.14 | 0.61 to 0.65 | 306 to 315 |
| account_takeover \| phishing_collect | 0.92 | 0.63 to 0.69 | 74 to 91 |

The other eight are within about 0.1. The refined counter's own posterior is calibrated too (Brier
0.131 and 0.140 against the border model's 0.127 and 0.139), so the counter is not overconfident
about borders: the uncertainty is real. All blocks come from the one train phase, so a per-border
bias reflects that border's surroundings in train.

## October

Mass of each border's most likely position, and the phones it can lie on either side of with
probability 0.01 or more:

| border | top position | band, phones |
|---|---|---|
| mule_account \| velocity_bot | 0.44 | 8 |
| velocity_bot \| layering_chain | 0.39 | 10 |
| layering_chain \| sim_swap | 0.82 | 3 |
| sim_swap \| account_takeover | 0.48 | 10 |
| account_takeover \| phishing_collect | 0.98 | 1 |
| phishing_collect \| qr_swap | 0.50 | 2 |
| qr_swap \| refund_abuse | 0.42 | 5 |
| refund_abuse \| fake_cashback | 0.21 | 7 |
| fake_cashback \| loan_app_extortion | 0.69 | 3 |
| loan_app_extortion \| merchant_collusion | 0.42 | 8 |

- 7 calls of `counter-news` sit on the unlikely side (posterior 0.5 or more for the other class),
  one or two rows per border.
- `border_shift` builds one file per border and position (`recipes/borders/`, 21 files that change
  a row). The layering_chain \| sim_swap files were dropped after the calibration above.
- A 13-row sim_swap to account_takeover move contains the 7 rows of an earlier upload that lost on
  public, and was dropped.
- In the uploads: `border_shift` with `border: 8, rank: 0` (1 row, loan_app_extortion to
  fake_cashback, about 0.7 to 0.8 after calibration). It is part of the stacked files in
  `docs/findings/october-events.md`; its row has not been read on its own.

## How border changes are judged

Decision 19 in `docs/decisions.md`: a border change moves no call in train, so no month-fold gain
is required of it. The test is the calibration above, at the test phase's size.

**Command.**

```bash
python analysis/october_rules.py replay 240
python analysis/october_rules.py calibration 300
python -m ringtrace.final build recipes/borders/border-8-r0.json --out <dir>
```

**What it changes.** A single-row border move cannot raise the private score through a public
read: a rise means the row is public, a tie means it is private and untested. Border files are
worth an upload only when several rows share one answer.
