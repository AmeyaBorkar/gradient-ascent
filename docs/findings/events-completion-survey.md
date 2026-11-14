# Event completion over October's own connections: nothing to complete

**Question.** The problem statement labels every row of a fraud event and allows building
connections inside `test.csv`. Our calls are made row by row and then partly propagated. Do the
calls of the best file (`blend-tabmF-bq`) leave parts of fraud events uncalled, in train or in
October? If so, a rule that calls the remaining rows with the event's type would add correct
calls. Also: why did the earlier event-completion rule change no October row?

**Answer.** No. Every label-free evidence event is called whole or not at all. Legit-called rows
linked to a fraud call (same sender, device or person receiver, or the money moving on or back) are
fraud at the base rate (2.2 to 3.8 %) and of the anchor's class at chance (0 to 0.45 %). The
break-even precision for adding a row to a class is about F1 / 2, about 0.25. Every completion
rule loses in every month on every target. October holds no excess of such links: per 100,000 rows
it has 0.47 to 0.53 times train, which is what a half-volume month predicts.

## 1. Evidence events are called whole

`python analysis/events.py partial`. Events are the label-free families in `rules.label_free_events`
(bot bursts, drains, flows, collect sessions, merchant rings, repeat payers).

| period | events | called whole | called not at all | called in part | legit rows in those |
|---|---|---|---|---|---|
| Jul | 3,277 | 2,887 | 389 | 1 | 1 |
| Aug | 3,164 | 2,815 | 349 | 0 | 0 |
| Sep | 3,008 | 2,640 | 367 | 1 | 1 |
| Oct | 1,396 | 1,254 | 142 | 0 | 0 |

- The events called not at all are collect sessions. On train they are 96 % legit (given and
  cleaned labels), so leaving them uncalled is right.
- Of the 24,345 generator-made train rows, 99.0 % are called fraud and 91.9 % get their type. The
  247 called legit sit outside events (218 rows): their silver label rests on the label and on the
  two columns that die in October, so they have no label-free handle.
- Type errors come in whole events, not as stray rows. 601 events (1,553 rows) are typed wrong as
  a whole, mostly with 1 to 3 rows. Only 81 events (190 rows) carry two generator types. Our calls
  hold a single type in all but 5 events.

**Why the earlier rule (`event_completion`) changed no October row.** The evidence families are
model features, and the generator-only target is built from the same evidence. So the model
already calls each event whole, and nothing is left inside an event for a rule to complete. The
fraud rows that the events do not reach are the ordinary-looking rows, and section 2 shows that
they are not attached to events.

## 2. Rows linked to fraud calls are not fraud

`python analysis/events.py links`. Query rows are legit-called, non-generator rows; the anchor is
the nearest fraud-called row of the same month. Train columns pool July to September (1,564,060
rows).

| link | window | train per 100k | cleaned fraud share | share of the anchor's class | October per 100k | October / train |
|---|---|---|---|---|---|---|
| same sender | 10 min | 2.0 | 0.031 | 0.000 | 1.2 | 0.56 |
| same sender | 1 h | 12.0 | 0.032 | 0.000 | 6.1 | 0.51 |
| same sender | 24 h | 246 | 0.022 | 0.0016 | 116 | 0.47 |
| same sender | 7 d | 1,472 | 0.023 | 0.0030 | 696 | 0.47 |
| same device | 1 h | 10.1 | 0.038 | 0.000 | 5.8 | 0.57 |
| same device | 24 h | 203 | 0.021 | 0.0013 | 98 | 0.48 |
| same person receiver | 1 h | 0.8 | 0.083 | 0.083 | 0.8 | 1.00 |
| same person receiver | 24 h | 24.4 | 0.010 | 0.0026 | 11.9 | 0.49 |
| onward (query's sender = anchor's receiver) | 1 h | 2.3 | 0.028 | 0.000 | 1.2 | 0.50 |
| onward | 24 h | 42.4 | 0.035 | 0.0045 | 21.9 | 0.52 |
| back (query pays the anchor's sender) | 24 h | 25.6 | 0.022 | 0.0025 | 10.7 | 0.42 |
| back | 7 d | 153 | 0.025 | 0.0017 | 85.6 | 0.56 |

- The fraud share of linked rows (2.1 to 3.8 %) is the share among all legit-called rows. Their
  same-class share (0.2 to 0.3 %) is one class's slice of that base rate. No cell of link kind,
  window, anchor class and transaction type with 15 or more rows reaches 0.25.
- The same-person-receiver link at 1 h is a single row out of 12.
- **October.** October has 0.494 times train's density, and its fraud events are halved with it.
  So a link from a legit row to a fraud call should be about half as frequent per 100,000 rows.
  The ratios are 0.42 to 0.57 (the four tiny cells aside). Fraud rows tied to October events
  beyond the ones we call would push them above 0.5.

**Completion rules** (`python analysis/events.py rules`): each linked row takes its anchor's class.
The change in macro-F1 is given per month, Jul / Aug / Sep.

| rule | rows train / Oct | given | cleaned | gen-only |
|---|---|---|---|---|
| sender, 10 min | 32 / 3 | -0.00012 / -0.00012 / -0.00020 | -0.00013 / -0.00013 / -0.00022 | -0.00040 / -0.00037 / -0.00065 |
| sender, 1 h | 187 / 16 | -0.00065 / -0.00109 / -0.00104 | -0.00070 / -0.00120 / -0.00114 | -0.00198 / -0.00357 / -0.00332 |
| device, 1 h | 158 / 15 | -0.00054 / -0.00096 / -0.00087 | -0.00059 / -0.00105 / -0.00095 | -0.00173 / -0.00315 / -0.00272 |
| person receiver, 1 h | 12 / 2 | -0.00007 / +0.00001 / -0.00005 | -0.00007 / +0.00001 / -0.00006 | -0.00019 / -0.00011 / -0.00015 |
| person receiver, 24 h | 382 / 31 | -0.00196 / -0.00185 / -0.00210 | -0.00211 / -0.00201 / -0.00228 | -0.00555 / -0.00567 / -0.00630 |
| onward, 1 h | 36 / 3 | -0.00016 / -0.00017 / -0.00015 | -0.00017 / -0.00018 / -0.00016 | -0.00046 / -0.00051 / -0.00048 |
| onward, 24 h | 663 / 57 | -0.00341 / -0.00331 / -0.00365 | -0.00368 / -0.00363 / -0.00400 | -0.01003 / -0.01065 / -0.01129 |
| back, 24 h | 400 / 28 | -0.00212 / -0.00194 / -0.00186 | -0.00230 / -0.00213 / -0.00204 | -0.00699 / -0.00652 / -0.00615 |

## 3. October's fraud pass is recognised as well as train's

`python analysis/events.py fraudblock`. Rows on a new single-use device numbered in the fraud block
are a fixed 3 % sample of the fraud pass: the device draw does not depend on the row. On these
rows, the share that carries label-free evidence (an event, a mark or a template) measures how
much of the fraud pass our evidence recognises.

| period | single-row fraud-block rows | with evidence | all fraud-block rows | with evidence |
|---|---|---|---|---|
| Jul | 576 | 0.325 | 1,137 | 0.658 |
| Aug | 600 | 0.323 | 1,082 | 0.625 |
| Sep | 541 | 0.335 | 1,013 | 0.645 |
| Oct | 320 | 0.338 | 564 | 0.624 |

On train, the fraud-block rows without evidence carry the legit pass's draws of the two
train-only columns:

| rows | count | app below 4.0 | IP risk above 0.5 | given fraud |
|---|---|---|---|---|
| fraud block, generator-made | 2,080 | 0.613 | 0.399 | 0.882 |
| fraud block, no evidence | 1,152 | 0.150 | 0.022 | 0.888 |
| legit block | 45,177 | 0.151 | 0.018 | 0.001 |

So they are rows written with the legit template and labelled fraud, which is the ordinary-row
mechanism. October's evidence share equals train's (standard error about 2.6 points on 320 rows).
So no October-only template family is going unrecognised: any such family is at most about 5
points of the fraud pass at 95 %.

## 4. Mule-shaped fan-ins and the bot deficit

`python analysis/events.py mf23`. The mf2+mf3 template (mule-shaped fan-ins) is written as
mule_account or velocity_bot.

| period | mf23 events per 100k | of which velocity_bot | bot bursts per 100k |
|---|---|---|---|
| Jul | 13.1 | 3 of 69 | 6.26 |
| Aug | 13.5 | 9 of 71 | 4.93 |
| Sep | 13.3 | 7 of 68 | 4.90 |
| Oct | 16.1 | ? of 42 | 2.69 |

- Velocity events per 100,000 rows add up to 6.3 to 6.8 in every train month (bursts plus
  velocity-labelled fan-ins).
- October's fan-in excess (+2.8 per 100k) equals its burst deficit (-2.7 per 100k).
- So one reading is that about 10 of October's 42 fan-ins are velocity_bot (25 %, against 9 % in
  train).
- On train, the velocity fan-ins match the mule ones on every event statistic: rows 20 against
  17, senders, receivers, duration, amount, login time and hour. October's fan-ins look like the
  mule ones.
- Even at 25 %, retyping fan-ins loses. Estimated from October's call counts, velocity would gain
  only about 0.006 F1 per 100 rows moved, while mule would lose about 0.04, because mule's precision on those rows (0.75) is far
  above its break-even (0.22). This agrees with the earlier rejected mf2+mf3 -> velocity_bot file.

## 5. What any structure-only system can reach

`python analysis/events.py ceiling`. Call exactly the generator-made rows, all of them right.
Count the ordinary rows as misses, with the unreported share (12 %) added back as on clean labels.
Legit F1 is set to 0.988.

| class | generator rows | ordinary rows | F1 ceiling |
|---|---|---|---|
| mule_account | 3,769 | 7,965 | 0.486 |
| velocity_bot | 3,693 | 5,561 | 0.570 |
| layering_chain | 3,280 | 4,486 | 0.594 |
| sim_swap | 1,727 | 2,859 | 0.547 |
| account_takeover | 1,719 | 2,860 | 0.546 |
| phishing_collect | 3,088 | 4,591 | 0.574 |
| qr_swap | 3,148 | 5,134 | 0.551 |
| refund_abuse | 1,507 | 2,261 | 0.571 |
| fake_cashback | 966 | 1,860 | 0.509 |
| loan_app_extortion | 669 | 1,539 | 0.465 |
| merchant_collusion | 779 | 2,073 | 0.429 |
| **macro** | | | **0.569** |

The same arithmetic with the best file's own train calls gives about 0.541. That is close to its
public 0.5499. A public score near 0.63 is therefore about 0.06 above anything that types the
generator-made rows perfectly. It needs the ordinary rows. Sections 2 and 3 close the
connection-based routes to them, in train and in October alike.

**Verdict.** No candidate file.
- Event completion has nothing to complete inside events.
- Completion over links loses in every month and on every target (by -0.0001 to -0.004 on given
  and cleaned labels).
- October shows no excess of linked rows and no unrecognised template family.

**Command.** `python analysis/events.py all` (about 5 s on the laptop). It reads
`$RINGTRACE_OUT/closelook/prep.npz` (the best file's out-of-fold and October calls with the
label-free evidence) and writes `$RINGTRACE_OUT/events/all.json`.
