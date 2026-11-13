# Event completion: legit-called rows inside mostly-fraud evidence events

**Question.** The generator labels every row of a fraud event. Inside a label-free evidence event
that the base calls mostly fraud, are the remaining legit-called rows fraud, and do they carry the
event's majority type? If so, a rule could call them that type.

**Answer (measured in this repo; October as aggregates only).** No. The events are already
complete: the bases leave almost no legit-called rows inside them.

## Survey

`python analysis/rescue.py evcomp [--base-run tabm]`, with `rules.completion_targets`:
- families: bot bursts, drains, mule bursts (flow shapes 2-3), chains (flow shapes 1 and 4),
  collect sessions, merchant rings, refund / lending repeats;
- an event is mostly fraud when a share of at least q of its rows is called fraud;
- events never join across the train/test split.

Bases:
- **blend:** the five-way blend of `recipes/blend-blankqr.json`, out of fold in train;
- **TabM:** the TabM run (`runs/tabm`, class weights off), decoded the same way;
- **October:** the `blend-tabmF-bq.csv` upload file.

Legit-called rows in mostly-fraud events. Train columns show rows, cleaned fraud rate, given fraud
rate, and the share whose cleaned label equals the event's majority call.

| family | q | blend, train | TabM, train | blend, October | upload file, October |
|---|---|---|---|---|---|
| bot | 0.5 / 0.67 / 0.8 | 4 rows each; 1.00 / 0.50 / 1.00 | 0 | 0 | 0 |
| drain | 0.5 | 3 rows; 1.00 / 1.00 / 0.67 | 0 | 1 | 0 |
| drain | 0.67 / 0.8 | 0 | 0 | 0 | 0 |
| collect | 0.5 | 1 row; 1.00 / 1.00 / 1.00 | 0 | 0 | 0 |
| collect | 0.67 / 0.8 | 0 | 0 | 0 | 0 |
| mule, chain, ring, repeat | every q | 0 | 0 | 0 | 0 |

Every event family is called whole or not at all. That matches the signature survey
(`signature-rescue.md`): out-of-fold legit-called shares inside events are 0 to 0.14 %.

## Rule

`rules.event_completion` (parameters `family`, `q`): legit-called rows of a mostly-fraud event take
the event's majority fraud call. The only cells with leftovers, and a rate above the call cut-off,
were checked on leave-one-month-out:

`python analysis/rescue.py rule --sig evcomp:<family>:<q>`. Per month Jul / Aug / Sep, 95 % event
bootstrap:

| rule | given | cleaned | gen-only | train / October rows changed | verdict |
|---|---|---|---|---|---|
| bot, q = 0.5 | +0 / +0 / +0.00004, [+0.00000, +0.00005] | +0 / +0 / +0.00012 | +0 / +0 / +0.00016 | 4 / 0 | never loses, but changes no October row; nothing to build |
| drain, q = 0.5 | +0.00007 / +0.00007 / -0.00002 | +0.00006 / +0.00007 / -0.00002 | +0.00008 / +0.00010 / -0.00006 | 3 / 1 on the blend, 0 on the upload file | **rejected**: September negative |

October-like was not run: neither rule changes an October row of the upload file, so neither could
give a file.

## Where the remaining labels are

The rows the problem statement labels as part of an event but that are not in these events (for
example a mule's onward transfers, or payments into a mule outside the detected burst) are not
reached by completing the events. The evidence modules' event definitions already take every row
they can find. Rows they miss look like ordinary rows: the knowledge base puts 63 % of fraud labels
on ordinary rows, with no label-free handle. Widening an event definition is a new evidence
module, not a decoding rule.

**Verdict.** Not adopted, and no file written: no family and no q changes any October row of
`blend-tabmF-bq.csv`.
