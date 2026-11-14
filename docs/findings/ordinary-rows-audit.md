# Ordinary fraud rows: no signal in train or October, and the ceiling it sets

**Question.** About 62 % of fraud rows look like legit rows. Is any part of them reachable from
the data, in train or in October? What score is reachable at most, and what would a public score
of 0.61 to 0.63 require?

**Answer** (measured in this repo; commands below, all on commit `463e163`). No: by every audit
the ordinary rows are legit rows with a fraud label, in both splits. The reachable ceiling in
October is about 0.556 to 0.565 macro-F1, and 0.6338 would need about 1,000 hidden rows found.

## Ground truth without labels

The device counter splits new phones into each split's fraud phase and legit block
(`ringtrace.audit.probes`). A one-row fraud-phase phone with no template evidence carries an
ordinary fraud-pass row; a legit-block phone carries a legit-pass row.

| split | ordinary probe rows | legit-block rows | probe rows labelled fraud |
|---|---|---|---|
| train | 1,152 | 45,157 | 88.98 % (the rest is the 12 % unreported) |
| October | 210 | 7,459 | (no labels) |

## Separability, entity links and October shift

| audit | what is compared | result |
|---|---|---|
| `ordinary`, train | probe rows vs legit-block rows, all 109 context columns + 10 raw columns (`app_version` and `ip_risk_score` included) | AUC 0.4904 (seeds 0.4951 / 0.5022 / 0.4806) |
| `ordinary`, October | same | AUC 0.4896 (seeds 0.5165 / 0.4868 / 0.4690) |
| `ordinary`, best single column | about 120 tests per split | train p = 0.012; October: day of month AUC 0.557, p = 0.0049 (none survives a multiple-test correction) |
| `links`, train | ordinary fraud vs ordinary legit rows: another ordinary-fraud row on the same sender, receiver or phone within 1 h / 1 d / 7 d | ratio 0.95 to 1.02 in all nine cells |
| `links`, history | September ordinary rows, by whether the key had ordinary fraud in July-August | fraud rate 2.28 % to 2.34 % in every bucket (sender, receiver, phone) |
| `shift` | September vs October own-phone rows without evidence, 21 density-free columns + minute, second, cents; 200,000 rows a side | AUC 0.5024; October share of the most October-like 0.1 / 0.5 / 1 / 5 %: 0.475 / 0.497 / 0.503 / 0.507 |

A hidden October-only group of ordinary rows would show in the `ordinary` October AUC or in the
`shift` tail shares; neither moves off chance.

## Composition and ceiling

`ceiling` estimates the hidden ordinary rows of a month from its probe rows, per class (counter
segment), using the train share of ordinary fraud rows that drew a new phone: **2.82 %**.

| period | rows | hidden ordinary | share of fraud | hidden / labelled ordinary | ceiling macro |
|---|---|---|---|---|---|
| July | 527,065 | 13,359 | 0.607 | 1.125 | 0.5710 |
| August | 526,852 | 14,015 | 0.627 | 1.172 | 0.5800 |
| September | 510,143 | 12,392 | 0.602 | 1.072 | 0.5941 |
| October | 260,582 | 7,249 | 0.648 | - | 0.5560 |

- The hidden / labelled ratio should be about 1 / 0.88 = 1.14, because the labelled count misses
  the 12 % unreported rows. The three train months give 1.07 to 1.17, so the estimator holds.
- The ceiling counts every anchor- or counter-reachable row as found and typed right (3,939
  October rows). October per class: legit 0.986, mule 0.451, bot 0.381, chain 0.623, sim 0.663,
  takeover 0.504, phishing 0.565, qr 0.566, refund 0.614, cashback 0.452, loan 0.544,
  collusion 0.322.
- For the five-member blend file (3,946 fraud calls, 95 % taken as right), the expected macro is
  **0.5475**; its public score was 0.5474. The ceiling with the same reachable rows is 0.5541.
- A public score of **0.6338 needs 1,006 extra correct rows**, 13.9 % of October's hidden
  ordinary rows. The audits above find no handle on any of them.
- The same check on the TabM fraud-only file (3,973 fraud calls) gives an expected 0.5491
  against its public 0.5499, and 991 extra rows for 0.6338.

## If the public part were picked by row position

Scores of 0.61 to 0.63 need no extra signal if the public part is the first 30 % of rows *of each
uploaded file* rather than a fixed set of ids. A file that happens to hold more than 30 % of its
fraud calls in its first rows then scores higher on public and lower on private, with the same
calls (`ceiling --front-shares`, TabM fraud-only file):

| fraud calls in the first 78,175 rows | public | private |
|---|---|---|
| 30 % (random order, every file of ours) | 0.5491 | 0.5491 |
| 40 % | 0.6110 | 0.5156 |
| 45 % | 0.6357 | 0.4967 |
| 50 % | 0.6572 | 0.4761 |

**Outcome: rejected.** The order probe of the TabM fraud-only file scored the same public score as
its source, so the public part is fixed by id and row order plays no part. A follow-up check also
finds no re-use of past fraud transactions: September rows that repeat a July-August ordinary
fraud row on (sender, amount) are 2.8 % fraud against 2.2 % for other repeats (106 rows), and no
October row repeats a past ordinary fraud row on (sender, receiver, amount).

The hypothesis was about the platform, not about the data. The order probe tested it with one
upload: `shuffle` keeps every call and permutes the rows. An identical public score means the
public part is fixed by id (the hypothesis is false); a different score means it moves with row
order. The probe's calls are the source's, so its expected private score is the same in either
case. A file must never be reordered to raise its public score: the final ranking scores the best
public file on its private part, which such an order lowers.

## Look-alike templates

`lookalike` turns each train event of a template into one row and asks a booster to tell the two
types apart (events on fraud-phase phones are already typed by the counter; "off phase" leaves
them out).

| template | pair (rule class first) | events | second-type share | AUC all | AUC off phase |
|---|---|---|---|---|---|
| mf1 | layering_chain / mule_account | 578 | 0.152 | 0.600 | 0.598 |
| drain1 | sim_swap / account_takeover | 542 | 0.306 | 0.598 | 0.590 |
| mf2+mf3 | mule_account / velocity_bot | 204 | 0.093 | 0.348 | 0.382 |
| col7 | fake_cashback / phishing_collect | 258 | 0.581 | 0.455 | 0.500 |
| col1 | phishing_collect / fake_cashback | 1,138 | 0.097 | 0.518 | 0.525 |
| rng1 | qr_swap / refund_abuse | 460 | 0.033 | 0.610 | 0.575 |

The sibling type is close to random per event (AUC at most 0.61; the small mf2+mf3 template
overfits below 0.5). An event-level typing model would gain little over the template rule.

**Method.**
- Groups: `ringtrace.audit.probes` on the counter columns (`ctr_block`, `ctr_dev_single`,
  `ctr_seg`) and the event / mark columns of the six event families and `marks`.
- Statistics: `ringtrace.audit.stats` (LightGBM, 300 trees, 5 folds x 3 seeds unless stated;
  Mann-Whitney and two-proportion tests; `searchsorted` link counts).
- Ceiling: `ringtrace.audit.ceiling`; reachable rows are the counter segment on fraud-phase phones,
  else the row anchor (`ringtrace.evidence.anchors.row_anchor`).
- Test rows are used through aggregates only.

**Command.**

```bash
python -m ringtrace.audit ordinary
python -m ringtrace.audit links
python -m ringtrace.audit shift
python -m ringtrace.audit ceiling --submission <file.csv> --target 0.6338 --front-shares 0.3,0.4,0.45,0.5
python -m ringtrace.audit lookalike
python -m ringtrace.audit shuffle <submission.csv>      # order probe, optional upload
```

Each writes `$RINGTRACE_OUT/audit/<command>.json`. With the feature families cached, wall time is
6 to 23 s per command on a laptop CPU with 16 threads (polars 2.1.0, numpy 2.5.4, lightgbm 4.7.0,
scikit-learn 1.9.1, scipy 1.18.1).

**What it changes.**
- **Ordinary rows are out of reach on this data.** Runs aimed at them are not worth spending. A
  new feature family that claims ordinary rows must first lift the `ordinary` AUC clearly above
  0.5 in both train and October, and leave `shift` at chance (`docs/decisions.md` item 16).
- **Public scores of 0.61 to 0.63 sit above what the data supports.** They need about a thousand
  hidden rows found; whatever produces them is not in the row or relational signal audited here.
- **The remaining gain is template-side and small:** about 0.007 between the blend's expected
  0.5475 and its ceiling of 0.5541. The look-alike AUCs support the template rule.
- **Planning:** `ceiling --submission` predicts a file's public score from its call counts alone
  (0.5475 against 0.5474 for the blend), a check worth running before any upload.
- **Order probe:** `shuffle` writes the same calls in a new order. Uploading it beside its source
  shows whether public rows are picked by position (any score difference). The calls are
  identical, so its expected private score equals the source's.
