# Counter news: the refined device counter moves a few fraud-phase calls, and every check agrees

**Question.** Inside a fraud phase, does each phone's allocation geometry place the class
boundaries better than the anchors alone? The geometry is the run of unused numbers just below a
phone, plus the phone's row count. And if it does, which October calls should move?

**Answer.** Yes on every check. The members keep the champion counter they were trained on, and
only the rules read the refined one.
- Retraining the members on the refined counter's features did not pass the gate (below).
- The October change against `research-combo-2` is 4 rows.

Aggregates only. October rows were used only through code.

## The refined counter (`ringtrace.evidence.refined`)

The blocks, the anchor emission table and the 11-state left-to-right HMM are the champion's. Each
phone also carries two pieces of allocation evidence:
- **the gap below it:** geometric, with a mean per segment, from sim_swap 0.41 (dense) to
  account_takeover 2.81 (sparse);
- **its row count:** a per-segment table; only the two drain segments carry phones with more than
  one row.

The constants were fitted once on the train months' fraud phase, with phones placed by a monotone
fit to the cleaned labels, and are fixed in the code. The counter stays label-free when it runs.
The sim_swap / account_takeover cut now comes from the same model instead of a separate change
point.

## Boundaries on four labelled passes

The passes are train's shared pass plus three September frames rebuilt at October's density.

| check | champion counter | refined counter |
|---|---|---|
| total boundary error, four passes | 82 phones | 58 phones |
| mule \| bot boundary | – | better on all four passes |
| sim_swap \| account_takeover boundary | – | better on two passes, never worse |

Where the two counters disagree, by the refined counter's posterior:

| posterior | phones | refined right | champion right |
|---|---|---|---|
| 0.5–0.6 | 11 | 4 | 7 |
| ≥ 0.6 | 38 | 34 | 3 |
| ≥ 0.7 | 27 | 25 | 2 |
| ≥ 0.8 | 13 | 12 | 1 |

That table sets the rule's threshold of 0.7.

## The rule (`counter_news`)

A fraud-phase row moves to the refined class when all three conditions hold:
- its call equals the champion counter's class;
- the refined posterior's mode is exactly one class away;
- that mode's posterior is at least 0.7.

A call that already differs from the champion segment is left alone. There the model beat the
segment 73 % to 20 % on train's out-of-fold calls.

## Validation against the uploaded `research-combo-2` calls

**Month folds**, refined counter in the rules, members unchanged:

| target | July / August / September | mean | 95 % event bootstrap |
|---|---|---|---|
| given | +0.00004 / +0.00008 / +0.00088 | +0.00033 | [+0.00004, +0.00099] |
| cleaned | +0.00003 / +0.00008 / +0.00104 | +0.00038 | [+0.00004, +0.00118] |
| gen-only | −0.000003 / −0.00001 / +0.00160 | +0.00053 | – |

**October-like frames**, three thinnings of September at October's density:
- Each frame rebuilds its counter as its own pass (both champion and refined).
- The rules read the frame's refined counter, and `counter_news` compares the frame's two counters.
- The members' cells are the production members' cells on those frames.

| target | frame s13 / s14 / s15 | mean | 95 % event bootstrap |
|---|---|---|---|
| given | +0.00102 / +0.00266 / +0.00008 | +0.00125 | [+0.00040, +0.00234] |
| cleaned | +0.00136 / +0.00276 / +0.00008 | +0.00140 | [+0.00045, +0.00258] |
| gen-only | +0.00191 / +0.00402 / +0.00001 | +0.00198 | [+0.00044, +0.00405] |

Rows changed per frame: 8, 18 and 1. By move, pooled over the three frames:

| move | rows | new call right (cleaned / given) | old call right (cleaned / given) |
|---|---|---|---|
| sim_swap → account_takeover | 18 | 18 / 15 | 0 / 0 |
| velocity_bot → mule_account | 8 | 7 / 7 | 0 / 0 |
| qr_swap → refund_abuse | 1 | 1 / 1 | 0 / 0 |

`fC1` also flipped these 3 October rows, among 13 changes. Its flips came from members retrained
on a density-matched counter, a different mechanism, and it was negative in one month. That upload
lost 0.0009 on public, which cannot tell which of its phones were wrong.

## October

Uploaded as `counter-news.csv`: public **0.5516**, against 0.5506 for `research-combo-2` (+0.0010).
Three of the four rows sit on one phone, so the public rows also speak for its private ones.

The change against `research-combo-2` is 4 rows:

| move | rows | why |
|---|---|---|
| sim_swap → account_takeover | 3 (one phone) | `counter_news`, posterior ≥ 0.8 |
| velocity_bot → mule_account | 1 | `mule_edge` reads the refined counter: 11 rows instead of 10 |

## Not adopted

| change | result |
|---|---|
| all six members retrained on the refined counter's features | July −0.00006 given, bootstrap bound −0.00001, 7 October rows |
| any adjacent-class move on the refined posterior alone (also overriding member calls) | July and August negative, October-like −0.0001 to −0.0003 |

## Reproduce

```bash
python -m ringtrace.final build recipes/counter-news.json --out <dir>   # sha256 b8c3183a...967799
```

The same commit rebuilds `bait-edge` and `research-combo-2` byte for byte. `phones.py` is unchanged
from the champion, so the member runs are too.
