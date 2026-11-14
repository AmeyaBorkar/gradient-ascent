# velocity_bot and layering_chain in October: fewer events, not missed ones

**Question.** The best file calls `velocity_bot` on 122 October rows per 100k against 209 to 257
in the train months (0.53 times) and `layering_chain` on 215 against 227 to 247 (0.90 times). Does
October have fewer bots and chains, or does the model miss bots that changed shape, or that the
half-density month hides?

**Answer (measured in this repo; October as aggregates only).** Fewer events. The model calls
every bot and chain the data shows, in October as in train. Nothing to upload.

## velocity_bot

- **The bursts are whole, not thinned.** The October bot-pool bursts (devices 420,000 to 422,999)
  have 29 / 37 / 58 rows (min / median / max). Train bursts have 20 / 40 / 60, and train bursts
  thinned to 50 % would have 2 / 20 / 39. A Mann-Whitney test of October's sizes against whole
  train bursts gives p = 0.85, against thinned bursts p = 1.8e-5. The chance that all 7 bursts
  reach 29 rows is 0.13 if bursts are whole and 3.4e-7 if they are thinned. So the half-density
  month does not hide bot rows: events keep their size, and only the legit background is thinner.
- **There are fewer bursts.** 7 bursts in October (2.68 per 100k rows) against 33 / 26 / 25 in
  train (6.26 / 4.93 / 4.90). At the train rate per row, 14.0 were expected (Poisson P(<= 7) =
  0.032). The bursts have the same shape as in train: 3 to 58 accounts, spans of 312 to 787 s,
  ecommerce 78 % and gaming 22 % (train 81 to 93 % and 7 to 19 %), amount median 253 (train 241 to
  250). One burst is at night.
- **No bot outside the pool.** Off the pool, no device or sender has a session of 3 or more small
  whole-rupee merchant payments (Rs 10 to 499) within 5 minutes in any month, at full density or
  thinned to 50 % (seeds 0 to 2). Relaxed versions find nothing either:
  - merchant payments of any amount, at gaps of 5, 15 or 60 minutes, per device or per sender;
  - ecommerce, gaming, travel or electronics rows of any type;
  - devices shared by 4 or more accounts in a month (October: 0).

  At most 5.5 rows per 100k fall in any of these groups, and none of them is called or labelled
  bot. Multi-row fraud sessions of any kind off the pool run at the full-density train rate in
  October (for example 27.6 per 100k rows in device sessions of 4 or more rows within 15 minutes,
  against 20.5 to 25.8 in train and 3.0 to 4.2 when train is thinned). That is a second sign that
  fraud events stay whole.
- **The calls.** All 269 October pool rows are called `velocity_bot`, as in every train month
  (1,307 / 968 / 1,066 rows, 100 %). The other 49 October bot calls are bot-zone rows on single-use
  fraud-pass devices (38) and bot-written fan-ins (11). In train the zone rows are 84 to 86 % bot
  on given labels.
- **The ordinary bot rows did not halve.** Single-use fraud-pass devices with no template carry
  about 3 % of each class's ordinary-looking rows, and their counter segment names the class.
  For `velocity_bot` they run at 9.1 / 11.0 / 10.8 per 100k in train and 10.7 in October (1.04
  times). So the burst routine halved, while the ordinary bot rows stayed at the train rate. Train
  ordinary bot rows match legit rows on type, category, receiver type, new payee, amount, hour and
  login: there is nothing to detect.
- **What this costs.** Bursts are a smaller share of October's bot rows than of train's, so
  `velocity_bot` F1 should fall from 0.583 (pooled train, cleaned) to about 0.42 in October. No
  label-free signal reaches the rest. Calling bot on more rows with a high P(bot) loses in every
  month:

  | rule | train rows | October rows | given Jul / Aug / Sep | cleaned | gen-only |
  |---|---:|---:|---|---|---|
  | P(bot) >= 0.35 | 68 | 0 | -0.0011 / 0.0000 / -0.0006 | -0.0011 / 0.0000 / -0.0006 | -0.0020 / 0.0000 / -0.0012 |
  | P(bot) >= 0.20 | 195 | 22 | -0.0028 / -0.0003 / -0.0010 | -0.0030 / -0.0003 / -0.0010 | -0.0053 / -0.0005 / -0.0019 |
  | P(bot) >= 0.10 | 386 | 39 | -0.0045 / -0.0008 / -0.0035 | -0.0049 / -0.0010 / -0.0038 | -0.0086 / -0.0018 / -0.0071 |

## layering_chain

Rows per 100k, July / August / September / October, by money-flow shape. Every shape row is
called layering or mule as it is in train.

| shape | Jul | Aug | Sep | Oct | called layering, train / Oct |
|---|---:|---:|---:|---:|---|
| chain, every hop within 25 min | 123.5 | 133.6 | 125.3 | 114.7 | 0.98 / 0.94 (the rest mule) |
| chain, a hop of 1 to 6 h | 35.3 | 40.6 | 38.2 | 49.9 | 1.00 / 1.00 |
| fan-in of 3 to 7 payers | 76.8 | 64.2 | 56.1 | 47.6 | 1.00 / 1.00 |
| mule burst (inflow) | 202.4 | 206.1 | 215.7 | 260.4 | 0.00 / 0.00 (mule) |
| zone row, no shape | 7.4 | 9.1 | 6.7 | 7.7 | 1.00 / 0.95 |

- The 0.90 is a change of mix. Fan-ins keep falling (a trend through the train months), fast
  chains are a little rarer, and slow chains are 1.3 times more common. The fast-chain rows
  called mule are whole events. In train they are mule on 100 % of cleaned labels (35 rows).
  Ordinary layering rows (zone) stay at 0.99 times.

## A check of the whole October picture

Rescale each class's train anatomy (cleaned target) to October with two label-free ratios:

- the structured part, by our October calls outside the zone rows;
- the ordinary part, by the zone-ordinary rows.

This predicts October macro-F1 **0.555** for the best file, against 0.566 pooled on train. The
public score is 0.5499 (single-file sd about 0.006).

The estimator, held out on each train month in turn and built from the other two, is off by
-0.013 / +0.006 / +0.011 on cleaned and -0.015 / +0.006 / +0.014 on given. It is noisy because
small classes have few zone rows. Within that error, the October picture holds:

- the structured events are found;
- the ordinary-looking rows are present at about the train rate (zone rows per 100k: mule 1.46,
  phishing 1.31, collusion 1.36, loan 0.61, sim 0.80 times; the others 0.91 to 1.12).

The per-class estimates for October are mule 0.46, bot 0.42, layering 0.59, sim 0.60, takeover
0.49, phishing 0.49, qr 0.59, refund 0.68, cashback 0.45, loan 0.59, collusion 0.32.

## One misplaced edge: the mule/bot cut on the device counter

The counter HMM decodes each split's fraud phase separately. October's mule zone is 1.21 times
train's share of the phase (positions 0 to 0.193, against 0 to 0.159). That matches the October
mule event volume (1.24 to 1.28 times). The other zones keep 0.81 to 1.11 times their train
share.

The model calls 10 October single-row zone devices `velocity_bot` although the HMM puts them in
the mule segment (posterior 0.86 to 0.96). They sit at positions 0.164 to 0.188. In train the
model learned the exact edge (the first bot row is at 0.1563) from the one fraud phase that all
three train months share. Out-of-fold scoring therefore never tests whether a position cut
transfers to a new phase.

Which reading is right?

- **For the HMM (mule).** The band from 0.156 to 0.196 holds 4 mule-template anchored devices
  per 12 zone rows in October. That density (0.33) is close to October's mule zone (0.36) and
  far above the start of train's bot zone (4 per 85, 0.047). At the train density,
  P(>= 4 anchors) is about 0.003. Under the bot reading, these four events would be bot-written
  fan-ins, and the model calls all four mule anyway.
- **For the model (bot).** With the HMM reading, ordinary mule zone rows run at 1.46 times train
  and bot at 1.04. With the model's reading they run at 1.20 and 1.41. Ordinary counts per class
  are otherwise flat (about 1.08 times), which favours the model's reading by about 2 to 1.
- **Combined.** The HMM reading is favoured by roughly 40 to 1.

The probe `velocity-muleedge` moves the 10 rows from bot to mule. On the train months the same
rule changes 6 rows, all at the true edge and all truly bot. It loses there:

| target | Jul | Aug | Sep |
|---|---|---|---|
| given | -0.00004 | -0.00008 | -0.00012 |
| cleaned | -0.00003 | -0.00008 | -0.00012 |
| gen-only | +0.000001 | +0.00001 | +0.00001 |

So it fails the bar and is a hypothesis probe only. It needs one assumption: October's zone edges
moved with its class mix. The evidence for that is the zone shares and the anchor density
above.

Expected full-October change: about +0.0007 if the HMM is right and -0.0011 if not, so about
+0.0006 at 40 to 1. The public slice holds about 3 of the 10 rows, so a public upload cannot
resolve it.

**Verdict.** No upload candidate. `velocity_bot` and `layering_chain` are saturated on every row the data
marks. Their October call rates are real changes in event volume (bot bursts 0.5 times,
fan-ins down, slow chains up). They are not drift and not missed events. A score above our
ceiling must come from the ordinary-looking rows, which these two classes cannot reach.

The mule/bot edge probe is the only disagreement left between
the calls and the counter. It is worth about +0.0006 in expectation and is not testable on the
train months.

**Command.** `python analysis/velocity.py` (about 30 s on 4 threads). The report goes to
`$RINGTRACE_OUT/velocity/report.json`, with sections `bursts`, `offpool`, `zones`, `layering`,
`estimate`, `probe` and `edge`. `python analysis/velocity.py build` writes the probe
`velocity-muleedge.csv`.
