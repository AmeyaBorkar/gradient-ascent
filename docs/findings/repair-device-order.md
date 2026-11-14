# Device-order class repair: no signal away from the look-alike edges

**Question.** The phones of a fraud phase are minted class by class in label-list order. A call can
fit its own rows yet clash with the classes of the phones on either side of it. Does retyping such
calls, using only the order and label-free anchors, raise macro-F1, and above all in October?

**Answer (measured in this repo; October as counts only).** No. Away from the two adjacent
look-alike edges (mule/bot and sim/takeover), every call already fits the device order, in the
dense months, in the October-like frames and in October. At those two edges the order cannot
settle the class, because the anchors there are look-alike rows and their class is the counter's
own guess. The repair is a coin flip there, and it loses on the faithful October-like test. Both
candidates fail the bar.

## Method (`ringtrace.repair`, `analysis/repair.py`)

1. **Shape sets.** Each row gets one shape: its template, else its first family shape, else its
   first one-class mark. A shape allows every class with at least 1.5 % of its fraud-labelled fit
   rows (given labels), and always allows its most common class. Rows with no shape allow all 11
   classes.
2. **Device sets.** A phone allows only the classes that all of its rows' shapes allow.
3. **Anchors.** Phones are ordered by number within each fraud phase. A phone is anchored in two
   cases:
   - a counter anchor: its own label-free anchor names the class the counter gives it, at
     posterior >= 0.80;
   - a template anchor: one of its template rows has a segment in the template's pair at
     posterior >= 0.80, and that class is handed to the event's other fraud-phase rows.

   The longest non-decreasing run of anchors is kept.
4. **Intervals.** Every other phone allows the classes between the kept anchors on either side of
   it, intersected with its device set.
5. **Retype.** A fraud call moves only when its class is outside a non-empty feasible set. Rows of
   one event (or, outside events, of one phone) that share the original class move together.
   - `conservative` moves only when exactly one class is feasible;
   - `full` moves to the feasible class with the highest summed decoder value.

On every test, both variants made the same moves: every feasible set that excluded a call held
exactly one class. The longest-run step never dropped an anchor, because each anchor's class is
the counter's monotone segment.

## Numbers

Calls of the best file (`blend-tabmF-bq`). Changes are in macro-F1, Jul / Aug / Sep. The bootstrap
is the event bootstrap of the mean monthly change, 2,000 draws, 95 %.

| test | rows moved | moves | given | cleaned | gen-only | cleaned precision (new class) |
|---|---:|---|---|---|---|---|
| dense months, stored counter | 1 / 2 / 3 | bot -> mule 6 | -0.00004 / -0.00008 / -0.00012 [-0.00015, -0.00002] | -0.00003 / -0.00008 / -0.00012 [-0.00015, -0.00002] | +0.0000005 / +0.00001 / +0.00001 | 0 / 6 (95 % 0 to 0.39); break-even 0.23 to 0.25 |
| dense months, each month's phones decoded as their own phase | 14 / 2 / 24 | mule -> bot 38, bot -> mule 2 | -0.00046 / -0.00008 / +0.00076 [-0.00047, +0.00076] | -0.00052 / -0.00008 / +0.00092 [-0.00052, +0.00093] | -0.00091 / +0.00001 / +0.00161 | 24 / 40 (0.45 to 0.74): July 0 / 14, September 24 / 24; break-even 0.21 to 0.31 |

The 6 dense-month moves are exactly the 6 rows of the `velocity-muleedge` probe. Each of them sits
just past train's true edge, before two mule-shaped bot-zone fan-ins that the counter puts in the
mule segment. In the month-alone test, every move is a whole mule-burst event at the mule/bot edge.
July's goes the wrong way and September's the right way.

**October-like test.** September is thinned (seeds 13 to 15 x sets r0 to r2), the models are
fitted on July and August, and the counter is rebuilt on the thinned frame. Rebuilding means:

- each phone keeps its block from the unthinned number line;
- the kept September phones are decoded as their own fraud phase from the frame's anchors.

The rebuilt segments agree with the held ones on 0.93 / 0.90 / 0.90 of the rows. The base calls
are the best recipe's. Its anchored template rule reads either the rebuilt counter (`rebuilt`,
which matches October) or the held one (`held`, which the existing gate uses). The lower bound is
pooled over the 9 cells.

| base | rows (9 cells) | moves | given mean / per seed / lower bound | cleaned mean / per seed / lower bound | gen-only mean | cleaned precision |
|---|---:|---|---|---|---|---|
| rebuilt | 24 | takeover -> sim 24 | -0.00038 / -0.00039, 0, -0.00074 / -0.00064 | -0.00049 / -0.00038, 0, -0.00109 / -0.00082 | -0.00083 | 0 / 24 (0 to 0.14); break-even 0.27 |
| held | 114 | mule -> bot 72, takeover -> sim 42 | -0.00017 / -0.00078, +0.00139, -0.00111 / -0.00073 | -0.00029 / -0.00094, +0.00171, -0.00163 / -0.00101 | -0.00034 | 72 / 114 (0.54 to 0.71): mule -> bot 72 / 72, takeover -> sim 0 / 42 |

The three seed sets of a thinning seed make the same moves, so a cell triple counts as one draw.
A retype also removes a call from the old class, so its real break-even sits well above the new
class's F1 / 2. With small classes on both sides it is near one half. In s15, nine wrong
takeover -> sim rows cost 0.0016 cleaned.

**Diagnostic: single-class anchors only.** Only phones whose own shapes allow one class are
anchored, so look-alike rows name a pair, not a class. This leaves:

- no move in the dense months;
- 1 move in the month-alone test (full variant, wrong);
- no move in any October-like cell;
- 1 move in October.

Outside the look-alike edges, then, the calls already fit the device order everywhere. The model
reads the counter's segment posteriors directly (worth +0.028), which explains why.

**October (counts only).** 410 fraud-phase phones and 164 anchored.

- Both variants retype 11 rows: velocity_bot -> mule_account 10 and qr_swap -> merchant_collusion 1.
- The 10 are the rows of the `velocity-muleedge` probe. The candidate differs from that probe on 1
  row.

## Verdict

Rejected. The repair fails the bar on the dense months: every month is negative on given and
cleaned, and all 6 moves are wrong. It also fails the October-like gate: the mean is negative, the
lower bound is below 0, and the faithful base makes 0 right of 24.

The candidate files are written for the record only:

- `repair-conservative.csv` and `repair-full.csv` are identical, with sha256
  `523eeb968727291a92630703acb8983d648aac156c951b1f04e1bdae44c7f395`;
- each has 11 rows changed against `blend-tabmF-bq.csv`.

Device order holds no information the model has not already used. The only open question left at
the edges is the mule/bot edge probe of `velocity-october.md`. This test does not resolve it: the
same edge goes the right way in September and the wrong way in July.

**Commands.**

- `python analysis/repair.py lomo alone octlike build` runs every test, about 30 minutes on 4
  threads. Most of that is the October-like decode, 14 minutes, and the dense-month event
  bootstrap, 5 to 10 minutes.
- `--strict` runs the single-class-anchor diagnostic.

Reports go to `$RINGTRACE_OUT/repair/`.
