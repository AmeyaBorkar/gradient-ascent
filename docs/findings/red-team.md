# Red team: how could a file score 0.557 public, 0.008 above the blend?

**Question.** The leaderboard top is 0.557 public; the blend with the blank-check merchant QR rule
scores 0.5486. List every way a file could be 0.008 or more above it, size each with our own
numbers, test the three most plausible ones, and build any real lever.

**Answer (measured in this repo unless marked; October as aggregates only).** No lever of 0.008
exists in the data that we can see. The most plausible explanation is public-slice luck plus a
small true edge. A file with the same true score as ours beats us by 0.0084 on the public slice
in 0.6-2.2 % of slices. With a true edge of +0.004 that rises to about 12 %. Over the ways a
generator row can still be lost, the whole budget is 0.025; 0.008 would mean fixing a third of
it, and the two scans for October-only misses found nothing beyond the 20 rows the blank-QR rule
already moves. No candidate file was built.

## The budget a rival has to find it in

From `fresh-error-anatomy.md` (cleaned labels, mean of the three held-out months):

| pocket | stake |
|---|---|
| ordinary fraud rows missed | 0.387 |
| generator rows called another fraud type | 0.021 |
| generator rows called legit | 0.0025 |
| legit rows called fraud | 0.0018 |

The public score of 0.5486 against the cleaned out-of-fold 0.5667 is a level shift that a rival
shares: October holds a larger ordinary share (earlier measurement: 2.87 % of rows against 2.65 %
in train), which lowers every file's ceiling equally.

## Ranked hypothesis table

"Max" is the macro-F1 the idea could move on October if fully true. Ranked by plausibility of
explaining 0.008.

| # | hypothesis | max | plausibility | test | result |
|---|---|---|---|---|---|
| 1 | **Public-slice luck** (plus a small true edge) | unbounded | **high** | `redteam.py noise`: slice gap against models 0.027-0.037 apart, slices of 78,175 September rows | sd of the gap error 0.0034-0.0042; P(gap error >= 0.0084) 0.6-2.2 %, >= 0.004 12-16 %. One file's slice sd is 0.0094, but 90-93 % of it is shared by every file (the ordinary rows that land in the slice) |
| 2 | **October shift hides generator rows from our model** (blank-QR-like cells) | about 0.003 (20-30 rows) | low, now tested | `redteam.py cells`: 1-, 2- and 3-way crosses of 12 row columns, cells >= 90 % cleaned fraud in train, October legit calls above the out-of-fold rate by >= 3 rows | one real cell: QR with a blank check, 20 called legit against 0.4 expected; this is the rule already in `blend-blankqr`. One weak cell: COLLECT with a just-under amount, 48 against 36 expected, but out of fold only 14 of its 216 called-legit train rows are cleaned fraud (6.5 %). Flipping them loses. Nothing else |
| 3 | **A pool of October generator rows we under-call** (October-specific fraud pool) | 0.01+ if a whole class | low, now tested | `redteam.py census`: calls per 100k, September out of fold against October | October totals 1,514 fraud calls per 100k against an expected 1,438 structured rows (earlier measurement); every class moves as the known pool shifts predict: bots 220 to 122 (bursts halved), mule 261 to 303 (bursts up), collusion 59 to 32 (3 rings). No class falls short |
| 4 | **Look-alike separator we missed** (six template pairs) | 0.021 (all of pocket c) | low | none new; earlier measurements | event-level AUC of the pairs 0.47-0.57, except bait pairs 0.76 (earlier measurement); 0.008 needs 40 % of the pocket fixed. A perfect separator of all six pairs is worth +0.029 / +0.030 |
| 5 | **Labelling convention we under-call** (onward transfers, first or last row, victim side) | 0.0025 (pocket b) + whatever hides in "ordinary" | very low | none new; numbers already in the repo | generator rows called legit cost only 0.0025 in total. A convention that marks other rows would put fraud-pass rows into the ordinary pool, but that pool's old-app rate is 15.0 %, the legit-pass rate (`validation-blind-spot.md`): fraud-pass share 0 %. Links of ordinary rows to event rows are 0.96-1.08 of chance (earlier measurement) |
| 6 | **Ordinary rows partly findable** | 0.387 | very low | none new | AUC 0.49-0.52 from nine angles, earlier measurement; old-app marker says they are legit-pass rows |
| 7 | **Better October fit of the counter zones** | about 0.003 | low | none new | about 320 single-row fraud-block devices in October, type accuracy 0.90 on unflagged September phones (earlier measurement): about 30 mistyped rows, worth about 0.0001 each on the full test |
| 8 | **Per-class decoding under the clean test labels** | about 0.001 | very low | none new | tuned multipliers and prior-shift decoders tie; an oracle October prior gains +0.00005 (earlier measurement) |
| 9 | **A regime nuance per class** (some class's ordinary rows not counted) | large for one class | very low | none new | the all-legit public 0.0816 fits the full 4.06 % fraud share (earlier measurement); a class without its ordinary rows would move the share by 0.1-0.3 points, inside the reading's resolution, so it is not ruled out by that upload, but nothing in train separates classes' ordinary rows by month or volume (slope -0.02) |

## How luck sizes up

The gap between two files on one public slice moves far less than either score, because the
ordinary rows that fall into the slice hit every file alike (correlation 0.90-0.93 between our
slice error and a rival's). For a rival file with true edge d:

| true edge d | P(public gap >= 0.0084), sd 0.0034 | sd 0.0042 |
|---|---|---|
| 0 | 0.7 % | 2.3 % |
| +0.002 | 3.0 % | 6.4 % |
| +0.004 | 9.8 % | 14.8 % |

The leader is the best of many teams, so a team at d = +0.002 to +0.004 that is one of several
near our level reaches 0.557 by chance at these rates. The final ranking uses the private 70 %,
where the same gap shrinks back towards d.

**Verdict.** Nothing to build. The only real lever the scans found is the blank-check merchant QR
rule, already in `recipes/blend-blankqr.json`. The COLLECT just-under cell is rejected (6.5 %
fraud among its called-legit train rows). A 0.008 public lead is best read as luck on top of an
edge of at most a few thousandths, which no change measured in this repo can match.

**Commands** (box, run `fresh-blend-blankqr`, 4 threads, each under 4 min):
`python analysis/redteam.py noise`, `census`, `cells`; outputs in
`$RINGTRACE_OUT/redteam/<cmd>-fresh-blend-blankqr.json`. Helpers in `analysis/redteam_core.py`,
tested in `tests/test_redteam_core.py`. The analysis decodes the blend without the blank-QR rule
(the fixed point and the anchored template rule), so the scan sees the 20 rows the rule moves.

**Caveats.** `evidence-nophones` and `champion-without-phones` gave identical calls; they count
as one comparison. The luck table uses a normal approximation of the gap error.
