# The counter's segments at October's size: the decoder gains, the rule does not

**Question.** October's fraud phase holds one sixth of train's phones, so its counter segments rest
on one sixth of the anchors. Thinning train's phase to that size loses 8.6 ± 5.1 of 171 unanchored
fraud rows (`farlook.md`, lever 1). Can what train's own phase teaches win those rows back on
October, and does that move October's calls?

**Answer (measured in this repo).** Yes for the segments: a prior on where each cut falls and a
per-segment gap emission raise thinned agreement by **+0.030** (z 5.1, 30 seeds; +0.036 on the
seeds not used for the choice). The recovery does not reach the calls:

- In October the prior moves the segment of 4 rows, against about 14 expected at October's size
  from the simulation.
- The rows a counter rule would then change are 5 to 10 rows where the model calls velocity_bot on
  a mule-segment phone.
- In train, when the model's fraud type disagrees with the counter's segment, the model is right on
  17 of 19 rows, and on 6 of 6 of this exact pattern.
- The rule fails leave-one-month-out in September.

**No candidate passes, and none is recommended for upload.**

## 1. The change

`phones.decode_phase(..., prior=None, edge=None)`. With `prior=None` (the default) the decoder is
bit-identical to before: the thinning baseline reproduces to the last digit, and the rebuilt
`phones` family equals every earlier cached copy. `phones.counter(..., teach=...)` learns the
prior from the train split's own fit and decodes the test split with it. The prior's two parts:

1. **Cut positions.** Train's ten cuts are fractions of its phase's number line. Each cut is the
   midpoint between the last phone of one segment and the first phone of the next:
   0.125, 0.211, 0.299, 0.373, 0.690, 0.763, 0.890, 0.929, 0.951, 0.975.

   A path that enters segment k between phones t - 1 and t takes the log of a normal's mass
   between their two positions (centre: train's cut; sd: the spread). A path that stops before k
   takes the mass beyond the last phone.

   This replaces the uniform prior over paths. It weights a cut by the width of the gap it sits in,
   which is right for a fixed cut. The gap that contains a fixed point is length-biased. That is why
   even a spread of 0.1, flat across any one gap, adds +0.016.
2. **Gap emission.** The unused numbers before each phone are geometric at the segment's occupancy.
   Occupancies are relative to the phase's mean (train: 1.28, 0.86, 1.10, 1.72, 0.64, 1.38, 0.82,
   1.19, 1.38, 1.18, 1.13). They are rescaled to the decoded phase's own mean, so a thinned or a
   short phase both read them correctly.

October's own plain fit puts its cuts later than train's by +0.003, +0.013, +0.013, +0.020,
+0.014, +0.028, +0.012, +0.010, +0.005 and -0.001. That is the mismatch a train-built prior faces
in October.

## 2. First gate: the thinning simulation (passes)

`python analysis/counter.py gate`, 35 s. Seeds 0 to 29 keep each phone of train's phase with
probability 1/6 and decode the kept phones without and with the prior. The prior of each draw is
learned from the phones outside it, so it sits on exactly the same number line as the draw.

That makes it more exact than any train-built prior can be in October. So every setting was also
run with the prior's cuts shifted by October's offset above (the "shifted" column).

The spread and the gap weight were chosen on seeds 0 to 14 and reported on seeds 15 to 29. Two
selection rules were used:

- **As specified (the best plain agreement)** picks spread 0.005, gap weight 0.5. It is the best in
  the plain simulation (+0.041) and the worst prior under the shift (+0.013).
- **Adopted: the larger of the worse of plain and shifted** picks **spread 0.01, gap weight 1.0**.

Baseline thinned fit: 0.897 ± 0.028 on unanchored fraud rows, 0.800 in the last four segments.
Full block: 0.942 and 0.885. Every setting tried:

| spread / gap weight | choose half (0-14): plain, shifted | report half (15-29): agreement change (z), last four | all 30: agreement change (z), last four, rows lost / won | shifted prior, all 30: agreement, last four | full block taught by itself / by October's fit |
|---|---|---|---|---|---|
| none / 0.25 | -0.0029, -0.0029 | +0.0035 (1.5), -0.002 | +0.0003 (0.2), -0.003, 0.4 / 0.4 | +0.0003, -0.003 | +0.0010 / +0.0000 |
| none / 0.5 | -0.0013, -0.0013 | +0.0108 (3.5), +0.039 | +0.0048 (2.1), +0.024, 0.7 / 1.5 | +0.0048, +0.024 | -0.0019 / +0.0068 |
| none / 1.0 | +0.0015, +0.0015 | +0.0143 (2.6), +0.046 | +0.0079 (2.4), +0.037, 1.9 / 3.2 | +0.0079, +0.037 | -0.0019 / +0.0048 |
| 0.005 / 0.0 | +0.0385, +0.0107 | +0.0440 (4.9), +0.107 | +0.0413 (7.7), +0.111, 1.3 / 8.4 | +0.0129, +0.062 | -0.0010 / +0.0010 |
| 0.005 / 0.25 | +0.0377, +0.0109 | +0.0440 (4.9), +0.112 | +0.0409 (7.6), +0.113, 1.4 / 8.4 | +0.0126, +0.059 | +0.0010 / +0.0010 |
| 0.005 / 0.5 | +0.0386, +0.0105 | +0.0433 (5.0), +0.116 | +0.0409 (7.7), +0.113, 1.6 / 8.6 | +0.0126, +0.059 | +0.0010 / +0.0010 |
| 0.005 / 1.0 | +0.0376, +0.0096 | +0.0404 (4.6), +0.115 | +0.0390 (7.1), +0.113, 2.0 / 8.7 | +0.0122, +0.061 | +0.0010 / +0.0048 |
| 0.01 / 0.0 | +0.0272, +0.0198 | +0.0381 (4.2), +0.096 | +0.0327 (6.0), +0.097, 2.0 / 7.6 | +0.0244, +0.073 | -0.0010 / -0.0010 |
| 0.01 / 0.25 | +0.0264, +0.0198 | +0.0385 (4.3), +0.096 | +0.0325 (6.0), +0.097, 2.0 / 7.6 | +0.0236, +0.073 | +0.0010 / +0.0010 |
| 0.01 / 0.5 | +0.0256, +0.0207 | +0.0368 (4.0), +0.096 | +0.0312 (5.6), +0.097, 2.1 / 7.4 | +0.0240, +0.070 | +0.0010 / +0.0010 |
| **0.01 / 1.0** | +0.0240, +0.0230 | **+0.0360 (3.7), +0.100** | **+0.0300 (5.1), +0.093, 2.7 / 7.9** | **+0.0265, +0.080** | +0.0010 / +0.0048 |
| 0.015 / 0.0 | +0.0234, +0.0200 | +0.0303 (3.8), +0.094 | +0.0269 (5.4), +0.094, 2.1 / 6.7 | +0.0228, +0.079 | -0.0010 / -0.0010 |
| 0.015 / 0.25 | +0.0224, +0.0212 | +0.0298 (3.8), +0.094 | +0.0261 (5.2), +0.091, 2.1 / 6.6 | +0.0244, +0.082 | +0.0010 / +0.0010 |
| 0.015 / 0.5 | +0.0224, +0.0204 | +0.0291 (3.6), +0.094 | +0.0257 (5.1), +0.091, 2.4 / 6.9 | +0.0243, +0.085 | +0.0010 / +0.0010 |
| 0.015 / 1.0 | +0.0219, +0.0205 | +0.0278 (3.3), +0.094 | +0.0248 (4.8), +0.089, 2.7 / 7.0 | +0.0236, +0.080 | +0.0010 / +0.0048 |
| 0.02 / 0.0 | +0.0204, +0.0197 | +0.0262 (3.0), +0.094 | +0.0233 (4.6), +0.086, 2.4 / 6.4 | +0.0236, +0.076 | -0.0010 / -0.0010 |
| 0.02 / 0.25 | +0.0201, +0.0193 | +0.0278 (3.4), +0.094 | +0.0239 (4.9), +0.086, 2.2 / 6.3 | +0.0243, +0.084 | +0.0010 / +0.0010 |
| 0.02 / 0.5 | +0.0181, +0.0185 | +0.0291 (3.6), +0.094 | +0.0236 (4.8), +0.086, 2.5 / 6.6 | +0.0244, +0.083 | +0.0010 / +0.0010 |
| 0.02 / 1.0 | +0.0176, +0.0172 | +0.0278 (3.3), +0.094 | +0.0227 (4.5), +0.084, 2.8 / 6.7 | +0.0221, +0.073 | +0.0010 / +0.0048 |
| 0.03 / 0.0 | +0.0169, +0.0169 | +0.0210 (2.8), +0.091 | +0.0189 (4.1), +0.084, 2.6 / 5.9 | +0.0210, +0.084 | -0.0010 / -0.0010 |
| 0.03 / 0.25 | +0.0177, +0.0165 | +0.0206 (2.5), +0.091 | +0.0192 (4.0), +0.085, 2.7 / 6.0 | +0.0208, +0.084 | +0.0010 / +0.0010 |
| 0.03 / 0.5 | +0.0181, +0.0169 | +0.0222 (2.8), +0.091 | +0.0202 (4.2), +0.085, 2.6 / 6.1 | +0.0216, +0.082 | +0.0010 / +0.0010 |
| 0.03 / 1.0 | +0.0172, +0.0156 | +0.0209 (2.5), +0.087 | +0.0191 (3.9), +0.081, 3.2 / 6.6 | +0.0205, +0.076 | +0.0010 / +0.0048 |
| 0.05 / 0.0 | +0.0149, +0.0149 | +0.0186 (2.6), +0.091 | +0.0168 (3.8), +0.084, 2.6 / 5.5 | +0.0174, +0.084 | -0.0010 / -0.0010 |
| 0.05 / 0.25 | +0.0158, +0.0150 | +0.0164 (2.4), +0.091 | +0.0161 (3.9), +0.085, 2.7 / 5.5 | +0.0160, +0.085 | +0.0010 / +0.0010 |
| 0.05 / 0.5 | +0.0145, +0.0149 | +0.0179 (2.6), +0.091 | +0.0162 (3.9), +0.082, 2.9 / 5.8 | +0.0180, +0.082 | +0.0010 / +0.0010 |
| 0.05 / 1.0 | +0.0153, +0.0153 | +0.0209 (2.5), +0.087 | +0.0181 (3.8), +0.081, 3.2 / 6.4 | +0.0179, +0.080 | +0.0010 / +0.0048 |
| 0.1 / 0.0 | +0.0129, +0.0129 | +0.0186 (2.6), +0.091 | +0.0157 (3.6), +0.084, 2.6 / 5.4 | +0.0157, +0.084 | -0.0010 / -0.0010 |
| 0.1 / 0.25 | +0.0141, +0.0137 | +0.0152 (2.4), +0.091 | +0.0146 (3.7), +0.085, 2.7 / 5.3 | +0.0145, +0.085 | +0.0010 / +0.0010 |
| 0.1 / 0.5 | +0.0145, +0.0145 | +0.0152 (2.3), +0.091 | +0.0148 (3.7), +0.082, 3.0 / 5.6 | +0.0148, +0.082 | +0.0010 / +0.0010 |
| 0.1 / 1.0 | +0.0153, +0.0153 | +0.0209 (2.5), +0.087 | +0.0181 (3.8), +0.081, 3.2 / 6.4 | +0.0179, +0.080 | +0.0010 / +0.0048 |

The full-block columns differ by one row of 1,032 (±0.0010), except that October's fit as teacher
gains 5 rows at gap weight 1.0.

**Verdict: spread 0.01 / gap weight 1.0 passes every part of the first gate.** On the report half
it gains +0.036 with z 3.7, and +0.030 with z 5.1 on all 30 seeds. The last four segments gain
+0.100 and +0.093. The full block does not fall: +0.001 taught by itself, +0.005 taught by
October's fit. Under the October shift it still gains +0.027, and +0.080 in the last four.

**Which rows the taught segment can be trusted on** (`python analysis/counter.py calib --spread
0.01 --gap-weight 1.0`, 30 seeds). These are the unanchored fraud rows whose taught confidence
reaches 0.9:

| rows | per draw | taught segment right | plain segment right | taught mule, right |
|---|---|---|---|---|
| plain confidence below 0.9 | 10.6 | 0.940 | 0.718 | 85 rows, 0.929 |
| the same, prior shifted by October's offset | 12.3 | 0.930 | 0.673 | 107 rows, 0.925 |
| plain confidence 0.9 or more | 141.4 | 0.957 | 0.957 (same segment) | |
| taught confidence below 0.9 | 18.5 | 0.695 | 0.542 | |

## 3. October, counted only

`python analysis/counter.py october --spread 0.01 --gap-weight 1.0`:

- The train split's columns are unchanged (bit-identical).
- Of 564 fraud-block rows, the taught fit moves the segment of **4**, all unanchored, all below 0.9
  under the plain fit and below 0.9 under the taught fit.
- Of 213 unanchored rows, 35 sit below 0.9 under the plain fit and 23 under the taught fit. 14 rise
  from below 0.9 to 0.9 or more, and 2 fall.

The simulation moves 11.0 rows a draw out of 171, about 14 at October's 213. October moves 4.

The model already follows the segment: the final recipe's October calls equal it on 545 of the 564
rows. Of the 4 moved rows, the call equals the plain segment on 3.

## 4. Second gate: the counter rule after `anchored_templates` (fails)

The challengers add the counter rule `{"name": "counter", "trusted": 11, "min_conf": 0.9,
"unanchored": true}` after the template rule, with `source: "cpr"` (the taught segments, family
`phones_prior`) unless noted. Three variants:

- **taught:** as above.
- **taught, plain below 0.9:** adds `below: 0.9`, so the rule only touches rows the plain counter
  was unsure of.
- **plain control:** the same rule on the plain counter (`source: "ctr"`).

On train the taught segments equal the plain ones, so the leave-one-month-out table measures the
rule itself.

`python analysis/counter.py table recipes/final.json recipes/cand/counter-final*.json --alias
final-tabm=tabm`: 276 s, laptop CPU, 8 threads, 2,000 event-bootstrap draws. Change per month is
Jul / Aug / Sep, with the 95 % interval of the mean:

| challenger | given | cleaned | gen-only | rows train / October | verdict |
|---|---|---|---|---|---|
| taught | -0.00004 / 0 / -0.00003 [-0.00007, +0.00002] | -0.00003 / 0 / -0.00003 [-0.00007, +0.00002] | +0.000001 / 0 / -0.00011 | 4 / 10 | FAIL |
| taught, plain below 0.9 | 0 / 0 / -0.00003 [-0.00003, 0] | 0 / 0 / -0.00003 [-0.00003, 0] | 0 / 0 / -0.00008 | 1 / 5 | FAIL |
| plain control | as taught | as taught | as taught | 4 / 5 | FAIL |

The same rules on the best file's recipe (`recipes/final-half.json`, which rebuilds
`blend-tabmF-bq.csv` with 0 of 260,582 rows different): `python analysis/counter.py table
recipes/final-half.json recipes/cand/counter-tabmF*.json --alias final-tabm=tabm`, 271 s.

| challenger | given | cleaned | gen-only | rows train / October | verdict |
|---|---|---|---|---|---|
| taught | -0.00004 / 0 / -0.00004 [-0.00007, 0] | -0.00003 / 0 / -0.00004 [-0.00006, 0] | 0 / 0 / 0 | 2 / 10 | FAIL |
| taught, plain below 0.9 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 5 | exact tie: train cannot see it |
| plain control | as taught | as taught | as taught | 2 / 5 | FAIL |

October-like is n/a on the laptop: the local cell set is incomplete. It could not see this change
anyway: the cells hold both counter families from the unthinned frame, where the taught segments
equal the plain ones.

The train rows the rules change (train rows are inspected):

- **The plain rule's 4 rows.** Two mule-segment rows (confidence 0.906 and 0.936) called
  velocity_bot are set to mule_account. Both are velocity_bot under the given and the cleaned labels,
  so both are wrong. One legit call on a layering_chain segment is set right. One legit call is
  forced to fake_cashback, where the truth is loan_app_extortion.
- **The `below` variant's one row** is that forced call. The rule's first part turns a legit call on
  a fraud-block phone into the model's best fraud type, and here the type was wrong.
- **On the best file's recipe** the plain rule changes only the two wrong bot -> mule rows.

## 5. Model against counter when they disagree

On train, the final recipe's out-of-fold calls on unanchored fraud-block rows (1,159) are a fraud
type different from the counter's segment on 19 rows:

| plain segment confidence | rows | segment right | call right |
|---|---|---|---|
| below 0.6 | 13 | 1 | 11 |
| 0.6 to 0.9 | 4 | 0 | 4 |
| 0.9 to 0.99 | 2 | 0 | 2 |
| segment mule_account, call velocity_bot (confidence 0.857 to 0.936) | 6 | 0 | 6 |

The model sees the bot's behaviour, and the counter only sees where the phone sits on the number
line. The simulation's 93 % holds for all rows with a confident taught segment, not for the rows
where the model has already disagreed. Every October row the rule changes is velocity_bot to
mule_account: the pattern the model wins 6 of 6 times on train.

## 6. Candidates

Built with `python analysis/counter.py build <base> <challengers> --alias final-tabm=tabm`. The
local `tabm` run is byte-identical to the box's `final-tabm` (oof sha256 `5bef8fbc...`, test
`64b2992c...`). The rebuilt base `final-half` reproduces `blend-tabmF-bq.csv`: every challenger
differs from the file and from the base on the same rows. All files are valid against the sample
submission.

| file (`candidates/`) | sha256 | rows vs `blend-tabmF-bq` | moves vs its base | plain confidence 0.9 or more among them | train rows | verdict |
|---|---|---|---|---|---|---|
| `counter-final.csv` (final + taught rule) | `1d5d5dae...e099` | 12 (10 bot -> mule, 2 phishing -> legit from `final`) | 10 bot -> mule | 5 | 4 | stop (changes rows confident under the plain fit); fails LOMO |
| `counter-final-below.csv` (final + taught rule, plain below 0.9) | `69927264...0f43` | 7 (5 bot -> mule, 2 phishing -> legit) | 5 bot -> mule | 0 | 1 | fails LOMO (Sep); the train pattern says these moves are wrong |
| `counter-final-plain.csv` (control) | `646f779c...2a78` | 7 | 5 bot -> mule | 5 | 4 | fails LOMO |
| `counter-tabmF.csv` (best file + taught rule) | `2eaa5e7a...d5f2` | 10 bot -> mule | 10 bot -> mule | 5 | 2 | stop |
| `counter-tabmF-below.csv` (best file + taught rule, plain below 0.9) | `cbd2986e...fe6` | 5 bot -> mule | 5 bot -> mule | 0 | 0 | LOMO is an exact tie; the train pattern says these moves are wrong |
| `counter-tabmF-plain.csv` (control) | `533e5a25...6a76` | 5 bot -> mule | 5 bot -> mule | 5 | 2 | stop |

Full sha256 are in `RESULTS.md`. **Nothing passes, and no file is worth a public slot.**

`counter-tabmF-below` is the closest: no train row moves, and the simulation favours its rows.
Section 5 makes it worse in expectation, though, and a file that is worse in expectation should not
be uploaded as a probe (`farlook.md`, section 6).

The one change the evidence does not oppose is the 4 moved segments. A rule limited to them would
change 3 October rows (velocity_bot -> mule_account, velocity_bot -> layering_chain,
loan_app_extortion -> fake_cashback; taught confidence 0.63 to 0.84). The simulation's moved rows
are right 7.9 times for every 2.7 wrong, which gives about +0.0001 to +0.0002 in expectation. That
is too small to tell apart on the public score. It was not built.

## 7. What this closes

The thinning loss in `farlook.md` was measured on segments. On October's calls it is mostly gone
already:

- The model reads the counter's posteriors and calls the segment on 545 of 564 fraud-block rows.
- Where the two disagree, the model is usually right.
- The taught decoder moves only 4 October segments.

Lever 1 is worth about 0 on October, not +0.0005. The decoder change stays in the code behind its
switch (default off). The `phones_prior` family and the rule's `source`, `unanchored` and `below`
options are used by no recipe.

**Commands.** `RINGTRACE_THREADS=8 PYTHONPATH=src python analysis/counter.py gate | calib | october
| table | build` (reports in `$RINGTRACE_OUT/counter/`). Helpers are in `analysis/counter_core.py`
(tests: `tests/test_counter_core.py`). Decoder tests are in `tests/test_phones_prior.py`, and rule
tests in `tests/test_rules.py`.
