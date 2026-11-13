# CV against public: the gaps and why near-identical files differ by chance

Summary: the champion's public score (0.5470) sits 0.0192 below its cleaned LOMO (0.5662) and 0.0344
above its given-label LOMO (0.5126). That is the expected pattern: the test labels are clean but
still include ordinary-looking fraud rows. The public slice has an sd of 0.010-0.027 for one upload
and cannot rank files that differ on a few dozen rows. All numbers are **earlier measurements**
except those marked as measured in this repo (see [RESULTS.md](../../../RESULTS.md)).

## The gaps

| Quantity | Value | Status |
|---|---|---|
| Champion LOMO, given labels | 0.5126 (0.5130 / 0.5093 / 0.5154) | earlier measurement; this repo's rebuild 0.5122, measured in this repo |
| Champion LOMO, cleaned labels | 0.5662 (0.5641 / 0.5635 / 0.5709) | earlier; rebuild 0.5665, measured in this repo |
| Champion public | **0.5470** | earlier |
| Forecast (A-clean, from the cleaned out-of-fold months, minus 0.019) | about 0.54 | earlier |
| Independent rebuild public | 0.5457 (-0.0013 vs champion) | earlier |
| Best public after blends | 0.5488 | earlier |

Why public sits below cleaned LOMO:
- The cleaned labels still treat some ordinary-looking fraud as legit. A truer "A-clean" proxy adds
  the 3,482 train fraud-pass rows that cleaning leaves legit as misses; it is about 0.019 lower than
  the cleaned score.
- Macro-F1 on a 30% slice is biased down: the public mean is about 0.002 (row split) to 0.006 (event
  split) below the month's score.
- October is expected 0 to 0.01 below the train months (bots and collusion are about halved, sim
  over-called by about 25 per 100,000).
- In the October-like September rehearsal, the transfer cost is almost nothing: the champion scores
  0.511-0.520 given and 0.568-0.576 cleaned, close to LOMO September (0.5154 / 0.5709). In this repo
  the rebuilt champion scores 0.5093 mean over nine October-like cells against 0.5097 for September
  at full density (measured in this repo; see RESULTS.md, `python -m ringtrace.octlike`).

CV and the board agreed on the champion and ranked the pair {champion, rebuild} within noise. After
each upload the CV-to-board ranking should be checked; if it stops matching, stop and look for a
shift, a leak or a metric difference before the next upload.

## Public-slice noise (September stood in for October)

Method: draw an October-sized month (260,582 rows, whole events) from each train month, split 30/70
by row or by whole event, 2,000 draws per month and split type. Truth: labels cleaned by detector
events and zones. Ranges are over July, August, September.

| Quantity (macro-F1) | Row split | Event split |
|---|---|---|
| sd of one public score | 0.013-0.014 (older run: 0.010) | 0.023-0.025 (older: 0.022-0.027) |
| sd of public minus whole-test score | 0.008-0.009 | 0.021-0.023 |
| sd of public minus private | 0.012-0.013 (older: 0.012-0.014) | 0.030-0.032 |
| sd of the paired public gain, two files about 300 rows apart | 0.0037-0.0039 | 0.0047-0.0050 |
| Smallest public gain that is not noise (1.645 sd) | 0.005 | 0.007-0.008 |
| Smallest gain shown 95% of the time, other estimate | 0.004-0.008 (given-like) / 0.006-0.013 | |

Probability that public and private order a pair differently, by true gap (files about 300 rows
apart):

| True gap | 0.002 | 0.005 | 0.010 | 0.015 |
|---|---|---|---|---|
| Row split | 34% | 5.5% | 0.1% | 0% |
| Event split | 42% | 10% | 0.5% | 0% |
| Row split: no visible public gain at all | 27% | 5.5% | 0.1% | 0% |

- A gap of 0.03 (rules plus zones over rules only) never flipped (0 of 6,000 draws).
- Belief: paired noise grows with the number of rows two files disagree on. For two different models
  (thousands of rows) the smallest real gain approaches 1.645 times the single-upload sd: about 0.015
  (row) or 0.036 (event).
- Whether the organisers split by row or by event was not answered; the thresholds differ by about
  1.4x for near-identical files and 2.5x for one upload.

## Why near-identical files differ by chance

- A public difference comes only from the rows two files label differently. Upload 3 changed 17 rows,
  about 5 of them in the public slice; upload 4 changed 9 rows relative to upload 3; upload 5 15.
- Rare-class sensitivity: in a public slice of about 78,000 rows, one extra right row moves
  macro-F1 by 0.0002 (mule_account) up to 0.0013 (loan_app_extortion) and 0.0011
  (merchant_collusion). So +0.0013 can be a single row, and upload 3's +0.0017 is a row or two.
- The independent rebuild differs from the champion on 59 September rows; the difference has sd
  0.0007 over public-sized slices (5-95% -0.0016 to +0.0007); -0.0013 or worse occurs in 11% of
  slices. The public-minus-private sd of one file is 0.0117.
- Uploads 4 and 5 tie at 0.5488 despite 15 different rows: their changes net to zero in the slice.

## Selection effects (best public becomes the entry)

Synthetic simulator (one month, macro-F1 0.515, 30% of fraud isolated and never found, events of
2-30 rows; c is the share of events a change rewires):

| Quantity | Row split | Event or time split |
|---|---|---|
| sd(public - private), one upload | 0.009 | 0.020 |
| sd(public gain - private gain), c = 4% / 15% / 40% | 0.003 / 0.006 / 0.010 | 0.007 / 0.013 / 0.023 |
| corr(public gain, private gain), equally good uploads | 0.63 | 0.00 |
| Private gain of the public-best of N = 5 / 10 / 20 equal-CV candidates (c 15%) | +0.006 / +0.008 / +0.010 | +0.001 / +0.002 / +0.002 |
| Public minus private of that pick | 0.004 / 0.005 / 0.006 | 0.013 / 0.017 / 0.020 |

- A candidate worse by 0.01 still beats the incumbent on public 10% (row split) to 19% (event split)
  of the time at c = 15%; so nothing goes up that CV puts below the incumbent.
- The best public of eight candidates spread over 0.013-0.019 is truly the best in 66-84% of draws
  and costs 0.002-0.003 on private.
- Expect the final private score 0.005 (row split) to 0.02 (event or time split) below the best
  public score, since the entry is selected on public.
- Across 120 public competitions, public and private agree within about 2%; exceptions had tiny
  effective test sets. For macro-F1 here the public slice holds a few thousand fraud rows and a few
  dozen events of the rarest classes, which is a small test set.
- Detecting +0.01 with 80% power in a paired event bootstrap needs about 43,000 x c events
  (1,700 for a tweak, 6,400 for a new blend member, 17,000 for a different model). Three validation
  months hold a few thousand events, so a different model must win by more than about 0.015 to be
  distinguished from a tie.

## Practical rules from this

1. Decide on LOMO and October-like cells with event-bootstrap intervals, never on public.
2. Treat a public difference under 0.005 (similar files) as zero.
3. Compute a change footprint (rows changed per class pair, expected count in the public slice)
   before uploading; a footprint bounding the public change under 0.005 is a lottery ticket.
4. One-factor uploads are readable on the public board (a +0.012 change touching few events is read
   correctly 96-100% of the time under a row split); bundles are not.

Decision rules built on these facts are in [decisions.md](decisions.md).
