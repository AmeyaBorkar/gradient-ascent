# The two label regimes, the regime-hybrid detector and the upload record

Status convention. "Measured in this repository" means this repo's own runs (its champion scores
0.5122 given / 0.5665 cleaned leave-one-month-out). Everything below is an "earlier measurement"
from earlier research, which a later pass may re-derive here. The regime gate itself was not
re-run in this repo.

Summary. The training labels contain a large block of "ordinary" fraud rows that look exactly like
legit rows (about 63% of fraud labels). Two readings of the hidden October labels were possible:
ordinary rows count as fraud (about 4.06% fraud share) or only generator-made rows count (about
1.5%). A detector gated to call fraud only where there is generator evidence wins 0.020 under the
second reading and loses 0.015 under the first. An all-legit upload settled it before any model file
was uploaded: it scored 0.0816, so ordinary rows count as fraud and the gated detector was never
uploaded. The "hybrid" of the two readings turned out to be the same model, and every real mix of
the two loses. The public leaderboard is too noisy to rank near-identical files: five uploads
ended at 0.5488 public, a +0.0018 gain that is a few rows of noise.

## Overview

| experiment | verdict | effect |
|---|---|---|
| All-legit upload to read the regime | helps (decisive) | 0.0816: ordinary rows count; about 40 sd from the other reading |
| Generator-evidence gate (regime-hybrid detector) | helps only if ordinary rows are legit | +0.020 / -0.015 |
| Decoder variants under the generator-only reading | tie | within +-0.001 |
| Real mixes of the two readings (per row, class, family, stack) | loses | -0.0008 to -0.106 vs the champion |
| Oracle gate (ceiling) | cannot pay | +0.0012 LOMO, +0.0015 October-like |
| Paired-check upload plan | not needed | planned, never run |
| Upload 2 (XGBoost champion) | as forecast | 0.5470 public |
| Upload 3 (three-family rank blend) | tie that won public | 0.5487 (+0.0017, 17 rows) |
| Uploads 4 and 5 | ties | 0.5488, 0.5488 |
| Upload rule "consistently better" | adopted | per-seed, per-month, three targets |
| Public leaderboard noise | large | sd 0.010-0.013 one file, 0.0007 near-identical pair |

## 1. What the two regimes are
- Background. In training, 36k fraud-labelled rows are "ordinary": no event, no mark, no counter
  zone, indistinguishable from legit rows (every method tried scored about 0.50 AUC on them). The
  test labels are stated to be clean, and some fraud in train was unreported or mis-typed.
- Ordinary-rows-count reading (the clean test regime): every fraud-pass row is fraud, ordinary
  rows included. Test fraud share 4.061% (10,582 rows). A noisy variant keeps the train noise
  (12% unreported): 3.654%.
- Ordinary-rows-legit reading (the generator-only regime): only rows made by the fraud generator
  are fraud; ordinary rows are legit. Fraud share about 1.50-1.52% (counts-based 1.520%, model-based
  1.465%).
- Why it mattered: the ordinary rows are the largest term of the information gap (an oracle that
  fixed them is worth +0.43 macro-F1). Under the legit reading the ceiling jumps from about 0.61
  to about 0.9 and the decoder lane has no effect; under the count reading it stays near 0.56-0.57.
  See [ordinary-rows.md](ordinary-rows.md) and [labels.md](labels.md).

## 2. Reading the regime with an upload slot
- Idea. A file that labels every row legit scores F = 2(1-s) / (12(2-s)) for a test fraud share s,
  so the public score of that file inverts to the share: s = (1-12F) / (1-6F). One public point of
  fraud share moves the score by 0.00043.
- Expected public scores of the all-legit file (public slice 78,175 rows): 0.081606 (ordinary rows
  count, 3,175 fraud rows in the slice), 0.081782 (noisy variant), 0.082695-0.082719 (legit
  reading, 1,145-1,188 fraud rows), 0.082772 (legit reading with noise).
- Sampling noise of that reading is tiny: sd 0.000026 under a row split, about 2.3 times that under an
  event split. Gap between the readings: about 40 sd by row, 18 sd by event. Four decimals suffice
  (0.0816 against 0.0827).
- Decision table prepared in advance: at or below 0.08170 means ordinary rows count; 0.08170-0.08185
  means the noisy variant; 0.08185-0.08260 is unexplained, so stop, run the label-free census audit
  and upload no model file; at or above 0.08260 means the legit reading.
- Result. The all-legit file (upload 1) scored 0.0816: ordinary rows count. The champion was then
  uploaded; the gated file was not. The gate's cost under this reading (-0.015) is what would have
  been paid by uploading it blind.
- Lesson. Spend one slot on a file that cannot win but decides the biggest modelling fork; compute
  the reading table first and set the stop line before the score appears.

## 3. The generator-evidence gate (regime-hybrid detector)
- Idea. Under the legit reading, ordinary rows are legit, so a fraud call on a row with no
  generator evidence is a false positive. The gate sends a fraud call to legit unless the row has a
  mark or a label-free event-family id.
- Measurement (earlier), leave-one-month-out against the generator-only target: 0.8913 to 0.9113,
  +0.020 in all three months (a leave-one-month-out choice among decoders always picks a gated rule,
  held-out 0.9115). Under the count reading it costs -0.015 (0.5662 to 0.5512 cleaned). The rows it
  switches off are fraud-block phones with no mark and no event: 1,146 in train, 1,019 of which are
  ordinary rows. Fraud calls on legit-labelled rows fall from 1,265 to 85; only 6 generator rows are lost.
- On test: about 221 rows (85 per 100k) differ, 210 of them in a counter zone; 564 fraud-block test
  rows, 354 called fraud by the gated file and all 564 by the ungated file.
- Verdict. Helps under the legit reading, loses under the count reading. Never uploaded.
- Lesson. A regime-specific stage is a bet on the labels; keep it behind a switch and let a cheap
  probe pick the path.

## 4. The hybrid: one model, and every real mix loses
- Idea. The gated detector wins generator-only rows; maybe it also types template rows better, so
  use it on evidence rows and the champion elsewhere (per row, per class, per event family, or a
  stacker over both).
- Measurement (earlier), October-like September, 3 thinning seeds x 3 model seeds, plus
  leave-one-month-out on five targets (baseline 0.5352 cleaned October-like).
  - The two files are the same model: the gate is the last step on top of identical probabilities,
    so "gated model on evidence rows, champion elsewhere" changes 0 rows in every cell.
  - Gate on all rows: -0.0182 cleaned October-like, -0.0152 LOMO, +0.0171 / +0.0204 on
    generator-only.
  - Gate only counter-zone rows: -0.0164 (carries the whole loss and the whole generator-only gain);
    gate only outside zones: -0.0017 October-like, -0.0001 LOMO (about 30 calls a month).
  - Confidence cuts: -0.0008 to -0.0028 October-like; the fitted cut is "no gate" in every LOMO month.
  - One class gated at a time: -0.0009 to -0.0034 October-like (every class, every thinning seed),
    -0.0010 to -0.0016 LOMO; best class set on the scored month is empty.
  - Stripping a family's evidence: -0.063 to -0.106.
  - Logistic stacker over the evidence and zone flags: -0.0030 October-like, -0.0007 LOMO.
  - Oracle that gates exactly the wrong calls: +0.0015 October-like, +0.0012 LOMO (a fifth of the
    +0.007 bar).
- Why the gate "wins": the generator-only target counts zone-only rows as ordinary. About 413 fraud
  calls per month on non-evidence rows, 93% on a counter zone; on cleaned labels 82% of them are right
  (329/357/324 of 413/427/399), on the generator-only target 96% are legit. So its +0.020 is the
  target's definition, not better typing. A proxy that keeps zone rows as fraud shows -0.018.
- Verdict. Loses. No hybrid file exists to build.
- Lesson. Check that a "second model" is not the same model with a post-processing step before
  designing a blend. Read generator-only gains with the zone-row blind spot in mind. See
  [rules.md](rules.md) and [counter.md](counter.md).

## 5. The regime playbook
- Candidates built and validated (aggregates only): the champion file, its gated twin, template
  variants of each, an anchor baseline, and the all-legit file. The gated twins differ from their
  ungated twins on exactly the gated rows (221 fraud-to-legit, plus 8 type changes in the oldest pair
  from a since-disabled stage).
- Expected public scores (leave-one-month-out, three months mean): ungated champion about 0.541
  (A-clean target), 0.560 on the cleaned-silver target, 0.507 on given labels, 0.891 under the
  generator-only reading; gated twin 0.526 / 0.545 / 0.491 / 0.911; anchor baseline 0.511 / 0.529 /
  0.474 / 0.873 (September only, optimistic by about 0.003).
- Decoder under the legit reading: fixed-point scale 0.5 to 3, a correction subtracting the ordinary
  mass, and thresholds 0.2-0.8 all land at 0.8872-0.8923; the gate is the only lever. Template rule
  under that reading: +0.0039 / +0.0032 (a tie unless the bootstrap clears it). What is left there is
  look-alike typing: 1,965 wrong-type rows.
- Options: count reading, upload the champion once (do not upload both near-identical template
  variants); noisy variant, same file, expect 0.507; legit reading, upload the gated file expecting
  about 0.91, then the ungated twin as a paired check (expect it 0.020 lower, paired sd at most
  0.005; if it were 0.008 or more higher, those 210 rows count as fraud and the gate goes); unexplained
  or second upload more than 0.045 below forecast: stop and audit.
- Do not upload: decoder variants of one model (ties within +-0.001, picking the best of several costs
  about 0.002-0.003 on private), two near-identical files hoping to pick by public score, the gated file
  under the count reading, stale duplicates, or anything chosen by a public gap below the noise floor.
- Pair noise in the simulation: public order of the gated/ungated pair never flipped in 2,400 draws.

## 6. Uploads and what they showed
- Upload 1, all-legit file: 0.0816. Ordinary rows count. About 10:17 local time on round day.
- Upload 2, XGBoost champion (cleaned LOMO 0.5662): 0.5470 public, inside the forecast of about 0.54
  +-0.02-0.04 and above the 0.495 stop line. Rule at that point: nothing else the same day.
- Upload 3, three-family rank blend (XGBoost, LightGBM, CatBoost; 17 test rows differ from upload 2:
  7 takeover to sim_swap, 5 qr_swap to legit, 2 legit to phishing and three singles): 0.5487, +0.0017.
  It was uploaded before the local rule check finished. That check later gave families-mode October-like
  +0.00017 [-0.00013, +0.00046] cleaned (thinning seeds +0.0004, -0.0001, +0.0002), LOMO cleaned
  +0.00019 / -0.00003 / +0.00047 (Jul / Aug / Sep); generator-only positive everywhere: it would
  have failed the rule on a hair. The public +0.0017 on 17 changed rows is a few right rows
  (about 5 in the public slice, one extra loan or collusion row is worth 0.0011-0.0013).
- Upload 4, blend plus a lower type score cut and the counter column dropped: 0.5488 (+0.0001); 9 rows
  from upload 3, 18 from upload 2; LOMO +0.0001 versus upload 3.
- Upload 5, a bundle without the pretrained piece (pools, self-training, 5-seed bag, one type per
  event): 0.5488, identical to upload 4 to four decimals; 15 rows differ, netting to zero.
- Final standing: uploads 4 and 5 tied for best public at 0.5488; the ordinary-rows-legit
  file was never uploaded, and a ten-upload day was never used in full.
- Lesson. Because the entry is the best public upload scored on private, every file uploaded is a
  potential final; a file 0.01 worse still wins public 10-19% of the time. Footprint (rows changed)
  bounds the public effect, so a 17-row change cannot be a measurement.

## 7. The upload rule "consistently better"
- Idea. The earlier bar (+0.007 on the October-like month) was set against validation noise and no change
  ever reaches it; the best candidates are stacks of +0.001 gains.
- Rule adopted. Upload a challenger only if all hold: (1) October-like, three thinning seeds x three model
  seeds, counter from the unthinned frame, 95% event-bootstrap lower bound above 0 and a positive
  seed-averaged gain on every thinning seed; (2) no loss in any LOMO month on given or cleaned labels;
  (3) no loss on generator-only rows; (4) each piece of a stack still helps left out (nested selection);
  (5) pre-upload audit with no failure, footprint recorded, forecast written before upload.
- Reference: the XGBoost-plus-LightGBM six-seed champion variant did not pass (pooled lower bound -0.0000;
  never below the champion in any month); a type-score cut shrank from +0.0004 (one seed) to +0.0001 / 0.0000.
- Verdict. Adopted, then waived for later uploads (upload 3 went out before its check finished).
- Lesson. Winner's curse: the public-best of several near-equal files is chosen partly on noise, so
  insist on consistency, not size.

## 8. Public leaderboard noise
- Simulation (earlier), October-sized month from each train month, 30/70 split by row or by event:

  | quantity | row split | event split |
  |---|---|---|
  | sd of one public score | 0.010-0.014 | 0.019-0.025 |
  | sd of public minus whole test | 0.008-0.009 | 0.017-0.023 |
  | sd of public minus private | 0.011-0.013 | 0.022-0.032 |
  | sd of public gain, pair differing on ~300 rows | 0.0037-0.0039 | 0.0047-0.0050 |
  | smallest public gain that is not noise (1.645 sd) | 0.005 | 0.007-0.008 |

- Near-identical files (59 rows apart): sd 0.0007 on a public-size slice; a gap of -0.0013 happens in 11%
  of slices; one right row moves macro-F1 by 0.0002 (mule) to 0.0013 (loan). An independent rebuild scoring
  0.5457 against 0.5470 is noise.
- Rank flips by true gap (pair differing on ~300 rows), row / event split: gap 0.002 flips 34% / 42%,
  0.005 flips 5.5% / 10%, 0.010 flips 0.1% / 0.5%, 0.015 never.
- Same file, public versus private: sd 0.0117. The public-best of N near-equal uploads gains about +0.001 to
  +0.002 on private, much less than its public edge.
- Verdict. The public board cannot rank files that differ on a few dozen rows; use it only to veto a
  gross failure.
- Lesson. Decide with leave-one-month-out plus the October-like cells, and treat each public gain below
  about 0.005 as zero. See [stress-test.md](stress-test.md) for the matching private-risk estimate and
  [ensembles.md](ensembles.md) for blend results.

## 9. Past the information limit (research)
- Literature check (earlier): with class-conditional noise the noisy posterior is a monotone transform
  of the clean one, so corrections only move thresholds the fixed point already re-solves; majority-vote
  gains are bounded by member disagreement (about 0.004 here); the F-optimal rule never calls ordinary
  rows and assigns each look-alike pool whole to one class. Expected remaining gain about +0.001.
- Untried candidates with small positive expectation: nested ensemble selection plus one type per event
  (+0.0005), an extra neural member (+0.0003), an event-level expected-utility decoder (+0.0001).
  Hours buy robustness (regime, hygiene, reproducibility) more than mean gain. See [open-ideas.md](open-ideas.md).
