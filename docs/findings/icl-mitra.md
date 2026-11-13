# Mitra-2 as an in-context blend member

**Question.** Does a pretrained in-context tabular model (Mitra-2,
`autogluon/mitra-classifier-2`, decision 16), used as a fraud-vs-legit scorer on the champion's
doubtful rows, help alone or as a sixth member of the equal-weight blend?

**Answer (measured in this repo).** Alone it is clearly worse than the champion. At weight 1/6
inside the blend it is non-negative in every month on every target, and the event bootstrap's
lower bound is just above zero. The gain is +0.0001, the smallest step that four decimals can show.
At weight 1/3 it is a tie that leans negative. It is **not adopted**. The October-like gate was
not run, so the bar is not met, and the arrays come from a run that was not yet bit-reproducible
(see below). Under the bar this counts as a tie: it needs the October-like cells before it can
be considered.

## Design (decision 17)

- Candidates: champion fraud probability >= tau = 2.18e-5 (train 95 % quantile).
  - Train: 78,296 rows (5.0 %), covering 41.0 % of the 57,648 fraud-labelled rows; 30.2 % of
    candidates are fraud-labelled.
  - Per month (candidates / fraud among them): July 26,977 / 8,113; August 22,901 / 7,778;
    September 28,418 / 7,726. October: 15,013 candidates.
- Context per fold: every fraud-labelled candidate of the other two months plus 0.5 legit rows
  per fraud row (seeded). That is about 23.3-23.8k rows per held-out month and 35.4k for October,
  with a legit keep rate of 0.20-0.23, corrected in the odds.
- Features: the top 32 of the champion's 230 by |AUC - 1/2| on the context rows.
  - 100 features with a 31k context ran out of memory on 24 GB.
  - October fits only at 32 features.
- Model: pure in-context, with no fine-tuning and the whole context as support. The 10-class
  limit rules out the 12-class variant.

## Numbers

Fraud ranking on each held-out month's candidates (AUC against the given labels):

| month | member | champion |
|---|---|---|
| July | 0.9416 | 0.9478 |
| August | 0.9337 | 0.9463 |
| September | 0.9292 | 0.9450 |

Leave-one-month-out macro-F1, decoded as the champion (fixed point 0.875, anchored templates):

| run | given Jul / Aug / Sep (mean) | cleaned Jul / Aug / Sep (mean) | gen-only Jul / Aug / Sep (mean) |
|---|---|---|---|
| champion | 0.5130 / 0.5090 / 0.5146 (0.5122) | 0.5645 / 0.5639 / 0.5712 (0.5665) | 0.9020 / 0.8996 / 0.9014 (0.9010) |
| Mitra member alone | 0.4962 / 0.4798 / 0.4837 (0.4866) | 0.5475 / 0.5342 / 0.5406 (0.5408) | 0.8960 / 0.9009 / 0.8987 (0.8985) |
| five-way blend | 0.5130 / 0.5091 / 0.5147 (0.5123) | 0.5647 / 0.5641 / 0.5714 (0.5667) | 0.9022 / 0.9000 / 0.9018 (0.9013) |
| blend + member at 1/6 | 0.5133 / 0.5091 / 0.5147 (0.5124) | 0.5649 / 0.5641 / 0.5714 (0.5668) | 0.9026 / 0.9001 / 0.9020 (0.9016) |
| blend + member at 1/3 | 0.5133 / 0.5086 / 0.5145 (0.5121) | 0.5650 / 0.5637 / 0.5710 (0.5666) | 0.9028 / 0.8999 / 0.9016 (0.9014) |

Paired event bootstrap: 2,000 draws, mean of the three months, 5 % / 95 % bounds.

| comparison | given | cleaned | gen-only |
|---|---|---|---|
| member vs champion | -0.0256 [-0.0273, -0.0241] | -0.0258 [-0.0274, -0.0243] | -0.0025 [-0.0045, +0.0000] |
| 1/6 vs blend | +0.0001 [+0.00001, +0.0002] | +0.0001 [+0.00002, +0.0002] | +0.0002 [+0.0001, +0.0004] |
| 1/3 vs blend | -0.0001 [-0.0003, +0.0001] | -0.0002 [-0.0004, +0.0000] | +0.0001 [-0.0002, +0.0004] |

October calls changed: member vs champion 378; 1/6 vs blend 3; 1/3 vs blend 15.

The member alone loses because its fraud probabilities on the candidates are not calibrated like
the champion's. The decoder's 0.875 scale and the template rule are tuned to the champion. Its
ranking is also slightly worse in every month. In the blend at 1/6 it moves only a few dozen rows.

## Determinism

Repeat runs of one fold slice (September, 2,048 queries) differed with the first code:
- max |delta p| 0.34, mean 0.008;
- 17 % of rows equal.

AutoGluon's Mitra "deterministic context" is an empty no-op. Seeding numpy's global generator
and torch (CPU and CUDA), with deterministic algorithms (warn only), cuDNN benchmark off and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, made two repeats **bit-identical**. That check ran at 32
features first, which ran out of memory beside another GPU tenant, so it was done at 16 features
on the same code path (51 s each). The scored arrays above predate this fix (commit `ea2c63d`).
A rerun at commit `060f6ce` would be reproducible.

## Commands

```bash
python -m ringtrace.icl prepare icl-mitra
python -m ringtrace.icl fit icl-mitra --model mitra --tag mitra-k32 --legit-ratio 0.5 --k 32 --query-batch 512
python -m ringtrace.icl assemble icl-mitra --tag mitra-k32
python -m ringtrace.icl score icl-mitra
```

**Next.** Gate the 1/6 blend on the October-like cells. That needs the member on the thinned
frames and the four other members' cells, none of which exist yet. Rerun first with the
deterministic code.
