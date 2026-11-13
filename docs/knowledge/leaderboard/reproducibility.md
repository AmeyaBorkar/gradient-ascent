# The reproducibility requirement

Summary: the organisers check the top teams' code by re-running the **best public upload**. A
submission must therefore be rebuildable, from one command and a pinned environment, into the
exact file that was uploaded. The platform chooses the final entry (the best public upload), so
whichever file scores best publicly is the one that must reproduce. Rules and evidence below are
**earlier measurements** unless marked as measured in this repo.

## What the organisers require

- A repository or notebook link **and a short approach note**, submitted before the deadline
  (15 November 10:00 IST, round 13 November 10:00 to 15 November 10:00). Only teams with a scored
  upload can submit. Whether the repository must be public was asked and not answered.
- The top teams' code is checked by reproducing their best upload; the best upload must be
  reproducible from the submitted code.
- Data rules: train only on the dataset provided; no external data; open-source libraries and
  public pretrained weights are allowed (record each before use); rules and features written in
  code are fine, hand-labelling test rows is not; using `test.csv` to build connections is allowed
  and expected; sharing code, predictions or labels between teams is disqualification. No compute
  limits were stated.
- Format: CSV with header `id,label`; exactly one row per test transaction id, in the sample order;
  labels spelled as the 12 classes; at most 15 MB.

## What a submission must satisfy (practical checklist)

1. One command rebuilds the file from the committed code (this repo:
   `python -m ringtrace.build recipes/champion.json`), with data supplied by the organisers.
2. Pinned environment: exact library versions (the original rebuilds used Python 3.13 and
   xgboost 3.4.1; the blends also lightgbm 4.7.0 and catboost 1.2.10; polars 2.1.0, numpy 2.5.4).
3. The best public file's whole recipe is in the code: if it is a blend, every member, seed count,
   rank-average recipe, decoder and post-decode rule must be committed. A file that depends on a
   scratch script or a local array cannot be reproduced.
4. No test row is typed by hand and nothing is fixed for a particular test row; every test label
   comes from code that also runs on the train months, where it is scored. Hard-coded device-counter
   boundaries are forbidden; segments are fitted per split.
5. Validator PASS with no warnings (warnings are a blocker for the final upload); the file's sha256
   recorded with the commit it rebuilds from; a tag per upload.
6. Features that could be ruled out (the device counter) are one switch-off family, with the
   counter-free model kept as a fallback.
7. Pretrained weights, if used, are public and recorded. Heavy dependencies not used by the final
   rebuild (graph or deep-learning libraries) should stay out of the locked environment.
8. A pre-upload audit: 44 PASS, 6 WARN (explained), 0 FAIL for the champion; calls per class against
   a label-free census (every class within 0.97-1.12x of the census except fake_cashback at 1.97x,
   an explained template-rule effect of at most 0.001-0.003 macro-F1).

## Evidence that rebuilds are byte-identical

| Rebuild | Result |
|---|---|
| Champion, clean checkout, empty cache, one GPU (RTX 5070 Ti), about 9 minutes | byte-identical to the uploaded file |
| Same on a second machine and GPU (RTX 2050 4 GB, 12 cores, 16 GB RAM), about 35 minutes, polars thread count capped at 3 and 16 XGBoost threads on 12 cores | byte-identical; LOMO given 0.5126 (0.5130 / 0.5093 / 0.5154) reproduced |
| Regime-B twin, second machine | byte-identical |
| Upload 4 file, clean checkout, XGBoost and CatBoost on GPU, about 14 minutes | byte-identical (0 rows differ); XGBoost matches bitwise on an RTX 5060 Ti and an A100 |
| Upload 4 file, all-CPU rebuild | 10 of 260,582 rows differ (XGBoost and CatBoost use different algorithms on CPU and GPU) |
| This repo's champion from a clean checkout, empty cache, fresh cleaned labels | same sha256 and the same cleaning statistics (3,629 rows changed): measured in this repo, `RESULTS.md` run 11 |

Reading: the file depends on neither the GPU model, the polars thread count nor oversubscribed
XGBoost threads; an organiser with the locked libraries and any CUDA GPU should get the same bytes.
A CPU-only rebuild of the champion was not tested; the CPU/GPU difference seen for the blend is
seed-level noise (10 rows), enough to change the sha256. GPU graph-network scatter operations are
not bit-reproducible (atomics), another reason to keep such models out of the final file.

## Process that supports it

- Each upload: file validated, sha256 recorded, a record with the hypothesis, commit, run ids, CV,
  forecast and read-out rule, then the public score; a tag on the commit that rebuilds it. The file
  itself is never committed.
- Two independent implementations exist: the original pipeline and this repository, written from
  the written record with no shared code. Both reproduce the champion's CV within 0.0004 and its
  public score within 0.0013 (noise; see [cv-vs-public.md](cv-vs-public.md)). The rebuild is a check
  that the written method suffices to reproduce the result.

## Open points to settle next

- Which of the tied best-public files (uploads 4 and 5 in [uploads.md](uploads.md)) will be
  re-run, and whether its full recipe is in the code that will be submitted.
- Whether the device counter is acceptable to the organisers; if not, switch it off and accept the
  measured loss (about 0.03 macro-F1, measured in this repo as 0.028 in the ablation).
- Whether the approach note and the repository link have been submitted before the deadline.
