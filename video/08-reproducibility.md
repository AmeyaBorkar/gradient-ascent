# Reproducibility

The organisers check the top teams by re-running the best public upload. So the best file must
rebuild, from one command, into the exact bytes that were uploaded.

## One command, from a clean checkout

Needs the organisers' files in `~/data` (or `RINGTRACE_DATA`), a CUDA GPU and Python 3.13, with an
empty cache.

```bash
uv venv --python 3.13 ~/venv && VIRTUAL_ENV=~/venv uv pip install -e ".[dev,nn]"
RINGTRACE_THREADS=8 bash scripts/build_final.sh recipes/stack/stack-burst12-sim45.json <dir>
```

It builds every feature family (cached by content hash), freezes the cleaned labels, runs the six
member recipes and writes the October file, validated against `sample_submission.csv`.

## Rebuild from saved member runs (seconds)

```bash
python -m ringtrace.final build recipes/stack/stack-burst12-sim45.json --out <dir>
# sha256 c278490094706e156dc4f4d573989fafda24de896f3a7ec147fb5fc886eb2efc
```

## Hashes of the uploads

| File | Public | sha256 |
|---|---|---|
| `final-half` | 0.5499 | `8a461426…fb926eb` |
| `bait-combo` | 0.5502 | `3b25869e…e1974` |
| `bait-edge` | 0.5506 | `b571af3e…2f9dfc` |
| `research-combo-2` | 0.5506 | `214926a627984cbfd9a4f3b2b18ff6e2dca8b3a34e14a9b35dc3d0eb48986344` |
| `counter-news` | 0.5516 | `b8c3183ab76d9c0730cd8d60dba60c36f8647633937b2a639a560accb0967799` |
| `stack-burst-b8` | 0.5531 | `31251e77f9607c417cfa4c14375e7501908d5c6a091423e5d18a061076eb63d3` |
| `stack-burst12-b8` | 0.5532 | `a9a3e35d0baca7fc2f92e1ddcd206b51883b3c105dffe3d706756b79226ed20b` |
| `stack-burst13-b8` | 0.5520 | `f51fe71b96c574931d088556ab8b28ce24a66ea4d906c2ae8a69837790ec3659` |
| **`stack-burst12-sim45`** | **0.5556** | **`c278490094706e156dc4f4d573989fafda24de896f3a7ec147fb5fc886eb2efc`** |
| `stack-sim45-b9r2` | 0.5556 | `438e0eb79c7b6a527e90eca7fb36289a853e8d4cec6120cfb9020b1cf91fd474` |
| `stack-sim45-rings8` | 0.5505 | `e7872f85cf6445b6e162748e393b5085c3c17083cc2697580da7eceb73bec4c3` |

## Measured rebuilds

| Rebuild | Result |
|---|---|
| clean tree at `afadf4b`, empty cache: six member runs + `counter-news` | 5,086 s, byte-identical to the upload |
| `stack-burst12-sim45` and `stack-sim45-b9r2`, built on those runs | the uploaded sha256 |
| earlier: `bait-edge` from scratch, one RTX 5060 Ti, code at `c7eba12` | 93 minutes, byte-identical, 0 rows different |

## What is and is not bit-exact

- **XGBoost on the GPU** reproduced byte for byte. LightGBM runs deterministic, with row-wise
  histograms.
- **TabM**: two fits on two GPUs of the same model are bit-identical. The uploaded run's arrays,
  fitted earlier in another environment, differ slightly from the rebuild's (oof sha256
  `5bef8fbc…` against `a78a87a1…`). No October call of the final file moves.
- **CatBoost on the GPU** is not bit-reproducible: two runs differ by at most 0.165 in one
  probability and change 2 of 510,143 argmax calls. It moved no call here.

## Environment

- Python 3.13.18, numpy 2.5.4, polars 2.1.0, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10,
  scikit-learn 1.9.1; torch 2.14.1+cu130 for TabM (the `nn` extra).
- Seeds 0, 1, 2 for every member, averaged. The decoder, the rules and the mix are deterministic.
- `RINGTRACE_THREADS` sets the CPU thread budget.

## Stated exceptions in the final recipe

- Every October label comes from code that also runs on the train months. No test row is typed by
  hand. Counter borders are fitted per split, never hard-coded.
- The base-rate rules are rules in code with a cap on the test rows they may change.
  `mule_edge`, `counter_news` and `blank_merchant_qr` rest on a test-phase argument that train
  cannot score.
- **The cuts of `fresh_bursts` (12 rows, SIM 45 to 50 hours) and the dropped border row were set
  with public reads.** The recipe states the cuts.
