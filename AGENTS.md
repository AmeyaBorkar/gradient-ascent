# AGENTS.md: rules for working in this repository

Read this first, then [`docs/onboarding.md`](docs/onboarding.md) (state of the project, code map,
how to reproduce results, next steps).

## Project

**GradientAscent** is a one-person project. Sole member and author: **Ameya Borkar**. The code is
the Python package `ringtrace`: a fraud-ring detection pipeline for the "UPI Fraud Rings" datathon.
The task is to label every October 2026 UPI transaction with one of 12 classes (legit plus 11 fraud
types), scored by macro-F1. Train is July-September; test is October.

## Data rules (hard)

- **Train rows may be inspected.** Look at them freely.
- **Test rows: aggregates only.** Use them through code and summary statistics (counts, rates,
  distributions). Never read out, print or reason about individual test rows.
- **Never hand-label** any row, train or test. Labels come from code that can be rerun.
- **Nothing derived from data goes in git**: no data files, no model files, no caches, no
  prediction files, no sample rows. Numbers in prose (scores, counts, rates) are fine.
- No external data.

## No outside names

The repository carries no names of other people, teams or repositories, and no user names, machine
paths, IP addresses or host names. Use "we" or the passive voice. Describe a model by what it is
("the XGBoost champion", "the XGBoost + LightGBM blend"), not by a private configuration label.
Uploads are "upload 1, upload 2, ...". A commit hook enforces part of this; if it refuses a commit,
remove the offending text and never work around the hook.

## How to run

Local (a CUDA GPU is needed for the full cascade; tests and small pieces run on CPU):

```bash
python -m venv .venv && source .venv/bin/activate     # or: uv venv --python 3.13
pip install -e ".[dev]"
pytest
ruff check .
```

Paths come from environment variables (defaults in `src/ringtrace/paths.py`):

| variable | meaning | default |
|---|---|---|
| `RINGTRACE_DATA` | the organisers' CSV files | `~/data` |
| `RINGTRACE_CACHE` | feature-family cache, silver labels | `~/cache/ringtrace` |
| `RINGTRACE_OUT` | runs, submissions | `~/outputs` |
| `RINGTRACE_THREADS` | thread budget for Polars/BLAS/XGBoost | 16 |

Compute box (shared GPU machine), driven by the scripts in `scripts/`. Set these in your shell,
never in a committed file:

```bash
export RINGTRACE_SSH_CONFIG=<path-to-ssh-config>   # required
export RINGTRACE_HOST=<ssh-host-alias>             # default: compute-box
export RINGTRACE_BOX_DIR=<work-dir-name>           # default: ringtrace
bash scripts/ship.sh                               # copy HEAD (+ tracked edits) to the box
bash scripts/launch.sh <job> python -m ringtrace.build recipes/champion.json
bash scripts/box.sh "tail -n 20 ~/logs/<job>.log"
```

`launch.sh` runs each job from its own frozen copy of the code, so a later `ship.sh` cannot change a
job that is already running. Always give jobs explicit thread budgets; the box is shared.

## Where things go

| what | where |
|---|---|
| every scored run, failures included | [`RESULTS.md`](RESULTS.md) (run number, commit, per-month scores, wall time) |
| findings: what we learned about the data or a method, with numbers and the command that shows it | `docs/findings/` (one file per finding; create the folder when adding the first) |
| design choices and their reasons | [`docs/decisions.md`](docs/decisions.md) |
| state of the rebuild | [`PROGRESS.md`](PROGRESS.md) |
| sources used | [`docs/research.md`](docs/research.md) |
| curated background from earlier research | `docs/knowledge/` (see onboarding) |

Label every claim in a document: **measured in this repo** (a run or finding here shows it; give the
command) or **earlier measurement** (carried over from earlier research; a later session may
re-derive it). Keep the numbers; they are the value.

## The validation bar

- Score by **leave-one-month-out** (train two months, score the third, for all three months) and
  report the mean macro-F1 over the three held-out months.
- Also score **October-like**: September thinned to October's density, fitted on July + August
  (`python -m ringtrace.octlike`). October's ordinary traffic is about half as dense as train, so a
  gain that exists only at full density is not trusted.
- A **small gain must hold everywhere**: in every held-out month, in the October-like frames, across
  seed sets, and (for a challenger) with a paired event bootstrap interval
  (`python -m ringtrace.compare`) that lies wholly above zero. One month or one seed is not evidence.
- Report both the given-label and cleaned-label scores; cleaned labels are for scoring only.
- Record failures and ties in `RESULTS.md` just as plainly as wins.

## Reproducibility

- Seeds are explicit in every recipe (`cascade.seeds`); never rely on a default or on wall-clock.
- One command builds a result from a clean checkout: `python -m ringtrace.build recipes/<name>.json`.
- A submission is verified by validating it against `sample_submission.csv` (ids, order, label set)
  and recording its **sha256**. Rebuild from a clean `git archive` with an empty cache and compare the
  sha256 before calling a result reproducible (run 11 in `RESULTS.md` did this).
- Record the commit, library versions and hardware next to the result.

## Git

- Small commits, each one change.
- **Conventional Commits** subjects (`feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`).
- **No attribution lines** in commit messages or PR text.
- Work on a branch, push it, and do not rewrite shared history.
