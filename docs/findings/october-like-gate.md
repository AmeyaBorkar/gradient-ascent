# October-like gate: the blend members and the blend

Question: on the October-like frames (September thinned to October's density, models fitted on
July + August), does any member, the five-way blend, or a four-way blend without CatBoost beat
the champion, under the adoption bar? That bar requires a positive change on every thinning seed,
for both given and cleaned labels, and a positive averaged 95 % event-bootstrap lower bound.

Numbers: `RESULTS.md`, section "October-like gate". Cells:
`$RINGTRACE_OUT/octlike/<member>_s<seed>_r<r>.npz`. Gate reports:
`$RINGTRACE_OUT/octlike/gate_<challenger>_vs_<base>.json`.

## Verdicts

| challenger | given per seed | cleaned per seed | lower bound (given / cleaned) | verdict |
|---|---|---|---|---|
| lgbm | -0.0003 / -0.0000 / +0.0005 | -0.0001 / +0.0001 / +0.0005 | -0.0007 / -0.0006 | reject |
| catboost | -0.0004 / -0.0006 / -0.0004 | -0.0004 / -0.0006 / -0.0005 | -0.0012 / -0.0014 | reject (loses on every seed) |
| clean-event | +0.0001 / +0.0002 / +0.0001 | +0.0001 / +0.0004 / -0.0001 | -0.0002 / -0.0003 | reject |
| clean-l2 | -0.0001 / +0.0004 / +0.0001 | +0.0000 / +0.0006 / +0.0001 | -0.0003 / -0.0002 | reject |
| five-way blend | +0.0001 / +0.0004 / +0.0002 | +0.0003 / +0.0005 / +0.0002 | -0.0001 / -0.0000 | reject (lower bound only) |
| four-way vs five-way | +0.0000 / -0.0001 / -0.0001 | -0.0001 / -0.0003 / -0.0001 | -0.0003 / -0.0005 | reject |

- **No member passes.** clean-event and clean-l2 are the closest: mostly positive by a few
  ten-thousandths, with one seed at or below zero and negative lower bounds.
- **CatBoost is the one member that is clearly worse**: -0.0004 to -0.0006 on every seed and
  target. This is the only consistent sign in the table.
- **The five-way blend is the most consistent challenger.** It is positive on all three seeds for
  given, cleaned and gen-only (+0.0002 / +0.0003 / +0.0003 mean). Its lower bounds are -0.0001 and
  -0.0000, so it fails the bar by its last criterion. It is the best measured option, but not a
  proven gain over the champion.
- **The four-way blend (no CatBoost) is not a tie in its favour.** It is at or below the five-way
  on every cleaned seed and on two of three given seeds. On leave-one-month-out it gains +0.0001
  in July and August and loses -0.0001 in September on all three targets; the given-label
  September interval lies entirely at or below zero. So dropping CatBoost costs about 0.0001.
  That is inside the noise, but always in the same direction. "Ties go to the simpler option"
  does not cleanly apply: the four-way is the simpler and reproducible build at a small,
  consistent cost, and choosing between them is a judgement, not a measured win.
  Against the five-way it changes only 3 October calls.

## Reproducibility

- `lgbm`'s boosters run on the CPU (`device_type` cpu, `deterministic` true, `force_row_wise` true),
  with every LightGBM seed tied to the cascade seed. Its screening score is XGBoost on CUDA.
- One cell refitted at 12 and at 16 threads was byte-identical to the original. The whole `lgbm`
  member, screen included, reproduces exactly on this box.
- CatBoost on the GPU is the only member here without that property.

## Fix made for this gate

- The October-like fit used to read only `train_on: cleaned`. The event-cleaned members
  (`event_clean`, `event_l2`) were trained on the given labels without warning, which would have
  made them copies of the champion in this gate.
- `octlike.train_target` now applies the event cleaning with only the fit months voting, and it
  rejects any unknown `train_on`.
- With the fix, the event-cleaned members' cells score differently from the champion's (for
  example s13 r0 given: clean-event 0.5241, clean-l2 0.5240, champion 0.5240; s14 r0: 0.5092 /
  0.5096 / 0.5092). The relabelled row counts are in the box run's `summary.json`
  (`<member>_s<seed>_changed`).

## Choices made
- **Outputs:** cells and gate reports live under the job's own output root (`~/outputs/gate`), next
  to the champion's nine cells copied from the laptop. An existing `~/outputs/octlike` on the box
  was left untouched.
- **Feature cache:** the frames' feature caches were built once in a private copy of the shared
  cache, before the two parallel fit jobs, so they never wrote the same files.
- **CatBoost memory:** a new optional `RINGTRACE_CATBOOST_GPU_RAM` caps CatBoost's GPU memory. Its
  recipe is unchanged; the cap does not change the model.
