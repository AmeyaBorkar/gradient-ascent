# Decisions

Short records of design choices.

1. **One stacked "ledger" frame with integer keys** (`ringtrace.tables`): accounts, merchants and
   devices become integers (merchants offset by 10,000,000 so one `receiver` key covers both).
   Window queries then become one `searchsorted` over a composite (entity, time) key
   (`ringtrace.windows.Timeline`) instead of per-group rolling joins: every count and sum window
   for 1.8M rows costs well under a second.
2. **JSON recipes, not TOML configs**: a recipe names its feature families, drop list, cascade and
   decoder settings and post-decode rules; `python -m ringtrace.run <recipe>` runs it end to end.
3. **Feature cache keyed by the module's own source hash** (plus declared dependencies), so editing
   one family rebuilds only that family; no manual version numbers.
4. **The cleaned-label teacher is a different model from the scored one** (plain multiclass booster
   on context + marks, not the cascade with all families), and the silver labels are frozen on
   first build. Scoring a model against labels cleaned by itself would turn its own false
   positives into true positives.
5. **Cross-fitted screen score is nested inside each month fold** (a 2-fold score over the fold's
   two training months), so no training filter ever saw the held-out month's labels.
6. **Evidence-family readings where the specification was silent or ambiguous** (each reproduces the target
   train and October event counts; see the module docstrings):
   - exact balance steps (0.25 / 0.35) are matched to the cent, which reproduces 1,592 / 1,602 rows;
     a 0.0075 tolerance over-fires;
   - a bot "farm" is a burst of more than 6 accounts with accounts >= 0.9 x rows (a few farms reuse
     one account); a 2-6-account burst is scripted when most logins take <= 6 s (the specification does not
     give this constant; the generator's scripted logins are short);
   - "two-bait" collect sessions are two or more baits of 1/2/5/10 without requiring a 10;
   - drain chaining uses connected components, so one chance 3-row chain merges where the specification
     counted two runs (607 vs 608 events; same rows);
   - the stale QR-check mark keeps the original rule (`mrk_odd`) and the corrected one
     (`mrk_odd_v2`, no blank-check QR-to-person rows) side by side.
7. **Context hygiene follows the October measurements** even though, without the event
   families, it cost 0.0095 given (run 2): month-identifying columns (weekday, account and device
   ages, whole-frame entity totals), raw device numbers and the two dead columns are out; gaps are
   capped; receiver windows are blanked on merchant receivers; amounts stay in rupees.
8. **Device counter emissions are a fixed table, not refitted per split.** The HMM's emission
   table comes from the measured unmarked shares per zone plus three cross terms (bot zone
   marks look like mule, sim zone like takeover, cashback zone like phishing); only the anchors
   and the number line are per split. It reproduces the target segment agreement (0.8456 / 0.9566
   of fraud-block / fraud-labelled train rows against 0.8447 / 0.9555) without a fit step.
9. **The template rule reads every family and the counter from the store, whatever the recipe's
   feature families are.** Ablations that drop a family from the features therefore still apply
   the rule with full evidence; the "no counter" ablation turns the rule off explicitly.
10. **Jobs on the box run from a frozen copy of the code** (`scripts/launch.sh` copies the shipped
    tree to `~/work/jobs/<name>`): a run that lazily imported a module after the next ship once
    died silently (run 3).
11. **Scores are reported against our own cleaned labels.** They are rebuilt from the
    recipe (independent teacher on context + marks; noise inversion with rho 0.0238 and
    u 0.122; likelihood ratios of the label, the old app version and the IP-risk bin; event
    majority, else teacher type) and change 3,629 rows. Small differences in which rows change move the cleaned score by a
    few thousandths, so cleaned scores from different cleanings are comparable, not identical.
12. **The bootstrap resamples events, decoded as each recipe says.** `ringtrace.compare` re-decodes
    both runs' saved probabilities (fixed point + rules) instead of trusting stored calls, uses
    the first family event of a row as its unit, and averages the per-month macro-F1 like the
    headline score. One sparse product per month gives all 2,000 resamples.
13. **October-like mode** (`python -m ringtrace.octlike build|fit|score`): September made to look
    like October, scored by models fitted on July + August only. Everything is built here from
    the normal ledger.
    - **Protected units.** September rows are grouped into units that are kept or dropped whole:
      generator events (rows sharing an event id in any of the six event modules, joined across
      modules where events share rows), and label units (connected groups of fraud-labelled rows
      plus the legit-labelled rows next to them: same sender or device within 1 h, same
      receiving person within 72 h, same sender-receiver pair within 7 days, money passed on
      within 24 h). A label unit and an event that share a row are one unit (connected
      components over both edge sets). Every other row, marked rows outside events included, is
      its own unit. Label units alone would cut merchant rings and repeat bursts (different
      senders, a merchant receiver); with both, no event is cut. A legit row next to two fraud
      groups joins them into one unit; legit-legit adjacency never joins anything.
    - **Thinning.** Each unit is kept with probability 0.494 (October's legit rows per day against
      September's), RNG seeded by the thinning seed s in {13, 14, 15}; July, August and October
      stay complete. Among kept September QR payments to a person, `qr_vpa_mismatch` is blanked
      on 44 % (separate seeded stream), as in October. The frame definition (kept stacked rows,
      blanked rows, unit of each row) is cached as `octlike/s<seed>.npz`; `build` prints its sizes.
    - **Features.** The frame's ledger (rows renumbered, `stacked_row` kept) is installed in the
      store under its own cache key (`store.use_frame`), so every family is rebuilt on it with
      the normal code, except the device counter: its columns are held at their values on the
      unthinned build, joined by stacked row (thinning breaks the block finder).
    - **Model.** The recipe is fitted on July + August only (the cascade's screening score is
      cross-fitted between those two months); seed set r in {0, 1, 2} uses cascade seeds 3r,
      3r + 1, 3r + 2. A `train_on = cleaned` recipe builds its labels on the frame from July +
      August only. September's pre-decoder probabilities go to
      `$RINGTRACE_OUT/octlike/<recipe>_s<seed>_r<r>.npz` (`rows` = stacked row, `prob` = n x 12).
    - **Decoding.** September is decoded on its own, like a test month: the fixed point at
      0.875, then the recipe's rules (the counter-anchored template rule, which reads the held
      counter and the frame's own event families). Blends use the same decoding.
    - **Gate** (`score <base> <challenger>`; a challenger is a recipe or a fixed-weight blend
      `a+b` / `a*0.6+b*0.4` of recipes' cells). Per cell: macro-F1 change on the given, cleaned
      and gen-only targets (cleaned and gen-only are the existing silver labels, read by stacked
      row), and a paired bootstrap over the protected units (1,000 Poisson draws, seeded per
      cell, 2.5-97.5 %). Reported: the change per thinning seed (mean over seed sets) and the
      lower bound averaged over the 9 cells. It passes when the change is positive on every
      thinning seed for given and cleaned, and the averaged lower bound is above 0 for both
      given and cleaned (gen-only is reported, not gated). Aggregates only.
14. **CatBoost stages (member 3, `recipes/catboost.json`).** The cascade's binary and type stages
    can be CatBoost (`"stages": "catboost"`): 300 iterations, depth 8, learning rate 0.08, 254
    borders, GPU device 0, Logloss / MultiClass, seed per model, seeds 0-2 averaged. The screening
    score stays the champion's XGBoost one, so the rows each stage trains on are identical to the
    champion's. CatBoost on the GPU is not bit-reproducible: two runs of fold September, seed 0
    (`scripts/catboost_repro.py`) differ by at most 0.165 in one probability, by 4.3e-6 on average,
    and change the argmax of 2 of 510,143 rows. Scores move in the fourth decimal at most; the
    saved `oof.npy` / `test.npy` are the reference copy for the blend.
15. **Event-cleaned training labels are rebuilt inside each month fold** (`ringtrace.eventclean`):
    only the fit months' rows vote and change, so an event spanning a month boundary cannot carry
    held-out labels into training. The events are the six event-id modules in the teacher's order
    (bursts, drains, flows, requests, merchant, repeats). The `l2` minority flip uses the same
    chosen event (highest fraud share) and needs no fraud-share threshold of its own.
