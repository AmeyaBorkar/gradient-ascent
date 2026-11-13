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
13. **October-like mode** (`python -m ringtrace.octlike`), on thinned frames (`~/data-oct/s13|s14|s15`: September thinned to
    October's density, fraud events whole, devices in the counter's legit block renumbered, the QR
    check blanked on ~150 QR payments to a person):
    - every family is rebuilt on the frame's own ledger (frame train + original test) with the
      normal code, in a cache per frame (`~/cache/octlike/s<seed>`), except the device counter:
      its columns come from the unthinned build, exported once and joined by stacked row (the
      block finder sees no legit block once September is thinned);
    - the champion recipe is fitted on July + August only; the screening score inside the
      cascade is cross-fitted between those two months; seed set r uses cascade seeds 3r..3r+2;
    - the cleaned-label variant (`recipes/champion-cleantrain.json` = champion trained on cleaned
      labels) builds its labels on the frame from July + August only, the teacher scoring July
      with a model fitted on August and vice versa (`silver.build(..., mask=...)`);
    - outputs: `~/outputs/octlike/{champion,cleantrain}_s<seed>_r<r>.npz` (`rows` = stacked row of
      each September row, `prob` = pre-decoder n x 12 probabilities) and `summary_s<seed>.json`
      (September given-label macro-F1 after the fixed-point decoder, and cleaning statistics).
      The template rule is not applied in this mode (it needs the counter's event view of the
      frame); the scores are fixed-point only.
