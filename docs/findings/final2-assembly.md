# The final file: bait rules and members trained at October's density

Question: of the components measured on 14 Nov, which pass the bar, do they still pass together on
the best recipe (`final-half`, file `blend-tabmF-bq.csv`), and which file should be uploaded?

Numbers and commands: `RESULTS.md`, section "Final assembly: the bait rules with the members
trained at October's density (`final2`)".

## Answer

- Two components passed alone: the bait rules (`recipes/bait-combo.json`, 7 October rows legit ->
  phishing_collect) and the three XGBoost members trained on the device counter at October's
  density (`recipes/density/final-half-chalf3b.json`, 11 October rows, 8 of them sim_swap ->
  account_takeover). The 0.04-cut cascade member, festival-scaled training and the prior
  corrections passed nothing.
- Their October rows are disjoint, and the combined recipe `recipes/final2.json` changes exactly
  their union: 18 rows against the best file.
- **Together they hold at October's density.** Month folds with the held-out month at October's
  density: given +0.00058 / +0.00069 / +0.00110 (95 % [+0.00025, +0.00136]), cleaned +0.00054 /
  +0.00093 / +0.00143 ([+0.00038, +0.00161]). October-like cells with September's counter rebuilt:
  given +0.00173 ([+0.00089, +0.00271]), cleaned +0.00178 ([+0.00092, +0.00281]), gen-only
  +0.00143 ([+0.00021, +0.00321]). The gains add: the joint change is the sum of the two
  components' changes to within 0.00004 per month.
- **It fails the production-counter month folds** (given -0.00027 / -0.00005 / -0.00010, cleaned
  -0.00029 / -0.00007 / -0.00011). All of that comes from the density members; the bait rules
  alone gain in every month there (+0.00033 / +0.00019 / +0.00018 given).
- `final2-bait-half3.csv` (sha256 `1010acd2...4191`) rebuilds byte for byte from a clean run, is
  valid against the sample submission, and its call counts give an expected public score of
  0.5497 against 0.5491 for the best file (`python -m ringtrace.audit ceiling`).

## The one judgement in the file

The density members pass every check made at October's density and fail the month folds that put
the held-out month on the shared full-density counter pass. October's counter is computed on
October alone, at about half a train month's density, so it is never on that shared pass. The
density study found that the same mismatch made the old October-like cells overstate the best
recipe by 0.0035 per cell. Reading the October-density checks as the honest ones, `final2` passes.
Reading the production-counter month folds as binding, only the bait rules pass, and the proven
file is `final2-bait.csv` (the same bytes as `bait-combo.csv`).

## What the 18 rows are

| move | rows | from |
|---|---|---|
| sim_swap -> account_takeover | 8 | density members |
| legit -> phishing_collect | 7 | bait rules (lone unanswered Rs 10 from an operator; first-time whole-rupee PAY of Rs 1,000 or more to an operator) |
| qr_swap -> refund_abuse | 1 | density members |
| qr_swap -> phishing_collect | 1 | density members |
| velocity_bot -> mule_account | 1 | density members |

About 30 % of these rows fall in the public slice, around 5 or 6 rows. One more right row in a small
class moves public macro-F1 by 0.0002 to 0.001 (`public-score-noise.md`), so the public score of
this file is mostly noise around its expected gain of about +0.0008 to +0.0017. The gain is the
October-density month folds and the rebuilt-counter cells.

## Upload ranking

The final ranking scores the best public upload on the private 70 %. A file is uploaded only if it
is at least as good as `blend-tabmF-bq.csv` in expectation. A file that another uploaded file
dominates in expectation is not uploaded, because it could win the public slice by luck alone.

Tonight (4 slots):

1. `final2-bait-half3.csv`: the joint file. It is the file to make final if its public score
   beats 0.5504.
2. `final2-bait.csv` (= `bait-combo.csv`), unless that file is already on the leaderboard. It is
   proven on every reading of the bar. Its public score against slot 1 also gives a small public
   reading of the density members on their 3 or so public rows.
3. and 4. Kept free. `density-half3.csv` alone is dominated by slot 1 (the same evidence, without
   the bait rules), so it is not uploaded. The mule/bot counter edge, the October rules, the
   0.04-cut member and the prior corrections fail the bar and are not uploaded.

Tomorrow (10 slots until 10:00 IST):

- TabM trained at October's density is still the largest lever measured: +0.0044 given on the one
  frame where both fits ran on the same GPU. Upload `final2-bait-half3t` only if its run and cells
  finish overnight and `bash scripts/final2_gate.sh tabm` passes against TabM refitted on the
  same GPU: cells on at least s13 and s14, month folds at October's density positive on given and
  cleaned in every month. Build it with `python -m ringtrace.final build
  recipes/final2/bait-half3t.json`.
- Otherwise there are no further uploads. Spending slots on near-ties only raises the chance that a
  lucky file becomes the final pick.

**The single file to make final:** `final2-bait-half3.csv`. If its public score does not beat
0.5504, the team's current best stays final.
