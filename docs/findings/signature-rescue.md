# Signature rescue: generator signatures the blend calls legit in October

**Question.** The blank-check merchant QR rule moved 20 October rows from legit to fraud and scored
+0.0012 on the public board. Do other label-free generator signatures have October rows that the
blend calls legit while train says they are near-pure fraud?

**Answer (measured in this repo; October as aggregates only).** Almost none are left. The blend
already calls nearly every October signature row fraud.

## Survey

`python analysis/rescue.py survey` measures every signature in `ringtrace.signatures`:
- the row marks: non-QR row with a QR check, blank-check merchant QR, person with a category,
  QR to a person with no category, balance 1.1x and 1.02x, amount above balance, 0.25 and 0.35
  balance steps, round merchant amounts, classic prices, baits;
- amounts above 100,000 to a person;
- every evidence family's event shapes, and membership of any shape;
- the six look-alike templates;
- the device counter's fraud block (any segment, and the sure segments);
- two certificates on one row, and five mark x counter conjunctions.

It runs on the blend of `recipes/blend-blankqr.json`: out-of-fold calls for train, October calls
for test. The table lists every signature with at least 50 train rows, a cleaned fraud rate of at
least 97 % and an October call profile that can be checked.

| signature | train rows | cleaned fraud | out-of-fold legit | October rows | October legit | legit rows |
|---|---:|---:|---:|---:|---:|---:|
| non-QR row with a QR check | 7,478 | 99.92 % | 0.03 % | 1,134 | 0.18 % | 2 |
| drain event, any shape | 3,463 | 99.86 % | 0.14 % | 578 | 0.17 % | 1 |
| drain shape 1 (template drain1) | 1,567 | 99.87 % | 0.26 % | 252 | 0.40 % | 1 |
| blank-check merchant QR (already ruled) | 678 | 100 % | 0 | 144 | 0 | 0 |
| every other qualifying signature (balance marks, steps, round amounts, bots, flows, collect shapes 1-4 and 6-9, merchant rings, repeats, templates, sure counter, two certificates, conjunctions) | 44 to 7,349 | 97.75 to 100 % | 0 to 0.49 % | 0 to 1,291 | **0** | 0 |

These signatures are excluded:
- the counter's fraud block (96.0 % fraud) and the bait / classic-price / collect shape 5
  signatures (7 to 81 %) fall below the 97 % bar;
- amounts above 100,000 to a person, because October scales legit amounts by 1.166 and could
  change the base rate by design (0 October rows are called legit there anyway);
- a blank check on QR to a person, which October creates by design. It is not in the catalogue
  as a fraud signature.

## Rules

`rules.signature_rescue` (parameter `signature`): a row carrying the signature that is called
legit takes its most likely fraud class, exactly like `blank_merchant_qr`. Leave-one-month-out
(`python analysis/rescue.py rule --sig <name>`); per month Jul / Aug / Sep, 95 % event bootstrap
(2,000 draws):

| rule | given | cleaned | gen-only | train / October rows changed | verdict |
|---|---|---|---|---|---|
| `mark:nonqr_check` | +0.00002 / 0 / 0, [-0.00001, +0.00004] | +0.00002 / 0 / 0, [-0.00001, +0.00003] | +0.00001 / 0 / 0, [-0.00003, +0.00004] | 2 / 2 (phishing_collect 2) | never loses; upload candidate |
| `drn:any` | +0.00003 / -0.00002 / -0.00002 | +0.00002 / -0.00002 / -0.00002 | -0.00002 / -0.00005 / -0.00006 | 5 / 1 (sim_swap 1) | **rejected**: negative months |
| both | +0.00005 / -0.00002 / -0.00002 | +0.00004 / -0.00002 / -0.00002 | -0.00001 / -0.00005 / -0.00006 | 7 / 3 | **rejected** |

October-like (`python analysis/rescue.py octlike --sig <name>`). The champion cells s13-15 x
r0-2 are decoded with the blend's rules, with and without the rescue rule:

- **`mark:nonqr_check`:** 0 rows changed in every cell; +0.00000 on every target and seed, lower
  bound +0.00000. A tie: no thinned September row carries the case.
- **Both rules:** 1 to 2 rows changed per cell. Given -0.00007 / -0.00004 / +0.00000 per seed,
  lower bound -0.00014; cleaned and gen-only negative in seeds 13 and 15. **Loses.**

**Verdict.**
- **`mark:nonqr_check`: untestable, an upload candidate only.** It never loses on
  leave-one-month-out, but its bootstrap lower bound is below 0, so it is not adopted by the bar.
  The case is like the blank-check merchant QR rule: the train base rate (99.92 % over 7,478
  rows) is the argument, and the stake is 2 October rows.
- **The drain rule is rejected.** The few drain rows the blend calls legit in train are the 0.14 %
  that really are legit or wrongly typed, so flipping them costs.
- **The sweep is closed.** The model's October misses on generator signatures were concentrated in
  the blank-check merchant QR rows that October's shift moved. No other pocket of more than 2
  rows exists.

**Files** (on top of `blend-blankqr.csv`, which this repo does not track):
- `rescue-nonqr.csv`:
  - valid against the sample submission;
  - sha256 `0f0a60e8e86838cb3602f0a8453d14ee767c989cfbf00e5c2dbaa1cc35449206`;
  - changes: legit -> phishing_collect 2;
  - train base rate 99.92 % fraud over 7,478 rows;
  - recipe `recipes/rescue-nonqr.json` (`python -m ringtrace.blend recipes/rescue-nonqr.json`).

  The same file comes out of
  `python analysis/rescue.py apply --sig mark:nonqr_check --base blend-blankqr.csv --out ...`,
  which applies the rule to any submission file, using the blend's probabilities for the class.
- `rescue-nonqr-on-tabmF.csv`: the same rule on the TabM blend upload file (template rule and
  blank-check merchant QR rule):
  - valid;
  - sha256 `715138bd2d1d6bc48338f34873363c310a2493556f1769ec0cc9a94d2f05fc3a`;
  - changes: legit -> phishing_collect 1 (that base already calls the other row fraud).
- No combined file is written: with the drain rule rejected, it would equal `rescue-nonqr.csv`.

**Ranking** (October rows rescued x train precision): `mark:nonqr_check` 2 x 0.9992 = 2.0;
`drn:any` 1 x 0.9986 = 1.0 (rejected); every other signature 0.
