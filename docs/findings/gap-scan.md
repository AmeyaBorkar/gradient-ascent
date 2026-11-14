# Gap scan: what a 0.63 public score would need

**Question.** The top public score is 0.6311; the best file here scores 0.5499. A perfect typist of
generator-made rows was put near 0.57 to 0.60. Which assumption about October is wrong, and what
would a file need to do to reach 0.63?

**Answer.** No assumption tested here is wrong, and no lever was found. The train picture explains
the best file's public score exactly, and it puts 0.63 out of reach of any file that types generator
rows only.

- **Our public score is what the train picture predicts.** Train out-of-fold calls score 0.5499 on an
  October-like truth that holds **1.091 times** the train rate of ordinary-looking fraud rows. That
  is 2.53 % of rows. The all-legit upload implies 4.06 % fraud in total, minus our 1.52 % of fraud
  calls, which gives 2.54 %. No drift loss and no slice luck are needed.
- **A perfect typist of generator-made rows gets 0.578 at that rate.** It gets 0.598 at the
  labelled train rate (r = 1) and 0.568 at r = 1.14.
- A structure-only typist, which calls the majority class inside each look-alike template, gets
  0.531 at r = 1.14. That is below our file (0.541).
- **To reach 0.6311, a file must catch ordinary-looking fraud rows with the right type:**

| starting from | precision of the extra calls | share of ordinary rows caught | October rows |
|---|---|---|---|
| our calls | 1.0 | 13.8 % | about 910 |
| our calls | 0.6 | 19.3 % | about 1,270 |
| perfect typist | 1.0 | 8.9 % | about 590 |
| perfect typist | 0.6 | 12.4 % | about 820 |
| either | 0.3 | unreachable even catching all of them | - |

  The second public score (0.5731) is our file plus about 3.8 % of ordinary rows, or simply a
  better typist: it sits inside the typist ceiling.
- **Every route to ordinary rows tested here is closed** (train rows; AUC of ordinary-looking fraud
  against legit, held-out month):

| test | result |
|---|---|
| gradient boosting on 48 raw, account, merchant, artefact and history features | AUC 0.509 (Sep), 0.500 (Jul); per class 0.48 to 0.53; precision at top 1,000 2.8 to 3.3 % against a 2.35 % base |
| receiver pays onward within 1 h (receiver becomes sender) | 0.57 % ordinary, 0.60 % legit |
| reverse-direction pair exists | 0 of 36,247 ordinary, 0.011 % legit |
| sender ever receives a P2P payment | 84.2 % against 84.2 % |
| `new_payee` against the observed pair history | mismatch 73.0 % in both: `new_payee` is drawn per row |
| account age against `opened_on` | consistent within 1 day on 100 % of every group |
| receiver account or merchant opened within 30 days | 0.01 % against 0.01 %; 0 against 0 |
| amount above balance | 0.00 % against 0.01 % |
| whole-rupee amounts, timestamp second 0, txn_id parity | 25.9 / 1.72 / 49.9 % against 26.0 / 1.67 / 50.0 % |
| own phone (device number = account number) | 92.4 % against 92.1 % |

- **Month-boundary filing gives nothing.** The 147 train rows dated October are all generator-made
  (18 labelled legit, 12 %, the unreported rate). The 46 October rows dated November are all
  anchored and all called fraud already. The first and last 24 h of each train month carry the
  mid-month ordinary rate (2.2 to 2.4 %); only 1 July has fewer generator rows (0.78 %), because no
  events carry over from June.
- **The look-alike minority is a real generator type, not filing noise.** Inside drain shape 3 (new
  phone, away from home), the blend's out-of-fold sim_swap / account_takeover split has AUC
  0.994 to 0.999 against the given type. The new phone's counter zone matches the label on 99.5 %
  of rows: sim_swap-labelled rows sit in the sim_swap zone and takeover-labelled rows in the
  takeover zone. So the sim_swap routine itself writes about 28 % of these events. Inside drain
  shape 1 (home phone), the few rows on a fraud-block device also sit in their label's zone (3.3 %
  and 2.6 %, no crossings). Retyping by template is therefore wrong on clean labels as well, which
  confirms the rejection of the template retype.
- **October's ordinary rows follow October's legit template.** The proxies are single-row
  fraud-block devices with no mark (213 rows). Their amounts carry the festival scale: the
  geometric-mean ratio to legit-block rows is 1.17 (95 % 0.96 to 1.41). If these rows kept the
  unscaled train amounts, the ratio would be 0.86. In train the same ratio is 1.02 (0.95 to 1.10).
- **Within-template signal left on the table is small.** Out-of-fold AUC of the type inside
  templates 1, 2, 5 and 6 is 0.54 to 0.67 (template 6 reaches 0.88 in August). The decoder calls the
  majority, as the F1 trade-off between class sizes requires. Template 4 calls fake_cashback on
  93.5 % of rows although 60 % are labelled phishing_collect. This is still right for macro-F1,
  because one fake_cashback row is worth about 2.7 phishing rows.

**Method.**
- `analysis/gap_core.py:shifted_f1` rescales each class's ordinary rows (cleaned fraud rows that the
  generator-only target leaves legit) by `r`. It also adds a caught share `q` at precision `prec`.
  It works on the counts of the pooled out-of-fold months (1,564,060 rows).
- `catch_needed` bisects on `q`. The implied `r` is the one at which our out-of-fold calls score
  0.5499.
- The calls and evidence arrays are the best file's (`blend-tabmF-bq`), in ledger order.
- October enters only as counts, rates and quantiles.

**Command.** `RINGTRACE_THREADS=4 python analysis/gapscan.py all` (70 s on the laptop). It writes
`$RINGTRACE_OUT/gapscan/{ceiling,omnibus,boundary,lookalike,festival}.json`.

**What it changes.**
- No upload candidate. Nothing here passes the bar, and nothing is a probe worth a slot.
- 0.63 is not a tuning gap. It needs about 900 extra correctly typed rows, and in train these rows
  show no handle better than 3 % precision. Chasing it with public-score probes would only fit the
  public slice.
- The private score is protected by keeping the gated file. Our public score matches its forecast
  to the third decimal.
