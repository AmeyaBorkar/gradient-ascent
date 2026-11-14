# What changes under clean October labels

**Question.** The test labels are clean; the train labels carry unreported fraud (filed legit) and
fraud filed under the wrong type. The models are trained, decoded and validated on noisy labels.
Does the noise depend on anything a model can see, and does decoding on clean-label posteriors
move the October file?

**Answer: nothing measurable changes.** The noise is uniform, so a model fitted on the given
labels learns nothing noise-specific, and inverting the noise moves at most one October row.

All numbers are measured in this repo on the train months (out-of-fold calls of the best file,
`blend-tabmF-bq`, rebuilt from the saved arrays with 0 October rows differing). October appears
only as counts. Code: `analysis/cleandec.py` (`noise`, `decode`, `public`, `census`, `build`).

## 1. The noise process

Three populations of rows that are fraud whatever their label says:

| population | rows | filed legit | filed another type |
|---|---|---|---|
| generator-only fraud (silver structured) | 24,345 | 12.09 % | 2.81 % |
| fraud-block phones, class = counter segment (label-free) | 3,195 | 11.64 % | 3.44 % |
| template rows of generator fraud | 11,385 | 11.68 % | 2.91 % |

- **Unreported rate.** Per group, chi-square of homogeneity on the generator rows:
  - flat by class (14.8 on 10 df), month (1.9 on 2), template (16.3 on 6), KYC (0.6 on 1),
    OS (0.2 on 1), app (4.1 on 4), payment type (5.0 on 4), home city (1.4 on 1), amount decade
    (4.3 on 5), new payee (0.8 on 1) and hour band (11.9 on 5);
  - bank is 40.9 on 29 df (p = 0.07). It is 25.8 on 29 (p = 0.64) on the label-free fraud-block
    phones, which are mostly one row per phone. The template rows' 52.5 on 29 is event
    clustering: an event's rows share one sender's bank;
  - over all 36 tests (12 groupings x 3 populations) the smallest p-values are 0.005 (that bank
    test), 0.012 (template on generator rows; amount decade on fraud-block rows) and 0.036 (hour
    band). That is about what 36 tests and event clustering give by chance.
- **Per class**, unreported is 9.7 % (sim_swap) to 13.0 % (loan_app_extortion), all inside the
  binomial noise.
- **Wrong type** is 2-5 % per class. It is spread over every other type, not concentrated on the
  look-alike partner. The largest cells are mule -> bot (33 rows), qr -> collusion (20) and
  phishing -> loan (20). On the label-free fraud-block phones the one partner cell
  (sim_swap -> takeover) holds 5 rows.
- **Our calls follow none of it.** On generator rows the out-of-fold fraud-call rate is 98-100 %
  in every bank. Its correlation with the bank's unreported rate is -0.22 (n.s.). No part of the
  call pattern is a learned suppression of noisy groups, because there are none.

## 2. A clean-label proxy

The counter proxy is the cleaned labels with the label-free counter imposed:
- a confident fraud-block row takes its segment class (126 rows the cleaned labels left legit,
  73 with another type);
- a bot-pool row is velocity_bot (1 row);
- then the A-clean expectation of the 3,482 unreported ordinary rows.

It changes 200 train rows. It scores the best file at 0.5498 / 0.5482 / 0.5560 (A-clean:
0.5462 / 0.5451 / 0.5525). It ranks every variant below exactly as A-clean does.

**Assumptions:**
- the counter segment is the clean class (85 % label agreement, the rest is the noise above);
- the ordinary rows stay unreachable.

## 3. Re-decoding under clean posteriors

`ringtrace.decode.clean_posterior` inverts the measured process, with `fixed_point(clean=...)`:
- q_k = (1 - u_k) t_k + r/11 t_0, solved for the clean posterior t, with legit-to-fraud
  r = 0.0007;
- optionally the uniform 3 % type flip is undone on the type split.

Change in 1e-5, Jul / Aug / Sep. In each cell the five numbers are given / cleaned / gen-only /
A-clean / counter proxy. Rows are train rows changed per month.

| decoder | Jul | Aug | Sep | rows | October |
|---|---|---|---|---|---|
| fraud scale 0.80 | -7 -4 -20 -3 -3 | -1 -2 -10 -2 -2 | 0 -1 -6 0 0 | 8 / 5 / 3 | 2 legit -> phishing |
| fraud scale 0.879 (= 1 - u) | 0 | +3 +3 +8 +3 +3 | 0 | 0 / 1 / 0 | 0 |
| fraud scale 1.0 | 0 | 0 +1 +7 +1 +1 | 0 -4 +7 -4 -4 | 0 / 3 / 4 | 2 phishing -> legit |
| clean, u = 0.121 | 0 | +3 +3 +8 +3 +3 | 0 | 0 / 1 / 0 | 1 legit -> phishing |
| clean, u by class | 0 | +3 +3 +8 +3 +3 | 0 | 0 / 1 / 0 | 1 legit -> phishing |
| clean, u by class, type flip 3 % | 0 | 0 | +4 +4 +5 +4 +4 | 0 / 0 / 1 | 1 legit -> phishing |
| type flip only (scale 0.875) | 0 | -1 -1 -4 -1 -1 | +4 +4 +5 +4 +4 | 0 / 1 / 1 | 1 legit -> phishing |
| hidden factor 1.1 | -6 -2 -16 -2 -2 | -3 -3 -7 -3 -3 | +4 +4 +5 +4 +4 | 7 / 2 / 1 | 2 legit -> phishing |

- The probabilities are bimodal, so every monotone correction moves 0 to 8 train rows a month.
- The model's fraud mass is 74 % of the A-clean fraud count in every month and in October
  (14,950 against 21,701 in July; 7,280 in October against about 10,580). The missing mass is the
  ordinary rows. Raising the decoder's class sizes to match (hidden factor above 1) calls more
  borderline rows and loses.
- **Candidate `cleandec-clean-uk-flip.csv`** is the full clean posterior (u by class, type flip
  3 %):
  - it passes the bar on given and cleaned (+0.00004 in September, 0 elsewhere);
  - it changes **1** October row (legit -> phishing_collect) against `blend-tabmF-bq.csv`;
  - sha256 `715138bd2d1d6bc48338f34873363c310a2493556f1769ec0cc9a94d2f05fc3a`.

  One phishing row is worth about ±0.0002 on the public slice. It is a tie, and it is not worth
  an upload slot.

## 4. What the public scores say about the label model

- **Level.** Six past files score 0.5408-0.5499 on the public slice. Mean of public minus each
  target's train-month score:
  - A-clean -0.0002;
  - given +0.035;
  - cleaned -0.020;
  - gen-only -0.355.

  Only A-clean predicts the level.
- **Differences.** Five of the six files are 7 to 55 rows apart, and their public gaps are noise.
  The one informative pair is the template-4 switch (bait answered within 4 h, fake_cashback ->
  phishing_collect). It moved 94 October rows and cost **-0.0078** public.
  - On the train months the same switch moves 155 / 153 / 198 rows. Of their fraud labels,
    46 / 40 / 44 % are fake_cashback.
  - Its train effect in 1e-4: given -32 / -13 / -19; cleaned and A-clean -35 / -22 / -36;
    counter proxy -34 / -23 / -35; gen-only -24 / **+9** / -18.
  - Under a **template-pure** label model (every switched fraud row is truly fake_cashback) the
    effect is -169 / -180 / -217.
- **Likelihood of -0.0078** on a public slice sized from the train months: 28 of the 94 rows fall
  in public, scored against the slice's class sizes.

  | fake_cashback share of the moved rows | mean public change | sd | P(change <= -0.0078) |
  |---|---|---|---|
  | 0.43 (train) | -0.0034 | 0.0034 | 0.098 |
  | 0.60 | -0.0095 | 0.0038 | 0.655 |
  | 0.80 | -0.0167 | 0.0041 | 0.990 |
  | 1.00 (template-pure) | -0.0238 | 0.0043 | 1.000 |

  The density at -0.0078 peaks at a share of 0.5-0.6. It is 0.044 at the train share and 0.000
  at 0.9 or above.
- **Reading.**
  - The clean test labels keep the partner types inside the look-alike templates. Template-pure
    labels would have cost about -0.024, against the observed -0.0078.
  - The train share is plausible (p = 0.10). A somewhat higher fake_cashback share in October is
    as likely.
  - This rejects the template-clean target, the only target on which re-typing whole templates
    gained.
  - The gen-only target got this pair's sign wrong in August. Of the offline targets, it is the
    worst predictor of public differences.

## 5. October, label-free

Measured with `census`:
- **Fraud-block rows with no generator evidence** (ordinary fraud on a new phone):
  73.8 / 77.1 / 70.6 per 100k in Jul / Aug / Sep and **81.4** in October. October's
  ordinary-fraud rate is the train rate (+10 %, Poisson sd 7 %). It is not lower.
- **Our calls on the counter blocks:**
  - every fraud-block and bot-pool row is called fraud in every month, October included;
  - every legit-block row is called legit.
- **The bot pool** carries 103 rows per 100k in October against 184-248 in train. Every phone
  with 5 or more payments within 5 minutes sits on the pool in every month, and every pool row is
  called velocity_bot. So October's 0.53x velocity_bot call rate is fewer bot events, not missed
  bots.
- **Ordinary rows match legit rows** on every check, in train:
  - cross-month entity history: sender, receiver, device and pair seen in another month, or with
    fraud there (for example receivers 52.9 % against 53.2 %);
  - new_payee against the pair's own earlier payments (4.2 % / 6.9 % against 4.4 % / 6.9 %);
  - account age against the account's opening date (78.4 % exact against 78.2 %);
  - device age against the pair's first sighting;
  - own-phone use (92.4 % against 92.1 %).

**Verdict.** Clean test labels do not change the decoding: the noise is uniform, its inversion
is a monotone rescaling, and the file moves by one row. The public col7 pair confirms the label
model used offline (A-clean, partner types kept) and rejects template-pure labels. Nothing here
explains a public score above the 0.58 ceiling. Such a score would need ordinary-looking fraud
rows, and every check above finds them identical to legit rows.

**Commands** (laptop, 4 threads, about 10 s each after a 9 s load):
`python analysis/cleandec.py noise | decode | public | census` and
`python analysis/cleandec.py build clean-uk-flip`. They write `$RINGTRACE_OUT/cleandec/*.json`
and `$RINGTRACE_OUT/candidates/cleandec-clean-uk-flip.csv`.
