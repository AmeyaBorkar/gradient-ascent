# Validation blind spot: are the contested rows generator-made?

**Question.** Two train-only columns are drawn per row from the generator pass that wrote the row:
`app_version` < 4.0 (old app) and `ip_risk_score` > 0.5. The old-app rate is about 61.8 % for
fraud-pass rows and 15.0 % for legit-pass rows. So a set's fraud-pass share is about
(old-app rate - 0.150) / 0.468. Two sets are checked:

- rows with generator evidence that the champion calls legit;
- the rows where a member disagrees with the champion.

Are these rows generator-made?

**Answer.** Train rows only. The 95 % interval is binomial on the old-app rate.

Reference sets, as a check of the formula:

| set | rows | old app | fraud-pass share |
|---|---|---|---|
| given fraud | 57,648 | 32.4 % | 37 % |
| given legit | 1,506,412 | 15.1 % | 0 % |
| given legit, cleaned fraud (unreported) | 2,944 | 60.6 % | 97 % |
| cleaned fraud, gen-only legit (ordinary) | 36,247 | 15.0 % | 0 % |

The cleaned labels themselves use these two columns as likelihood ratios, so the last two rows
are partly circular. The first two rows are not. The 37 % fraud-pass share among fraud labels
agrees with the ~63 % ordinary share.

Rows with generator evidence that the champion calls legit (held-out months). The evidence is a
family-shape or one-class-mark anchor, or a fraud-block phone.

| evidence | rows | old app | fraud-pass share (95 %) | given fraud | cleaned fraud |
|---|---|---|---|---|---|
| anchor | 28 | 82.1 % | 100 % (100-100) | 89 % | 100 % |
| fraud-block phone | 4 | 25 % | 21 % (0-100) | 75 % | 75 % |
| in a look-alike template | 16 | 81.2 % | 100 % | 100 % | 100 % |
| any | 31 | 74.2 % | 100 % (94-100) | 87 % | 97 % |

The disagreement cells (from `member-disagreement.md`):

| member | member fraud, champion legit | member legit, champion fraud | both fraud, type differs |
|---|---|---|---|
| catboost | 21 rows, share 100 % (79-100) | 60 rows, 89 % (62-100) | 14 rows, 100 % (52-100) |
| clean-event | 36 rows, 87 % (52-100) | 21 rows, 90 % (45-100) | 7 rows, 100 % |
| clean-l2 | 41 rows, 88 % (55-100) | 20 rows, 75 % (28-100) | 7 rows, 100 % |
| lgbm | 23 rows, 100 % (76-100) | 13 rows, 83 % (25-100) | 12 rows, 93 % (33-100) |

The large phishing_collect <-> legit cells sit near one half:

- catboost, phishing_collect -> legit, 32 rows: 55 % (18-91);
- clean-event, legit -> phishing_collect, 18 rows: 51 % (3-99).

**Method.**
- The share and its interval come from `analysis/core.py:fraud_pass_share`.
- The evidence comes from `ringtrace.evidence.anchors` (`row_anchor`, `template_of`) and from the
  counter (`ctr_block == 2`).
- Only train rows are used. These columns are dead in October and are never used as features.

**Command.** The same as in `public-score-noise.md` (key `blind_spot`).

**What it changes.**
- **Anchored rows called legit.** The champion leaves 31 rows with evidence as legit, and they
  are generator-made: 74 % old app, a fraud-pass share of 100 % (interval 94-100). On October's
  clean labels they are missed fraud, so a rule "anchored row -> its anchor class" is right. It
  touches only about 10 of 510,000 rows a month. Scaled to October's size, that is about 2 rows of
  the public slice, worth under 0.001.
- **Member calls fraud, champion calls legit.** These rows are mostly generator-made (share
  87-100 %). Calling them fraud is right on clean labels; the given labels undercount these rows.
- **phishing_collect versus legit splits.** These rows are a coin flip (about 50 %), so there is
  no evidence either way. This fits the public-noise finding that such differences cannot be
  measured.
- Each set holds only tens of rows, so the intervals are wide.
