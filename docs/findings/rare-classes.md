# Rare classes just below their decode cut-off (fused mass, A-clean proxy)

**Verdict: rejected, for all five classes.** No band of rows just below a rare class's cut-off has
train precision near the class's F1/2, and calling them loses macro-F1 on the A-clean proxy in
almost every month. Nothing was added to October.

Code: `python -m ringtrace.aclean rare` (`src/ringtrace/aclean.py`, commit 85c2d30).

## Method

- Probabilities: the fused mass of `blend-tabmF-bq` (five-way types, TabM-arm-c fraud mass).
- Calls: the fixed point at 0.875 per month, then the template and blank-QR rules.
- Margin: for a row not called k, its class-k probability (after the 0.875 scale) divided by the
  fixed point's label-free admission cut F_k/2. F_k is computed from the month's own
  probabilities, so the margin needs no labels. A margin of 1 is the cut.
- Bands: margin 0.25-0.5, 0.5-0.75 and 0.75-1.
- Precision: the share of a band's train out-of-fold rows whose label is k. On the proxy it also
  credits the hidden-fraud chance for cleaned-legit rows.
- Gain: the proxy macro-F1 change, per month, of calling the whole band k.
- October: row counts per band only.

## Results

| class | band (x F/2) | train rows | Oct rows | prec given / cleaned / proxy | class F1/2 (proxy) | proxy gain Jul / Aug / Sep |
|---|---|---|---|---|---|---|
| loan_app_extortion | 0.25-0.5 | 76 | 22 | 0.09 / 0.12 / 0.12 | 0.226 | -0.00077 / -0.00094 / -0.00180 |
| | 0.5-0.75 | 14 | 5 | 0.14 / 0.14 / 0.14 | | +0.00009 / -0.00021 / -0.00035 |
| | 0.75-1 | 7 | 0 | 0.29 / 0.29 / 0.29 | | -0.00014 / 0 / +0.00009 |
| merchant_collusion | 0.25-0.5 | 9 | 5 | 0 / 0 / 0 | 0.217 | -0.00019 / -0.00018 / -0.00027 |
| | 0.5-0.75 | 1 | 0 | 0 / 0 / 0 | | 0 / 0 / -0.00017 |
| | 0.75-1 | 0 | 1 | - | | 0 |
| fake_cashback | 0.25-0.5 | 217 | 49 | 0.06 / 0.08 / 0.08 | 0.214 | -0.00297 / -0.00368 / -0.00342 |
| | 0.5-0.75 | 60 | 7 | 0.10 / 0.13 / 0.13 | | -0.00074 / -0.00076 / -0.00080 |
| | 0.75-1 | 22 | 3 | 0.05 / 0.05 / 0.05 | | -0.00036 / -0.00043 / -0.00040 |
| velocity_bot | 0.25-0.5 | 208 | 35 | 0.13 / 0.14 / 0.14 | 0.280 | -0.00161 / -0.00077 / -0.00299 |
| | 0.5-0.75 | 104 | 16 | 0.18 / 0.20 / 0.20 | | -0.00150 / -0.00022 / -0.00049 |
| | 0.75-1 | 52 | 16 | 0.17 / 0.21 / 0.21 | | -0.00075 / -0.00011 / -0.00021 |
| refund_abuse | 0.25-0.5 | 58 | 21 | 0.16 / 0.16 / 0.16 | 0.287 | -0.00048 / -0.00200 / -0.00029 |
| | 0.5-0.75 | 16 | 12 | 0 / 0 / 0 | | -0.00018 / -0.00070 / -0.00008 |
| | 0.75-1 | 14 | 5 | 0.21 / 0.21 / 0.21 | | -0.00006 / -0.00032 / -0.00009 |

- Every band but one has precision below its class's F1/2 (0.21-0.29). The exception, loan
  0.75-1, is 2 right rows of 7 (0.29 against 0.226) with gains -0.00014 / 0 / +0.00009, a tie,
  and it holds no October rows.
- The proxy and cleaned precisions are almost equal. The rows just below a rare class's cut-off
  are rows already called another fraud type (97-100% per band; only refund's top band
  has 4 legit-called rows), and the proxy's hidden-fraud credit applies
  only to cleaned-legit rows. Calling them is a type switch that costs the other class a true
  positive.
- The October rows in these bands (0-49 per band) are therefore left alone. The public gains from
  the rare classes came from the QR rule and the fused fraud mass, which are already in
  `blend-tabmF-bq`.
