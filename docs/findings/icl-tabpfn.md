# In-context tabular model as a blend member (TabPFN v2): rejected

**Verdict: rejected under the bar.** Alone it loses to the champion in every month and target.
Inside the blend it ties at 1/6 and 1/3 weight, and neither weight is non-negative everywhere.
Measured in this repo (`RESULTS.md`, section "In-context tabular member"). Evaluation only: see
`docs/decisions.md` item 16.

## What was tested
- `python -m ringtrace.icl lomo --name icl-tabpfn-v2 --context 20000`. The champion is kept
  everywhere except its candidate rows: champion fraud probability >= tau, which admits 5 % of
  train rows.
- On candidate rows, TabPFN v2 (binary fraud vs legit, given labels) gives p(fraud). Its
  context is the candidate rows of the other train months; October's context is all three months.
- Features: the champion matrix, top 100 by |AUC - 0.5| on the context.
- Context: every fraud row, plus legit rows sampled up to 20,000 rows, with the odds corrected
  for the sampling rate.
- Fraud types are split as the champion splits them.
- TabPFN-3 and LimiX-2, the intended models, were not run. TabPFN-3 needs an account and a
  licence acceptance; LimiX-2 has no installable package. TabPFN v2 stands in for them.

## Numbers (leave-one-month-out, after the fixed point and the template rule)

| | given | cleaned | gen-only |
|---|---|---|---|
| member - champion | -0.0111 / -0.0068 / -0.0011 (**-0.0063**) | -0.0115 / -0.0083 / -0.0010 (**-0.0069**) | -0.0170 / -0.0107 / -0.0006 (**-0.0094**) |
| blend6 - blend | +0.0001 / 0.0000 / 0.0000 (+0.0001) | +0.0001 / +0.0001 / 0.0000 (+0.0001) | +0.0002 / +0.0001 / **-0.0001** (+0.0001) |
| blend3 - blend | 0.0000 / **-0.0001** / 0.0000 | 0.0000 / **-0.0001** / 0.0000 | +0.0001 / -0.0001 / -0.0001 |

Event bootstrap 5 % bounds:
- member vs champion: -0.0072 (given) and -0.0079 (cleaned);
- blend6 vs blend: -0.0000 for both;
- blend3 vs blend: -0.0001 for both.

No lower bound is above 0. The October-like gate was not finished: every variant already fails
the leave-one-month-out part of the bar.

## What it says
- **The candidates are where the trees are already sure.** At tau = 2e-5 the champion's top 5 %
  of rows hold only 41 % of fraud-labelled rows. The other 59 % sit below a fraud probability of
  2e-5: these are the ordinary-looking fraud rows, which no feature separates.
- **On the candidate rows, a 20k-row in-context fit is a weaker classifier than the cascade.**
  The cascade fits on every row with the event families. The in-context model sees 100 columns
  and about 4,300 sampled legit rows per fold.
- **The loss is smallest in September.** September (context July + August) loses
  0.0011 given; July and August lose 0.0068-0.0111. The cause is not measured here.
- **Inside the blend, at most 1/3 weight, the member moves 4-12 October calls against each
  tree.** That is the size of the blend's own disagreement, and inside the noise.

## Cost
- The full run took 77 min of GPU on a shared RTX A6000 (16-23 min per fold).
- A fold on an idle card took about 8 min.
- Two runs of one fold slice gave identical probabilities.
