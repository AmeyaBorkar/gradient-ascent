# TabM config sweep: 32 heads help a little, nothing else does

**Question.** Is there a TabM configuration that beats the base (`recipes/tabm.json`: k = 16,
width 512, 3 blocks, periodic embeddings, 10 epochs, lr 2e-3, no class weights, given labels,
direct 12-class)?

**Protocol** (`python -m ringtrace.tabmsweep <config>`).
- One TabM fit per part: seed 0, July + August, identical data for every config.
- Four parts:
  - September of the normal ledger (leave-one-month-out fold 9);
  - the three October-like frames (thinning seeds 13 / 14 / 15).
- Decoding: fixed point at 0.875, then the anchored template rule.
- Score: the change against the base config on the same parts.
- Fraud-only role (September only): TabM sets each row's fraud mass at 1/2 with each tree
  member at 1/10, and the five-way blend keeps the type split.
- The table and arrays are in `RESULTS.md`.

## Verdicts (change against base; Sep, then October-like s13 / s14 / s15)

- **k = 32: best config, a candidate.**
  - Given: Sep +0.0003; October-like +0.0007 / +0.0003 / +0.0003.
  - Cleaned: Sep +0.0003; October-like +0.0002 / +0.0003 / +0.0003.
  - Gen-only, October-like: +0.0012 / +0.0003 / +0.0011.
  - **Gate against base** (r0, one seed): positive on every seed, but the lower bounds are
    -0.0000 given and -0.0003 cleaned. **FAIL on bounds:** a consistent tie-plus.
  - Taken to full runs (below).
- **Short (6 epochs, lr 3e-3): positive everywhere, smaller on September.**
  - Given: Sep +0.0001; October-like +0.0007 / +0.0004 / +0.0002.
  - Cleaned: Sep +0.0001; October-like +0.0003 / +0.0004 / +0.0002.
  - It also halves the fit time. Not taken further (time box); the first follow-up if k32 holds.
- **Small (width 256, 2 blocks): rejected.** It loses on September cleaned (-0.0001) and on
  October-like s15 (-0.0002 given, -0.0001 cleaned).
- **Long (20 epochs, lr 1e-3): rejected.**
  - Sep -0.0003 / -0.0003 (given / cleaned); October-like s13 -0.0034 / -0.0034.
  - More training overfits the label noise: training loss 0.166 against 0.169 at 10 epochs.
    This answers "10 epochs is unconverged": converging further hurts.
- **Piecewise-linear embeddings (48 quantile bins): rejected.**
  - Sep -0.0124 / -0.0142; October-like s13 -0.0099 / -0.0113.
  - The periodic embedding is clearly better here.
- **Two-stage (P(fraud) x P(type | fraud) on the champion's screening rows): rejected.**
  - Sep -0.0077 / -0.0082.
  - The binary stage collapses (training loss 0.0005): it learns the screen.
  - TabM's value is in learning P(fraud) from all rows directly.
- **Event-cleaned `l2` training labels: tie** (Sep -0.00003 / -0.00002). Same as for the trees:
  the event cleaning changes few rows that TabM does not already handle.
- **Recency (August weighted x2): tie** (Sep +0.00000 / +0.00001).
- **Not finished:** k = 64, class weights (sqrt, full) and the larger network (width 1024,
  4 blocks).
  - The second machine ran 3-10x slower than the first (shared GPU): k64 took about 5 min an
    epoch there, and the large network about 16 min.
  - These stay open.

## Fraud-only role

In the fraud-only role every config ties the base on September: within ±0.00005 given and
cleaned, about +0.0003 / +0.0004 over the five-way blend. The exception is two-stage, at
+0.00007 / +0.00015. The role uses only TabM's legit column, and every config's legit column
ranks the margin rows almost identically. So the sweep cannot improve that role. Only the
direct 12-class blend role shows config differences.

## Not tried

- **Early stopping** on validation F1 of the clean rows (rows inside generator events where
  the given and cleaned labels agree).
- **A variant without the quantile transform** (robust-scaled raw values), to test an
  interaction between the transform and the density shift.

Both are open ideas; neither was run in the time box.
