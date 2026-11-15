# GradientAscent: UPI Fraud Rings, video context pack

**Team:** GradientAscent. **Member:** Ameya Borkar.

Context for a 5-minute video about the final solution. Every number comes from the repository's
documents. October rows were used only through code and aggregates.

## The story in ten lines

1. **The task.** Label 260,582 October UPI payments as legit or one of 11 fraud types. Scored by
   12-class macro-F1, so every class counts the same.
2. **The wall.** About 62 % of fraud labels sit on rows that look exactly like legit payments.
   Every audit gives AUC about 0.49 to 0.50, in train and in October. October's reachable ceiling is
   about 0.556 to 0.565.
3. **Connect the dots.** Label-free evidence modules rebuild the generator's fraud routines from how
   rows link across accounts, phones, receivers and time. On generator-made labels the system
   scores macro-F1 0.90.
4. **The device counter.** Phone numbers are issued in order. The fraud routines' phones sit in one
   numbered block, written class by class. An HMM reads the block without labels. Its features are
   worth +0.028 macro-F1 on held-out months.
5. **Models.** Five boosted cascades set the fraud type. A TabM network sets half of the
   legit-against-fraud split. About 230 features per row.
6. **Decoding.** An expected-macro-F1 fixed point per month. Fraud probabilities are divided by
   0.875, because train labels miss about 12 % of fraud and test labels are clean.
7. **Validation.** Month folds on given, cleaned and generator-only labels; September thinned to
   October's density; labelled counter passes at October's size. Model and feature changes were
   never chosen on the public board; the last rule steps were (line 9).
8. **Upload path.** 0.5499 → 0.5502 → 0.5506 → 0.5516 → 0.5531 → 0.5532 → **0.5556**.
9. **Best file.** `stack-burst12-sim45`, public **0.5556**: `counter-news` plus one base-rate rule,
   `fresh_bursts` (two October bursts, 40 rows, mule_account → velocity_bot). The burst cuts and
   the dropped border row were set with public reads.
10. **Reproducible.** It rebuilds byte for byte, sha256 `c2784900…86eb2efc`. A clean tree and an
    empty cache rebuilt the six member runs and `counter-news` in 5,086 s, byte-identical; the
    best file, built on those runs, has its uploaded sha256.

## Files

| File | What it holds |
|---|---|
| `01-problem-and-data.md` | the task, the data, the metric and the October twists |
| `02-architecture.md` | the pipeline layer by layer, with a mermaid diagram of the final system |
| `03-device-counter.md` | the key insight, stated simply, with its evidence |
| `04-validation.md` | month folds, October-like frames, labelled counter passes, with the numbers |
| `05-upload-history.md` | every upload: what it changed, public score, what it taught |
| `06-key-findings.md` | one slide's worth each |
| `07-tried-and-rejected.md` | what did not work, with numbers |
| `08-reproducibility.md` | commands, hashes and the measured from-scratch rebuild |
| `09-numbers-cheatsheet.md` | the key numbers on one page |
| `10-video-script.md` | slide by slide for about 5 minutes, with timings and what to say |
| `images/pipeline.png` | diagram of the final system (members, mix, decoder, rule layers, the 0.5556 file) |
| `images/fraud-phase.png` | October's fraud phase: 410 fresh phones in class order |
| `images/score-path.png` | the public score path of the uploads |
| `images/october-calls-by-type.png` | chart: October calls per fraud type against train rates |
