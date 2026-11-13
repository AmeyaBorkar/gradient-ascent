# Training techniques for the neural member

**Verdict: nothing adopted.** The fraud-only TabM stays as it is. Numbers are in `RESULTS.md`,
"Neural training techniques", and sources and the prior ranking in `docs/research.md`.

## What was tested

Each technique is a switch on `tabm.TabMSpec`, all off by default, and the plain path is
unchanged step for step. `ringtrace.nnexp` fits it once on July + August (seed 0) and scores
September in four cells (the fold-9 month and the r0 October-like frames s13-15). It is scored
in two roles, the fraud-only blend member and TabM alone, against the plain TabM with the same
seed.

| technique | setting | verdict | why |
|---|---|---|---|
| SWA, last 3 epochs | `swa_last=3` | passed the screen, **failed the full bar** | screen: never loses in either role, but moves 0-1 fraud-only calls per cell. 3-seed member: fraud-only July -0.00004 given / cleaned, -0.00013 gen-only |
| EMA of weights | `ema_decay=0.999` | rejected | fraud-only Sep -0.00013 given |
| QR check blanked on QR-to-person rows | `qr_blank_p=0.44` | rejected | fraud-only never loses (+0.00002 to +0.00056); alone s14 gen-only -0.00126 and given -0.00011 |
| distillation, alpha 0.3 | `distill_alpha=0.3` | rejected | fraud-only s15 -0.00008 given |
| distillation, alpha 0.7 | `distill_alpha=0.7` | rejected | fraud-only negative in all four cells |
| mixup of embeddings | `mixup_alpha=0.2` | rejected | fraud-only s13 -0.00003 given |
| input noise | `noise_std=0.05` | rejected (partial) | fraud-only Sep -0.00006 given |
| masked-feature pretraining on train + October inputs | `pretrain_epochs=3` | rejected (partial) | fraud-only Sep -0.00004 given |
| label smoothing across legit / fraud | `smooth_fraud=0.03` | rejected | fraud-only s15 -0.00044 given |
| logit adjustment | `loss=logit_adjust` | rejected | about -0.3 macro-F1 |
| focal loss | `loss=focal` | rejected | about -0.05 to -0.06 |
| RealMLP-style network as the fraud-mass source | `ringtrace.realmlp` | rejected | fraud-only s15 -0.00071 given |

## What it means

- **Anything that moves the probabilities away from calibration loses badly.** Logit adjustment,
  focal loss and, more mildly, label smoothing all do this, and the decoder needs calibrated
  inputs. This matches the published caveats on label smoothing and logit adjustment.
- **Variance reduction (SWA, EMA) helps TabM alone, by +0.0001 to +0.0004 in every cell.** It does
  almost nothing in the fraud-only role: TabM is only half of the legit column there, so a
  smoother TabM changes 0-7 calls.
- **The October-shaped QR augmentation is the most consistent fraud-only result** (positive in
  every cell and target). It fails only because TabM alone loses on s14. It is the one worth a
  3-seed check if fraud-only ever becomes the only judge.
- **Distillation from an out-of-fold booster adds no fraud mass the trees don't already have.**
  At alpha 0.7 the member becomes a copy of its teacher and loses everywhere in the fraud-only
  role.

## Decisions taken

- **Seed 0 only, not seeds 0-2.** The screen used one seed per fit (base and technique share it)
  to fit the time box. Two 3-seed jobs on one GPU took 14 min per cell.
- **Distillation teacher.** The teacher is `teacher.teacher_oof` on the TabM features, cross-fitted
  between July and August, not the five-way blend. The blend's July and August month-fold
  probabilities come from models that saw September, which would leak the held-out labels. The
  champion cascade could not serve either, because it needs two training months.
- **Trees in the October-like cells.** The fraud-only role there uses the four tree members with
  r0 cells; clean-l2 has none.
- **Not run:** the full member's October-like cells, and the s15 cells of noise and pretraining
  (s14 and s15 for pretraining). The machines were shut down before these could run.
