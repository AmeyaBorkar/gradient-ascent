# Weight-space methods on the TabM member: nothing to adopt

Question: can the neural member be improved by operating on its trained weights (interpolating,
extrapolating, averaging or re-fitting them) instead of on its predictions?

**Verdict (measured in this repo):** no. Four methods were run; every arm either ties the base
inside the noise of a plain re-fit (8 of 1.56 M train calls) or breaks the network. Nothing passes
even the fast judge, so no October file was written. Numbers: `RESULTS.md`, section "Weight-space
methods".

## Setup

The base is the blend with the neural member in the fraud-only role: five tree members at 1/10
each, TabM at 1/2 setting only the fraud mass of each row, the fixed point at fraud scale 0.875,
the anchored template rule, then `blank_merchant_qr`. Every arm replaces only TabM's legit
probability. The paired base is our own re-fit of the given-label TabM (seeds 0-2, arm `base`); it
differs from the stored member on 8 train calls and 1 October call, which is the noise floor of
everything below.

A context is one fit: `fold7`, `fold8`, `fold9` (leave one month out on the normal ledger) or
`s13`, `s14`, `s15` (the October-like frames: fitted on July + August, thinned September held out;
seed set r0). Scores come from the final harness (`ringtrace.final.Bench`): change per month on the
given, cleaned and gen-only targets, the paired event bootstrap of the mean change (2,000 draws,
95 %), and the three October-like cells (1,000 draws).

```bash
python -m ringtrace.wspace fit fold7 fold8 fold9 s13 s14 s15   # endpoints and arms (GPU)
python -m ringtrace.wspace vector fold9 s13 s14 s15            # fine-tuned label vector (GPU)
python -m ringtrace.wspace rebasin fold9 s13 s14 s15           # seed averaging in weight space
python -m ringtrace.wspace head                                # legit-column rescale
python -m ringtrace.wspace score                               # every arm against the base
```

## 1. Task arithmetic for the label regime

theta_g is trained on the given labels and theta_c on event-cleaned labels of the fit months
(`event_l2`: about 1,710 of 1.05 M fit rows relabelled, 1,530 of them legit to fraud), from the
same seed, so the same initial weights, batch order and dropout stream. The arm `ta<alpha>` is
theta_g + alpha (theta_c - theta_g).

- **The two runs end far apart.** |theta_g| = 301 and |theta_c - theta_g| = 60 to 61 in every
  context.
  The +-1 member scales hold 91 % of the squared norm and are shared by both runs, so the distance
  is 69 % of the norm of everything else (88). Changing 0.16 % of the labels is enough for AdamW
  with dropout to send the same start to a different solution.
- **They are still linearly connected.** The held-out loss along the straight path never rises
  above the line between its ends by more than 0.002 (0 on the three folds), so the pair shares a
  basin. The loss against the given labels rises along the path (0.1725 to 0.1922 on September),
  which is the label regime itself: the cleaned model is sure of rows the given labels file as
  legit.
- **The blend does not move.** alpha = 1, the cleaned-label model itself, changes 8 train calls:
  given +0.00006 / +0.00004 / +0.00001, October-like 0 / 0 / -0.00006. alpha = 0.5 is negative
  on the given labels in every month and cell (-0.00003 to -0.00014).
- **Extrapolation breaks the network.** alpha = 1.5 costs -0.046 / -0.020 / -0.002 on the
  October-like cells while moving 21 calls on the full months; alpha = 2 loses 0.13 to 0.34
  everywhere. The damage is larger on the thinned frames, so extrapolated weights are least stable
  exactly under the density shift that separates October from train.

A cleaner vector was tried next (`python -m ringtrace.wspace vector`): two short fine-tunes of
theta_g (2 epochs, rate 2e-4 falling to 0, same seed), one on the given and one on the cleaned
labels; `tv<alpha>` = theta_g + alpha (theta_cf - theta_gf). The vector is no cleaner: the two
fine-tunes end 4.8 apart while each moved only 3.7 and 5.4 from theta_g, and its cosine with
theta_c - theta_g is 0.10 (one frame). In the blend, on September and the three cells: alpha = 1
changes 3 calls (-0.00001); alpha = 2 is +0.00006 on September but -0.00008 on every October-like
seed; alpha = 4 is +0.00002 on September and -0.012 to -0.026 on the cells; alpha = 8 breaks the
network (-0.30). The given-label fine-tune alone (`ft_g`) changes 1 call.

Why the label regime has no leverage here: it changes how sure the network is on rows that are
already decoded as fraud. The rows it could add are the unreported rows inside events, which the
trees and the given-label network already call; the fraud rows that look like legit traffic stay
out of reach under either set of labels.

## 2. Interpolation across time

theta_r is theta_g fine-tuned for 2 epochs on the fit month nearest the scored month (August for
July and September, September for August); `ti<alpha>` = theta_g + alpha (theta_r - theta_g).
The step is small (|theta_r - theta_g| = 2.95, 1 % of the norm), there is no barrier, and the
held-out loss moves by less than 0.001. In the blend alpha = 0.5, 1, 1.5 and 2 change 2 to 8 train
calls and at most +-0.00006 in any month or cell. Adding both vectors (`ta1+ti1`) equals `ta1`.
Leaning toward the most recent month does nothing measurable.

## 3. Re-basin averaging

Seeds 0-2 of theta_g were aligned to seed 0 by weight matching over every symmetry of the network
(member order, hidden units of each block, embedding units, frequencies, and the sign gauges
between a shared weight and its member scales), then averaged into one model.

- Independent seeds are 426 apart (sqrt(2) times the norm). Matching brings that to 307 in 5 to 7
  sweeps.
- The barrier barely moves: the loss at the midpoint between two seeds is 1.17 before matching and
  1.06 after, against 0.172 at the ends.
- The averaged model is broken: held-out loss 1.28 without alignment and 1.25 with it, against
  0.172 for the mean of the three seeds' predictions. In the blend it loses 0.24 on September
  and 0.24 to 0.27 on the cells, aligned or not.

The reason is the architecture. Each member's input and output scales start as random +-1 and stay
there (mean |scale| 0.992 after training); between two seeds they agree on 49.9 % of the entries,
which is chance. No permutation or sign gauge can align two independent random sign patterns over
16 members, so the average sets about half of the scales to 0. Re-basin needs symmetries that
cover the whole difference between two runs, and this network's member scales are outside them.
Prediction averaging stays the only way to combine seeds.

## 4. Head re-optimisation

With the body frozen, the legit column of the output layer is rescaled by `a` and its bias shifted
by `c` (the same two numbers for every member and seed), which is the only part of the head that
the fraud-only role reads. `(a, c)` is chosen on a 5 x 7 grid to maximise the decoded macro-F1 of
the blend on two months, from logits that are out of fold for those months, and applied to the
third; the October-like cells use September's choice.

The surface is flat and the choice does not transfer. Over the whole grid the change against
`(1, 0)` lies between -0.0004 and +0.0002 in any month, and the best mean over three months is
+0.00002. The chosen points gain +0.00002 to +0.00007 on their two fit months and return
+0.00000 / -0.00010 / -0.00015 on the held-out month; the October-like cells lose -0.0005 /
-0.0002 / -0.0004 on the given labels (56 train calls changed). Only this two-parameter form
was run: a full re-fit of the head's 513 x 16 legit weights needs body features that are out of
sample for the rows it is fitted on, which a single fit does not give.

## Do weight-space methods generalise here?

No, and the October-like cells say so most clearly: every arm scores 0 or below there, and the two
that extrapolate (`ta1.5`, `tv4`) lose most on the thinned frames while looking harmless on the
full months. Three things stand in the way, all measured above:

1. The blend's calls are insensitive to the neural member's weights: a move that keeps the network
   working changes 1 to 56 of 1.56 M train calls, and a plain re-fit changes 8.
2. Training is chaotic at this scale: two runs from one seed that differ in 0.16 % of the labels
   end as far apart as unrelated solutions of the shared weights, so a difference of weights is
   mostly noise and extrapolating along it leaves the basin by alpha = 1.5 to 2.
3. The member scales make independently trained networks impossible to align, so weights cannot be
   averaged across seeds.

## Not run

Seed sets r1 and r2 of the October-like cells, the label vector and re-basin on the July and
August folds, and an October file: nothing passed the fast judge, so the full bar was not spent
on any arm.
