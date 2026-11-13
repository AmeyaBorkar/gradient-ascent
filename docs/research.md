# Research and sources

Inputs: the organisers' problem statement and starter notebook. External sources used:

- Lipton, Elkan, Narayanaswamy (2014), "Thresholding Classifiers to Maximize F1 Score",
  https://arxiv.org/abs/1402.1892 : the optimal F1 threshold is half the optimal F1, the basis of
  the "call a class when 2p > F" form of the decoder.
- Narasimhan, Ramaswamy, Saha, Agarwal (2015), "Consistent Multiclass Algorithms for Complex
  Performance Measures", ICML, https://proceedings.mlr.press/v37/narasimhanb15.html : plug-in
  (weighted argmax) classifiers for linear-fractional metrics such as macro-F1, solved by
  linearising the metric around the current confusion matrix, which is what our fixed point does.
- XGBoost GPU training (`device="cuda"`, `tree_method="hist"`, `QuantileDMatrix`),
  https://xgboost.readthedocs.io/en/stable/gpu/index.html
- Polars rolling windows by group (`DataFrame.rolling(..., group_by=...)`) and `join_asof`,
  https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.rolling.html
- scikit-learn isotonic regression for calibrating the teacher,
  https://scikit-learn.org/stable/modules/isotonic.html
- Rabiner (1989), "A Tutorial on Hidden Markov Models and Selected Applications in Speech
  Recognition", Proc. IEEE 77(2): forward-backward and Viterbi used by the device counter.
- Hanley and MacGibbon (2006), "Creating non-parametric bootstrap samples using Poisson
  frequencies", Computer Methods and Programs in Biomedicine 83(1): Poisson(1) weights per
  resampling unit, used by `ringtrace.compare` with generator events as units.
- Gorishniy, Kotelnikov, Babenko (2024), "TabM: Advancing Tabular Deep Learning with
  Parameter-Efficient Ensembling", https://arxiv.org/abs/2410.24210 : an MLP whose k members
  share weights with per-member scaling adapters, trained jointly; the basis of `ringtrace.tabm`.
- Wen, Tran, Ba (2020), "BatchEnsemble: an Alternative Approach to Efficient Ensemble and Lifelong
  Learning", https://arxiv.org/abs/2002.06715 : the rank-one per-member adapters (r, s).
- Gorishniy, Rubachev, Babenko (2022), "On Embeddings for Numerical Features in Tabular Deep
  Learning", https://arxiv.org/abs/2203.05556 : periodic (sin/cos with learned frequencies)
  numeric embeddings followed by a linear layer and ReLU.
- PyTorch reproducibility notes (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`),
  https://pytorch.org/docs/stable/notes/randomness.html

## Training techniques for the neural member (2023-2026 practice)

Question: which *training* techniques (not architecture or config) should make TabM stronger
and steadier on 1.05 M imbalanced training rows with about 12 % label noise and a shifted test
month? The member's job is fixed by the fraud-only design: it supplies each row's total fraud
probability, and the trees choose the type. So a technique is judged by what it does to
fraud-vs-legit calls and to calibration, which the macro-F1 decoder relies on.

Sources:

- Gorishniy, Kotelnikov, Babenko (2025), "TabM: Advancing Tabular Deep Learning with
  Parameter-Efficient Ensembling", ICLR 2025, https://arxiv.org/abs/2410.24210 : the k members
  already average out much of the variance that weight averaging and snapshot ensembles target.
  The recipe is plain cross-entropy with AdamW, and the paper's tuning space has no augmentation
  or special losses.
- Holzmüller, Grinsztajn, Steinwart (2024), "Better by Default: Strong Pre-Tuned MLPs and Boosted
  Trees on Tabular Data" (RealMLP), NeurIPS 2024, https://arxiv.org/abs/2407.04491 : a bag of
  tricks tuned on 118 datasets. These are robust scaling followed by smooth clipping, a learnable
  per-feature scale, PBLD numeric embeddings and a parametric activation. Training uses a
  multi-cycle learning-rate schedule, label smoothing (0.1) for classification, and selecting
  the best epoch on validation. In their ablations, label smoothing helps classification error
  but hurts cross-entropy, so it trades calibration for accuracy.
- Erickson et al. (2025), "TabArena: A Living Benchmark for Machine Learning on Tabular Data",
  NeurIPS 2025 Datasets and Benchmarks, https://arxiv.org/abs/2506.16791 : TabM and RealMLP are
  the strongest neural models and match tuned boosted trees. Ensembles of different families
  are best, and the best single models are not always the best ensemble members.
- Izmailov et al. (2018), "Averaging Weights Leads to Wider Optima and Better Generalization"
  (SWA), https://arxiv.org/abs/1803.05407 ; PyTorch `torch.optim.swa_utils` (`AveragedModel`,
  `get_ema_multi_avg_fn`), https://pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema
- Müller, Kornblith, Hinton (2019), "When Does Label Smoothing Help?",
  https://arxiv.org/abs/1906.02629 : smoothing improves accuracy but changes the logit geometry
  and hurts calibration and distillation.
- Menon et al. (2021), "Long-tail learning via logit adjustment", ICLR,
  https://arxiv.org/abs/2007.07314 : add tau x log prior to the logits in training; this is
  consistent for balanced error, at the price of probabilities that no longer match the base
  rates.
- Lin et al. (2017), "Focal Loss for Dense Object Detection", https://arxiv.org/abs/1708.02002
- Hinton, Vinyals, Dean (2015), "Distilling the Knowledge in a Neural Network",
  https://arxiv.org/abs/1503.02531 : soft targets from a teacher. On tabular data, distilling
  tree ensembles into networks mainly transfers the trees' smooth decision structure.
- Bahri, Jiang, Tay, Metzler (2022), "SCARF: Self-Supervised Contrastive Learning using Random
  Feature Corruption", ICLR, https://arxiv.org/abs/2106.15147 ; Rubachev, Alekberov, Gorishniy,
  Babenko (2022), "Revisiting Pretraining Objectives for Tabular Deep Learning",
  https://arxiv.org/abs/2207.03208 : label-free reconstruction pretraining gives small or no
  gains once the labelled set is large. Target-aware objectives do better, and the gains shrink
  with data size.
- Zhang et al. (2018), "mixup: Beyond Empirical Risk Minimization",
  https://arxiv.org/abs/1710.09412 ; Kadra, Lindauer, Hutter, Grabocka (2021), "Well-tuned Simple
  Nets Excel on Tabular Datasets", https://arxiv.org/abs/2106.11189 : in their regularisation
  cocktails, mixup and similar augmentations are dataset-dependent and often not selected.

Ranking by expected gain here, judged on fraud mass and calibration:

1. **Weight averaging (EMA / SWA).** Cheap and close to calibration-neutral. It addresses
   late-epoch noise that k-member averaging does not, but the k members already shrink the room
   for it, so the expected gain is small.
2. **Distillation from out-of-fold tree probabilities.** The soft targets carry the trees'
   estimate of how noisy each label is (about 12 % of labels are wrong). That is the "fraud
   mass" signal the blend needs. A risk is that the member becomes a copy of the trees and adds
   less diversity.
3. **October-shaped augmentation (QR check blanked on QR-to-person rows).** Specific to the
   known shift. Our features already neutralise that column, except through the generator
   marks, so the expected effect is small but should not hurt.
4. **Label smoothing across the fraud / legit boundary only (0.02-0.05).** Matches the noise
   model. It moves calibration, which the decoder's fraud scale (0.875) partly absorbs.
5. **Imbalance losses (logit adjustment, focal).** They target balanced error, but the decoder
   already optimises macro-F1 on calibrated probabilities, so they mostly distort calibration.
   Expected neutral to negative.
6. **Mixup / input noise.** Mixed evidence on tabular data, often not selected by tuning.
   Expected neutral.
7. **Label-free pretraining (masked-feature / SCARF-style reconstruction).** Weak at a million
   labelled rows. Run last.
8. **RealMLP as a second neural member.** The strongest diversity bet in TabArena, but a new
   model rather than a technique. Only if time remains.

Protocol choice: the experiment protocol in the brief stands (single fit on July + August per
variant, seeds 0-2, judged on September and the three r0 October-like cells, alone and in the
fraud-only role). One change is that the distillation teacher is the champion recipe
cross-fitted between July and August, not the five-way blend. The blend's month-fold
probabilities for July and August come from models that saw September, which would leak the
held-out month's labels into the soft targets.
