# Research and sources

Ideas come from the team's research notes (prose only; no code) and the organisers' problem statement
and starter notebook. External sources used:

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
- Dietterich (2000), "Ensemble Methods in Machine Learning", MCS 2000, LNCS 1857: averaging the
  probabilities of models with partly independent errors lowers variance; the basis of the
  equal-weight five-member blend (`ringtrace.blend`).
