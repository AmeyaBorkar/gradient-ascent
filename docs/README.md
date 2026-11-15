# GradientAscent documents

**Team:** GradientAscent  ·  **Member:** Ameya Borkar

Every number in these documents was measured; October rows were used only through code and aggregates.

## Start here

| Document | What it holds |
|---|---|
| [`../README.md`](../README.md) | the approach, the uploads and how to rebuild each one |
| [`report/GradientAscent_Documentation.pdf`](report/GradientAscent_Documentation.pdf) | the full report: approach, architecture, validation, results (source: `report/*.md`) |
| [`../video/README.md`](../video/README.md) | the video context pack: story, architecture, validation, uploads, script |
| [`onboarding.md`](onboarding.md) | code map, how to reproduce results, how the pieces fit |
| [`decisions.md`](decisions.md) | design decisions and why |
| [`research.md`](research.md) | sources and prior art |
| [`../RESULTS.md`](../RESULTS.md) | every scored run, failures included |
| [`../PROGRESS.md`](../PROGRESS.md) | the project log |

## Findings

| Finding | Question and answer |
|---|---|
| [`bait-operator-roles`](findings/bait-operator-roles.md) | Unanswered baits and first-time payments to operators: two small phishing rules that pass |
| [`counter-news`](findings/counter-news.md) | Counter news: the refined device counter moves a few fraud-phase calls, and every check agrees |
| [`final-system`](findings/final-system.md) | The final system: one recipe, the evidence for each component, and the stress tests |
| [`fresh-blank-merchant-qr`](findings/fresh-blank-merchant-qr.md) | QR payments to a merchant with a blank QR check |
| [`fresh-error-anatomy`](findings/fresh-error-anatomy.md) | Error anatomy of the blend: where its macro-F1 loss sits |
| [`fresh-loan-in-col1`](findings/fresh-loan-in-col1.md) | loan_app_extortion rows called phishing_collect: the col1 template again |
| [`fresh-october-profile`](findings/fresh-october-profile.md) | October call profiles against train, and the sim_swap night window |
| [`headroom-per-class`](findings/headroom-per-class.md) | Headroom per class |
| [`member-disagreement`](findings/member-disagreement.md) | Where the members disagree with the champion |
| [`more-data`](findings/more-data.md) | More data: learning curve, fold mean against refit, refit October |
| [`october-borders`](findings/october-borders.md) | October's class borders: a posterior for each border, and how far it can be trusted |
| [`october-drift`](findings/october-drift.md) | October drift: the class-volume shifts are the generator's October mix, not missed fraud |
| [`october-events`](findings/october-events.md) | October events with no fraud-phase phone: bot bursts and merchant rings at the size cap |
| [`october-like-gate`](findings/october-like-gate.md) | October-like gate: the blend members and the blend |
| [`ordinary-rows-audit`](findings/ordinary-rows-audit.md) | Ordinary fraud rows: no signal in train or October, and the ceiling it sets |
| [`pair-templates`](findings/pair-templates.md) | Pair templates: the counter posterior inside a template's pair changes no call |
| [`pairs-partner-labels`](findings/pairs-partner-labels.md) | Look-alike pairs: the partner label is the routine, and a small takeover flip |
| [`public-score-noise`](findings/public-score-noise.md) | Public-score noise |
| [`refit-test`](findings/refit-test.md) | Refit test: one model on all months against the fold mean, October-like |
| [`squeeze-blind-spot-rule`](findings/squeeze-blind-spot-rule.md) | The blind-spot rule: anchored rows called legit |
| [`squeeze-ensemble-selection`](findings/squeeze-ensemble-selection.md) | Nested ensemble selection over the five members |
| [`squeeze-event-decoder`](findings/squeeze-event-decoder.md) | Event-level expected-utility decoder (open idea 6): not run |
| [`squeeze-event-types`](findings/squeeze-event-types.md) | One fraud type per label-free event, on top of the blend |
| [`squeeze-sim-ato-edge`](findings/squeeze-sim-ato-edge.md) | The sim_swap / takeover edge in October (aggregates only) |
| [`validation-blind-spot`](findings/validation-blind-spot.md) | Validation blind spot: are the contested rows generator-made? |

## Knowledge base

| Note | Topic |
|---|---|
| [`knowledge/data/classes-and-templates.md`](knowledge/data/classes-and-templates.md) | The 11 fraud classes: template signatures and look-alikes |
| [`knowledge/data/competition.md`](knowledge/data/competition.md) | The competition and the tables |
| [`knowledge/data/device-counter.md`](knowledge/data/device-counter.md) | The device-number counter, its zones, and the bot pool |
| [`knowledge/data/generator.md`](knowledge/data/generator.md) | How the data was generated |
| [`knowledge/data/index.md`](knowledge/data/index.md) | Knowledge base: the data and how it was generated |
| [`knowledge/data/label-noise.md`](knowledge/data/label-noise.md) | Label noise, the regimes, and what the all-legit upload showed |
| [`knowledge/data/october.md`](knowledge/data/october.md) | October: what changes in the test month |
| [`knowledge/data/ordinary-rows.md`](knowledge/data/ordinary-rows.md) | Ordinary fraud rows: 63% of fraud labels that no model can find |
| [`knowledge/experiments/baselines.md`](knowledge/experiments/baselines.md) | Baselines: the reference scores and what each layer adds |
| [`knowledge/experiments/counter.md`](knowledge/experiments/counter.md) | The device counter and its anchors |
| [`knowledge/experiments/decoders.md`](knowledge/experiments/decoders.md) | Decoders and prior-shift methods |
| [`knowledge/experiments/ensembles.md`](knowledge/experiments/ensembles.md) | Ensembles: seeds, other learners, rank blends, a second fit of the method, stacking |
| [`knowledge/experiments/index.md`](knowledge/experiments/index.md) | The experiment ledger: what was tried, what it did |
| [`knowledge/experiments/labels.md`](knowledge/experiments/labels.md) | Labels: noise model, cleaning and noise-robust training |
| [`knowledge/experiments/lookalikes.md`](knowledge/experiments/lookalikes.md) | Look-alike (sibling) separation |
| [`knowledge/experiments/more-data.md`](knowledge/experiments/more-data.md) | Getting more out of the data and the test month |
| [`knowledge/experiments/open-ideas.md`](knowledge/experiments/open-ideas.md) | Open ideas: what was never tried, ranked by expected value |
| [`knowledge/experiments/ordinary-rows.md`](knowledge/experiments/ordinary-rows.md) | Ordinary rows: every attempt to separate them from legit |
| [`knowledge/experiments/regime-hybrid.md`](knowledge/experiments/regime-hybrid.md) | The two label regimes, the regime-hybrid detector and the upload record |
| [`knowledge/experiments/rules.md`](knowledge/experiments/rules.md) | Rule and evidence-family experiments |
| [`knowledge/experiments/stress-test.md`](knowledge/experiments/stress-test.md) | Generalisation stress test and the October drift audit |
| [`knowledge/leaderboard/cv-vs-public.md`](knowledge/leaderboard/cv-vs-public.md) | CV against public: the gaps and why near-identical files differ by chance |
| [`knowledge/leaderboard/decisions.md`](knowledge/leaderboard/decisions.md) | Decisions that matter to modelling and validation |
| [`knowledge/leaderboard/index.md`](knowledge/leaderboard/index.md) | Leaderboard, uploads and decisions: index |
| [`knowledge/leaderboard/regime.md`](knowledge/leaderboard/regime.md) | How the all-legit upload read the regime |
| [`knowledge/leaderboard/reproducibility.md`](knowledge/leaderboard/reproducibility.md) | The reproducibility requirement |
| [`knowledge/leaderboard/uploads.md`](knowledge/leaderboard/uploads.md) | Every upload, in order |
| [`knowledge/modeling/cascade.md`](knowledge/modeling/cascade.md) | The two-stage cascade |
| [`knowledge/modeling/decoding.md`](knowledge/modeling/decoding.md) | Decoding for macro-F1 |
| [`knowledge/modeling/features.md`](knowledge/modeling/features.md) | Features: families, counter and leakage hygiene |
| [`knowledge/modeling/index.md`](knowledge/modeling/index.md) | Modelling, decoding and validation: overview |
| [`knowledge/modeling/public-noise.md`](knowledge/modeling/public-noise.md) | The public-score noise model |
| [`knowledge/modeling/training-labels.md`](knowledge/modeling/training-labels.md) | Training labels: given, cleaned and event-cleaned |
| [`knowledge/modeling/upload-rule.md`](knowledge/modeling/upload-rule.md) | The upload rule: consistently better, not just large |
| [`knowledge/modeling/validation.md`](knowledge/modeling/validation.md) | Validation |
