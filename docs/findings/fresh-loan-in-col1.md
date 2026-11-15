# loan_app_extortion rows called phishing_collect: the col1 template again

**Question.** The blend's largest untried wrong-type pocket on the cleaned target is generator
loan_app_extortion rows called phishing_collect: 66 rows over three months, about 0.0022 macro-F1
at stake. Can a label-free signal separate them?

**Answer (measured in this repo, train rows only).** No. Of the 66 rows:
- 61 are in the col1 template (tmpl5, just-under collect request), 62 are COLLECT payments, and
  all 66 go to a person;
- 63 carry the phishing_collect anchor;
- none carries a repeat-payer shape.

The 592 loan rows the blend gets right are all repeat-payer shapes 2 and 3, mostly PAY to a
merchant. Inside col1, phishing_collect generator rows outnumber loan ones about 29 to 1 (1,762
to 61). The rows are a third class written in the col1 look-alike template, at chance inside the
template like the known pairs. The blend gives them P(loan) 0.03 on average.

**Verdict: rejected without a run.** It is a look-alike pocket. The only label-free route is the
template shape, which points the other way 29 times out of 30.

**Command.** The probe script was not committed. It read the `analysis/fresh.py` loaders:
calls of run `blend` by `row_anchor`, `template_of`, `col_shape`, `rep_shape`, txn and receiver
type, on train rows with gen-only label loan_app_extortion.
