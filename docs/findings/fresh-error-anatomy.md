# Error anatomy of the blend: where its macro-F1 loss sits

**Question.** On the held-out months, which rows cost the five-member blend its macro-F1, and
which of those pockets has no fix tried yet?

**Answer (measured in this repo).** We decoded the blend as for the submission: fixed point at
fraud scale 0.875 per month, then the anchored template rule. We split each fraud class's loss
1 - F1 = (FN + FP) / (2 TP + FN + FP) row by row. The table gives the stake in each pocket: macro-F1
over 12 classes, mean of July, August and September. A generator row is one whose gen-only
(`structured`) silver label is fraud.

| pocket | given | cleaned |
|---|---|---|
| (a) fraud rows without generator evidence, missed | 0.3991 | 0.3874 |
| (b) generator rows called legit | 0.0022 | 0.0025 |
| (c) generator rows called another fraud type | 0.0279 | 0.0206 |
| (d) legit rows called fraud | 0.0331 | 0.0018 |
| (e) another fraud class's rows called this class (mirror of a-c) | 0.0244 | 0.0200 |

Blend per month: given 0.5130 / 0.5091 / 0.5147; cleaned 0.5647 / 0.5641 / 0.5714.

**Where (c) goes (cleaned, rows pooled over the three months):**
- account_takeover -> sim_swap: 416 (drain-1 template);
- mule_account -> layering_chain: 373 (chain-hop template);
- phishing_collect -> fake_cashback: 285; fake_cashback -> phishing_collect: 168 (col7 / col1);
- velocity_bot -> mule_account: 274 (mule-burst template);
- loan_app_extortion -> phishing_collect: 66;
- refund_abuse -> qr_swap: 45; merchant_collusion -> qr_swap: 33 (merchant-round template).

**(b) and (d), cleaned.**
- (b): phishing_collect has 90 generator rows called legit and qr_swap has 109, almost all with
  no label-free evidence.
- (d): mostly fraud-block phones (about 110 rows over all classes) on the cleaned target. On the
  given target, (d) is dominated by rows inside templates and anchors that the given labels call
  legit; the cleaned labels call them fraud, so this is label noise and not a pocket.

**Which pockets are untried.**
- (a) is the information limit: ordinary rows, AUC about 0.50 from every angle.
- The six template pairs in (c) are the known look-alike pairs (not worth trying, knowledge base).
- Feature-less generator rows in (b) are the "255 rows without evidence" pool (not worth trying).
- Two pockets were not in the ledger:
  - loan_app_extortion -> phishing_collect: rejected, see `fresh-loan-in-col1.md`;
  - QR-to-merchant rows with a blank QR check: tested, see `fresh-blank-merchant-qr.md`.

**Command.** `python analysis/fresh.py anatomy` (box, run `blend`; writes
`$RINGTRACE_OUT/fresh/anatomy-blend.json`). Helpers: `analysis/fresh_core.py`, tested in
`tests/test_fresh_core.py`.
