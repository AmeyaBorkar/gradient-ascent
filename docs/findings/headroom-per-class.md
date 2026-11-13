# Headroom per class

**Question.** On the cleaned labels, how far is the champion from a model that types every
generator-made row correctly and misses every ordinary fraud row?

**Answer.** Pooled over the three held-out months (July, August and September; 1,564,060 train
rows), the champion scores **0.5677** macro-F1 on cleaned labels. The ceiling is **0.5980**, so
the gap is **0.030**. The per-month mean (0.5665) differs slightly from the pooled score. Most of
the gap is in three classes:

| class | F1 now | ceiling | gap | ordinary share of cleaned fraud rows |
|---|---|---|---|---|
| legit | 0.988 | 0.988 | 0.000 | - |
| mule_account | 0.486 | 0.518 | 0.032 | 65.0 % |
| velocity_bot | 0.583 | 0.601 | 0.019 | 57.0 % |
| layering_chain | 0.611 | 0.624 | 0.014 | 54.6 % |
| sim_swap | 0.553 | 0.579 | 0.026 | 59.3 % |
| account_takeover | 0.485 | 0.577 | **0.092** | 59.4 % |
| phishing_collect | 0.551 | 0.605 | **0.053** | 56.7 % |
| qr_swap | 0.576 | 0.582 | 0.006 | 58.9 % |
| refund_abuse | 0.598 | 0.602 | 0.004 | 56.9 % |
| fake_cashback | 0.446 | 0.541 | **0.095** | 62.9 % |
| loan_app_extortion | 0.474 | 0.497 | 0.023 | 66.9 % |
| merchant_collusion | 0.461 | 0.461 | 0.000 | 70.1 % |
| **macro** | **0.5677** | **0.5980** | **0.030** | |

**Method.**
- The ceiling's calls are the gen-only labels of `ringtrace.silver` (`structured`), scored against
  the cleaned labels (`label`). The code is `analysis/core.py:headroom`.
- The ordinary share is the fraction of a class's cleaned fraud rows that are legit in the
  gen-only target. These rows look like legit rows, so no model is expected to find them. The
  ceiling is the F1 left once they are all missed.

**Command.** The same as in `public-score-noise.md` (key `headroom`).

**What it changes.**
- The champion already reaches the ceiling for qr_swap, refund_abuse, merchant_collusion and
  legit.
- Most of the remaining 0.030 is in telling apart two look-alike pairs:
  - sim_swap and account_takeover (gaps 0.026 + 0.092);
  - phishing_collect and fake_cashback (gaps 0.053 + 0.095).
- Put the work into those two pairs (anchors, templates, counter segments).
- Better legit-versus-fraud recall can add only a little, in mule_account, velocity_bot and
  loan_app_extortion.
- The ceiling is only as good as the silver labels, which judge about 63 % of fraud labels
  ordinary.
