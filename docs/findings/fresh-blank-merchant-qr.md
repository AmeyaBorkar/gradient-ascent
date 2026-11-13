# QR payments to a merchant with a blank QR check

**Question.** Can a label-free rule close the qr_swap pocket of generator rows called legit, or
the October rows with a blank QR check that the blend calls legit?

**Answer (measured in this repo; October as aggregates only).**

- **QR to a person.** Train never has a blank check here. October has 140 such rows, simulated
  in the October-like frames, and the blend calls all 140 qr_swap (all carry the P2P-category
  mark). None is called legit, so there is nothing to fix.
- **QR to a merchant.**

  | check | train rows | gen-only qr_swap | cleaned legit | October rows | October called legit |
  |---|---|---|---|---|---|
  | 0 | 550,885 | 201 | 537,299 | 92,177 | 92,040 |
  | 1 | 1,720 | 79 | 1,591 | 263 | 263 |
  | blank | 678 | 306 | **0** | 144 | **20** |

  A mismatch of 1 is only 5 % qr_swap, so it cannot carry a rule. A blank check is fraud on every
  cleaned train row, yet the blend calls 20 of October's 144 legit, against 1 of 678 in train.
- **The rule** (`rules.blank_merchant_qr`): a blank-check QR payment to a merchant that is called
  legit takes its most likely fraud class.
  - Leave-one-month-out, given / cleaned / gen-only: +0.00000 / +0.00000 / +0.00004 per month on
    each target (1 train row changed). 95 % event-bootstrap interval [+0.00000, +0.00004].
  - October-like, champion cells s13-15 x r0-2: +0.0000 in every cell (0 rows changed), lower
    bound +0.0000.
  - October: 20 calls change.

**Verdict: untestable, not adopted.** The rule never loses, but it fails the bar's "positive on
every seed" and "lower bound above 0" tests, because no validation frame contains the case. Train
and September never call these rows legit, so the train months cannot show the gain. The only
argument for it is the train base rate (678 of 678 fraud), and something in October makes the
model call 14 % of them legit. A tie goes to the simpler option.

The 20 rows are worth keeping in mind for an upload slot if the regime question allows one: at
train's base rate they are missed fraud.

**Command.** `python analysis/fresh.py rule --rule blank_merchant_qr`; October-like:
`python -m ringtrace.octlike score champion fresh-blankqr` (recipe `recipes/fresh-blankqr.json`).
