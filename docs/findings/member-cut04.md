# The 0.04-cut cascade member: no role passes the bar

Question: does a cascade trained with a lower type cut add anything to the best file
(`recipes/final-half.json`), as a sixth tree in the type split, in the fraud mass, or in place of
the champion?

## What the member is

The member is the champion recipe with one setting changed, `type_min_score` 0.04 instead of the
default 0.10 (`recipes/member-cut04.json`; `recipes/member-x5.json` is the same spec with seeds
0-4). Every other setting is the champion's: the features, the booster, seeds 0-2, the decoder,
the anchored template rule, and the legit-side drop `drop_legit_min_score` 0.70.

- **Stage one.** Both cascades leave out the legit-labelled rows whose cross-fitted screening
  score is 0.70 or more (about 2,000 rows per fit). Stage one's target is 1 on the fraud rows
  that reach the type cut. The fraud rows scoring between 0.04 and 0.10 are therefore positives
  for the member, where the champion counts them as legit.
- **Stage two.** It trains on the fraud rows scoring at least the cut. On the October-like frames
  that is 15,353 to 15,376 of 38,858 fraud rows at 0.04, about 270 to 430 more rows per fit than
  at 0.10.

## Device: like for like without refitting the base

- The champion's seeds 0-2, refitted on the GPU that built its saved arrays, give its saved cell
  s14 r0 byte for byte.
- A second GPU (RTX 4060), refitting member seed 0 on frame 14, gives the first GPU's array byte
  for byte.
- So every member seed comes from the same deterministic computation as the saved champion. The
  saved champion is the like-for-like base, and the swap needs no champion refit.

## Results

The gate is `python analysis/member_gate.py gate --tabm-run tabm`. It compares against
`final-half`, which rebuilds `blend-tabmF-bq.csv`. The tests are:

- leave-one-month-out on given, cleaned and gen-only, with the event bootstrap (2,000 draws);
- the nine October-like cells (s13-15 x r0-2), with the pooled 95 % lower bound (2,000 draws).

Full numbers are in `RESULTS.md`, section "The 0.04-cut cascade member".

| role | given Jul / Aug / Sep | cleaned Jul / Aug / Sep | October-like given / cleaned mean (lb) | rows train / Oct | verdict |
|---|---|---|---|---|---|
| sixth tree, type split (`member-type6`) | +0.00000 / +0.00004 / +0.00001 | +0.00000 / +0.00005 / +0.00001 | -0.00004 (-0.00010) / -0.00004 (-0.00009) | 5 / 3 | FAIL: October-like negative on every thinning seed |
| fraud mass (`member-mass6`) | -0.00004 / +0.00000 / +0.00005 | -0.00004 / +0.00000 / +0.00005 | -0.00001 (-0.00003) / +0.00001 (-0.00002) | 7 / 0 | FAIL: July |
| both (`member-full6`) | -0.00001 / +0.00007 / +0.00001 | -0.00001 / +0.00008 / +0.00001 | -0.00001 (-0.00006) / +0.00000 (-0.00005) | 6 / 3 | FAIL: July |
| replacing the champion (`member-swap`) | -0.00001 / +0.00000 / +0.00003 | -0.00001 / +0.00001 / -0.00002 | -0.00001 (-0.00007) / +0.00001 (-0.00005) | 7 / 1 | FAIL: July, cleaned September |
| alone vs the champion alone (`member-solo-cut04`) | +0.00008 / +0.00014 / -0.00011 | +0.00004 / +0.00009 / +0.00003 | +0.00025 (+0.00004) / +0.00037 (+0.00014) | 59 / 12 | FAIL: given September |

The five-seed member (`member-x5`: seeds 0-4, and cells from seeds 0-4 / 5-9 / 10-14) gives
the same verdicts in all four blend roles:

| role | given Jul / Aug / Sep | October-like given / cleaned mean (lb) | rows train / Oct | verdict |
|---|---|---|---|---|
| `member-x5-type6` | +0.00000 / +0.00004 / +0.00001 | -0.00003 (-0.00009) / -0.00003 (-0.00008) | 5 / 3 | FAIL: October-like |
| `member-x5-mass6` | -0.00004 / +0.00000 / +0.00001 | -0.00001 (-0.00003) / +0.00001 (-0.00002) | 6 / 0 | FAIL: July |
| `member-x5-full6` | -0.00001 / +0.00008 / +0.00001 | -0.00001 (-0.00007) / -0.00000 (-0.00005) | 5 / 3 | FAIL: July |
| `member-x5-swap` | -0.00001 / +0.00000 / +0.00003 | -0.00002 (-0.00007) / +0.00000 (-0.00005) | 7 / 2 | FAIL: July |

So the failure is not seed noise in the member.

## Reading

- **Inside the blend the member is a tie.** It moves 5 to 7 train rows and 0 to 3 October rows.
  Every per-month change is within ±0.00008, and every bootstrap interval straddles zero.
- **The lower cut does change the cascade on its own.** Alone, it is ahead on all three
  October-like thinnings, with both lower bounds above zero. But it loses 0.00011 given in
  September and moves 59 train rows, so its leave-one-month-out record fails.
- **The blend absorbs what the cut changes.** The five trees and the TabM fraud mass already
  carry the same signal, so the extra fraud rows in stage two change almost no call.
- **The October moves are look-alike type swaps:** qr_swap <-> phishing_collect,
  refund_abuse -> qr_swap, loan_app_extortion -> fake_cashback, one row each. No legit/fraud
  boundary moves at all in the type-split role.

**Rejected in every role.** No candidate file is delivered, and `final-half` stays the best file.
