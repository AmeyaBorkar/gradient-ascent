# Pair templates: the counter posterior inside a template's pair changes no call

**Question.** A template event is written by one of its two classes and no other. Does reading
each fraud-phase phone's counter posterior *inside that pair*, and adding the phones of one event,
type more shared-template events (mule burst against bot, drain, chain hop, baits, small merchant
round) than the anchored rule, which needs the counter's own segment to be one of the pair with
confidence 0.8?

**Answer (measured in this repo, `research/boundary-events`, commit `94618d6`).** No. The rule
`pair_templates` (`ringtrace.pairpost`) replaces `anchored_templates` in the `counter-news`
recipe and changes **0 train rows and 0 October rows** in all 24 variants: pair share 0.6, 0.7,
0.8, 0.9; pair mass 0.5 and 0.2; all six templates, or the mule burst, the drain or the chain hop
alone. With no call changed, every score, the bootstrap and the October-like cells are identical
to `counter-news`; nothing is uploaded.

## Why nothing moves

Phones inside template events, under the refined counter the rules read (aggregates only):

| template | train phones | segment in pair, confidence >= 0.8 | October phones | segment in pair, confidence >= 0.8 |
|---|---|---|---|---|
| mf1 (chain hop) | 70 | 70 | 16 | 16 |
| drain1 | 47 | 47 | 9 | 8 |
| mf2+mf3 (mule burst) | 127 | 123 | 21 | 21 |
| col7 | 26 | 26 | 2 | 2 |
| col1 | 52 | 52 | 9 | 9 |
| rng1 | 32 | 32 | 5 | 5 |
| all | 354 | 350 | 62 | 61 |

- The event already enters the counter as an anchor emission, so the posterior at an event's
  phone is decided: the second class's share inside the pair is below 0.05 or above 0.95 on all
  but five train phones and one October phone.
- The four undecided mule-burst phones in train and the one drain phone in October do not flip an
  event: two hold almost no pair mass, and the others sit on the template's own class or in an
  event another phone already anchors.
- The errors that remain in the shared templates are in events with **no** fraud-phase phone.
  A row draws a new phone with probability 0.03, so most events have none, and the counter cannot
  speak for them at any threshold.

**Command.**

```bash
python -m ringtrace.final build recipes/counter-news.json --out <dir>     # sha256 b8c3183a...967799
python -m ringtrace.final table recipes/counter-news.json recipes/research/pt-*.json
```

The first line confirmed the bench: the box rebuilt `counter-news.csv` with the uploaded sha256
before the comparison ran. Rows changed per recipe and the phone table come from the bench's
calls and the rules' ledger (`Bench.calls`, `Bench._rules_led`).

**What it changes.** Joint reading of the counter and the event is exhausted at the event's own
phones. Further gain in the shared templates would need evidence for events without a
fraud-phase phone, where every event-level separator measured so far is at chance
(`docs/findings/ordinary-rows-audit.md`, look-alike templates).
