# Video script: about 5 minutes

Eleven slides. Narration is about 650 words at a calm pace.

---

## Slide 1. Title (0:00 to 0:15)

**On screen:** "GradientAscent · UPI Fraud Rings". Ameya Borkar. Best public 0.5556.

**Say:** "I'm Ameya Borkar, team GradientAscent. This is how I labelled a month of UPI payments for
fraud, and why the final file is built the way it is."

---

## Slide 2. The task (0:15 to 0:40)

**On screen:** 260,582 October rows. 12 classes. Macro-F1. Train July to September.

**Say:** "The test month is October: 260,582 payments, each legit or one of eleven fraud types. The
score is macro-F1, so a rare type counts as much as legit. October is different. It has half the
traffic per day, festival amounts, two dead columns, and a new fraud mix."

---

## Slide 3. The wall (0:40 to 1:05)

**On screen:** 0.90 on generator-made fraud against 0.57 on cleaned labels. AUC 0.49.

**Say:** "About 62 percent of fraud labels sit on rows that look exactly like normal payments.
Every audit I ran gives an AUC of about 0.5, in train and in October. The fraud the generator
actually writes, I find almost perfectly: macro-F1 0.90. So October's ceiling is about 0.556 to
0.565. The job is to find every reachable fraud row, and type it right."

---

## Slide 4. Connecting the dots (1:05 to 1:35)

**On screen:** the evidence families, one line each.

**Say:** "Each fraud type follows a routine, and the routine leaves traces across rows. Layering
hops pass the money on. Mule accounts collect from many senders. Bots fire bursts from one phone.
Drains step a balance down by exactly 0.25 or 0.35. Collect scams sit just under round limits.
Merchant rings use round amounts. Every module rebuilds these without labels, inside each month."

---

## Slide 5. The device counter (1:35 to 2:10)

**On screen:** the number line: bot pool, fraud phase in class order, legit block.

**Say:** "The key insight: phone numbers are handed out in order. The fraud routines' phones were
created together, class by class. So a phone's place on the number line says fraud, and which type.
A hidden Markov model reads those zones without labels, separately for October. That is worth 0.028
macro-F1 on held-out months. A refined version also reads the gaps between phones. It cut boundary
errors from 82 phones to 58."

---

## Slide 6. Models and decoding (2:10 to 2:40)

**On screen:** the mermaid diagram from `02-architecture.md`.

**Say:** "Five boosted cascades decide the fraud type. A TabM network sets half of 'is it fraud'.
Train labels miss about 12 percent of fraud, and test labels are clean. So the decoder divides fraud
probabilities by 0.875, and picks the labels that maximise expected macro-F1, month by month. Rules
come last: look-alike events take their counter zone's class."

---

## Slide 7. Validation (2:40 to 3:15)

**On screen:** month folds; October-like frames; labelled counter passes.

**Say:** "Every model change had to hold on three views. Each held-out month, on given and cleaned
labels. September thinned to October's density, with the counter rebuilt, as October's is. And for
counter borders, which train cannot see, calibration at October's size. The counter-news rule
passed all of them: pooled October-like gain plus 0.0016, with a lower bound above zero."

---

## Slide 8. The upload path (3:15 to 4:00)

**On screen:** 0.5499 → 0.5502 → 0.5506 → 0.5516 → 0.5531 → 0.5532 → 0.5556.

**Say:** "Here is the path. TabM on the fraud split: 0.5499. Operator bait rules: 0.5502. October's
wider mule zone: 0.5506. The refined counter: 0.5516. Then the last steps. Two October bursts with
no fraud-phase phone are bots, not mules: 40 rows. To be plain: the burst cuts, and the border row I
dropped, were set with public reads. One event has one answer, so a public score reads the whole
event. That gave 0.5556."

---

## Slide 9. What did not work (4:00 to 4:25)

**On screen:** three rejected ideas with their numbers.

**Say:** "Plenty did not work. Retyping look-alike pairs. Extra neural members like TabPFN and
TabDPT: small losses on held-out months. Three merchant rings as collusion lost 0.0051 on public, so they
stayed refund abuse. And no audit found any signal in the ordinary-looking fraud."

---

## Slide 10. Reproducible (4:25 to 4:45)

**On screen:** the one command and sha256 `c2784900…86eb2efc`.

**Say:** "The best file rebuilds byte for byte from one command. From a clean tree and an empty
cache, the six member runs rebuilt in about 85 minutes, and the file matched the upload."

---

## Slide 11. Close (4:45 to 5:00)

**On screen:** "Find every reachable row. Type it right. Prove it."

**Say:** "The remaining fraud carries no signal in this data. Everything else, I tried to find, type
correctly and measure. Thank you."

---

## Timing check

| Slides | Time |
|---|---|
| 1 to 3 | 1:05 |
| 4 to 6 | 1:35 |
| 7 to 8 | 1:20 |
| 9 to 11 | 1:00 |
| **Total** | **5:00** |
