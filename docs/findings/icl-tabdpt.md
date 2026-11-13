# TabDPT as an in-context blend member

**Question.** Same as `icl-mitra.md`, with TabDPT (`Layer6/TabDPT`, weights 1.3, decision 18) in
place of Mitra-2, under the identical design (decision 17).

**Answer (measured in this repo).** Alone it is worse than both the champion and Mitra. At 1/6 in
the blend it gives +0.0001 and is non-negative in every month and target, but the given-label
bootstrap lower bound is 0.00000, not above zero. At 1/3 it is a tie. It is **not adopted**, and
the October-like gate was not run. It is bit-reproducible, and far faster than Mitra: 35 s of
model time for all folds against 1,136 s.

Fraud ranking on the held-out candidates (AUC against the given labels), with the champion's in
brackets:
- July: 0.9379 (0.9478)
- August: 0.9309 (0.9463)
- September: 0.9223 (0.9450)

| run | given Jul / Aug / Sep (mean) | cleaned Jul / Aug / Sep (mean) | gen-only Jul / Aug / Sep (mean) |
|---|---|---|---|
| champion | 0.5130 / 0.5090 / 0.5146 (0.5122) | 0.5645 / 0.5639 / 0.5712 (0.5665) | 0.9020 / 0.8996 / 0.9014 (0.9010) |
| TabDPT member alone | 0.4932 / 0.4781 / 0.4788 (0.4834) | 0.5441 / 0.5315 / 0.5340 (0.5365) | 0.8933 / 0.8985 / 0.8905 (0.8941) |
| five-way blend | 0.5130 / 0.5091 / 0.5147 (0.5123) | 0.5647 / 0.5641 / 0.5714 (0.5667) | 0.9022 / 0.9000 / 0.9018 (0.9013) |
| blend + member at 1/6 | 0.5132 / 0.5091 / 0.5147 (0.5124) | 0.5649 / 0.5642 / 0.5714 (0.5668) | 0.9026 / 0.9001 / 0.9020 (0.9016) |
| blend + member at 1/3 | 0.5133 / 0.5088 / 0.5146 (0.5122) | 0.5650 / 0.5638 / 0.5710 (0.5666) | 0.9029 / 0.9001 / 0.9015 (0.9015) |

Paired event bootstrap: 2,000 draws, 5 % / 95 % bounds.

| comparison | given | cleaned | gen-only |
|---|---|---|---|
| member vs champion | -0.0288 [-0.0305, -0.0273] | -0.0300 [-0.0317, -0.0285] | -0.0069 [-0.0090, -0.0043] |
| 1/6 vs blend | +0.0001 [+0.0000, +0.0002] | +0.0001 [+0.00001, +0.0002] | +0.0002 [+0.0001, +0.0004] |
| 1/3 vs blend | -0.0000 [-0.0002, +0.0002] | -0.0001 [-0.0003, +0.0001] | +0.0002 [-0.0002, +0.0005] |

October calls changed: member vs champion 400; 1/6 vs blend 2; 1/3 vs blend 14.

**Determinism.** Two runs of September (all 28,418 queries) are bit-identical.

**Command.** `python -m ringtrace.icl fit icl-mitra --model tabdpt --tag tabdpt-k32 --legit-ratio 0.5 --k 32`,
then `assemble icl-mitra --tag tabdpt-k32 --run icl-mitra-tabdpt` and
`score icl-mitra --run icl-mitra-tabdpt`.
