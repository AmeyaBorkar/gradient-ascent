# Tried and rejected, with numbers

Each idea was measured on the same bar and set aside. Month-fold numbers are changes per month,
July / August / September.

## Models and blending

| Idea | Result | Verdict |
|---|---|---|
| TabM as a full member (fraud and type) | public 0.5483, against 0.5499 for fraud-only | type switches hurt |
| TabM at 2/3, 4/5 or all of the fraud mass | generator-only August −0.00013 to −0.00070 | too much weight |
| 6 or 9 TabM seeds | identical calls | 3 seeds kept |
| SWA TabM | −0.00004 / −0.00001 / +0.00004, 0 October rows | rejected |
| Mitra-2 in the fraud mass | −0.00009 / −0.00015 / −0.00018 | rejected |
| TabPFN v2 in the fraud mass (1/6) | −0.00018 / −0.00014 / −0.00006 | rejected |
| TabDPT in the fraud mass | −0.00005 / −0.00012 / −0.00018 | rejected |
| four-way blend without CatBoost | October-like cleaned −0.00010 / −0.00027 / −0.00012 | rejected |
| nested greedy weights | +0.00011 / +0.00005 / −0.00009, October-like bound −0.00020 | rejected |
| refit or hedge trees for October | refit test fails the bar; both files public 0.5482 | rejected |
| all six members retrained on the refined counter | July −0.00006 given, bound −0.00001, 7 October rows | rejected |

## Decoding

| Idea | Result | Verdict |
|---|---|---|
| fraud scale 0.85 / 0.90 instead of 0.875 | 0 / 0 / +0.00003 and −0.00004 / 0 / −0.00004 | ties |
| one fraud type per event before the rule | +0.00033 / +0.00014 / 0, 0 October rows | tie |
| anchor rescue | October-like bound −0.00009 | rejected |

## Rules and October moves

| Idea | Result | Verdict |
|---|---|---|
| `col7` template flipped to phishing (94 rows) | public 0.5408, −0.0078 | the template's majority stands |
| any adjacent-class move on the refined posterior | July and August negative; October-like −0.0001 to −0.0003 | overrides members where they beat the counter |
| top-ups with phishing_collect (`research-combo-1` rules) | month folds positive; October-like bound given −0.00021 | does not hold at October's density |
| operator-role recovery stage | fails the given labels in August and September | rejected |
| `pair_templates` (24 recipes) | 0 train and 0 October rows changed | rejected |
| `capped_rings`: three 8-row rings → merchant_collusion | public 0.5505 against 0.5556, −0.0051 | the rings are refund_abuse |
| the 15-row burst as velocity_bot | read as mule_account on public | removed |
| `border_shift` row (loan_app_extortion → fake_cashback) | removal fits it being public and wrong | removed |
| 2-row border move (`stack-sim45-b9r2`) | public tie, 0.5556 | unread, not adopted |
| flipping every phone-less burst to velocity_bot | lost 0.02 on public (earlier measurement) | mixes many answers |
| members retrained at October's density (`fC1`, 13 rows) | lost 0.0009 on public; negative in one month | rejected |

## Hunting the ordinary-looking fraud

| Audit | Result |
|---|---|
| probe rows against legit rows, 109 context + 10 raw columns | AUC 0.4904 train, 0.4896 October |
| best single column, about 120 tests per split | nothing survives a multiple-test correction |
| links through the same sender, receiver or phone (1 h / 1 d / 7 d) | ratio 0.95 to 1.02 in all nine cells |
| history in earlier months | fraud rate 2.28 % to 2.34 % in every bucket |
| September against October | AUC 0.5024 |
| look-alike pairs, event-level model | AUC at most 0.61 |

The conclusion holds across every audit: ordinary-looking fraud labels sit on rows that carry no
signal. Runs aimed at them are not worth spending.
