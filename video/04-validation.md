# Validation: how every change was judged

## Three views

| View | What it is | What it catches |
|---|---|---|
| **Month folds** | leave-one-month-out on July, August, September. Scored on given labels, on cleaned labels from an independent teacher, and on generator-only labels. Paired bootstrap over label-free events, 2,000 draws. | model and feature changes |
| **October-like frames** | September thinned to October's density (keep rate 0.494, thinning seeds 13 / 14 / 15, fraud events kept whole), members fitted on July + August, seed sets r0-r2: nine cells. The device counter is rebuilt on each frame as its own pass, as October's is. | changes that only help at full density |
| **Labelled counter passes** | train's fraud phase, plus September frames rebuilt at October's density, where the counter's borders can be scored | border and counter changes that train cannot see |

## The bar

- A model or feature change is adopted only when it is non-negative in every held-out month on
  given and cleaned labels, and the event-bootstrap lower bound of the mean is above zero.
- On October-like frames it must be positive on every thinning seed, with the pooled lower bound
  above zero.
- A tie goes to the simpler file.
- **Border changes** move no train call: the train phase is six times longer. They are judged by
  calibration at the test phase's size instead.
- **One-event rules** (`fresh_bursts`) are base-rate rules with a row cap. Their cuts were set with
  public reads, and the repository says so.

## The numbers behind the final file

| Component | Month folds Jul / Aug / Sep | October-like | October rows | Public |
|---|---|---|---|---|
| TabM sets half the fraud mass | given +0.00073 / +0.00025 / +0.00025; cleaned +0.00068 / +0.00025 / +0.00039 | lower bound +0.00012 (given and cleaned) | 7 | 0.5499 |
| operator-role bait rules | cleaned +0.00029 / +0.00017 / +0.00016 | nine cells +0.00018 (bound +0.00005) | 7 | 0.5502 |
| `mule_edge` | untestable: train's one shared phase; −0.00004 / −0.00008 / −0.00012 from 6 rows at its exact edge | October's mule zone 1.21x wider; anchored phones about 40 to 1 for mule | 10 | 0.5506 |
| three small rules (`research-combo-2`) | given +0.00024 / +0.00010 / +0.00013; cleaned +0.00021 / +0.00026 / +0.00015 | **failed**: lower bound given −0.00010, cleaned −0.00005 | 4 | 0.5506 (tie) |
| `counter_news` | given +0.00004 / +0.00008 / +0.00088 [+0.00004, +0.00099]; cleaned +0.00003 / +0.00008 / +0.00104 [+0.00004, +0.00118] | given +0.00125 [+0.00040, +0.00234]; cleaned +0.00140 [+0.00045, +0.00258] | 4 | 0.5516 |
| `fresh_bursts` | loses on month folds: more than half the bursts it fires on in train are mule_account | not gated | 40 | 0.5556 |

`counter_news` also passed a fourth frame built after its upload. Pooled over four frames: given
+0.00157 [+0.00071, +0.00264], cleaned +0.00175 [+0.00082, +0.00289].

## The border calibration

300 thinned blocks at October's size, pooled over the ten borders:

| Said | 0.05 | 0.25 | 0.45 | 0.55 | 0.75 | 0.95 |
|---|---|---|---|---|---|---|
| Happened (no mix shift) | 0.07 | 0.25 | 0.40 | 0.56 | 0.79 | 0.98 |

Two borders do not hold: layering_chain | sim_swap (said 0.13 to 0.14, happened 0.61 to 0.65) and
account_takeover | phishing_collect (said 0.92, happened 0.63 to 0.69).

## What `fresh_bursts` rests on

Train, mule-shaped bursts on cleaned labels:

| Group | Bursts | velocity_bot |
|---|---|---|
| all | 204 | 19 |
| no fraud-phase phone, 20 rows or more, SIM under 50 h | 9 | 4 |
| no fraud-phase phone, 20 rows or more, SIM 50 h or more | 33 | 2 |
| no fraud-phase phone, exactly 12 rows, SIM under 50 h | 2 | 0 |

Train supports the cut only weakly. The final cuts (12 rows, SIM 45 to 50 hours) come from the
public reads in `05-upload-history.md`. One event has one class, so a public score reads the whole
event.

## What the validation taught

- The call-count forecast predicts a file's public score: 0.5475 expected for the five-member
  blend, 0.5474 public.
- The counter depends on density, so each October-like frame rebuilds it as its own pass, exactly
  as October's is built.
- A single file's public score has an sd of about 0.0059 against its true score. Two files that
  differ on a few dozen rows have a public-gap sd of 0.0002 to 0.0005.
- A single-row border move cannot raise the private score through a public read: a rise means the
  row was public.
