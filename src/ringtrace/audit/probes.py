"""Label-free row groups from the device counter, for audits that need ground truth on test rows.

The counter (``ringtrace.evidence.phones``) splits the phones numbered above 420,000 into the bot
block, each split's fraud phase and its legit block. A fraud-phase phone was minted while the fraud
pass wrote its rows; a legit-block phone while the legit pass wrote its rows. So:

* a row on a one-row fraud-phase phone with no template evidence is an *ordinary* fraud-pass row
  (the legit row template, labelled with a fraud class), and its counter segment names its class;
* a row on a legit-block phone is a legit-pass row.

Both groups exist in every split, October included, and neither needs a label. Comparing them asks
whether a feature can tell ordinary fraud from legit traffic. They differ from the bulk of rows only
in sitting on a new phone, so phone-history features are not comparable across the groups; every
other feature is.

Row kinds: own home phone, borrowed home phone (another account's), bot block, fraud phase, legit
block, and extra phones the counter could not place.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.tables import HOME_DEVICES

# event-id columns of the label-free families (``<prefix>_event``, null or -1 when outside)
EVENT_PREFIXES = ("bot", "drn", "mf", "col", "rng", "rep")

OWN, BORROWED, BOT, FRAUD_PHASE, LEGIT_BLOCK, UNPLACED = range(6)
KIND_NAMES = ("own", "borrowed", "bot", "fraud_phase", "legit_block", "unplaced")

# ordinary-probe codes
ORDINARY, LEGIT, OUTSIDE = 1, 0, -1

# counter block codes (ringtrace.evidence.phones)
_CTR_BOT, _CTR_FRAUD, _CTR_LEGIT = 1, 2, 3


def evidence(events: list[np.ndarray], marks_any: np.ndarray) -> np.ndarray:
    """True where any family places the row in an event (id >= 0) or any one-row mark fires."""
    out = np.asarray(marks_any, dtype=np.float64)
    out = np.nan_to_num(out, nan=0.0) > 0
    for ev in events:
        e = np.nan_to_num(np.asarray(ev, dtype=np.float64), nan=-1.0)
        out |= e >= 0
    return out


def device_kind(device: np.ndarray, sender: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Kind code of each row's phone (see ``KIND_NAMES``)."""
    device = np.asarray(device, dtype=np.int64)
    sender = np.asarray(sender, dtype=np.int64)
    block = np.asarray(block, dtype=np.int64)
    kind = np.full(len(device), UNPLACED, dtype=np.int8)
    home = device <= HOME_DEVICES
    kind[home & (device == sender)] = OWN
    kind[home & (device != sender)] = BORROWED
    extra = ~home
    kind[extra & (block == _CTR_BOT)] = BOT
    kind[extra & (block == _CTR_FRAUD)] = FRAUD_PHASE
    kind[extra & (block == _CTR_LEGIT)] = LEGIT_BLOCK
    return kind


def ordinary_probe(kind: np.ndarray, dev_single: np.ndarray, evid: np.ndarray) -> np.ndarray:
    """1 = one-row fraud-phase phone without evidence, 0 = legit-block row without evidence,
    -1 = any other row."""
    kind = np.asarray(kind)
    single = np.asarray(dev_single).astype(bool)
    evid = np.asarray(evid).astype(bool)
    out = np.full(len(kind), OUTSIDE, dtype=np.int8)
    out[(kind == FRAUD_PHASE) & single & ~evid] = ORDINARY
    out[(kind == LEGIT_BLOCK) & ~evid] = LEGIT
    return out


def segment_class(seg: np.ndarray) -> np.ndarray:
    """Fraud class code (1..11) of a counter segment (0..10); 0 where the row has no segment."""
    s = np.nan_to_num(np.asarray(seg, dtype=np.float64), nan=-1.0).astype(np.int64)
    return np.where(s >= 0, s + 1, 0).astype(np.int8)


def groups(led: pl.DataFrame) -> pl.DataFrame:
    """Kind, evidence, ordinary-probe code and counter class of every ledger row."""
    from ringtrace.evidence import anchors

    cols = [f"{p}_event" for p in EVENT_PREFIXES] + [
        "mrk_any",
        "ctr_block",
        "ctr_dev_single",
        "ctr_seg",
    ]
    fam = anchors.family_columns(led, cols)
    evid = evidence(
        [fam[f"{p}_event"].to_numpy() for p in EVENT_PREFIXES], fam["mrk_any"].to_numpy()
    )
    kind = device_kind(
        led["device"].to_numpy(), led["sender"].to_numpy(), fam["ctr_block"].to_numpy()
    )
    probe = ordinary_probe(kind, fam["ctr_dev_single"].to_numpy(), evid)
    return pl.DataFrame(
        {
            "kind": pl.Series(kind, dtype=pl.Int8),
            "evidence": pl.Series(evid, dtype=pl.Boolean),
            "probe": pl.Series(probe, dtype=pl.Int8),
            "segment": pl.Series(segment_class(fam["ctr_seg"].to_numpy()), dtype=pl.Int8),
        }
    )
