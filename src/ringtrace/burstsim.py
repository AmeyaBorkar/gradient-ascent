"""Large mule-shaped bursts with a fresh SIM and no fraud-phase phone, called velocity_bot.

Mule fan-ins and bot bursts are written in one shape (template mf2+mf3). When a burst has a phone
in the fraud phase the counter's zone names its class; when it has none the template rule calls
it mule_account, the pair's majority. ``fresh_bursts`` calls such a burst velocity_bot when it has
at least ``min_rows`` rows and its youngest sender SIM is under ``max_sim_hours`` hours old.

Train measures this weakly: of the 9 bursts without a phase phone that meet both cuts, 4 are
velocity_bot on the cleaned labels (2 of the other 33 with 20 rows or more). One event has one
class, so its rows share one answer, and the rule enters as a base-rate rule with a row cap.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.evidence import anchors
from ringtrace.evidence.anchors import BOT, MULE

BURST_TEMPLATE = 3


def fresh_bursts(
    pred, proba, led, log=print, min_rows: int = 20, max_sim_hours: float = 50.0
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.int64)
    tmpl, event = anchors.templates(led)
    block = anchors.family_columns(led, ["ctr_block"])["ctr_block"].fill_null(0).to_numpy()
    inside = np.asarray(tmpl) == BURST_TEMPLATE
    out = pred.copy()
    if inside.any():
        ev = (
            pl.DataFrame(
                {
                    "ev": event[inside],
                    "phase": block[inside] == 2,
                    "sim": led["sim_age_hours"].to_numpy().astype(np.float64)[inside],
                }
            )
            .group_by("ev")
            .agg(pl.len().alias("rows"), pl.col("phase").any(), pl.col("sim").min())
            .filter(
                ~pl.col("phase") & (pl.col("rows") >= min_rows) & (pl.col("sim") < max_sim_hours)
            )
        )
        hit = inside & np.isin(event, ev["ev"].to_numpy()) & np.isin(pred, (MULE, BOT))
        out[hit] = BOT
        is_test = led["is_test"].to_numpy()
        ch = out != pred
        log(
            f"  fresh bursts: {ev.height} events, "
            f"train {int((ch & ~is_test).sum())} / test {int((ch & is_test).sum())} rows"
        )
    return out
