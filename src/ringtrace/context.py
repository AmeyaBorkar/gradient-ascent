"""Relational context of every row: what the same sender, receiver, phone and pair did around it.

Everything here is label-free and computed on the stacked ledger (train and test together), in
both time directions, because an event is only visible whole. Columns that identify the month
(absolute dates, raw identifiers, the two columns that die in October) never leave this module.
Long-window counts are divided by the month's background density, because October's ordinary
traffic is about half as dense as the train months'.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.tables import HOME_DEVICES, MERCHANT_OFFSET
from ringtrace.windows import Timeline, distinct_in_window

MIN = 60
HOUR = 3600
DAY = 86400
# previous/next gaps beyond these windows say more about the month than about the row
GAP_CAP = {"snd": DAY, "dev": DAY, "rcv": 3 * DAY, "pair": 7 * DAY}

CATEGORICALS = ("txn_type", "merchant_category", "app", "s_kyc_level", "s_os", "s_bank_code",
                "m_category", "r_kyc")


def _codes(col: pl.Series) -> np.ndarray:
    return col.cast(pl.Utf8).fill_null("~").rank("dense").to_numpy().astype(np.float32)


def month_deflator(led: pl.DataFrame) -> np.ndarray:
    """Factor that maps a row's amount to the train months' scale (October spends 1.166x)."""
    amt = led["amount"].to_numpy()
    period = led["period"].to_numpy()
    med = np.ones_like(amt)
    for p in np.unique(period):
        m = period == p
        med[m] = np.median(amt[m])
    ref = np.median(amt[period <= 9])
    return ref / med


def row_view(led: pl.DataFrame) -> dict[str, np.ndarray]:
    """Single-row facts. Amounts stay in rupees: the festival scaling touched legit traffic only,
    so a month-relative amount would mostly encode the month."""
    f: dict[str, np.ndarray] = {}
    amt = led["amount"].to_numpy()
    bal = led["balance_before"].to_numpy()
    f["log_amount"] = np.log1p(amt)
    f["log_balance"] = np.log1p(np.maximum(bal, 0))
    cents = np.round(amt * 100).astype(np.int64)
    whole = cents % 100 == 0
    rupees = cents // 100
    f["amt_whole"] = whole
    f["amt_whole_not10"] = whole & (rupees % 10 != 0)
    f["amt_mod10"] = np.where(whole, rupees % 10, -1)
    f["amt_mod100"] = np.where(whole, rupees % 100, -1)
    f["amt_mod1000"] = np.where(whole, rupees % 1000, -1)
    f["amt_round100"] = whole & (rupees % 100 == 0)
    f["amt_tiny"] = amt <= 10
    with np.errstate(divide="ignore", invalid="ignore"):
        f["amt_over_bal"] = amt / bal
        f["bal_over_amt"] = bal / amt
    f["amt_gt_bal"] = amt > bal + 1e-9
    f["log_bal_after"] = np.sign(bal - amt) * np.log1p(np.abs(bal - amt))
    ts = led["timestamp"]
    f["hour"] = (ts.dt.hour() + ts.dt.minute() / 60).to_numpy()
    # calendar-trending columns (weekday, account and device ages) identify the month: left out
    for c in ("sim_age_hours", "new_payee", "pin_retries", "login_to_txn_sec"):
        f[c] = led[c].to_numpy()
    f["sim_fresh"] = led["sim_age_hours"].to_numpy() <= 36
    f["phone_fresh"] = led["device_age_days"].to_numpy() < 2
    date = ts.dt.date()
    age_true = (date - led["s_opened_on"]).dt.total_days().to_numpy()
    f["acct_age_err"] = led["sender_account_age_days"].to_numpy() - age_true
    f["to_person"] = led["to_person"].to_numpy()
    dev = led["device"].to_numpy()
    f["phone_extra"] = dev >= HOME_DEVICES
    f["phone_own"] = dev == led["sender"].to_numpy()
    f["away"] = (led["device_city"] != led["s_home_city"]).fill_null(True).to_numpy()
    dest_city = led.select(pl.coalesce("m_city", "r_home_city")).to_series()
    f["dest_city_match"] = (led["device_city"] == dest_city).fill_null(False).to_numpy()
    f["cat_mismatch"] = (led["merchant_category"] != led["m_category"]).fill_null(False).to_numpy()
    # The QR check means the same in October only for merchant receivers; its stale state
    # (filled on a non-QR row, or empty on a QR merchant row) is a generator mark everywhere.
    qr = led["qr_vpa_mismatch"].to_numpy()
    is_qr = (led["txn_type"] == "QR").to_numpy()
    to_person = led["to_person"].to_numpy()
    f["qr_flag_merchant"] = np.where(to_person, np.nan, qr)
    f["qr_stale"] = (~is_qr & ~np.isnan(qr)) | (is_qr & np.isnan(qr) & ~to_person)
    for c in CATEGORICALS:
        f[f"cat_{c}"] = _codes(led[c])
    return f


def density(led: pl.DataFrame) -> np.ndarray:
    """Rows per day of the row's period relative to the train months' mean."""
    period = led["period"].to_numpy()
    per = pl.DataFrame({"p": period, "d": led["timestamp"].dt.date()}).group_by("p").agg(
        (pl.len() / pl.col("d").n_unique()).alias("rate"))
    rate = dict(zip(per["p"].to_list(), per["rate"].to_list(), strict=True))
    base = np.mean([rate[p] for p in rate if p <= 9])
    lut = np.zeros(int(max(rate)) + 1)
    for p, r in rate.items():
        lut[int(p)] = r / base
    return lut[period]


def entity_view(led: pl.DataFrame) -> dict[str, np.ndarray]:
    f: dict[str, np.ndarray] = {}
    ts = led["ts"].to_numpy()
    amt = led["amount"].to_numpy()
    dens = density(led)
    sender = led["sender"].to_numpy()
    receiver = led["receiver"].to_numpy()
    device = led["device"].to_numpy()
    pair = sender * (2 * MERCHANT_OFFSET) + receiver
    keys = {"snd": sender, "rcv": receiver, "dev": device, "pair": pair}
    windows = {"10m": 10 * MIN, "1h": HOUR, "1d": DAY, "7d": 7 * DAY}
    for name, key in keys.items():
        tl = Timeline(key, ts)
        for wn, w in windows.items():
            back = tl.count(w, 0)
            ahead = tl.count(0, w)
            if w >= DAY:
                back = back / dens
                ahead = ahead / dens
            f[f"{name}_n_back_{wn}"] = back
            f[f"{name}_n_ahead_{wn}"] = ahead
            if w <= HOUR or name == "pair":
                f[f"{name}_amt_back_{wn}"] = np.log1p(tl.total(amt, w, 0))
        cap = GAP_CAP[name]
        for d, nm in ((-1, "prev"), (+1, "next")):
            g = tl.gap(d)
            f[f"{name}_gap_{nm}"] = np.where(g <= cap, g, np.nan)
    for wn, w in (("1h", HOUR), ("1d", DAY)):
        f[f"rcv_dsnd_back_{wn}"] = distinct_in_window(receiver, sender, ts, w)
        f[f"rcv_dsnd_ahead_{wn}"] = distinct_in_window(receiver, sender, ts, 0, w)
        f[f"dev_dsnd_back_{wn}"] = distinct_in_window(device, sender, ts, w)
        f[f"dev_dsnd_ahead_{wn}"] = distinct_in_window(device, sender, ts, 0, w)
        f[f"snd_drcv_back_{wn}"] = distinct_in_window(sender, receiver, ts, w)
        f[f"snd_ddev_back_{wn}"] = distinct_in_window(sender, device, ts, w)
    # Receiver windows on merchants mostly measure the merchant's size, which moves with the
    # month's density: keep them for person receivers only.
    merchant = ~led["to_person"].to_numpy()
    for k in list(f):
        if k.startswith("rcv_"):
            f[k] = np.where(merchant, np.nan, f[k])
    f.update(reverse_payments(sender, receiver, ts, amt))
    return f


def reverse_payments(sender, receiver, ts, amt, window: int = 7 * DAY) -> dict[str, np.ndarray]:
    """Payments in the opposite direction (receiver -> sender) within +-7 days of the row."""
    fwd = sender.astype(np.int64) * (2 * MERCHANT_OFFSET) + receiver
    rev = receiver.astype(np.int64) * (2 * MERCHANT_OFFSET) + sender
    keys, inv = np.unique(np.concatenate([fwd, rev]), return_inverse=True)
    del keys
    kf, kr = inv[: len(fwd)], inv[len(fwd):]
    t = ts.astype(np.int64) - int(ts.min())
    comp = kf.astype(np.int64) * (1 << 33) + t
    order = np.argsort(comp, kind="stable")
    comp_sorted = comp[order]
    q = kr.astype(np.int64) * (1 << 33) + t
    lo = np.searchsorted(comp_sorted, q - window, side="left")
    hi = np.searchsorted(comp_sorted, q + window, side="right")
    n = (hi - lo).astype(np.float32)
    prev_i = np.searchsorted(comp_sorted, q, side="left") - 1
    has_prev = (prev_i >= 0) & (prev_i >= lo)
    j = order[np.clip(prev_i, 0, None)]
    out = {
        "rev_n_7d": n,
        "rev_prev_dt": np.where(has_prev, t - t[j], np.nan),
        "rev_prev_amt_ratio": np.where(has_prev, amt / amt[j], np.nan),
    }
    nxt_i = np.searchsorted(comp_sorted, q, side="right")
    has_next = nxt_i < hi
    j2 = order[np.clip(nxt_i, 0, len(order) - 1)]
    out["rev_next_dt"] = np.where(has_next, t[j2] - t, np.nan)
    out["rev_next_amt_ratio"] = np.where(has_next, amt[j2] / amt, np.nan)
    return out


def flow_view(led: pl.DataFrame) -> dict[str, np.ndarray]:
    """Money moving on: what the receiver sent next, and what the sender received just before."""
    f: dict[str, np.ndarray] = {}
    base = led.select("row", "sender", "receiver", "ts", "amount", "to_person").with_columns(
        pl.col("ts").cast(pl.Int64))
    outgoing = base.select(pl.col("sender").alias("acct"), pl.col("ts").alias("t_out"),
                           pl.col("amount").alias("x_out")).sort("t_out")
    incoming = base.filter(pl.col("to_person")).select(
        pl.col("receiver").alias("acct"), pl.col("ts").alias("t_in"),
        pl.col("amount").alias("x_in")).sort("t_in")
    left = base.sort("ts")
    nxt = left.join_asof(outgoing, left_on="ts", right_on="t_out", by_left="receiver",
                         by_right="acct", strategy="forward", allow_exact_matches=False
                         ).sort("row")
    person = nxt["to_person"].to_numpy()
    f["flow_next_out_dt"] = np.where(person, (nxt["t_out"] - nxt["ts"]).to_numpy(), np.nan)
    f["flow_next_out_ratio"] = np.where(person, (nxt["x_out"] / nxt["amount"]).to_numpy(),
                                        np.nan)
    prv = left.join_asof(incoming, left_on="ts", right_on="t_in", by_left="sender",
                         by_right="acct", strategy="backward", allow_exact_matches=False
                         ).sort("row")
    f["flow_prev_in_dt"] = (prv["ts"] - prv["t_in"]).to_numpy()
    f["flow_prev_in_ratio"] = (prv["amount"] / prv["x_in"]).to_numpy()
    as_rcv = np.where(base["to_person"].to_numpy(), base["receiver"].to_numpy(), -1)
    tl_in = Timeline(as_rcv, base["ts"].to_numpy())
    f["flow_in_n_back_1h"] = tl_in.count(HOUR, 0)
    return f


def build(led: pl.DataFrame) -> pl.DataFrame:
    cols: dict[str, np.ndarray] = {}
    for view in (row_view, entity_view, flow_view):
        cols.update(view(led))
    return pl.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in cols.items()})
