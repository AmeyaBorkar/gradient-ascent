"""Label-free generator signatures: one boolean mask per row for every marker the repo computes.

``catalogue(led)`` returns ``{name: mask}`` over every ledger row (train and October). A signature
reads only label-free columns: the raw row, the row marks, the evidence families' event shapes,
the look-alike templates and the device counter. Names:

* ``mark:<m>``: a row mark of ``evidence.marks`` (``nonqr_check`` is a filled QR check on a non-QR
  row, ``qr_merchant_blank`` a QR payment to a merchant with a blank check);
* ``big_p2p``: an amount above 100,000 to a person;
* ``<prefix>:shape<k>`` / ``<prefix>:any``: rows in an event of that family shape / of any shape;
* ``tmpl:<name>``: rows of a look-alike template;
* ``ctr:fraud_block`` / ``ctr:fraud_block_sure``: rows on a fraud-block phone (any segment / a
  trusted segment with ``ctr_conf >= 0.9``);
* ``and:<a>&<b>`` / ``marks2``: conjunctions (two certificates on one row).

``mask(led, name)`` evaluates one name, conjunctions included.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.evidence import anchors

MARKS = (
    "p2p_category",
    "qr_p2p_nocat",
    "bal110",
    "bal102",
    "over_balance",
    "step025",
    "step035",
    "round5",
    "classic12",
    "bait",
)
FAMILY_SHAPES = {
    "bot": (1, 2, 3, 4),
    "drn": (1, 2, 3, 4),
    "mf": (1, 2, 3, 4),
    "col": (1, 2, 3, 4, 5, 6, 7, 8, 9),
    "rng": (1, 2, 3, 4, 5),
    "rep": (1, 2, 3),
}
BIG_P2P = 100_000.0
SURE_CONF = 0.9
TRUSTED_SEGMENTS = 7
CONJUNCTIONS = (
    ("mark:bal110", "ctr:fraud_block"),
    ("mark:over_balance", "ctr:fraud_block"),
    ("mark:nonqr_check", "ctr:fraud_block"),
    ("mark:p2p_category", "ctr:fraud_block"),
    ("mark:bal110", "mark:over_balance"),
)


def _raw(led: pl.DataFrame) -> dict[str, np.ndarray]:
    qr = (led["txn_type"] == "QR").fill_null(False).to_numpy()
    merchant = (led["receiver_type"] == "MERCHANT").fill_null(False).to_numpy()
    check = led["qr_vpa_mismatch"].cast(pl.Float64).fill_nan(None).is_not_null().to_numpy()
    person = ~merchant
    amount = led["amount"].to_numpy().astype(np.float64)
    return {
        "mark:nonqr_check": ~qr & check,
        "mark:qr_merchant_blank": qr & merchant & ~check,
        "big_p2p": person & (amount > BIG_P2P),
    }


def catalogue(led: pl.DataFrame) -> dict[str, np.ndarray]:
    """Every single and conjunction signature, as boolean arrays in ledger order."""
    out = _raw(led)
    mk = anchors.family_columns(led, [f"mrk_{m}" for m in MARKS])
    for m in MARKS:
        out[f"mark:{m}"] = mk[f"mrk_{m}"].fill_null(False).cast(pl.Boolean).to_numpy()
    cols = anchors.family_columns(led, [f"{p}_shape" for p in FAMILY_SHAPES])
    for p, shapes in FAMILY_SHAPES.items():
        sh = cols[f"{p}_shape"].fill_null(0).to_numpy().astype(np.int64)
        out[f"{p}:any"] = sh > 0
        for k in shapes:
            out[f"{p}:shape{k}"] = sh == k
    tmpl = anchors.template_of(led)
    for t, name in anchors.TEMPLATE_NAMES.items():
        out[f"tmpl:{name}"] = tmpl == t
    c = anchors.family_columns(led, ["ctr_block", "ctr_seg", "ctr_conf"])
    block = c["ctr_block"].fill_null(0).to_numpy()
    seg = c["ctr_seg"].cast(pl.Float64).fill_null(-1.0).to_numpy()
    conf = c["ctr_conf"].cast(pl.Float64).fill_null(-1.0).to_numpy()
    out["ctr:fraud_block"] = block == 2
    out["ctr:fraud_block_sure"] = (
        (block == 2) & (seg >= 0) & (seg < TRUSTED_SEGMENTS) & (conf >= SURE_CONF)
    )
    certs = [
        "mark:nonqr_check",
        "mark:qr_merchant_blank",
        "mark:p2p_category",
        "mark:qr_p2p_nocat",
        "mark:bal110",
        "mark:bal102",
        "mark:over_balance",
        "mark:step025",
        "mark:step035",
    ]
    out["marks2"] = np.sum([out[k] for k in certs], axis=0) >= 2
    for a, b in CONJUNCTIONS:
        out[f"and:{a}&{b}"] = out[a] & out[b]
    return out


def mask(led: pl.DataFrame, name: str) -> np.ndarray:
    """One signature by name; ``a&b`` is a conjunction and ``a|b`` a union (``&`` binds first)."""
    cat = catalogue(led)
    if name in cat:
        return cat[name]
    terms = [t.removeprefix("and:").split("&") for t in name.split("|")]
    missing = [p for t in terms for p in t if p not in cat]
    if missing:
        raise KeyError(f"unknown signatures {missing}; known: {sorted(cat)}")
    return np.logical_or.reduce([np.logical_and.reduce([cat[p] for p in t]) for t in terms])
