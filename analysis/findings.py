"""Run the four findings analyses on the member runs and write aggregates to JSON.

    python analysis/findings.py [--out DIR] [--draws 2000]

Reads ``$RINGTRACE_OUT/runs/<member>/{oof,test}.npy``, decodes each member as the run does
(fixed point at 0.875 per month, then the anchored template rule) and writes
``findings.json`` with aggregates only (counts, rates, scores).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import (  # noqa: E402
    disagreement,
    fraud_pass_share,
    headroom,
    one_row_gain,
    slice_scores,
)

from ringtrace import paths, silver, store  # noqa: E402
from ringtrace import rules as rulebook  # noqa: E402
from ringtrace.classes import CLASSES  # noqa: E402
from ringtrace.decode import decode_by_period  # noqa: E402
from ringtrace.evidence import anchors  # noqa: E402
from ringtrace.scoring import macro_f1, per_class_f1  # noqa: E402

MEMBERS = ("champion", "catboost", "clean-event", "clean-l2", "lgbm")
PUBLIC_ROWS = 78_175
TRAIN_MONTHS = (7, 8, 9)


def decode_member(name: str, led, period, tr_idx, te_idx) -> np.ndarray:
    d = paths.out_dir() / "runs" / name
    proba = np.zeros((led.height, len(CLASSES)))
    proba[tr_idx] = np.load(d / "oof.npy")
    proba[te_idx] = np.load(d / "test.npy")
    pred = np.zeros(led.height, dtype=np.int64)
    pred[tr_idx] = decode_by_period(proba[tr_idx], period[tr_idx], fraud_scale=0.875)
    pred[te_idx] = decode_by_period(proba[te_idx], period[te_idx], fraud_scale=0.875)
    return rulebook.apply(["anchored_templates"], pred, proba, led, log=lambda _m: None)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args(argv)
    t0 = time.time()
    out_dir = Path(args.out) if args.out else paths.out_dir() / "findings-analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    led = store.ledger()
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    tr_idx, te_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    given = led["y"].to_numpy().astype(np.int64)
    clean = silver.load(led, "label")
    gen = silver.load(led, "structured")
    members = [m for m in MEMBERS if (paths.out_dir() / "runs" / m / "test.npy").exists()]
    calls = {m: decode_member(m, led, period, tr_idx, te_idx) for m in members}
    rep: dict = {"members": members, "classes": list(CLASSES)}

    # scores per member, to check the decode matches the runs
    rep["scores"] = {}
    for m in members:
        s = {}
        for tname, y in (("given", given), ("cleaned", clean), ("gen", gen)):
            per = [macro_f1(y[period == p], calls[m][period == p]) for p in TRAIN_MONTHS]
            s[tname] = {"months": per, "mean": float(np.mean(per))}
        rep["scores"][m] = s

    # 1. public-score noise on September
    sep = np.flatnonzero(period == 9)
    n_oct = len(te_idx)
    frac = PUBLIC_ROWS / n_oct
    preds = [calls[m][sep] for m in members]
    sl = slice_scores(clean[sep], preds, frac, args.draws, seed=0)
    pub, priv = sl["public"], sl["private"]
    noise: dict = {
        "public_fraction": frac,
        "sep_rows": len(sep),
        "draws": args.draws,
        "pairs": {},
        "one_file": {},
    }
    for j, m in enumerate(members):
        noise["one_file"][m] = {
            "full": float(macro_f1(clean[sep], preds[j])),
            "public_sd": float(pub[:, j].std()),
            "pub_minus_priv_sd": float((pub[:, j] - priv[:, j]).std()),
        }
        if j == 0:
            continue
        d = pub[:, j] - pub[:, 0]
        full_gap = macro_f1(clean[sep], preds[j]) - macro_f1(clean[sep], preds[0])
        dpriv = priv[:, j] - priv[:, 0]
        noise["pairs"][m] = {
            "rows_differ": int((preds[j] != preds[0]).sum()),
            "full_gap": float(full_gap),
            "public_gap_sd": float(d.std()),
            "public_gap_mean": float(d.mean()),
            "p_abs_dev_ge": {
                f"{g:.4f}": float((np.abs(d - full_gap) >= g).mean())
                for g in (0.0005, 0.001, 0.002, 0.003, 0.005)
            },
            "public_and_private_disagree_on_sign": float((np.sign(d) != np.sign(dpriv)).mean()),
        }
    # per-class value of one extra correct row on the public slice
    oct_calls = np.bincount(calls["champion"][te_idx], minlength=12) * frac
    sep_mix = np.bincount(clean[sep], minlength=12) / len(sep)
    n_true = sep_mix * PUBLIC_ROWS
    f1_sep = per_class_f1(clean[sep], calls["champion"][sep])
    gain = one_row_gain(n_true, oct_calls, f1_sep)
    noise["one_row_gain"] = {
        c: {
            "public_true": float(n_true[i]),
            "public_calls": float(oct_calls[i]),
            "sep_f1": float(f1_sep[i]),
            "gain": float(gain[i]),
        }
        for i, c in enumerate(CLASSES)
    }
    rep["noise"] = noise

    # 2. headroom per class (champion, pooled held-out months, cleaned labels)
    tr_clean, tr_gen = clean[tr_idx], gen[tr_idx]
    h = headroom(tr_clean, tr_gen, calls["champion"][tr_idx])
    rep["headroom"] = {
        "per_class": {
            c: {k: float(h[k][i]) for k in ("now", "ceiling", "gap", "ordinary_share")}
            for i, c in enumerate(CLASSES)
        },
        "macro_now": h["macro_now"],
        "macro_ceiling": h["macro_ceiling"],
        "macro_gap": h["macro_ceiling"] - h["macro_now"],
    }

    # 3. disagreements vs the champion
    rep["disagree"] = {}
    ref = calls["champion"]
    for m in members[1:]:
        rep["disagree"][m] = {
            "heldout": disagreement(ref[tr_idx], calls[m][tr_idx], tr_clean),
            "october": disagreement(ref[te_idx], calls[m][te_idx]),
        }
        g = disagreement(ref[tr_idx], calls[m][tr_idx], given[tr_idx])
        rep["disagree"][m]["heldout_given"] = {
            k: g[k] for k in ("rows", "truth_other", "truth_ref", "truth_neither")
        }

    # 4. validation blind spot (train only; these columns are dead in October)
    app_old = led["app_version"].to_numpy()[tr_idx] < 4.0
    ip_hi = led["ip_risk_score"].to_numpy()[tr_idx] > 0.5
    anchor = anchors.row_anchor(led)[tr_idx]
    block = anchors.family_columns(led, ["ctr_block"])["ctr_block"].fill_null(0).to_numpy()
    block = block[tr_idx]
    tmpl = anchors.template_of(led)[tr_idx]
    champ_tr = ref[tr_idx]

    def check(mask: np.ndarray) -> dict:
        r = fraud_pass_share(app_old[mask])
        r["ip_high"] = float(ip_hi[mask].mean()) if mask.any() else float("nan")
        r["given_fraud"] = float((given[tr_idx][mask] > 0).mean()) if mask.any() else None
        r["cleaned_fraud"] = float((tr_clean[mask] > 0).mean()) if mask.any() else None
        return r

    blind: dict = {
        "reference": {
            "given_fraud": check(given[tr_idx] > 0),
            "given_legit": check(given[tr_idx] == 0),
            "cleaned_legit_given_legit": check((given[tr_idx] == 0) & (tr_clean == 0)),
            "unreported_cleaned": check((given[tr_idx] == 0) & (tr_clean > 0)),
            "ordinary_fraud": check((tr_clean > 0) & (tr_gen == 0)),
        },
        "evidence_called_legit": {},
    }
    legit_call = champ_tr == 0
    for name, mask in (
        ("anchor", anchor > 0),
        ("fraud_block", block == 2),
        ("template", tmpl > 0),
        ("any", (anchor > 0) | (block == 2)),
    ):
        blind["evidence_called_legit"][name] = check(mask & legit_call)
        blind["evidence_called_legit"][name + "_given_legit"] = check(
            mask & legit_call & (given[tr_idx] == 0)
        )
    blind["disagree_cells"] = {}
    for m in members[1:]:
        o = calls[m][tr_idx]
        cells = {}
        diff = o != champ_tr
        cells["member_fraud_champ_legit"] = check(diff & (champ_tr == 0) & (o > 0))
        cells["member_legit_champ_fraud"] = check(diff & (champ_tr > 0) & (o == 0))
        cells["both_fraud_type_differs"] = check(diff & (champ_tr > 0) & (o > 0))
        for c in rep["disagree"][m]["heldout"]["cells"][:4]:
            sel = diff & (champ_tr == c["ref"]) & (o == c["other"])
            cells[f"{CLASSES[c['ref']]}->{CLASSES[c['other']]}"] = check(sel)
        blind["disagree_cells"][m] = cells
    rep["blind_spot"] = blind
    rep["runtime_s"] = time.time() - t0
    (out_dir / "findings.json").write_text(json.dumps(rep, indent=1))
    print(f"done in {rep['runtime_s']:.0f}s -> {out_dir / 'findings.json'}")
    return rep


if __name__ == "__main__":
    main()
