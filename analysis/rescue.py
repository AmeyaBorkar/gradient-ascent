"""Signature rescue: generator signatures whose October rows the blend calls legit.

    python analysis/rescue.py survey                 # every signature: train and October aggregates
    python analysis/rescue.py rule --sig <name>      # leave-one-month-out check of one rescue rule
    python analysis/rescue.py octlike --sig <name>   # October-like check on the champion cells

The blend is the equal-weight mean of the five members' saved arrays, decoded with the fixed
point and the rules of ``recipes/blend-blankqr.json``. October rows are only ever counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ringtrace import paths, signatures, silver, store
from ringtrace.blend import decode_all, load_member, load_recipe, weighted_mean
from ringtrace.classes import CLASSES, N_CLASSES

ROOT = Path(__file__).resolve().parent.parent
RECIPE = ROOT / "recipes" / "blend-blankqr.json"
MONTHS = (7, 8, 9)
MIN_TRAIN = 50
MIN_RATE = 0.97
CHUNK = 100


def load_blend():
    led = store.ledger()
    is_test = led["is_test"].to_numpy()
    recipe = load_recipe(RECIPE)
    parts = [
        load_member(paths.out_dir() / "runs" / m["run"], int((~is_test).sum()), int(is_test.sum()))
        for m in recipe["members"]
    ]
    w = [m["weight"] for m in recipe["members"]]
    proba = np.zeros((led.height, N_CLASSES))
    proba[~is_test] = weighted_mean([p[0] for p in parts], w)
    proba[is_test] = weighted_mean([p[1] for p in parts], w)
    calls = decode_all(
        proba,
        led["period"].to_numpy(),
        is_test,
        recipe["decoder"],
        recipe["rules"],
        led,
        log=lambda m: None,
    )["rules"]
    return led, proba, calls


def _mix(y: np.ndarray, top: int = 3) -> str:
    fr = y[y > 0]
    if len(fr) == 0:
        return "-"
    cls, n = np.unique(fr, return_counts=True)
    order = np.argsort(-n)[:top]
    return ", ".join(f"{CLASSES[cls[i]]} {n[i]}" for i in order)


def survey_rows(cat, y_clean, calls, is_test) -> list[dict]:
    tr, te = ~is_test, is_test
    rows = []
    for name, m in cat.items():
        a, b = m & tr, m & te
        n_tr, n_te = int(a.sum()), int(b.sum())
        rate = float((y_clean[a] > 0).mean()) if n_tr else float("nan")
        oof_legit = float((calls[a] == 0).mean()) if n_tr else float("nan")
        oct_legit = float((calls[b] == 0).mean()) if n_te else float("nan")
        qualifies = (
            n_tr >= MIN_TRAIN
            and rate >= MIN_RATE
            and n_te > 0
            and oct_legit > oof_legit
            and int((calls[b] == 0).sum()) > 0
        )
        rows.append(
            {
                "sig": name,
                "train": n_tr,
                "fraud_rate": rate,
                "mix": _mix(y_clean[a]),
                "oof_legit": oof_legit,
                "october": n_te,
                "oct_legit": oct_legit,
                "oct_legit_rows": int((calls[b] == 0).sum()),
                "expected": rate * int((calls[b] == 0).sum()),
                "qualifies": qualifies,
            }
        )
    return rows


def cmd_survey() -> list[dict]:
    led, _, calls = load_blend()
    is_test = led["is_test"].to_numpy()
    y_clean = silver.load(led, "label").astype(np.int64)
    rows = survey_rows(signatures.catalogue(led), y_clean, calls, is_test)
    print(
        f"{'signature':<52}{'train':>8}{'rate':>8}{'oof_leg':>9}{'oct':>7}{'oct_leg':>9}"
        f"{'rows':>6}  mix"
    )
    for r in sorted(rows, key=lambda r: -r["expected"]):
        print(
            f"{r['sig']:<52}{r['train']:>8}{r['fraud_rate']:>8.4f}{r['oof_legit']:>9.4f}"
            f"{r['october']:>7}{r['oct_legit']:>9.4f}{r['oct_legit_rows']:>6}"
            f"  {'*' if r['qualifies'] else ' '} {r['mix']}"
        )
    return rows


def cmd_rule(sig: str, draws: int = 2000) -> dict:
    """The blend with and without the rescue rule for ``sig``: per-month deltas on the three
    targets, a paired event bootstrap, and the October rows changed with their class mix."""
    import scipy.sparse as sp

    from ringtrace import rules as rulebook
    from ringtrace.blend import month_scores
    from ringtrace.compare import _counts, _macro
    from ringtrace.teacher import union_events

    led, proba, base = load_blend()
    chal = rulebook.apply(
        [{"name": "signature_rescue", "signature": sig}], base, proba, led, log=print
    )
    te = led["is_test"].to_numpy()
    tr = np.flatnonzero(~te)
    period = led["period"].to_numpy()[tr]
    targets = {"given": led["y"].to_numpy().astype(np.int64)[tr]}
    for t, w in (("cleaned", "label"), ("gen", "structured")):
        targets[t] = silver.load(led, w).astype(np.int64)[tr]
    sa, sb = month_scores(base[tr], targets, period), month_scores(chal[tr], targets, period)
    ev = union_events(led.height)[tr]
    groups = np.unique(np.where(ev >= 0, ev, -1 - np.arange(len(tr))), return_inverse=True)[1]
    rng = np.random.default_rng(0)
    changed = base[te] != chal[te]
    rep = {
        "sig": sig,
        "train_changed": int((base[tr] != chal[tr]).sum()),
        "october_changed": int(changed.sum()),
        "october_to": _mix(chal[te][changed], top=11),
    }
    for t, y in targets.items():
        deltas = np.zeros(draws)
        for m in MONTHS:
            s = period == m
            gm = np.unique(groups[s], return_inverse=True)[1]
            ng = int(gm.max()) + 1
            ca, cb = _counts(gm, y[s], base[tr][s], ng), _counts(gm, y[s], chal[tr][s], ng)
            for lo_d in range(0, draws, CHUNK):  # chunks keep the weight matrix small
                k = min(CHUNK, draws - lo_d)
                w = sp.csr_matrix(rng.poisson(1.0, size=(k, ng)).astype(np.float32))
                deltas[lo_d : lo_d + k] += (
                    _macro(np.asarray((w @ cb).todense(), dtype=np.float64))
                    - _macro(np.asarray((w @ ca).todense(), dtype=np.float64))
                ) / 3
        diff = [b - a for a, b in zip(sa[t]["months"], sb[t]["months"], strict=True)]
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        rep[t] = {"months": diff, "mean": float(np.mean(diff)), "lo": float(lo), "hi": float(hi)}
        print(
            f"[rescue] {t:<8} per month "
            + " / ".join(f"{d:+.5f}" for d in diff)
            + f"  mean {np.mean(diff):+.5f}  95% [{lo:+.5f}, {hi:+.5f}]"
        )
    print(
        f"[rescue] rows changed: train {rep['train_changed']}, October "
        f"{rep['october_changed']} ({rep['october_to']})"
    )
    return rep


def cmd_octlike(sig: str, cells: str = "champion") -> dict:
    """Champion cells decoded as in the gate, with and without the rescue rule."""
    from ringtrace import octlike

    rules = ["anchored_templates", "blank_merchant_qr"]
    rec = {"name": "signature_rescue", "signature": sig}
    out = {}
    for seed in octlike.SEEDS:
        frame = octlike.Frame(seed)
        units = frame.unit[frame.hold]
        for r in octlike.SEED_SETS:
            _, prob = octlike.cell_prob(octlike.parse_blend(cells), seed, r)
            ya = octlike.decode(frame, prob, None, rules)
            yb = octlike.decode(frame, prob, None, [*rules, rec])
            cell = {"changed": int((ya != yb).sum())}
            for t, truth in frame.targets.items():
                cell[t] = octlike.bootstrap(units, truth[frame.hold], ya, yb, seed=seed * 100 + r)
            out[f"s{seed}_r{r}"] = cell
            print(
                f"[octlike] s{seed} r{r} changed {cell['changed']}  "
                + "  ".join(
                    f"{t} {cell[t]['delta']:+.5f} [{cell[t]['lo']:+.5f}]" for t in frame.targets
                )
            )
    for t in ("given", "cleaned", "gen"):
        per_seed = [
            float(np.mean([out[f"s{s}_r{r}"][t]["delta"] for r in octlike.SEED_SETS]))
            for s in octlike.SEEDS
        ]
        lb = float(np.mean([c[t]["lo"] for k, c in out.items() if k.startswith("s1")]))
        out[f"summary_{t}"] = {"per_seed": per_seed, "lower_bound": lb}
        print(f"[octlike] {t:<8} per seed " + " / ".join(f"{v:+.5f}" for v in per_seed), end="")
        print(f"  lower bound {lb:+.5f}")
    return out


def cmd_apply(sig: str, base_csv: str, out_csv: str) -> dict:
    """The rescue rule on top of a submission file: October rows of ``base_csv`` that carry the
    signature and are called legit take the five-way blend's most likely fraud class."""
    import hashlib

    import polars as pl

    from ringtrace.blend import load_member, load_recipe, weighted_mean

    led = store.ledger()
    is_test = led["is_test"].to_numpy()
    te = led.filter(pl.col("is_test")).select("txn_id")
    recipe = load_recipe(RECIPE)
    parts = [
        load_member(paths.out_dir() / "runs" / m["run"], int((~is_test).sum()), int(is_test.sum()))
        for m in recipe["members"]
    ]
    proba = weighted_mean([p[1] for p in parts], [m["weight"] for m in recipe["members"]])
    base = pl.read_csv(base_csv, infer_schema=False)
    code = {c: i for i, c in enumerate(CLASSES)}
    calls = te.join(base.rename({"id": "txn_id"}), on="txn_id", how="left", maintain_order="left")
    if calls["label"].null_count():
        raise ValueError("base file does not cover every October row")
    pred = np.array([code[c] for c in calls["label"].to_list()], dtype=np.int64)
    hit = signatures.mask(led, sig)[is_test] & (pred == 0)
    new = pred.copy()
    new[hit] = proba[hit, 1:].argmax(axis=1) + 1
    lut = dict(zip(calls["txn_id"].to_list(), [CLASSES[k] for k in new], strict=True))
    out = base.with_columns(pl.col("id").replace_strict(lut).alias("label"))
    out.write_csv(out_csv)
    sample = pl.read_csv(paths.data_dir() / "sample_submission.csv", infer_schema=False)
    problems = []
    if out["id"].to_list() != sample["id"].to_list():
        problems.append("ids or order differ from the sample")
    if not set(out["label"].unique().to_list()) <= set(CLASSES):
        problems.append("unknown labels")
    pairs: dict[str, int] = {}
    for a, b in zip(pred[hit], new[hit], strict=True):
        key = f"{CLASSES[a]} -> {CLASSES[b]}"
        pairs[key] = pairs.get(key, 0) + 1
    rep = {
        "sig": sig,
        "base": Path(base_csv).name,
        "out": Path(out_csv).name,
        "changed": pairs,
        "valid": not problems,
        "problems": problems,
        "sha256": hashlib.sha256(Path(out_csv).read_bytes()).hexdigest(),
    }
    print(json.dumps(rep, indent=1))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["survey", "rule", "octlike", "apply"])
    ap.add_argument("--sig", default="")
    ap.add_argument("--base", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.cmd == "apply":
        rep = cmd_apply(a.sig, a.base, a.out)
    elif a.cmd == "survey":
        rep = cmd_survey()
    elif a.cmd == "rule":
        rep = cmd_rule(a.sig)
    else:
        rep = cmd_octlike(a.sig)
    out = paths.out_dir() / "rescue"
    out.mkdir(parents=True, exist_ok=True)
    tag = a.sig.replace(":", "-").replace("&", "+").replace("|", "_or_")
    (out / f"{a.cmd}{'-' + tag if tag else ''}.json").write_text(
        json.dumps(rep, indent=1, default=str)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
