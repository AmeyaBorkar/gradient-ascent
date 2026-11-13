"""How the TabM and tree fraud masses are combined, and where TabM's adds are right.

The base is the fraud-only TabM blend (``blend-tabmF-bq``): the legit column is a fixed average
(TabM 1/2, each of the five trees 1/10); the 11 fraud types are the five-way tree blend's
distribution rescaled to that fraud mass; then the fixed point (0.875), the anchored template rule
and ``blank_merchant_qr``. Every challenger here changes only the fraud mass (or, for ``types``,
one class's share of it), so the rest of the pipeline is shared.

    python -m ringtrace.lens combine          # Q1: mass combiners and the nested stacker
    python -m ringtrace.lens seeds RUN [RUN]  # Q2: TabM seed sets, October adds and churn
    python -m ringtrace.lens adds             # Q3: the adds split by class, template, family
    python -m ringtrace.lens types            # Q4: TabM's type share for single classes, nested
    python -m ringtrace.lens write NAME       # October file of a named challenger

Every scored line: per-month change on given / cleaned / gen-only and the A-clean proxy, the paired
event bootstrap (95 %) of the mean change, and the nine October-like cells. October is used through
aggregates only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from ringtrace.classes import CLASSES, N_CLASSES

TREES = ["blend-champion-m1", "lgbm", "catboost", "clean-event", "clean-l2"]
CELLS = {
    "blend-champion-m1": "champion",
    "lgbm": "lgbm",
    "catboost": "catboost",
    "clean-event": "clean-event",
    "clean-l2": "clean-l2",
}
TABM = "tabm"
TABM_SHARE = 0.5
RULES = ["anchored_templates", "blank_merchant_qr"]
DECODER = {"fraud_scale": 0.875}
MONTHS = (7, 8, 9)
EPS = 1e-7
HOWS = ("prob", "logit", "geo", "noisy_or", "max")


# ------------------------------------------------------------------------------ pure parts


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def fraud_mass(proba: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.asarray(proba, dtype=np.float64)[:, 0], 0.0, 1.0)


def tree_types(trees: list[np.ndarray]) -> np.ndarray:
    """The five-way tree blend's fraud-type distribution (rows sum to 1; uniform if empty)."""
    t = np.mean([a[:, 1:] for a in trees], axis=0)
    tot = t.sum(axis=1, keepdims=True)
    return np.where(tot > 0, t / np.where(tot > 0, tot, 1.0), 1.0 / t.shape[1])


def combine(t: np.ndarray, trees: np.ndarray, how: str, share: float = TABM_SHARE) -> np.ndarray:
    """Fraud mass from TabM's ``t`` (n,) and the trees' masses ``trees`` (n, m).

    ``prob``: share * t + (1 - share) * mean(trees) (the base). ``logit``: the same weights on
    logits. ``geo``: the same weights on log masses (unnormalised geometric mean). ``noisy_or``:
    1 - (1 - t)(1 - mean(trees)). ``max``: max(t, mean(trees))."""
    trees = np.asarray(trees, dtype=np.float64)
    b = trees.mean(axis=1)
    w_tree = (1.0 - share) / trees.shape[1]
    if how == "prob":
        return share * t + (1 - share) * b
    if how == "logit":
        return sigmoid(share * logit(t) + w_tree * logit(trees).sum(axis=1))
    if how == "geo":
        lt = np.log(np.clip(t, EPS, 1.0))
        return np.exp(share * lt + w_tree * np.log(np.clip(trees, EPS, 1.0)).sum(axis=1))
    if how == "noisy_or":
        return 1.0 - (1.0 - t) * (1.0 - b)
    if how == "max":
        return np.maximum(t, b)
    raise ValueError(f"unknown combiner {how!r}")


def assemble(mass: np.ndarray, types: np.ndarray) -> np.ndarray:
    """12-class rows: legit = 1 - mass, fraud types = mass * types."""
    out = np.empty((len(mass), types.shape[1] + 1))
    out[:, 0] = 1.0 - mass
    out[:, 1:] = mass[:, None] * types
    return out


def stack_features(t: np.ndarray, trees: np.ndarray, spread: bool = True) -> np.ndarray:
    lt = logit(np.asarray(trees))
    cols = [logit(t)[:, None], lt]
    if spread:
        cols.append(lt.std(axis=1, keepdims=True))
    return np.hstack(cols)


def fit_stacker(x: np.ndarray, y: np.ndarray, c: float = 0.01) -> tuple[np.ndarray, float]:
    """L2 logistic regression (strength 1 / c), deterministic lbfgs."""
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=c, max_iter=500, tol=1e-8)
    lr.fit(x, y.astype(np.int64))
    return lr.coef_.ravel().astype(np.float64), float(lr.intercept_[0])


def stack_mass(x: np.ndarray, coef: np.ndarray, icpt: float) -> np.ndarray:
    return sigmoid(x @ coef + icpt)


def with_class_share(types: np.ndarray, other_types: np.ndarray, classes: list[int]) -> np.ndarray:
    """``types`` with the share of each fraud class in ``classes`` (1..11) taken from
    ``other_types``; the remaining classes keep their relative shares."""
    if not classes:
        return types
    idx = [k - 1 for k in classes]
    out = types.copy()
    take = other_types[:, idx]
    rest = np.ones(types.shape[1], dtype=bool)
    rest[idx] = False
    rest_tot = types[:, rest].sum(axis=1, keepdims=True)
    left = np.clip(1.0 - take.sum(axis=1, keepdims=True), 0.0, 1.0)
    scaled = np.where(rest_tot > 0, types[:, rest] / np.where(rest_tot > 0, rest_tot, 1.0), 0.0)
    out[:, rest] = scaled * left
    out[:, idx] = take
    s = out.sum(axis=1, keepdims=True)
    return out / np.where(s > 0, s, 1.0)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    return len(a & b) / max(len(a | b), 1)


def bucket_table(keys: np.ndarray, right: np.ndarray, period: np.ndarray) -> list[dict]:
    """Per key: adds, right adds, per month (adds, right)."""
    out = []
    for k in np.unique(keys):
        sel = keys == k
        row = {"key": k.item() if hasattr(k, "item") else k, "n": int(sel.sum())}
        row["right"] = int(right[sel].sum())
        row["months"] = {
            int(m): [int((sel & (period == m)).sum()), int((sel & right & (period == m)).sum())]
            for m in MONTHS
        }
        out.append(row)
    return sorted(out, key=lambda r: -r["n"])


def verdict(lomo: dict, octlike: dict | None) -> dict:
    months_ok = all(min(lomo[t]["change"]) >= 0 for t in ("given", "cleaned", "gen", "aclean"))
    boot_ok = all(lomo[t]["lo"] > 0 for t in ("given", "cleaned"))
    oct_ok = None
    if octlike is not None:
        oct_ok = bool(
            octlike["complete"]
            and all(v > 0 for t in ("given", "cleaned") for v in octlike["per_seed"][t].values())
            and all(octlike["lower_bound"][t] > 0 for t in ("given", "cleaned"))
        )
    return {
        "months_non_negative": months_ok,
        "bootstrap_above_zero": boot_ok,
        "octlike_positive": oct_ok,
        "passes": bool(months_ok and boot_ok and oct_ok),
    }


# ------------------------------------------------------------------------------ the lab


def _out() -> Path:
    from ringtrace import paths

    d = paths.out_dir() / "lens"
    d.mkdir(parents=True, exist_ok=True)
    return d


class Lab:
    """Arrays loaded once; challengers are functions of (TabM proba, tree probas, rows-key)."""

    def __init__(self, tabm_runs=(TABM,), cells_dir: Path | None = None, log=print):
        from ringtrace import final, paths
        from ringtrace.aclean import hidden_counts

        self.log = log
        self.runs = paths.out_dir() / "runs"
        self.cells_dir = cells_dir or Path(
            os.environ.get("RINGTRACE_LENS_CELLS", str(paths.out_dir() / "lens-cells"))
        )
        self.bench = final.Bench.from_store(self.runs, None, log=log)
        b = self.bench
        self.led, self.tr, self.te, self.period = b.led, b.tr, b.te, b.period
        self.ptr = self.period[self.tr]
        self.targets = b.targets
        self.hidden = hidden_counts(self.targets["cleaned"], self.targets["gen"], self.ptr)
        self.trees = [self._load(r) for r in TREES]
        self.tabm = self.tabm_mean(tabm_runs)
        self.types = tree_types(self.trees)
        self.tree_mass = np.stack([fraud_mass(a) for a in self.trees], axis=1)
        self._boot = None
        self.base_pred = self.decode(self.mass_of("prob"))
        self.base_lomo = self.month_scores(self.base_pred)

    def _load(self, run: str) -> np.ndarray:
        from ringtrace.blend import load_member

        oof, test = load_member(self.runs / run, len(self.tr), len(self.te))
        full = np.zeros((self.led.height, N_CLASSES))
        full[self.tr], full[self.te] = oof, test
        return full

    def tabm_mean(self, runs) -> np.ndarray:
        return np.mean([self._load(r) for r in runs], axis=0)

    def mass_of(self, how: str, tabm: np.ndarray | None = None) -> np.ndarray:
        t = fraud_mass(self.tabm if tabm is None else tabm)
        return combine(t, self.tree_mass, how)

    def decode(self, mass: np.ndarray, types: np.ndarray | None = None) -> np.ndarray:
        from ringtrace import rules as rulebook
        from ringtrace.decode import fixed_point

        proba = assemble(mass, self.types if types is None else types)
        pred = np.zeros(self.led.height, dtype=np.int64)
        for k in (*MONTHS, "october"):
            rows = self.te if k == "october" else self.tr[self.ptr == k]
            pred[rows] = fixed_point(proba[rows], **DECODER)
        return rulebook.apply(RULES, pred, proba, self.led, log=lambda *_: None)

    def month_scores(self, pred: np.ndarray) -> dict:
        from ringtrace.aclean import per_month

        return per_month(pred[self.tr], self.targets, self.ptr, self.hidden)

    def compare(self, pred: np.ndarray, base: np.ndarray | None = None) -> dict:
        from ringtrace.pairboot import PairedBoot

        if base is None:
            base, sa = self.base_pred, self.base_lomo
            if self._boot is None:
                self._boot = PairedBoot(
                    self.bench.groups, self.targets, base[self.tr], self.ptr, draws=2000
                )
            boot = self._boot
        else:
            sa = self.month_scores(base)
            boot = PairedBoot(self.bench.groups, self.targets, base[self.tr], self.ptr, draws=2000)
        sb = self.month_scores(pred)
        bs = boot.compare(pred[self.tr])
        out = {}
        for t in sb:
            ch = [y1 - y0 for y0, y1 in zip(sa[t], sb[t], strict=True)]
            out[t] = {"score": sb[t], "change": ch, **bs.get(t, {"lo": None, "hi": None})}
        a, b = base, pred
        out["rows"] = {
            "train": int((a[self.tr] != b[self.tr]).sum()),
            "october": int((a[self.te] != b[self.te]).sum()),
        }
        from ringtrace.final import pair_counts

        out["october_pairs"] = pair_counts(a[self.te], b[self.te])
        return out

    # October-like ------------------------------------------------------------------------

    def octlike(self, makers: dict, seeds=(13, 14, 15), sets=(0, 1, 2), draws=1000) -> dict:
        """``makers[name](tabm_cell, tree_cells) -> (mass, types)``; the base is ``prob``."""
        from ringtrace import octlike as ol
        from ringtrace import store
        from ringtrace.pairboot import PairedBoot
        from ringtrace.scoring import macro_f1

        cells: dict = {n: {} for n in makers}
        try:
            for seed in seeds:
                frame = ol.Frame(seed)
                units = frame.unit[frame.hold]
                truth = {t: y[frame.hold] for t, y in frame.targets.items()}
                for r in sets:

                    def load(name, seed=seed, r=r, frame=frame):
                        f = np.load(self.cells_dir / f"{name}_s{seed}_r{r}.npz")
                        if not np.array_equal(f["rows"], frame.stacked[frame.hold]):
                            raise ValueError(f"{name} s{seed} r{r}: rows differ from the frame")
                        return f["prob"].astype(np.float64)

                    tc = load("tabm")
                    trs = [load(CELLS[x]) for x in TREES]
                    tm = np.stack([fraud_mass(a) for a in trs], axis=1)
                    types = tree_types(trs)
                    base_p = assemble(combine(fraud_mass(tc), tm, "prob"), types)
                    ya = ol.decode(frame, base_p, DECODER, RULES)
                    boot = PairedBoot(units, truth, ya, draws=draws, seed=seed * 100 + r)
                    for name, mk in makers.items():
                        mass, ty = mk(tc, trs, tm, types)
                        yb = ol.decode(frame, assemble(mass, ty), DECODER, RULES)
                        bs = boot.compare(yb)
                        cell = {"rows": int((ya != yb).sum())}
                        for t, yt in truth.items():
                            cell[t] = {"base": macro_f1(yt, ya), "chal": macro_f1(yt, yb), **bs[t]}
                        cells[name][(seed, r)] = cell
                    self.log(f"[lens] octlike s{seed} r{r} done")
        finally:
            store.use_frame(None)
        out = {}
        for name, c in cells.items():
            tg = ("given", "cleaned", "gen")
            out[name] = {
                "complete": len(c) == len(seeds) * len(sets),
                "per_seed": {
                    t: {
                        s: float(np.mean([v[t]["delta"] for (s2, _), v in c.items() if s2 == s]))
                        for s in seeds
                    }
                    for t in tg
                },
                "lower_bound": {t: float(np.mean([v[t]["lo"] for v in c.values()])) for t in tg},
                "rows": int(sum(v["rows"] for v in c.values())),
            }
        return out


# ------------------------------------------------------------------------------ reports


def fmt(name: str, cmp: dict, oct_: dict | None = None) -> str:
    def m(t):
        c = cmp[t]["change"]
        s = " / ".join(f"{x:+.5f}" for x in c)
        if cmp[t].get("lo") is not None:
            s += f" [{cmp[t]['lo']:+.5f}, {cmp[t]['hi']:+.5f}]"
        return s

    parts = [f"{name}: rows {cmp['rows']['train']}/{cmp['rows']['october']}"]
    for t in ("given", "cleaned", "gen", "aclean"):
        parts.append(f"  {t:<8} {m(t)}")
    if oct_:
        for t in ("given", "cleaned", "gen"):
            ps = " / ".join(f"{v:+.5f}" for v in oct_["per_seed"][t].values())
            parts.append(f"  oct {t:<8} {ps} (lb {oct_['lower_bound'][t]:+.5f})")
        parts.append(f"  verdict {verdict(cmp, oct_)}")
    return "\n".join(parts)


def adds_split(lab: Lab, base: np.ndarray, pred: np.ndarray) -> dict:
    """Train rows legit in ``base`` and fraud in ``pred``: precision by called class, template,
    evidence family; pocket b = given legit, generator fraud."""
    from ringtrace.evidence import anchors
    from ringtrace.squeeze import RULE_COLUMNS

    a, b = base[lab.tr], pred[lab.tr]
    add = (a == 0) & (b > 0)
    y = lab.targets
    right_c = add & (y["cleaned"] > 0)
    right_type = add & (y["cleaned"] == b)
    pocket = add & (y["given"] == 0) & (y["gen"] > 0)
    tmpl = anchors.template_of(lab.led)[lab.tr]
    cols = anchors.family_columns(lab.led, RULE_COLUMNS)
    fam = np.full(len(lab.tr), "none", dtype=object)
    order = ["bot", "drn", "mf", "col", "rng", "rep"]
    for p in order[::-1]:
        v = cols[f"{p}_shape"].fill_null(0).to_numpy()[lab.tr] > 0
        fam[v] = p
    mark = np.zeros(len(lab.tr), dtype=bool)
    for c in ("mrk_p2p_category", "mrk_qr_p2p_nocat", "mrk_bal102", "mrk_step025", "mrk_step035"):
        mark |= cols[c].fill_null(0).to_numpy()[lab.tr] > 0
    fam[(fam == "none") & mark] = "mark"
    ctr = cols["ctr_block"].fill_null(0).to_numpy()[lab.tr]
    i = np.flatnonzero(add)
    p = lab.ptr[i]
    rc = right_c[i]
    return {
        "adds": int(add.sum()),
        "right_cleaned": int(right_c.sum()),
        "right_given": int((add & (y["given"] > 0)).sum()),
        "right_type_cleaned": int(right_type.sum()),
        "pocket_b": int(pocket.sum()),
        "by_called": bucket_table(np.array([CLASSES[k] for k in b[i]]), rc, p),
        "by_template": bucket_table(tmpl[i].astype(np.int64), rc, p),
        "by_family": bucket_table(fam[i].astype(str), rc, p),
        "by_counter_block": bucket_table(ctr[i].astype(np.int64), rc, p),
    }


def _jdump(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, np.integer | np.floating):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(json.dumps(obj, indent=1, default=default))


def cmd_combine(args) -> dict:
    t0 = time.time()
    lab = Lab()
    tree_only = lab.decode(lab.tree_mass.mean(axis=1))
    res: dict = {"base_scores": lab.base_lomo}
    preds = {how: lab.decode(lab.mass_of(how)) for how in HOWS if how != "prob"}
    # nested stacker: fit on two months, score the third; October fit on all three
    t = fraud_mass(lab.tabm)
    x = stack_features(t, lab.tree_mass)
    yfit = lab.targets[args.target][:] > 0
    ytr_x = x[lab.tr]
    for c in args.c:
        mass = lab.mass_of("prob").copy()
        coefs = {}
        for held in (*MONTHS, "october"):
            fit = lab.ptr != held if held != "october" else np.ones(len(lab.tr), bool)
            coef, icpt = fit_stacker(ytr_x[fit], yfit[fit], c)
            rows = lab.te if held == "october" else lab.tr[lab.ptr == held]
            mass[rows] = stack_mass(x[rows], coef, icpt)
            coefs[str(held)] = [*coef.round(4).tolist(), round(icpt, 4)]
        preds[f"stack-c{c}"] = lab.decode(mass)
        res[f"stack-c{c}-coef"] = coefs
    for name, pred in preds.items():
        cmp = lab.compare(pred)
        cmp["adds_vs_trees"] = {
            k: v for k, v in adds_split(lab, tree_only, pred).items() if not k.startswith("by_")
        }
        res[name] = cmp
    res["base_adds_vs_trees"] = {
        k: v for k, v in adds_split(lab, tree_only, lab.base_pred).items() if not k.startswith("by")
    }

    # October-like: the stacker uses the September fold's fit (July + August)
    makers = {}
    for how in HOWS:
        if how != "prob":
            makers[how] = lambda tc, trs, tm, ty, how=how: (combine(fraud_mass(tc), tm, how), ty)
    for c in args.c:
        fit = lab.ptr != 9
        coef, icpt = fit_stacker(ytr_x[fit], yfit[fit], c)
        makers[f"stack-c{c}"] = lambda tc, trs, tm, ty, coef=coef, icpt=icpt: (
            stack_mass(stack_features(fraud_mass(tc), tm), coef, icpt),
            ty,
        )
    if not args.no_octlike:
        oc = lab.octlike(makers)
        for n in oc:
            res[n]["octlike"] = oc[n]
    lines = []
    for name in preds:
        lines.append(fmt(name, res[name], res[name].get("octlike")))
        lines.append(f"  adds vs trees-only: {res[name]['adds_vs_trees']}")
    lines.append(f"base adds vs trees-only: {res['base_adds_vs_trees']}")
    print("\n".join(lines))
    res["wall_s"] = time.time() - t0
    _jdump(res, _out() / "combine.json")
    return res


def cmd_seeds(args) -> dict:
    """TabM as the mean of k seed runs; October adds against the trees-only calls, churn between
    seed sets, and the leave-one-month-out change against the 3-seed base."""
    t0 = time.time()
    lab = Lab()
    tree_only = lab.decode(lab.tree_mass.mean(axis=1))
    te = lab.te
    res: dict = {"runs": args.runs}
    adds = {}
    for k in range(1, len(args.runs) + 1):
        runs = args.runs[:k]
        tabm = lab.tabm_mean(runs)
        pred = lab.decode(lab.mass_of("prob", tabm))
        add = np.flatnonzero((tree_only[te] == 0) & (pred[te] > 0))
        rem = int(((tree_only[te] > 0) & (pred[te] == 0)).sum())
        adds[f"{3 * k}"] = add
        cmp = lab.compare(pred)
        tr_add = (tree_only[lab.tr] == 0) & (pred[lab.tr] > 0)
        res[f"seeds{3 * k}"] = {
            "runs": runs,
            "october_adds": len(add),
            "october_removes": rem,
            "october_changed_vs_trees": int((tree_only[te] != pred[te]).sum()),
            "train_adds": int(tr_add.sum()),
            "train_adds_right_cleaned": int((tr_add & (lab.targets["cleaned"] > 0)).sum()),
            "vs_base": cmp,
        }
        print(fmt(f"seeds{3 * k}", cmp))
    singles = {}
    for r in args.runs:
        pred = lab.decode(lab.mass_of("prob", lab.tabm_mean([r])))
        singles[r] = np.flatnonzero((tree_only[te] == 0) & (pred[te] > 0))
    names = list(singles)
    res["single_set_adds"] = {r: len(v) for r, v in singles.items()}
    res["single_set_jaccard"] = {
        f"{a}|{b}": jaccard(singles[a], singles[b])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }
    ks = list(adds)
    res["nested_jaccard"] = {
        f"{a}|{b}": jaccard(adds[a], adds[b]) for a, b in zip(ks, ks[1:], strict=False)
    }
    res["wall_s"] = time.time() - t0
    print(json.dumps({k: v for k, v in res.items() if "jaccard" in k or k.endswith("adds")}))
    for k in ks:
        r = res[f"seeds{k}"]
        print(
            f"seeds{k}: Oct adds {r['october_adds']} removes {r['october_removes']}; "
            f"train adds {r['train_adds']} right {r['train_adds_right_cleaned']}"
        )
    _jdump(res, _out() / "seeds.json")
    return res


def _gated_mass(lab: Lab, allowed: np.ndarray) -> np.ndarray:
    """The base mass on ``allowed`` rows, the trees' mean mass elsewhere."""
    return np.where(allowed, lab.mass_of("prob"), lab.tree_mass.mean(axis=1))


def cmd_adds(args) -> dict:
    """Q3: where the fraud-only adds are right; then a nested, label-free gate by evidence family:
    TabM's mass is used only in the families whose adds were right in both fit months."""
    from ringtrace.evidence import anchors
    from ringtrace.squeeze import RULE_COLUMNS

    t0 = time.time()
    lab = Lab()
    tree_only = lab.decode(lab.tree_mass.mean(axis=1))
    split = adds_split(lab, tree_only, lab.base_pred)
    res = {"split": split}
    for key in ("by_called", "by_template", "by_family", "by_counter_block"):
        print(key)
        for r in split[key]:
            print(f"  {r['key']!s:<20} n {r['n']:>3} right(cleaned) {r['right']:>3} {r['months']}")
    print({k: v for k, v in split.items() if not k.startswith("by")})
    # family of every ledger row (label-free)
    cols = anchors.family_columns(lab.led, RULE_COLUMNS)
    fam = np.full(lab.led.height, "none", dtype=object)
    for p in ["bot", "drn", "mf", "col", "rng", "rep"][::-1]:
        fam[cols[f"{p}_shape"].fill_null(0).to_numpy() > 0] = p
    fams = np.unique(fam)
    a, b = tree_only, lab.base_pred
    add = np.zeros(lab.led.height, dtype=bool)
    add[lab.tr] = (a[lab.tr] == 0) & (b[lab.tr] > 0)
    rightc = np.zeros(lab.led.height, dtype=bool)
    rightc[lab.tr] = lab.targets["cleaned"] > 0
    per = lab.period
    allowed = np.zeros(lab.led.height, dtype=bool)
    chosen = {}
    for held in (*MONTHS, "october"):
        fit = [m for m in MONTHS if m != held]
        keep = []
        for f in fams:
            ok = True
            for m in fit:
                s = add & (fam == f) & (per == m)
                if s.sum() and rightc[s].mean() < args.min_precision:
                    ok = False
            keep.append(f) if ok else None
        rows = lab.bench.is_test if held == "october" else (~lab.bench.is_test & (per == held))
        allowed[rows & np.isin(fam, keep)] = True
        chosen[str(held)] = [str(f) for f in keep]
    print("families kept per held month:", chosen)
    pred = lab.decode(_gated_mass(lab, allowed))
    cmp = lab.compare(pred)
    res["family_gate"] = {"chosen": chosen, "cmp": cmp}

    print(fmt("family-gate", cmp))
    res["wall_s"] = time.time() - t0
    _jdump(res, _out() / "adds.json")
    return res


def cmd_types(args) -> dict:
    """Q4: for each fraud class, TabM's share of that class in place of the trees' share; the
    classes are chosen per held month on the other two (non-negative on given and cleaned in both
    fit months, and positive in at least one)."""
    from ringtrace.scoring import macro_f1

    t0 = time.time()
    lab = Lab()
    tabm_types = tree_types([lab.tabm])
    mass = lab.mass_of("prob")
    per_class = {}
    preds = {}
    for k in range(1, N_CLASSES):
        ty = with_class_share(lab.types, tabm_types, [k])
        pred = lab.decode(mass, ty)
        preds[k] = pred
        cmp = lab.compare(pred)
        per_class[CLASSES[k]] = cmp
        print(fmt(f"tabm-type {CLASSES[k]}", cmp))

    def month_gain(pred, m, t):
        s = lab.ptr == m
        y = lab.targets[t][s]
        return macro_f1(y, pred[lab.tr][s]) - macro_f1(y, lab.base_pred[lab.tr][s])

    chosen = {}
    for held in (*MONTHS, "october"):
        fit = [m for m in MONTHS if m != held]
        keep = []
        for k, pred in preds.items():
            g = [month_gain(pred, m, t) for m in fit for t in ("given", "cleaned")]
            if min(g) >= 0 and max(g) > 0:
                keep.append(k)
        chosen[str(held)] = keep
    print("classes kept per held month:", {h: [CLASSES[k] for k in v] for h, v in chosen.items()})
    final_types = lab.types.copy()
    for held in (*MONTHS, "october"):
        rows = lab.te if held == "october" else lab.tr[lab.ptr == held]
        final_types[rows] = with_class_share(lab.types[rows], tabm_types[rows], chosen[str(held)])
    pred = lab.decode(mass, final_types)
    cmp = lab.compare(pred)
    print(fmt("nested-types", cmp))
    res = {"per_class": per_class, "chosen": chosen, "nested": cmp, "wall_s": time.time() - t0}
    _jdump(res, _out() / "types.json")
    return res


def cmd_write(args) -> dict:
    """October file of a challenger, on top of the base's calls."""
    from ringtrace import paths
    from ringtrace.blend import write_submission

    lab = Lab()
    if args.name in HOWS:
        pred = lab.decode(lab.mass_of(args.name))
    else:
        raise SystemExit(f"unknown challenger {args.name}")
    ids = lab.led["txn_id"].to_numpy()[lab.te]
    path = Path(args.out) / f"lens-{args.name}.csv"
    info = write_submission(path, ids, pred[lab.te], paths.data_dir() / "sample_submission.csv")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(path.name, sha, info)
    return {"sha256": sha}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ringtrace.lens")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("combine")
    c.add_argument("--c", type=float, nargs="+", default=[0.01, 1.0])
    c.add_argument("--target", default="cleaned", choices=["given", "cleaned"])
    c.add_argument("--no-octlike", action="store_true")
    s = sub.add_parser("seeds")
    s.add_argument("runs", nargs="+")
    a = sub.add_parser("adds")
    a.add_argument("--min-precision", type=float, default=0.5)
    sub.add_parser("types")
    w = sub.add_parser("write")
    w.add_argument("name")
    w.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {
        "combine": cmd_combine,
        "seeds": cmd_seeds,
        "adds": cmd_adds,
        "types": cmd_types,
        "write": cmd_write,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
