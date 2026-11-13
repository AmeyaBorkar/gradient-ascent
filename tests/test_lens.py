import numpy as np
import pytest

from ringtrace import lens


def _rng():
    return np.random.default_rng(0)


def test_combine_prob_is_the_fixed_average():
    r = _rng()
    t = r.random(50)
    trees = r.random((50, 5))
    got = lens.combine(t, trees, "prob")
    assert np.allclose(got, 0.5 * t + 0.1 * trees.sum(axis=1))


@pytest.mark.parametrize("how", lens.HOWS)
def test_combine_bounds_and_agreement(how):
    r = _rng()
    t = r.random(40)
    trees = r.random((40, 5))
    m = lens.combine(t, trees, how)
    assert m.shape == (40,) and np.all(m >= 0) and np.all(m <= 1)
    same = np.full((3, 5), 0.3)
    m2 = lens.combine(np.full(3, 0.3), same, how)
    if how == "noisy_or":
        assert np.allclose(m2, 1 - 0.7 * 0.7)
    else:
        assert np.allclose(m2, 0.3)


def test_combine_order_noisy_or_and_max_dominate():
    r = _rng()
    t = r.random(100)
    trees = r.random((100, 5))
    b = trees.mean(axis=1)
    assert np.all(lens.combine(t, trees, "max") >= lens.combine(t, trees, "prob") - 1e-12)
    assert np.all(lens.combine(t, trees, "noisy_or") >= np.maximum(t, b) - 1e-12)


def test_combine_unknown():
    with pytest.raises(ValueError):
        lens.combine(np.zeros(2), np.zeros((2, 5)), "median")


def test_assemble_and_tree_types():
    r = _rng()
    trees = [r.dirichlet(np.ones(12), 30) for _ in range(5)]
    trees[0][0] = 0
    trees[0][0, 0] = 1
    ty = lens.tree_types(trees)
    assert np.allclose(ty.sum(axis=1), 1)
    p = lens.assemble(np.full(30, 0.2), ty)
    assert np.allclose(p.sum(axis=1), 1) and np.allclose(p[:, 0], 0.8)
    empty = lens.tree_types([np.eye(12)[[0, 0]]])
    assert np.allclose(empty, 1 / 11)


def test_stacker_is_deterministic_and_learns():
    r = _rng()
    n = 2000
    y = r.random(n) < 0.3
    t = np.where(y, r.uniform(0.5, 1, n), r.uniform(0, 0.5, n))
    trees = np.clip(t[:, None] + r.normal(0, 0.2, (n, 5)), 0, 1)
    x = lens.stack_features(t, trees)
    assert x.shape == (n, 7)
    c1 = lens.fit_stacker(x, y, 0.01)
    c2 = lens.fit_stacker(x, y, 0.01)
    assert np.array_equal(c1[0], c2[0]) and c1[1] == c2[1]
    m = lens.stack_mass(x, *c1)
    assert m[y].mean() > m[~y].mean()


def test_with_class_share():
    r = _rng()
    a = r.dirichlet(np.ones(11), 20)
    b = r.dirichlet(np.ones(11), 20)
    out = lens.with_class_share(a, b, [3])
    assert np.allclose(out.sum(axis=1), 1)
    assert np.allclose(out[:, 2], b[:, 2])
    rest = [i for i in range(11) if i != 2]
    ratio = out[:, rest] / a[:, rest]
    assert np.allclose(ratio, ratio[:, :1])
    assert lens.with_class_share(a, b, []) is a


def test_jaccard_and_buckets():
    assert lens.jaccard(np.array([1, 2]), np.array([2, 3])) == pytest.approx(1 / 3)
    assert lens.jaccard(np.array([]), np.array([])) == 0
    keys = np.array(["a", "a", "b"])
    right = np.array([True, False, True])
    per = np.array([7, 8, 9])
    tab = lens.bucket_table(keys, right, per)
    assert tab[0]["key"] == "a" and tab[0]["n"] == 2 and tab[0]["right"] == 1
    assert tab[0]["months"][7] == [1, 1] and tab[0]["months"][8] == [1, 0]


def test_verdict():
    good = {t: {"change": [0.0, 1e-4, 2e-4], "lo": 1e-5} for t in ("given", "cleaned")}
    good.update({t: {"change": [0.0, 0.0, 1e-5]} for t in ("gen", "aclean")})
    oct_ = {
        "complete": True,
        "per_seed": {t: {13: 1e-4, 14: 1e-4, 15: 2e-4} for t in ("given", "cleaned")},
        "lower_bound": {"given": 1e-5, "cleaned": 1e-5},
    }
    assert lens.verdict(good, oct_)["passes"]
    bad = {**good, "gen": {"change": [-1e-6, 0, 0]}}
    assert not lens.verdict(bad, oct_)["passes"]
    assert not lens.verdict(good, None)["passes"]
