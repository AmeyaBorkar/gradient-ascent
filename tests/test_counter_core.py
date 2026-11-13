import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from counter_core import choose, class_moves, paired, passes  # noqa: E402


def test_paired():
    s = paired(np.array([0.01, 0.03, 0.02, 0.04]))
    assert s["n"] == 4
    assert np.isclose(s["mean"], 0.025)
    assert np.isclose(s["sd"], np.std([0.01, 0.03, 0.02, 0.04], ddof=1))
    assert np.isclose(s["se"], s["sd"] / 2)
    assert np.isclose(s["z"], s["mean"] / s["se"])
    assert paired(np.zeros(3))["z"] == 0.0
    assert paired(np.full(3, 0.5))["z"] == float("inf")
    assert paired(np.full(3, -0.5))["z"] == float("-inf")


def _s(mean):
    return {"mean": mean}


def test_choose_plain_and_maximin():
    stats = {
        "a": {"agreement": _s(0.04), "stress": _s(0.01), "guard": _s(0.0)},
        "b": {"agreement": _s(0.03), "stress": _s(0.025), "guard": _s(0.01)},
        "c": {"agreement": _s(0.05), "stress": _s(0.05), "guard": _s(-0.01)},
    }
    assert choose(stats, keys=("agreement",)) == "c"
    assert choose(stats, keys=("agreement",), guards=("guard",)) == "a"
    assert choose(stats, keys=("agreement", "stress"), guards=("guard",)) == "b"
    # ties go to the first setting listed
    tie = {"x": {"agreement": _s(0.02)}, "y": {"agreement": _s(0.02)}}
    assert choose(tie) == "x"
    with pytest.raises(ValueError):
        choose({"a": {"agreement": _s(0.1), "guard": _s(-1.0)}}, guards=("guard",))


def test_passes():
    assert passes({"mean": 0.03, "z": 2.5})
    assert not passes({"mean": 0.019, "z": 5.0})
    assert not passes({"mean": 0.03, "z": 2.0})


def test_class_moves():
    before = np.array([0, 1, 2, 2, 7, 7])
    after = np.array([0, 2, 2, 3, 8, 8])
    assert class_moves(before, after) == {"1->2": 1, "2->3": 1, "7->8": 2}
    assert class_moves(before, before) == {}


def test_class_moves_named():
    names = ["legit", "mule", "bot"]
    assert class_moves(np.array([0, 1, 1]), np.array([1, 2, 1]), names) == {
        "legit->mule": 1,
        "mule->bot": 1,
    }
