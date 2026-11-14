"""The final recipe is exactly its two gated components on the best recipe, and nothing else."""

from pathlib import Path

import pytest

from ringtrace import final

ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str) -> dict:
    return final.load_recipe(ROOT / rel)


def test_final2_is_bait_rules_on_the_density_members():
    best = _load("recipes/final-half.json")
    bait = _load("recipes/bait-combo.json")
    half3 = _load("recipes/density/final-half-chalf3b.json")
    f2 = _load("recipes/final2.json")
    # the members (and their weights) are the density blend's, the rules are the bait recipe's
    assert f2["members"] == half3["members"]
    assert f2["weights"] == half3["weights"] == best["weights"]
    assert f2["rules"] == bait["rules"]
    assert f2["decoder"] == best["decoder"] == bait["decoder"] == half3["decoder"]
    assert f2["base_rate_rules"] == best["base_rate_rules"]
    # the bait recipe adds rules only; the density blend swaps three member runs only
    assert bait["members"] == best["members"]
    changed = {a["name"] for a, b in zip(best["members"], half3["members"], strict=True) if a != b}
    assert changed == {"champion", "clean-event", "clean-l2"}
    assert final.recipe_key(f2) not in {final.recipe_key(r) for r in (best, bait, half3)}


@pytest.mark.parametrize(
    "rel",
    [
        "recipes/final2/cpu3-bait.json",
        "recipes/final2/r1-bait.json",
        "recipes/final2/bait-half3t.json",
        "recipes/final2/bait-half3-t4060.json",
    ],
)
def test_gate_recipes_carry_the_bait_rules(rel):
    bait = _load("recipes/bait-combo.json")
    r = _load(rel)
    assert r["rules"] == bait["rules"]
    assert r["base_rate_rules"] == bait["base_rate_rules"]


def test_tabm_follow_up_differs_from_its_baseline_in_tabm_only():
    a = _load("recipes/final2/bait-half3-t4060.json")
    b = _load("recipes/final2/bait-half3t.json")
    f2 = _load("recipes/final2.json")
    diff = {x["name"] for x, y in zip(a["members"], b["members"], strict=True) if x != y}
    assert diff == {"tabm"}
    trees = [m for m in f2["members"] if m["name"] != "tabm"]
    assert [m for m in b["members"] if m["name"] != "tabm"] == trees
