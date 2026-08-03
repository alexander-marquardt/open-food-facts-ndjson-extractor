"""Unit tests for the run-wide canonical parent map.

``build_canonical_parent_map`` collapses the Open Food Facts category DAG to a
spanning forest **once per run**, so that a category's address never depends on
which product you are looking at. These use small synthetic taxonomies, each
built to isolate one rule of the selection; the property tests that this exists
to serve run on real data in ``test_category_addressing.py``.

Run with ``pytest tests/`` or directly: ``python tests/test_canonical_parents.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from off_demo_extract.taxonomy import (  # noqa: E402
    AddressAudit,
    build_canonical_parent_map,
    build_category_path,
    canonical_ancestry,
    category_chain,
    category_path_entries,
)

EXCLUDE = {"en:null", "en:unknown", "en:undefined"}


def _n(name: str, parents: list[str]) -> dict:
    return {"name": {"en": name}, "parents": parents}


# A miniature DAG modelled on the real tea structure: en:teas has two parents.
TAXONOMY = {
    "en:beverages": _n("Beverages", []),
    "en:plant-based-foods-and-beverages": _n("Plant-based foods and beverages", []),
    "en:hot-beverages": _n("Hot beverages", ["en:beverages"]),
    "en:plant-based-beverages": _n(
        "Plant-based beverages",
        ["en:beverages", "en:plant-based-foods-and-beverages"],
    ),
    "en:teas": _n("Teas", ["en:hot-beverages", "en:plant-based-beverages"]),
    "en:tea-bags": _n("Tea bags", ["en:teas"]),
    "fr:tisanes": _n("Tisanes", ["en:teas"]),
}


def test_untagged_ancestors_are_materialised() -> None:
    """A product that skipped an intermediate tag still gets the full chain.

    This is the ancestor gap that put the same category at two addresses: the old
    walk stopped at whatever the product happened to hold, so ``en:tea-bags``
    filed under a one-segment path here and a four-segment one on a product that
    tagged more of the same lineage.
    """
    sparse = build_category_path(["en:tea-bags"], TAXONOMY, EXCLUDE, lang="en")
    complete = build_category_path(
        ["en:beverages", "en:hot-beverages", "en:teas", "en:tea-bags"],
        TAXONOMY,
        EXCLUDE,
        lang="en",
    )
    assert sparse == complete, "the same leaf must resolve to the same path"
    assert sparse == [
        "Beverages",
        "Beverages/Hot beverages",
        "Beverages/Hot beverages/Teas",
        "Beverages/Hot beverages/Teas/Tea bags",
    ]


def test_canonical_parent_map_is_a_spanning_forest() -> None:
    """Every node keeps exactly one parent; roots stay roots; nothing is orphaned."""
    canonical = build_canonical_parent_map(TAXONOMY)
    assert set(canonical) == set(TAXONOMY), "every node must be covered"
    for node, parent in canonical.items():
        if parent is None:
            assert not TAXONOMY[node]["parents"], (
                f"{node} was made a root but has parents in the taxonomy"
            )
        else:
            assert parent in TAXONOMY[node]["parents"], (
                f"{node}'s canonical parent {parent} is not one of its taxonomy parents"
            )


def test_canonical_parent_map_is_acyclic_and_reaches_a_root() -> None:
    canonical = build_canonical_parent_map(TAXONOMY)
    for node in TAXONOMY:
        chain = canonical_ancestry(canonical, node)
        assert chain[-1] == node
        assert len(chain) == len(set(chain)), f"{node}'s ancestry loops: {chain}"
        assert canonical[chain[0]] is None, f"{node}'s ancestry does not reach a root"


def test_canonical_parent_prefers_fewest_hops_to_a_root() -> None:
    """Depth, not authored order, decides — a shortcut parent wins outright."""
    taxonomy = {
        "en:root": _n("Root", []),
        "en:mid": _n("Mid", ["en:root"]),
        "en:deep": _n("Deep", ["en:mid"]),
        # Two parents at different depths: en:root is two hops closer.
        "en:leaf": _n("Leaf", ["en:deep", "en:root"]),
    }
    assert build_canonical_parent_map(taxonomy)["en:leaf"] == "en:root"


def test_tie_break_is_the_smallest_canonical_id() -> None:
    """When parents tie on depth, the lexicographically smallest id wins.

    42% of the real taxonomy's multi-parent nodes tie, so this rule decides
    nearly half of them — it is not a corner case.
    """
    taxonomy = {
        "en:zebra": _n("Zebra", []),
        "en:alpha": _n("Alpha", []),
        "en:middle": _n("Middle", []),
        "en:leaf": _n("Leaf", ["en:zebra", "en:middle", "en:alpha"]),
    }
    assert build_canonical_parent_map(taxonomy)["en:leaf"] == "en:alpha"


def test_tie_break_survives_a_reordered_parents_list() -> None:
    """The stability property: upstream re-ordering must not move an address.

    A category's address is what previously authored merchandising rules match
    against, so a tie-break sensitive to the order of the ``parents`` list would
    silently break them on the next taxonomy refresh.
    """
    parents = ["en:zebra", "en:middle", "en:alpha"]
    base = {
        "en:zebra": _n("Zebra", []),
        "en:alpha": _n("Alpha", []),
        "en:middle": _n("Middle", []),
    }
    addresses = set()
    for order in (parents, list(reversed(parents)), sorted(parents)):
        taxonomy = dict(base, **{"en:leaf": _n("Leaf", list(order))})
        addresses.add(tuple(build_category_path(["en:leaf"], taxonomy, EXCLUDE, "en")))
    assert len(addresses) == 1, f"address moved with parents order: {addresses}"


def test_a_cycle_does_not_swallow_its_component() -> None:
    """A future taxonomy refresh introducing a cycle must not drop those nodes.

    Today's taxonomy has none, but a cycle has no root to reach, so an unguarded
    BFS would leave the whole component unreachable and silently pathless.
    """
    taxonomy = {
        "en:a": _n("A", ["en:c"]),
        "en:b": _n("B", ["en:a"]),
        "en:c": _n("C", ["en:b"]),
        "en:d": _n("D", ["en:c"]),
    }
    canonical = build_canonical_parent_map(taxonomy)
    assert set(canonical) == set(taxonomy)
    for node in taxonomy:
        chain = canonical_ancestry(canonical, node)
        assert len(chain) == len(set(chain)), f"{node}'s ancestry loops: {chain}"
        assert canonical[chain[0]] is None
    # Broken at the lexicographically smallest node, deterministically.
    assert canonical["en:a"] is None


def test_prebuilt_map_matches_the_on_demand_one() -> None:
    """Threading the run-wide map must not change any answer."""
    tags = ["en:tea-bags", "en:plant-based-beverages"]
    canonical = build_canonical_parent_map(TAXONOMY, exclude=EXCLUDE)
    assert category_chain(
        tags, TAXONOMY, EXCLUDE, keep_prefixes={"en"}, canonical_parents=canonical
    ) == category_chain(tags, TAXONOMY, EXCLUDE, keep_prefixes={"en"})


def test_address_audit_flags_a_category_at_two_addresses() -> None:
    """The extract-time property-2 check must actually fire, not just exist."""
    audit = AddressAudit()
    audit.record(category_path_entries(["en:tea-bags"], TAXONOMY, EXCLUDE, "en"))
    assert audit.conflict_count == 0

    # Same node, a different address: exactly what a regression would look like.
    audit.record([("en:tea-bags", "Somewhere else/Tea bags")])
    assert audit.conflict_count == 1
    summary = audit.summary()
    assert summary["categories_at_multiple_addresses"] == 1
    assert summary["examples"][0]["category"] == "en:tea-bags"
    assert len(summary["examples"][0]["addresses"]) == 2


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
