"""Regression tests for the hierarchical category-path builder.

These use a small synthetic taxonomy (no network / no 4.5 MB file needed) that
reproduces the structural traps of the real Open Food Facts taxonomy:

* a node with **multiple parents** (a DAG, not a tree) — the exact reason a naive
  ``"/".join(tags)`` produces a nonsense path;
* parallel roots in the same product's tag union;
* a foreign-language-only node that must be excluded from an English path.

Run with ``pytest tests/``. ``tests/conftest.py`` puts ``src`` on ``sys.path``,
so the import below is an ordinary top-of-file import.
"""

from __future__ import annotations

from off_demo_extract.taxonomy import (
    build_category_path,
    build_primary_category_path,
    category_chain,
)


def _n(name: str, parents: list[str]) -> dict:
    return {"name": {"en": name}, "parents": parents}


# A miniature DAG modelled on the real OFF tea/spread structure.
TAXONOMY = {
    "en:beverages": _n("Beverages", []),
    "en:plant-based-foods-and-beverages": _n("Plant-based foods and beverages", []),
    "en:hot-beverages": _n("Hot beverages", ["en:beverages"]),
    "en:plant-based-beverages": _n(
        "Plant-based beverages",
        ["en:beverages", "en:plant-based-foods-and-beverages"],
    ),
    # en:teas has TWO parents — the multi-parent / DAG case.
    "en:teas": _n("Teas", ["en:hot-beverages", "en:plant-based-beverages"]),
    "en:tea-bags": _n("Tea bags", ["en:teas"]),
    # A French-only node parented into the chain — must be filtered for en.
    "fr:tisanes": _n("Tisanes", ["en:teas"]),
}

EXCLUDE = {"en:null", "en:unknown", "en:undefined"}


def test_chain_steps_are_real_parent_child_edges() -> None:
    """Every adjacent pair in the emitted chain is a real taxonomy parent→child
    edge, so the breadcrumb order always matches the category tree (e.g. Oranges
    can never appear above Citrus). Guaranteed by construction; asserted here so
    a future change to the walk can't silently break the ordering."""
    tags = ["en:beverages", "en:hot-beverages", "en:teas", "en:tea-bags"]
    chain = category_chain(tags, TAXONOMY, EXCLUDE, keep_prefixes={"en"})
    assert len(chain) >= 2
    for parent, child in zip(chain, chain[1:]):
        assert parent in TAXONOMY[child]["parents"], (
            f"{child!r} is not a taxonomy child of {parent!r}: "
            "the path order does not match the tree"
        )
    # The cumulative display path preserves that same root→leaf order.
    path = build_primary_category_path(tags, TAXONOMY, EXCLUDE, "en")
    labels = [p.split("/")[-1] for p in path]
    assert labels == [TAXONOMY[c]["name"]["en"] for c in chain]


def test_the_primary_is_one_clean_chain_from_the_dag() -> None:
    """A product's flat tag union yields ONE cumulative root→leaf primary path.

    The emitted field is a union of every address, but the address the product
    page leads with is a single chain — and that chain is what the tag union
    collapses to, which is the property this test was written for.
    """
    tags = [
        "en:plant-based-foods-and-beverages",
        "en:beverages",
        "en:hot-beverages",
        "en:plant-based-beverages",
        "en:teas",
        "en:tea-bags",
    ]
    path = build_primary_category_path(tags, TAXONOMY, EXCLUDE, lang="en")

    # Cumulative paths: each element extends the previous by exactly one segment.
    assert path, "expected a non-empty hierarchy"
    for i, entry in enumerate(path):
        assert entry.count("/") == i, f"path[{i}] is not cumulative: {entry!r}"
    # Leaf is the deepest node; root is a real top-level category.
    assert path[-1].endswith("Tea bags")
    assert path[0] in {"Beverages", "Plant-based foods and beverages"}
    # The whole thing is a single chain ending at the leaf.
    assert path[-1].split("/")[-1] == "Tea bags"
    # And the union leads with it, so a consumer reading the head of the emitted
    # field gets exactly this chain and nothing spliced in front of it.
    assert build_category_path(tags, TAXONOMY, EXCLUDE, lang="en")[: len(path)] == path


def test_no_naive_flatten() -> None:
    """The result must NOT be every tag joined — that would mix sibling branches."""
    tags = ["en:beverages", "en:hot-beverages", "en:plant-based-beverages", "en:teas"]
    chain = category_chain(tags, TAXONOMY, EXCLUDE, keep_prefixes={"en"})
    # Only one of the two parallel "Teas" parents may appear in a single chain.
    assert not ({"en:hot-beverages", "en:plant-based-beverages"} <= set(chain)), (
        "chain wrongly contains BOTH parents of en:teas"
    )
    assert "en:teas" in chain


def test_foreign_language_node_excluded() -> None:
    """An English product must not be *filed under* a French-only node.

    ``fr:tisanes`` is a leaf here, so the language filter is what keeps it out.
    It is not a claim that no French node can ever appear in an English path — an
    *ancestor* can, and must (``tests/test_language_blind_traversal.py``); cutting
    those was what severed chains. The rule is about what a product is filed
    under, not about which segments the taxonomy makes it pass through.
    """
    tags = ["en:beverages", "en:hot-beverages", "en:teas", "fr:tisanes"]
    path = build_category_path(tags, TAXONOMY, EXCLUDE, lang="en")
    assert all("Tisanes" not in entry for entry in path), path


def test_localized_node_kept_for_matching_persona() -> None:
    """For the French persona, the fr: node is allowed back into the subgraph."""
    tags = ["en:beverages", "en:hot-beverages", "en:teas", "fr:tisanes"]
    chain = category_chain(tags, TAXONOMY, EXCLUDE, keep_prefixes={"en", "fr"})
    assert "fr:tisanes" in chain


def test_no_known_category_returns_empty() -> None:
    assert build_category_path(["en:null"], TAXONOMY, EXCLUDE, lang="en") == []
    assert build_category_path(["en:not-in-taxonomy"], TAXONOMY, EXCLUDE, lang="en") == []
