"""Regression tests for the hierarchical category-path builder.

These use a small synthetic taxonomy (no network / no 4.5 MB file needed) that
reproduces the structural traps of the real Open Food Facts taxonomy:

* a node with **multiple parents** (a DAG, not a tree) — the exact reason a naive
  ``"/".join(tags)`` produces a nonsense path;
* parallel roots in the same product's tag union;
* a foreign-language-only node that must be excluded from an English path.

Run with ``pytest tests/`` or directly: ``python tests/test_taxonomy.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from off_demo_extract.taxonomy import build_category_path, category_chain  # noqa: E402


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


def test_single_clean_chain_from_dag() -> None:
    """A product's flat tag union collapses to ONE cumulative root→leaf path."""
    tags = [
        "en:plant-based-foods-and-beverages",
        "en:beverages",
        "en:hot-beverages",
        "en:plant-based-beverages",
        "en:teas",
        "en:tea-bags",
    ]
    path = build_category_path(tags, TAXONOMY, EXCLUDE, lang="en")

    # Cumulative paths: each element extends the previous by exactly one segment.
    assert path, "expected a non-empty hierarchy"
    for i, entry in enumerate(path):
        assert entry.count("/") == i, f"path[{i}] is not cumulative: {entry!r}"
    # Leaf is the deepest node; root is a real top-level category.
    assert path[-1].endswith("Tea bags")
    assert path[0] in {"Beverages", "Plant-based foods and beverages"}
    # The whole thing is a single chain ending at the leaf.
    assert path[-1].split("/")[-1] == "Tea bags"


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
    """An English path must not surface a French-only taxonomy node."""
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
