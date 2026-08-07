"""The acceptance gate for restoring the source DAG: no primary breadcrumb moved.

Restoring the taxonomy's alternate addresses changes the ``category_path`` of
35-77% of every catalog, depending on locale. A regression hiding inside a change
that large is not findable by eye, so the rebuild is made auditable by a property
instead: **every product's primary address is byte-identical to what the
collapsed build produced.** Every path that existed still exists, in the same
position, and anything else in the diff is an addition. A primary that moves is a
defect, not noise.

That property was measured over the whole 66 GB source dump — 4,241,020 lines,
one streaming pass, the pre-change module and the current one run side by side on
the same curated tag sets — and **zero** primaries moved, across 108,380 English,
31,913 Spanish and 223,036 French records. This file is the durable half of that
measurement: the census cannot run in CI, so a sample of the same corpus travels
with the repository and is checked on every commit.

Why this is not the implementation restating itself
---------------------------------------------------
The expected values in ``fixtures/off_primary_addresses.json`` were produced by
``taxonomy.py`` as it stood at the last commit before a product could hold more
than one address. That code is no longer in the tree, so nothing here can drift
into agreement with it: the fixture is an oracle, and the only way to satisfy it
is to place every product where the collapsed build placed it.

The fixture also has to keep *biting*. A gate over products that happen to sit at
one address each would pass on a build that had quietly dropped the restoration
altogether, so :func:`test_the_fixture_still_exercises_multi_address_products`
pins that the sample really does contain products the change moves.

Run with ``pytest tests/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from off_demo_extract.taxonomy import (  # noqa: E402
    build_category_path,
    build_primary_category_path,
)

_FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "off_primary_addresses.json").read_text(
        encoding="utf-8"
    )
)
TAXONOMY: Dict[str, Any] = _FIXTURE["taxonomy"]
PRODUCTS: List[Dict[str, Any]] = _FIXTURE["products"]
EXPECTED: Dict[str, Dict[str, List[str]]] = _FIXTURE["expected_primary"]

# The extractor's builds pass no --category-exclude, so the fixture was cut
# against an empty one and is checked against the same.
EXCLUDE: set = set()

LANGS = ("en", "es", "fr")


def test_no_products_primary_address_moved() -> None:
    """The gate. Every primary is byte-identical to the pre-restoration build."""
    moved = []
    for lang in LANGS:
        for product in PRODUCTS:
            actual = build_primary_category_path(
                product["categories_tags"], TAXONOMY, EXCLUDE, lang
            )
            expected = EXPECTED[lang][product["code"]]
            if actual != expected:
                moved.append(
                    {"lang": lang, "code": product["code"], "was": expected, "now": actual}
                )
    assert not moved, (
        f"{len(moved)} primary addresses moved:\n"
        + json.dumps(moved[:5], ensure_ascii=False, indent=2)
    )


def test_the_primary_still_leads_the_emitted_field() -> None:
    """A stable primary that no consumer can find is not a stable primary.

    ``category_path`` is the field a facet counts over and the field a breadcrumb
    renderer reads, so the pre-restoration path has to be recoverable *from it* —
    at its head, in order — and not merely somewhere inside it.
    """
    for lang in LANGS:
        for product in PRODUCTS:
            expected = EXPECTED[lang][product["code"]]
            union = build_category_path(
                product["categories_tags"], TAXONOMY, EXCLUDE, lang
            )
            assert union[: len(expected)] == expected, (
                f"[{lang}] {product['code']}: category_path does not lead with the "
                f"pre-restoration path ({union[: len(expected)]} vs {expected})"
            )


def test_the_fixture_still_exercises_multi_address_products() -> None:
    """Guard on the guard: a sample of single-address products proves nothing.

    Without this, a build that had reverted the restoration entirely — every
    product back at one address — would satisfy every assertion above, because a
    primary that never moved is exactly what a reverted build produces.
    """
    for lang in LANGS:
        widened = [
            product["code"]
            for product in PRODUCTS
            if len(build_category_path(product["categories_tags"], TAXONOMY, EXCLUDE, lang))
            > len(EXPECTED[lang][product["code"]])
        ]
        assert len(widened) >= len(PRODUCTS) // 3, (
            f"[{lang}] only {len(widened)} of {len(PRODUCTS)} fixture products gained "
            "an address; the stability gate above would pass on a build that had "
            "dropped the restoration"
        )


def test_the_fixture_taxonomy_is_a_faithful_slice() -> None:
    """The closure is complete under ``parents``, so no fork was pruned away.

    Every parent named by a fixture node is itself a fixture node. If it were not,
    a node's shortest distance to a root inside the fixture could differ from its
    distance in the full taxonomy, which would move the canonical parent and make
    the whole gate measure a different graph than the one the catalog is built on.
    """
    for node, value in TAXONOMY.items():
        for parent in value["parents"]:
            assert parent in TAXONOMY, f"{node} has parent {parent} outside the fixture"
    tagged = {tag for product in PRODUCTS for tag in product["categories_tags"]}
    assert tagged <= set(TAXONOMY), sorted(tagged - set(TAXONOMY))[:5]
