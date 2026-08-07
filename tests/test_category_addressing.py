"""Property tests for where a category lands in the emitted ``category_path``.

Unlike ``test_taxonomy.py``, which uses a hand-built miniature DAG, these run on
**real** Open Food Facts data: ``fixtures/off_real_categories.json`` holds the
``categories_tags`` of 19 real products taken from the public Open Food Facts
JSONL export, together with the ancestor closure of those tags taken from the
public category taxonomy. Nothing about the fixture is synthesised — it is a slice
of the same data an extraction run sees, cut small enough to check in.

Every one of those products reproduced the defect under the pre-fix chain logic:
``en:sodas``, ``en:beverages``, ``en:carbonated-drinks``,
``en:plant-based-beverages`` and ``en:juices-and-nectars`` each landed at two
different path addresses depending on which product you looked at. ``en:sodas``
is the sharpest case — its lineage has exactly one parent at every hop, no DAG
fork anywhere — which is why picking a canonical parent cannot be the whole fix:
the walk also has to be anchored to a **global** taxonomy root.

The ancestor closure is closed under ``parents``, so shortest-path depths inside
the fixture are identical to depths in the full 14,457-node taxonomy. Pruning
cannot flatter the result.

Run with ``pytest tests/`` or directly: ``python tests/test_category_addressing.py``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from off_demo_extract.taxonomy import (  # noqa: E402
    build_category_path,
    build_primary_category_path,
    category_chain,
)

_FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "off_real_categories.json").read_text(
        encoding="utf-8"
    )
)
TAXONOMY: Dict[str, dict] = _FIXTURE["taxonomy"]
PRODUCTS: List[dict] = _FIXTURE["products"]

# The extractor's default --category-exclude.
EXCLUDE = {"en:null", "en:unknown", "en:undefined"}

# Categories that sat at two addresses before the fix, and the language each was
# reported in. Spanish is included because the bundled Spanish smoke catalog
# carries the divergence for the last two.
DUAL_ADDRESS_BEFORE = [
    "en:sodas",
    "en:beverages",
    "en:carbonated-drinks",
    "en:plant-based-beverages",
    "en:juices-and-nectars",
]


def _addresses_by_category(lang: str) -> Dict[str, Dict[str, List[str]]]:
    """``{canonical_id: {path: [product codes]}}`` over the whole fixture."""
    seen: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for product in PRODUCTS:
        tags = product["categories_tags"]
        chain = category_chain(
            tags, TAXONOMY, EXCLUDE, keep_prefixes={"en", lang} if lang else {"en"}
        )
        paths = build_primary_category_path(tags, TAXONOMY, EXCLUDE, lang)
        assert len(chain) == len(paths), (
            f"{product['code']}: chain and path lengths disagree ({chain} vs {paths})"
        )
        for node, path in zip(chain, paths):
            seen[node][path].append(product["code"])
    return seen


def test_property_2_one_primary_address_per_category() -> None:
    """A category has exactly ONE primary position, on every product carrying it.

    The category may also sit at alternate addresses — the source taxonomy is a
    DAG and 2,545 of its nodes have several parents — but which address is
    *primary* is decided once per run over the whole graph, so it cannot differ
    between two products that both carry the node. A primary that moved would
    split the breadcrumb a product page leads with across two positions, which is
    the defect this was written for: on the pre-fix walk, 407 categories in the
    English catalog (246 es, 581 fr) sat at two or more addresses.
    """
    for lang in ("en", "es"):
        offenders = {
            node: dict(by_path)
            for node, by_path in _addresses_by_category(lang).items()
            if len(by_path) > 1
        }
        assert not offenders, (
            f"[{lang}] {len(offenders)} categories resolved to more than one "
            f"address: {json.dumps(offenders, ensure_ascii=False, indent=2)}"
        )


def test_property_2_covers_the_categories_that_were_broken() -> None:
    """Guard on the guard: the fixture must still exercise the broken categories.

    Property 2 above passes vacuously if the fixture stops covering the
    categories that used to diverge — e.g. if a future leaf rule dropped them
    from every chain. Pin that they are all still emitted.
    """
    seen = _addresses_by_category("en")
    for node in DUAL_ADDRESS_BEFORE:
        assert node in seen, (
            f"{node} no longer appears in any emitted chain, so the property-2 "
            "test no longer covers the case it was written for"
        )
        carriers = sum(len(codes) for codes in seen[node].values())
        assert carriers >= 2, (
            f"{node} appears on only {carriers} product(s); at least two are "
            "needed for a same-address assertion to mean anything"
        )


def test_the_primary_address_is_one_root_to_leaf_chain() -> None:
    """The address a product page leads with is still exactly one chain.

    ``category_path`` is a union now, but the breadcrumb is not: it is one
    cumulative sequence, entry *i* carrying *i* separators and extending its
    predecessor by one segment. Nothing downstream can render "one address plus
    also categorized as …" if the "one address" is itself a union.
    """
    for lang in ("en", "es", "fr"):
        for product in PRODUCTS:
            paths = build_primary_category_path(
                product["categories_tags"], TAXONOMY, EXCLUDE, lang
            )
            assert paths, f"[{lang}] {product['code']}: expected a resolved path"
            for i, entry in enumerate(paths):
                assert entry.count("/") == i, (
                    f"[{lang}] {product['code']}: primary[{i}] is not a single "
                    f"cumulative chain: {paths}"
                )
            for shorter, longer in zip(paths, paths[1:]):
                assert longer.startswith(shorter + "/"), (
                    f"[{lang}] {product['code']}: {longer!r} does not extend "
                    f"{shorter!r} — the primary is not a single branch: {paths}"
                )


def test_property_3_the_emitted_field_is_a_prefix_closed_union() -> None:
    """``category_path`` carries every ancestor prefix of every address it holds.

    This is what the single-chain assertion became. It is not a weaker check: the
    cumulative-path contract is what every hierarchy facet downstream is entitled
    to under either shape, and a union that dropped a prefix would leave a
    breadcrumb level with no bucket to render. The primary leads the list, so a
    consumer reading the head gets the address the product page shows.
    """
    for lang in ("en", "es", "fr"):
        for product in PRODUCTS:
            tags = product["categories_tags"]
            paths = build_category_path(tags, TAXONOMY, EXCLUDE, lang)
            primary = build_primary_category_path(tags, TAXONOMY, EXCLUDE, lang)
            assert paths, f"[{lang}] {product['code']}: expected a resolved path"
            assert paths[: len(primary)] == primary, (
                f"[{lang}] {product['code']}: the union does not lead with the "
                f"primary address ({paths} vs {primary})"
            )
            values = set(paths)
            for entry in paths:
                if "/" not in entry:
                    continue
                parent = entry.rsplit("/", 1)[0]
                assert parent in values, (
                    f"[{lang}] {product['code']}: {entry!r} is emitted but its "
                    f"ancestor {parent!r} is not — the union is not prefix-closed"
                )
            assert len(paths) == len(set(paths)), (
                f"[{lang}] {product['code']}: a path is emitted twice: {paths}"
            )


def test_sodas_is_anchored_to_a_global_root() -> None:
    """The regression case: a lineage with no DAG fork anywhere, at two addresses.

    Every parent hop from ``en:sodas`` up is single-parent, so choosing a
    canonical parent changes nothing here. The two addresses came purely from
    products that omitted the ``en:beverages-and-beverages-preparations`` tag,
    which made the walk stop at a local root and invent a shorter path. The fix
    is to materialise the ancestors the product never tagged.
    """
    carriers = [p for p in PRODUCTS if "en:sodas" in p["categories_tags"]]
    assert len(carriers) >= 2, "fixture must carry more than one soda product"
    tagged_root = [
        p
        for p in carriers
        if "en:beverages-and-beverages-preparations" in p["categories_tags"]
    ]
    untagged_root = [
        p
        for p in carriers
        if "en:beverages-and-beverages-preparations" not in p["categories_tags"]
    ]
    assert tagged_root and untagged_root, (
        "fixture must contain both a product that tagged the top-level ancestor "
        "and one that did not — that gap is the whole bug"
    )

    expected = [
        "Beverages and beverages preparations",
        "Beverages and beverages preparations/Beverages",
        "Beverages and beverages preparations/Beverages/Carbonated drinks",
        "Beverages and beverages preparations/Beverages/Carbonated drinks/Sodas",
    ]
    for product in carriers:
        paths = build_category_path(product["categories_tags"], TAXONOMY, EXCLUDE, "en")
        assert paths == expected, (
            f"{product['code']}: sodas resolved to {paths} instead of the "
            f"global-root-anchored {expected}"
        )


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
