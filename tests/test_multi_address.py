"""A product reachable by more than one breadcrumb, with one of them primary.

What is being tested
--------------------
The Open Food Facts category taxonomy is a DAG — 2,545 of its 14,457 nodes have
more than one parent — and this extractor used to emit exactly one breadcrumb per
product. It did so at **two** altitudes, and relaxing either alone buys nothing:

* **the graph**, in ``build_canonical_parent_map``: one parent per node, so a node
  had one address even where the source gives it several;
* **the leaf**, in ``category_chain``: one leaf per product, so a product filed
  under two disjoint branches still emitted one chain.

Fix the graph and the leaf rule still picks one chain. Fix the leaf and each chain
still has one address. So both are exercised here, separately and together.

The shapes
----------
Three DAG shapes read very differently and all three have to work. They are built
as miniature taxonomies so that a failure points at the rule rather than at the
data, and every one of them is modelled on a real Open Food Facts structure named
in its own docstring:

* **pure two-root divergence** — the fork's parents sit under different global
  roots and share no ancestor at all (``en:cheeses``: ``Dairies/…`` and
  ``Fermented foods/…``). 412 of the 2,545 multi-parent nodes are this shape, and
  to a shopper it reads as two unrelated departments, not two routes through one.
* **true reconvergent diamond** — the branches split *below* a common ancestor and
  rejoin at the fork (``en:peanut-butters``). 2,133 of the 2,545 are this shape.
* **nested diamond** — a shared node whose own subtree contains a second diamond,
  which multiplies rather than adds: four addresses from two forks.

Why the end-to-end run is here and not only the unit assertions
---------------------------------------------------------------
The last two tests drive the real ``extract.main()`` over a real input file. The
unit assertions below them call the taxonomy functions directly, which proves the
*rule* — it does not prove that the extractor calls it, that the addresses reach
the emitted document, or that the primary is written to a field of its own. A fix
that is correct in the function and never reaches the record is a fix that ships
inert, and no amount of green on the pure functions can see it.

Run with ``pytest tests/``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from off_demo_extract.extract import main  # noqa: E402
from off_demo_extract.taxonomy import (  # noqa: E402
    AddressExplosionError,
    AddressIndex,
    build_canonical_parent_map,
    build_category_path,
    build_primary_category_path,
    category_leaves,
    category_path_entries,
    primary_category_path_entries,
)

PRICING_CONFIG = REPO_ROOT / "config" / "pricing_buckets.json"
EXCLUDE: set = set()


def _n(name: str, parents: List[str]) -> Dict[str, Any]:
    return {"name": {"en": name}, "parents": parents}


# --------------------------------------------------------------------------- #
# the three shapes
# --------------------------------------------------------------------------- #

# Pure divergence, modelled on ``en:cheeses`` (1,632 English products): its parent
# ``en:fermented-milk-products`` sits under two global roots that share nothing.
DIVERGENCE = {
    "en:dairies": _n("Dairies", []),
    "en:fermented-foods": _n("Fermented foods", []),
    "en:fermented-milk-products": _n(
        "Fermented milk products", ["en:dairies", "en:fermented-foods"]
    ),
    "en:cheeses": _n("Cheeses", ["en:fermented-milk-products"]),
}

# A true reconvergent diamond, modelled on ``en:peanut-butters``: the branches
# split at ``Plant-based foods`` and rejoin at ``Legume butters``, entirely inside
# one root. ``en:spreads`` adds the second, disjoint root the real node also has,
# so this fixture carries BOTH shapes at once exactly as the real product does.
DIAMOND = {
    "en:plant-based-foods-and-beverages": _n("Plant-based foods and beverages", []),
    "en:spreads": _n("Spreads", []),
    "en:plant-based-foods": _n(
        "Plant-based foods", ["en:plant-based-foods-and-beverages"]
    ),
    "en:legumes-and-their-products": _n(
        "Legumes and their products", ["en:plant-based-foods"]
    ),
    "en:plant-based-spreads": _n(
        "Plant-based spreads", ["en:plant-based-foods", "en:spreads"]
    ),
    "en:oilseed-purees": _n("Oilseed purees", ["en:plant-based-spreads"]),
    "en:legume-butters": _n(
        "Legume butters", ["en:legumes-and-their-products", "en:oilseed-purees"]
    ),
    "en:peanut-butters": _n("Peanut butters", ["en:legume-butters"]),
}

# A nested diamond: ``en:d`` is shared by the first fork, and its own subtree
# contains a second one. Two forks multiply to four addresses, which is also the
# "reachable via three or more selections" case.
NESTED = {
    "en:a": _n("A", []),
    "en:b": _n("B", ["en:a"]),
    "en:c": _n("C", ["en:a"]),
    "en:d": _n("D", ["en:b", "en:c"]),
    "en:e": _n("E", ["en:d"]),
    "en:f": _n("F", ["en:d"]),
    "en:g": _n("G", ["en:e", "en:f"]),
}

# Two leaves on wholly disjoint branches: neither is an ancestor of the other, so
# the leaf rule has to choose, and what it discards is the second breadcrumb.
TWO_LEAVES = {
    "en:beverages": _n("Beverages", []),
    "en:snacks": _n("Snacks", []),
    "en:sodas": _n("Sodas", ["en:beverages"]),
    "en:crisps": _n("Crisps", ["en:snacks"]),
}


def _index(taxonomy: Dict[str, Any]) -> Tuple[Dict[str, Any], AddressIndex]:
    canonical = build_canonical_parent_map(taxonomy, exclude=EXCLUDE)
    return canonical, AddressIndex(taxonomy, canonical, exclude=EXCLUDE)


def _paths(taxonomy: Dict[str, Any], tags: List[str]) -> List[str]:
    return build_category_path(tags, taxonomy, EXCLUDE, "en")


def _primary(taxonomy: Dict[str, Any], tags: List[str]) -> List[str]:
    return build_primary_category_path(tags, taxonomy, EXCLUDE, "en")


def _terminals(paths: List[str]) -> List[str]:
    """The addresses in a prefix-closed union: values nothing else extends."""
    return [p for p in paths if not any(o.startswith(p + "/") for o in paths if o != p)]


# --------------------------------------------------------------------------- #
# shape 1 — pure two-root divergence
# --------------------------------------------------------------------------- #
def test_pure_two_root_divergence_gives_two_addresses() -> None:
    """``en:cheeses``: two roots, no shared ancestor, both breadcrumbs emitted."""
    paths = _paths(DIVERGENCE, ["en:cheeses"])
    assert _terminals(paths) == [
        "Dairies/Fermented milk products/Cheeses",
        "Fermented foods/Fermented milk products/Cheeses",
    ]
    # Both roots are present as values of their own, so a drill-down facet has a
    # top level to render for each.
    assert "Dairies" in paths and "Fermented foods" in paths


def test_the_divergent_primary_is_the_one_the_forest_chose() -> None:
    """Fewest hops, then lexicographically smallest id — unchanged, and leading.

    Both parents of ``en:fermented-milk-products`` are roots, so the tie-break
    decides, and it must still decide it the same way: this is the address the
    collapsed build emitted, and the whole rebuild is auditable only because it
    did not move.
    """
    primary = _primary(DIVERGENCE, ["en:cheeses"])
    assert primary == [
        "Dairies",
        "Dairies/Fermented milk products",
        "Dairies/Fermented milk products/Cheeses",
    ]
    assert _paths(DIVERGENCE, ["en:cheeses"])[: len(primary)] == primary


# --------------------------------------------------------------------------- #
# shape 2 — true reconvergent diamond
# --------------------------------------------------------------------------- #
def test_a_reconvergent_diamond_gives_the_branches_that_rejoin() -> None:
    """``en:peanut-butters``: split at Plant-based foods, rejoin at Legume butters.

    This is the shape a two-root check cannot see. Both addresses share the first
    two segments and diverge below them, so a test that only asserted "the roots
    differ" would pass on a build that had dropped the diamond entirely.
    """
    paths = _paths(DIAMOND, ["en:peanut-butters"])
    terminals = _terminals(paths)
    assert terminals == [
        "Plant-based foods and beverages/Plant-based foods/Legumes and their "
        "products/Legume butters/Peanut butters",
        "Spreads/Plant-based spreads/Oilseed purees/Legume butters/Peanut butters",
        "Plant-based foods and beverages/Plant-based foods/Plant-based spreads/"
        "Oilseed purees/Legume butters/Peanut butters",
    ]
    # The reconvergence itself: two of the three addresses share a strict prefix
    # and both end at the same node, which is what "rejoin" means.
    inside_one_root = [t for t in terminals if t.startswith("Plant-based foods and")]
    assert len(inside_one_root) == 2
    assert inside_one_root[0].split("/")[1] == inside_one_root[1].split("/")[1]
    assert inside_one_root[0].split("/")[2] != inside_one_root[1].split("/")[2]


def test_the_diamond_and_the_divergence_arrive_together() -> None:
    """The real product carries both shapes, so the fixture must too.

    ``Creamy Peanut Butter`` is filed at three addresses: two that split and
    rejoin inside ``Plant-based foods and beverages``, and one from the separate
    root ``Spreads`` with no shared ancestor at all.
    """
    terminals = _terminals(_paths(DIAMOND, ["en:peanut-butters"]))
    roots = {t.split("/")[0] for t in terminals}
    assert roots == {"Plant-based foods and beverages", "Spreads"}
    assert len(terminals) == 3


def test_the_union_carries_every_cumulative_prefix() -> None:
    """Prefix-closure, which is what a hierarchy facet is entitled to.

    An address whose intermediate levels were not emitted leaves the drill-down
    with a level that has no bucket, which is worse than not restoring it at all.
    """
    paths = _paths(DIAMOND, ["en:peanut-butters"])
    values = set(paths)
    assert len(values) == len(paths), "no address prefix may be emitted twice"
    for entry in paths:
        if "/" in entry:
            assert entry.rsplit("/", 1)[0] in values, entry


# --------------------------------------------------------------------------- #
# shape 3 — nested diamond, and three-or-more selections
# --------------------------------------------------------------------------- #
def test_a_nested_diamond_multiplies_rather_than_adds() -> None:
    """Two forks in one lineage give four addresses, not three.

    A shared node whose own subtree contains a second diamond. Adding the
    alternates of each fork independently would produce three; the addresses are
    the *paths*, so they compose.
    """
    terminals = _terminals(_paths(NESTED, ["en:g"]))
    assert sorted(terminals) == ["A/B/D/E/G", "A/B/D/F/G", "A/C/D/E/G", "A/C/D/F/G"]
    assert len(terminals) >= 3, "this is also the three-or-more-selections case"


def test_the_nested_primary_still_follows_the_canonical_parent_at_every_hop() -> None:
    """One address of the four is the primary, and it is the forest's."""
    assert _primary(NESTED, ["en:g"]) == ["A", "A/B", "A/B/D", "A/B/D/E", "A/B/D/E/G"]


# --------------------------------------------------------------------------- #
# the second collapse — the leaf
# --------------------------------------------------------------------------- #
def test_two_disjoint_leaves_both_become_breadcrumbs() -> None:
    """The downstream dedupe. Relaxing the graph alone would not have removed it.

    A product tagged under two branches that are not ancestors of one another used
    to emit one chain, because the leaf rule picked the longest and discarded the
    rest. Restoring the DAG changes nothing here — neither leaf is multi-parent —
    so this is the half of the work a graph-only fix silently misses.
    """
    tags = ["en:sodas", "en:crisps"]
    # The primary leaf's address leads; the discarded one follows.
    assert _terminals(_paths(TWO_LEAVES, tags)) == ["Snacks/Crisps", "Beverages/Sodas"]
    # The rule that chose between them is unchanged and still names the primary:
    # equal chain lengths, so the lexicographically smallest id wins.
    assert category_leaves(tags, TWO_LEAVES, EXCLUDE, keep_prefixes={"en"}) == [
        "en:crisps",
        "en:sodas",
    ]
    assert _primary(TWO_LEAVES, tags) == ["Snacks", "Snacks/Crisps"]


def test_an_alternate_leaf_on_the_primarys_own_chain_adds_nothing() -> None:
    """A redundant tag must not turn into a redundant value.

    ``en:beverages`` is an ancestor of ``en:sodas``; its address is already a
    prefix of the primary, so the union de-duplicates it away rather than emitting
    the shallower filing as if it were a second address.
    """
    assert _paths(TWO_LEAVES, ["en:beverages", "en:sodas"]) == [
        "Beverages",
        "Beverages/Sodas",
    ]


# --------------------------------------------------------------------------- #
# the properties the restoration must not cost
# --------------------------------------------------------------------------- #
def test_a_categorys_addresses_do_not_depend_on_the_product() -> None:
    """Global, product-independent addressing, which is why the map is built once.

    The point of deciding the graph per run rather than per product. Three
    products carrying ``en:peanut-butters`` with wildly different tag sets — one
    bare, one that tagged half its lineage, one that tagged a sibling branch too —
    must all place the node at the same set of addresses.
    """
    bare = _paths(DIAMOND, ["en:peanut-butters"])
    partial = _paths(
        DIAMOND, ["en:plant-based-foods", "en:legume-butters", "en:peanut-butters"]
    )
    wide = _paths(
        DIAMOND,
        ["en:spreads", "en:oilseed-purees", "en:peanut-butters", "en:plant-based-foods"],
    )
    assert bare == partial == wide
    assert _primary(DIAMOND, ["en:peanut-butters"]) == _primary(
        DIAMOND, ["en:plant-based-foods", "en:peanut-butters"]
    )


def test_the_alternate_order_does_not_move_when_upstream_reorders_parents() -> None:
    """The stability argument the tie-break was chosen for, extended to alternates.

    Re-ordering a node's ``parents`` list is the most common churn in the upstream
    file. If it moved an address, previously authored policies would break on a
    taxonomy refresh with nothing in the diff to explain it.
    """
    shuffled = {
        node: {"name": value["name"], "parents": list(reversed(value["parents"]))}
        for node, value in DIAMOND.items()
    }
    assert _paths(shuffled, ["en:peanut-butters"]) == _paths(
        DIAMOND, ["en:peanut-butters"]
    )
    assert _primary(shuffled, ["en:peanut-butters"]) == _primary(
        DIAMOND, ["en:peanut-butters"]
    )


def test_the_id_is_kept_beside_every_address_not_re_derivable_from_it() -> None:
    """``(canonical_id, cumulative_path)`` survives one id landing at two paths.

    Going breadcrumb → category id is not an inversion of the path string, and was
    not one before the alternates existed: on the pinned snapshot a French
    catalog's ``…/Vins italiens/Chianti`` is claimed by both ``en:chianti`` and
    ``it:chianti``. Here the ambiguity is the other way round — one id at three
    paths — and the pairing has to carry it either way.
    """
    entries = category_path_entries(["en:peanut-butters"], DIAMOND, EXCLUDE, "en")
    leaf_entries = [(node, path) for node, path in entries if node == "en:peanut-butters"]
    assert len(leaf_entries) == 3, leaf_entries
    assert len({path for _node, path in leaf_entries}) == 3
    assert len(entries) == len(set(entries)), "pairs must be de-duplicated as pairs"
    primary = primary_category_path_entries(
        ["en:peanut-butters"], DIAMOND, EXCLUDE, "en"
    )
    assert entries[: len(primary)] == primary


def test_a_pathological_taxonomy_is_refused_rather_than_truncated() -> None:
    """The circuit breaker: enumeration is exponential in the worst case.

    The pinned snapshot's worst node has 28 addresses, so this cannot fire on real
    data. It exists so that a future refresh which *is* pathological stops the
    build by name rather than silently emitting a truncated address set, which
    would be a facet that lies.
    """
    taxonomy: Dict[str, Any] = {"en:root": _n("Root", [])}
    previous = ["en:root"]
    # Each layer doubles the number of paths: 2**12 == 4,096 addresses at the leaf.
    for layer in range(12):
        left, right = f"en:l{layer}a", f"en:l{layer}b"
        taxonomy[left] = _n(f"L{layer}a", list(previous))
        taxonomy[right] = _n(f"L{layer}b", list(previous))
        previous = [left, right]
    taxonomy["en:leaf"] = _n("Leaf", list(previous))
    canonical = build_canonical_parent_map(taxonomy, exclude=EXCLUDE)
    try:
        AddressIndex(taxonomy, canonical, exclude=EXCLUDE)
    except AddressExplosionError as exc:
        assert "addresses" in str(exc)
    else:
        raise AssertionError("a 4,096-address node was accepted without complaint")


# --------------------------------------------------------------------------- #
# end to end, through the real extractor
# --------------------------------------------------------------------------- #
def _product(code: str, tags: List[str]) -> Dict[str, Any]:
    """A product that clears every filter the extractor applies before addressing."""
    return {
        "code": code,
        "lang": "en",
        "product_name_en": "Creamy Peanut Butter",
        "generic_name_en": "Smooth peanut butter, 462g jar",
        "categories_tags": tags,
        "images": {"front_en": {"rev": "3", "sizes": {"400": {"w": 400, "h": 400}}}},
    }


def _run_extract(
    tmp: Path, taxonomy: Dict[str, Any], products: List[Dict[str, Any]], *extra: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run the real extractor CLI; return (records, report)."""
    taxonomy_path = tmp / "categories.json"
    taxonomy_path.write_text(json.dumps(taxonomy), encoding="utf-8")
    input_path = tmp / "products.jsonl"
    input_path.write_text(
        "".join(json.dumps(p) + "\n" for p in products), encoding="utf-8"
    )
    output_path = tmp / "out.ndjson"
    report_path = tmp / "report.json"
    rc = main(
        [
            "--input", str(input_path),
            "--output", str(output_path),
            "--report", str(report_path),
            "--taxonomy", str(taxonomy_path),
            "--allow-unpinned-taxonomy",
            "--pricing-config", str(PRICING_CONFIG),
            "--progress-every", "0",
            "--progress-seconds", "0",
            "--yes",
            *extra,
        ]
    )
    assert rc == 0, f"extractor exited {rc}"
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records, json.loads(report_path.read_text(encoding="utf-8"))


def test_the_emitted_record_really_carries_every_address() -> None:
    """The transport, not the rule: the addresses have to reach the document.

    Everything above calls the taxonomy functions directly. This drives
    ``extract.main()`` over a real input file and reads the written NDJSON, which
    is the only thing that shows the extractor calls the new code at all — a
    restoration that is correct in ``taxonomy.py`` and never reaches
    ``category_path`` ships completely inert, and every assertion above would
    still be green.
    """
    with tempfile.TemporaryDirectory(prefix="off-multi-address-") as d:
        records, report = _run_extract(
            Path(d), DIAMOND, [_product("0050428476284", ["en:peanut-butters"])]
        )

    assert len(records) == 1
    record = records[0]
    assert len(_terminals(record["category_path"])) == 3, record["category_path"]
    assert record["category_path"] == _paths(DIAMOND, ["en:peanut-butters"])

    # The primary is a field of its own, and it is the head of the union. A
    # breadcrumb renderer must not have to know that "the primary is the first N
    # values" of a multi-valued keyword field.
    primary = record["category_path_primary"]
    assert primary == _primary(DIAMOND, ["en:peanut-butters"])
    assert record["category_path"][: len(primary)] == primary
    assert len(primary) == 5, primary

    # And the run says so, in numbers a downstream aggregation is sized against.
    addresses = report["category_path_addresses"]
    assert addresses["products_at_multiple_addresses"] == 1
    assert addresses["categories_at_multiple_primary_addresses"] == 0
    assert addresses["distinct_category_paths"] == len(set(record["category_path"]))
    assert addresses["max_category_path_values"] == len(record["category_path"])


def test_the_run_does_not_report_the_alternates_as_a_property_2_violation() -> None:
    """Two products, one shared multi-address node: a shape, not a conflict.

    The audit used to fire on a node seen at two addresses, which is now the
    normal case. It has to keep firing on a node whose *primary* differs between
    products and stop firing on this — and a check that had simply been disabled
    would look identical here, so the alternates are asserted to be counted.
    """
    with tempfile.TemporaryDirectory(prefix="off-multi-address-") as d:
        _records, report = _run_extract(
            Path(d),
            DIAMOND,
            [
                _product("0050428476284", ["en:peanut-butters"]),
                _product("0050428476291", ["en:legume-butters"]),
            ],
        )

    addresses = report["category_path_addresses"]
    assert addresses["categories_at_multiple_primary_addresses"] == 0
    assert addresses["categories_with_alternate_addresses"] >= 1
    assert addresses["max_addresses_for_a_category"] == 3
    assert report["counters"]["categories_at_multiple_primary_addresses"] == 0
    assert report["counters"]["products_at_multiple_addresses"] == 2


def test_the_report_describes_both_category_fields_and_their_two_shapes() -> None:
    """The report is the run's own self-description; it must describe this run.

    ``category_path`` stopped being one root→leaf chain the moment the alternates
    were restored, and ``category_path_primary`` did not exist before. A report
    that still calls the first a single path — or never names the second at all —
    hands a downstream consumer the wrong shape for the field it is reading, and
    nothing else in this suite reads the ``filters`` block, so nothing else would
    notice. The description drifted exactly this way once already.
    """
    with tempfile.TemporaryDirectory(prefix="off-multi-address-") as d:
        records, report = _run_extract(
            Path(d), DIAMOND, [_product("0050428476284", ["en:peanut-butters"])]
        )

    # The run being described really does emit the two different shapes, so what
    # follows is about a distinction this report had to make, not a hypothetical.
    record = records[0]
    assert len(record["category_path"]) > len(record["category_path_primary"])

    filters = report["filters"]
    assert "category_path_primary" in filters, sorted(filters)

    union_desc = filters["category_path"]
    primary_desc = filters["category_path_primary"]
    # The union is not a chain, and the chain is not the union. Asserted on the
    # wording because the wording is the whole artifact: this block is prose a
    # consumer reads, and a description that names the wrong shape is the defect.
    assert "union" in union_desc, union_desc
    assert "root->leaf" not in union_desc, union_desc
    assert "root->leaf" in primary_desc, primary_desc
    assert union_desc != primary_desc
