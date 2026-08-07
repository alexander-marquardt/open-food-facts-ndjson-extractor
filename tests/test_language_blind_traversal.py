"""A chain must walk through a foreign-language ancestor, not stop at it.

The defect
----------
The catalog's language filter used to be applied to the taxonomy *graph*. Keeping
only ``en``-prefixed nodes for an English run does not just hide those nodes — it
deletes every parent edge through them, so a node whose only parent was foreign
was promoted to a **root of the pruned graph** and its chain stopped there. An
English run walked a forest with 161 roots where the taxonomy has 92, a Spanish
one 161 and a French one 130, so the same category sat at a different depth in
each catalog and a segment that was mid-chain in one was top-level in another.

``en:pate`` really sits under ``fr:charcuteries-diverses`` under
``en:prepared-meats``; an English catalog filed it as a top-level ``Pâté``.
Refusing those products (``--require-category-path``, tested next door in
``test_root_anchored_gate.py``) contained the damage but discarded real products
rather than giving them their real lineage.

The fix, and what these tests pin
---------------------------------
Traversal is language-blind: one parent map over all 14,457 nodes, for every
catalog. The language filter moved to where the localization argument actually
holds — which tags a product may be **filed under**, and which values the flat
``categories`` field may carry — and labels are localized at render time by
``display_label``'s existing ``lang`` → ``en`` → ``xx`` → slug fallback.

So there are three separable claims here and each has its own test: chains reach
a real root, the leaf is still language-filtered, and a foreign segment renders
through the one labeller rather than a second one.

Why the fixture and not a hand-built taxonomy
---------------------------------------------
``tests/fixtures/off_unanchored_chains.json`` carries verbatim
``categories_tags`` from the public export together with the **complete upward
closure** of those tags from the public taxonomy. Complete is the load-bearing
word: in a trimmed closure every chain head looks like a legitimate root, so
``global_roots`` returns whatever the fixture happens to top out at and an
anchoring assertion asserts nothing. The synthetic taxonomy below is kept
alongside it to isolate the mechanism — it exists so a failure says *which* rule
broke, not merely that the data moved.

Run with ``pytest tests/`` or directly:
``python tests/test_language_blind_traversal.py``.
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
    build_canonical_parent_map,
    build_category_path,
    category_path_entries,
    display_label,
    global_roots,
    unanchored_head,
)

PRICING_CONFIG = REPO_ROOT / "config" / "pricing_buckets.json"
REAL_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "off_unanchored_chains.json"

EXCLUDE = {"en:null", "en:unknown", "en:undefined"}
LANGS = ("en", "es", "fr")


def _n(name: Dict[str, str], parents: List[str]) -> Dict[str, Any]:
    return {"name": dict(name), "parents": parents}


# The mechanism in miniature. ``fr:charcuteries`` is a real node with a real
# English parent and no English name — the exact shape that used to sever a
# chain, since an English run could neither keep it nor name it.
TAXONOMY = {
    "en:prepared-meats": _n({"en": "Prepared meats", "fr": "Charcuteries"}, []),
    "fr:charcuteries-diverses": _n(
        {"fr": "Charcuteries diverses"}, ["en:prepared-meats"]
    ),
    "en:pate": _n({"en": "Pâté", "fr": "Pâté"}, ["fr:charcuteries-diverses"]),
    "en:snacks": _n({"en": "Snacks", "fr": "Snacks"}, []),
}


def _real_fixture() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    data = json.loads(REAL_FIXTURE.read_text(encoding="utf-8"))
    return data["taxonomy"], data["products"]


# ---------------------------------------------------------------------------
# Claim 1 — the chain reaches a real taxonomy root
# ---------------------------------------------------------------------------

def test_a_foreign_ancestor_is_traversed_not_severed() -> None:
    """The chain crosses ``fr:charcuteries-diverses`` instead of starting below it.

    Both halves matter. The path must be the *anchored* one, and it must be the
    same length in every catalog — asserting only that ``en:pate`` is no longer
    first would also pass on a chain that gained a segment for some other reason.
    """
    path = build_category_path(["en:pate"], TAXONOMY, EXCLUDE, lang="en")
    assert path == [
        "Prepared meats",
        "Prepared meats/Charcuteries diverses",
        "Prepared meats/Charcuteries diverses/Pâté",
    ], path

    ids = [node for node, _p in category_path_entries(["en:pate"], TAXONOMY, EXCLUDE, "en")]
    assert ids[0] == "en:prepared-meats", ids
    assert ids == ["en:prepared-meats", "fr:charcuteries-diverses", "en:pate"], ids


def test_every_real_chain_reaches_a_taxonomy_root_in_every_catalog() -> None:
    """No product, in any of the three catalogs, files below a real root.

    This is the property the whole change exists to restore, asserted against the
    fixture's complete upward closure so that "a real root" means the taxonomy's
    answer and not the fixture's horizon.
    """
    taxonomy, products = _real_fixture()
    roots = global_roots(taxonomy)
    assert len(roots) == 7, sorted(roots)

    for lang in LANGS:
        for product in products:
            ids = [
                node
                for node, _p in category_path_entries(
                    list(product["categories_tags"]), taxonomy, EXCLUDE, lang
                )
            ]
            assert ids, f"{product['code']} resolved no chain at all in {lang}"
            assert unanchored_head(ids, roots) is None, (
                f"{product['code']} starts at {ids[0]!r} in the {lang} catalog, "
                f"which is not a taxonomy root"
            )


def test_the_walked_graph_has_exactly_the_taxonomys_roots() -> None:
    """The forest a run walks is the taxonomy's forest, not a locale-shaped one.

    The per-product test above can only see the roots its products happen to sit
    under. This one is the whole graph, which is where a phantom root is created.
    """
    taxonomy, _products = _real_fixture()
    canonical = build_canonical_parent_map(taxonomy, exclude=EXCLUDE)
    walked_roots = {node for node, parent in canonical.items() if parent is None}
    assert walked_roots == global_roots(taxonomy), (
        f"phantom roots: {sorted(walked_roots - global_roots(taxonomy))}"
    )
    assert set(canonical) == set(taxonomy), "every node must be traversable"


def test_every_catalog_files_a_category_at_the_same_address() -> None:
    """The same product resolves the same chain of ids in en, es and fr.

    Labels differ per catalog and must; the *structure* may not. Cross-locale
    comparability is what a locale-shaped forest destroyed — 92 nodes of the real
    taxonomy had a different ancestry in the English and French maps.
    """
    taxonomy, products = _real_fixture()
    for product in products:
        by_lang = {
            lang: [
                node
                for node, _p in category_path_entries(
                    list(product["categories_tags"]), taxonomy, EXCLUDE, lang
                )
            ]
            for lang in LANGS
        }
        assert len(set(map(tuple, by_lang.values()))) == 1, (
            f"{product['code']} resolves differently per catalog: {by_lang}"
        )


# ---------------------------------------------------------------------------
# Claim 2 — the leaf is still language-filtered
# ---------------------------------------------------------------------------

def test_a_foreign_only_tag_is_still_not_a_leaf() -> None:
    """Traversing a foreign node must not make it filable.

    The fix moves the language filter; it does not remove it. Without this, the
    change would quietly undo the refusal of ``fr:charcuteries-cuites`` from an
    English catalog — a real node that still has no business being the category
    an English product is filed under.
    """
    assert build_category_path(["fr:charcuteries-diverses"], TAXONOMY, EXCLUDE, "en") == []
    # The French catalog does file under it, which is what makes the refusal a
    # language rule rather than a blanket one.
    assert build_category_path(["fr:charcuteries-diverses"], TAXONOMY, EXCLUDE, "fr") == [
        "Charcuteries",
        "Charcuteries/Charcuteries diverses",
    ]


def test_a_foreign_tag_does_not_deepen_an_english_products_filing() -> None:
    """A product carrying both tags is filed under the English one, not the French.

    The tie-break prefers the longest chain, so a French tag one hop deeper would
    win the leaf if the filter had been dropped instead of moved.
    """
    path = build_category_path(
        ["en:prepared-meats", "fr:charcuteries-diverses"], TAXONOMY, EXCLUDE, "en"
    )
    assert path == ["Prepared meats"], path


# ---------------------------------------------------------------------------
# Claim 3 — one labeller, with the localized fallback doing the work
# ---------------------------------------------------------------------------

def test_a_foreign_segment_renders_through_the_one_labeller() -> None:
    """Segments come from ``display_label`` alone, fallback chain included.

    A traversed foreign node is the case that most invites a second rule — some
    caller "fixing up" a French string in an English path. Asserting the segment
    is byte-identical to ``display_label``'s answer is what forbids that.
    """
    taxonomy, _products = _real_fixture()
    tags = ["en:meals", "en:chinese-dumplings"]
    for lang in LANGS:
        entries = category_path_entries(tags, taxonomy, EXCLUDE, lang)
        for node, cumulative in entries:
            assert cumulative.rsplit("/", 1)[-1] == display_label(taxonomy, node, lang), (
                f"{node} rendered as a path segment differently from display_label "
                f"in the {lang} catalog"
            )

    # ``xx:dumplings`` is the all-languages node: named once, read by everyone.
    assert build_category_path(tags, taxonomy, EXCLUDE, "en") == [
        "Meals",
        "Meals/Dumplings",
        "Meals/Dumplings/Chinese dumplings",
    ]
    # ``fr:cereales-preparees`` has only a French name, so English falls all the
    # way through to the slug. That is the visible cost of the trade, pinned
    # rather than left to be discovered in a built index.
    rice = ["en:meals", "en:prepared-rices"]
    assert build_category_path(rice, taxonomy, EXCLUDE, "en") == [
        "Meals",
        "Meals/Cereales preparees",
        "Meals/Cereales preparees/Prepared rices",
    ]
    assert build_category_path(rice, taxonomy, EXCLUDE, "fr") == [
        "Plats préparés",
        "Plats préparés/Céréales préparées",
        "Plats préparés/Céréales préparées/Riz préparés",
    ]


# ---------------------------------------------------------------------------
# End to end, through the CLI the extraction actually runs
# ---------------------------------------------------------------------------

def _product(code: str, categories_tags: List[str]) -> Dict[str, Any]:
    """A product that clears every filter *except* possibly the category gate."""
    return {
        "code": code,
        "lang": "en",
        "product_name_en": "Country Style Pork Pate",
        "generic_name_en": "Coarse pork pate, 180g terrine",
        "categories_tags": categories_tags,
        "images": {"front_en": {"rev": "7", "sizes": {"400": {"w": 400, "h": 400}}}},
    }


def _run_extract(
    tmp: Path, taxonomy: Dict[str, Any], products: List[Dict[str, Any]], *extra: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
            # A hand-built fixture is deliberately not the pinned snapshot.
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


def test_the_gate_refuses_nothing_because_the_chains_are_now_right() -> None:
    """Every fixture product survives the anchoring gate.

    These are the records the gate used to drop — four of the seven, under the
    four worst heads measured over the first 300,000 records of the January 2026
    dump. They are kept now because their chains reach a root, not because the
    gate was loosened: it is still on, and ``test_root_anchored_gate.py`` still
    proves it fires. The run is English because the fixture's titles and images
    are; the per-catalog claim is asserted on the chain itself above, which is
    where it lives.
    """
    taxonomy, raw = _real_fixture()
    products = [_product(p["code"], list(p["categories_tags"])) for p in raw]

    with tempfile.TemporaryDirectory(prefix="off-langblind-") as d:
        records, report = _run_extract(Path(d), taxonomy, products)

    assert {r["id"] for r in records} == {p["code"] for p in raw}, (
        "products dropped that should now anchor"
    )
    counters = report["counters"]
    assert counters["unanchored_category_path"] == 0, counters
    assert counters["missing_category_path"] == 0, counters

    anchoring = report["category_path_anchoring"]
    assert anchoring["products_with_unanchored_path"] == 0, anchoring
    assert anchoring["phantom_roots"] == 0, anchoring
    assert anchoring["traversal_roots"] == anchoring["taxonomy_roots"] == 7, anchoring

    # The one that gave the issue its name: filed under its real lineage rather
    # than as a top-level Pâté of its own.
    by_id = {r["id"]: r for r in records}
    assert by_id["0055652001899"]["category_path"][0] == "Meats and their products"
    assert by_id["0017400140328"]["category_path_primary"] == [
        "Meals",
        "Meals/Cereales preparees",
        "Meals/Cereales preparees/Prepared rices",
        "Meals/Cereales preparees/Prepared rices/Fried rice",
    ]


def test_the_report_names_a_phantom_root_before_any_product_hits_one() -> None:
    """The forest is reported, not only the rows that fell foul of it.

    A run whose input happened to carry nothing under a phantom root would report
    a clean zero over a broken graph. So the counts of both forests are in the
    report, and this pins that they move when the graph really is pruned —
    without it, the assertion above is satisfied by an audit that reports 0/0.
    """
    products = [_product("3017620422003", ["en:pate"])]
    with tempfile.TemporaryDirectory(prefix="off-langblind-") as d:
        _records, report = _run_extract(
            Path(d),
            TAXONOMY,
            products,
            "--category-exclude", "fr:charcuteries-diverses",
            "--no-require-category-path",
        )

    anchoring = report["category_path_anchoring"]
    assert anchoring["taxonomy_roots"] == 2, anchoring
    assert anchoring["traversal_roots"] == 3, anchoring
    assert anchoring["phantom_roots"] == 1, anchoring
    assert anchoring["top_phantom_roots"] == ["en:pate"], anchoring


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
