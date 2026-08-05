"""An ``xx:``-prefixed category belongs to every catalog, not to none of them.

The defect
----------
``default_keep_prefixes(lang)`` returned ``{"en", lang}``, and that set decides
which ids a product may be **filed under** and which values the flat
``taxonomy_tags`` field may carry. Open Food Facts also uses an ``xx:`` prefix,
which is not a language: it marks a node whose name is *the same in every
language* — ``xx:tofu``, ``xx:dumplings``, ``xx:sake``. There are 34 of them in
the pinned snapshot, no catalog's keep-prefixes ever contained ``xx``, so those
34 categories could never be emitted by any locale.

That is a different thing from the refusal ``fr:pates-a-tartiner`` gets from an
English catalog, which is deliberate: a French-only node genuinely has no
business being a searchable English category. A language-neutral node belongs to
**every** catalog by construction. The inconsistency was already visible one
function away — ``display_label`` reads an ``xx`` *name* happily, so ``xx`` was
accepted as a way to name a node while being refused as a way to be one, and
``xx:tofu`` was reported in the ``out_of_language`` bucket, which is the opposite
of the truth about it.

What is asserted, and why each half is needed
---------------------------------------------
Both directions, because each alone is satisfiable by a wrong fix:

* **The neutral node is now reachable.** ``xx:sake`` resolves into
  ``category_path`` in the English, Spanish *and* French catalogs, at its real
  address under ``en:alcoholic-beverages``, and a real product carrying it is
  filed there end to end. Before the change all three return ``[]``.
* **The foreign-only node is still refused.** ``fr:pates-a-tartiner`` and
  ``fr:charcuteries-cuites`` are still not filable by an English or Spanish
  catalog — and ``fr:pates-a-tartiner`` *is* filable by the French one, so the
  control also shows the filter is still per-locale rather than merely still
  present. ``ca:cervesa-ipa`` is refused by all three. A fix that admitted every
  prefix would pass the first half and fail this one.

The strict-forest properties are asserted rather than assumed, because admitting
nodes changes which categories exist and therefore which addresses a catalog
holds: every eligible node resolves to exactly one address, and no two of them
render the same label, in each of the three catalogs. Measured over the whole
pinned snapshot as well as over this slice, in every locale: eligible nodes
8,939 -> 8,973 (en), 9,327 -> 9,361 (es), 11,747 -> 11,781 (fr); nodes reachable
on a path 8,965 -> 8,996, 9,354 -> 9,385, 11,753 -> 11,784; labels shared by more
than one node, 0 before and 0 after in all three. The parent map is language-
blind already, so no node's ancestry moves — what changes is only which nodes a
chain may *end* at.

The data
--------
``fixtures/off_language_neutral_nodes.json`` holds the 34 ``xx:`` nodes, three
foreign-language control nodes, real products carrying them verbatim from the
public export, and the **complete** upward closure of every id named. Complete is
load-bearing: in a trimmed closure every chain head looks like a root, so a
depth or anchoring assertion over it asserts nothing.

Labels are asserted on ``xx:sake``, whose rendering is unaffected by the casing
rule issue #16 is about, so the two changes cannot invalidate each other's
literals. Where ``xx:tofu`` is checked it is checked through ``display_label``,
because what is under test here is whether the node is *eligible*, not what it is
called.

Run with ``pytest tests/`` or directly:
``python tests/test_language_neutral_nodes.py``.
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
    build_category_path,
    canonical_ancestry,
    build_canonical_parent_map,
    default_keep_prefixes,
    display_label,
    eligible_nodes,
)

PRICING_CONFIG = REPO_ROOT / "config" / "pricing_buckets.json"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "off_language_neutral_nodes.json"

# The extractor's default --category-exclude.
EXCLUDE = {"en:null", "en:unknown", "en:undefined"}
LANGS = ("en", "es", "fr")

_FIXTURE = json.loads(FIXTURE.read_text(encoding="utf-8"))
TAXONOMY: Dict[str, Any] = _FIXTURE["taxonomy"]
PRODUCTS: List[Dict[str, Any]] = _FIXTURE["products"]
NEUTRAL_NODES: List[str] = _FIXTURE["language_neutral_nodes"]
FOREIGN_ONLY: List[str] = _FIXTURE["foreign_only_controls"]

# One real product per case, named so a failure says which case broke.
SAKE_PRODUCT = "0084391434088"      # xx:sake is its most specific tag
TOFU_PRODUCTS = ("0018513003388", "0025484000124", "0025484006577")
SPREAD_PRODUCT = "0000101209159"    # carries fr:pates-a-tartiner
IPA_PRODUCT = "0013189953166"       # carries ca:cervesa-ipa


def _input_record(product: Dict[str, Any]) -> Dict[str, Any]:
    """One real product's tags, wrapped so it clears the pre-category filters."""
    record: Dict[str, Any] = {
        "code": product["code"],
        "lang": "en",
        "categories_tags": product["categories_tags"],
        "images": {},
    }
    for lang in LANGS:
        record[f"product_name_{lang}"] = f"Product {product['code']}"
        record[f"generic_name_{lang}"] = f"Description for {product['code']}"
        record["images"][f"front_{lang}"] = {
            "rev": "7",
            "sizes": {"400": {"w": 400, "h": 400}},
        }
    return record


def _run_extract(tmp: Path, lang: str) -> Tuple[Dict[str, dict], dict]:
    """Run the real CLI over the fixture; return (records by id, report)."""
    taxonomy_path = tmp / "categories.json"
    taxonomy_path.write_text(json.dumps(TAXONOMY), encoding="utf-8")

    input_path = tmp / "products.jsonl"
    input_path.write_text(
        "".join(json.dumps(_input_record(p)) + "\n" for p in PRODUCTS), encoding="utf-8"
    )

    output_path = tmp / "out.ndjson"
    report_path = tmp / "report.json"

    rc = main(
        [
            "--input", str(input_path),
            "--output", str(output_path),
            "--report", str(report_path),
            "--taxonomy", str(taxonomy_path),
            "--pricing-config", str(PRICING_CONFIG),
            "--lang", lang,
            "--progress-every", "0",
            "--progress-seconds", "0",
            "--yes",
        ]
    )
    assert rc == 0, f"extractor exited {rc}"

    records = {
        json.loads(line)["id"]: json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return records, report


def _extract_all() -> Dict[str, Tuple[Dict[str, dict], dict]]:
    out: Dict[str, Tuple[Dict[str, dict], dict]] = {}
    for lang in LANGS:
        with tempfile.TemporaryDirectory(prefix="off-neutral-test-") as d:
            out[lang] = _run_extract(Path(d), lang)
    return out


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------

def test_the_keep_prefixes_admit_xx_in_every_catalog() -> None:
    """``xx`` rides alongside English and the persona language, in every locale.

    Asserted as an exact set so a future widening — say, admitting every prefix
    — fails here rather than being absorbed by the reachability tests below.
    """
    assert default_keep_prefixes("en") == {"en", "xx"}
    assert default_keep_prefixes("es") == {"en", "es", "xx"}
    assert default_keep_prefixes("fr") == {"en", "fr", "xx"}
    assert default_keep_prefixes("") == {"en", "xx"}


def test_every_language_neutral_node_is_eligible_in_every_catalog() -> None:
    """All 34, not just the one the products happen to carry."""
    assert len(NEUTRAL_NODES) == 34, len(NEUTRAL_NODES)
    for lang in LANGS:
        eligible = eligible_nodes(
            TAXONOMY, keep_prefixes=default_keep_prefixes(lang), exclude=EXCLUDE
        )
        missing = [n for n in NEUTRAL_NODES if n not in eligible]
        assert not missing, f"{lang}: {len(missing)} language-neutral nodes refused: {missing}"


# ---------------------------------------------------------------------------
# Reachability — the half the defect broke
# ---------------------------------------------------------------------------

def test_a_language_neutral_node_resolves_into_a_path_in_every_catalog() -> None:
    """``xx:sake`` reaches its real address in en, es and fr.

    The whole path is asserted, not merely that the last segment appeared: a
    leaf that resolved at some other address would be a different defect wearing
    the same green tick. Before the change every one of these is ``[]``.
    """
    expected = {
        "en": [
            "Beverages and beverages preparations",
            "Beverages and beverages preparations/Beverages",
            "Beverages and beverages preparations/Beverages/Alcoholic beverages",
            "Beverages and beverages preparations/Beverages/Alcoholic beverages/Sake",
        ],
        "es": [
            "Bebidas y preparaciones de bebidas",
            "Bebidas y preparaciones de bebidas/Bebidas",
            "Bebidas y preparaciones de bebidas/Bebidas/Bebidas alcohólicas",
            "Bebidas y preparaciones de bebidas/Bebidas/Bebidas alcohólicas/Sake",
        ],
        "fr": [
            "Boissons et préparations de boissons",
            "Boissons et préparations de boissons/Boissons",
            "Boissons et préparations de boissons/Boissons/Boissons alcoolisées",
            "Boissons et préparations de boissons/Boissons/Boissons alcoolisées/Saké",
        ],
    }
    for lang in LANGS:
        got = build_category_path(["xx:sake"], TAXONOMY, EXCLUDE, lang=lang)
        assert got == expected[lang], f"{lang}: {got}"


def test_the_extractor_files_a_real_product_under_a_language_neutral_leaf() -> None:
    """End to end, through the CLI, on a real product's verbatim tags.

    ``0084391434088`` carries ``en:beverages-and-beverages-preparations``,
    ``en:beverages``, ``en:alcoholic-beverages`` and ``xx:sake``. Before the
    change its chain stopped at ``Alcoholic beverages`` — the ``xx:`` tag was
    refused, so the product lost the only specificity it had.
    """
    runs = _extract_all()
    for lang in LANGS:
        records, _report = runs[lang]
        record = records[SAKE_PRODUCT.zfill(13)]
        leaf_label = display_label(TAXONOMY, "xx:sake", lang)
        assert record["category_path"][-1].endswith("/" + leaf_label), record["category_path"]
        assert len(record["category_path"]) == 4, record["category_path"]
        assert leaf_label in record["taxonomy_tags"], record["taxonomy_tags"]


def test_a_language_neutral_node_reaches_the_flat_field_of_a_real_product() -> None:
    """The other emitted surface, on the tag that actually occurs in the dump.

    Upstream retired ``en:tofu`` in favour of ``xx:tofu`` and the shipped alias
    map follows it, so a product tagged ``en:tofu`` arrives at the vocabulary
    check as ``xx:tofu`` and was refused there — the alias moved the tag from the
    ``not_in_taxonomy`` bucket into the ``out_of_language`` one and no further.

    The expected value is read from ``display_label`` rather than written out,
    because what is under test is whether the node is *eligible*; how it is cased
    is a separate rule with its own tests.
    """
    runs = _extract_all()
    for lang in LANGS:
        records, report = runs[lang]
        label = display_label(TAXONOMY, "xx:tofu", lang)
        for code in TOFU_PRODUCTS:
            record = records[code.zfill(13)]
            assert label in record["taxonomy_tags"], (lang, code, record["taxonomy_tags"])
        refused = {
            entry["tag"]
            for entry in report["category_tag_curation"]["top_out_of_language_tags"]
        }
        assert "xx:tofu" not in refused, (lang, sorted(refused))


# ---------------------------------------------------------------------------
# The control — the half a fix that admitted everything would break
# ---------------------------------------------------------------------------

def test_a_foreign_only_node_is_still_refused_as_a_leaf() -> None:
    """The language filter still discriminates, and still does so per locale.

    ``fr:pates-a-tartiner`` is unfilable by an English or a Spanish catalog and
    filable by the French one. Asserting only the refusals would also pass on a
    filter that had stopped keeping anything at all.
    """
    for node in ("fr:pates-a-tartiner", "fr:charcuteries-cuites"):
        assert build_category_path([node], TAXONOMY, EXCLUDE, lang="en") == []
        assert build_category_path([node], TAXONOMY, EXCLUDE, lang="es") == []
        assert build_category_path([node], TAXONOMY, EXCLUDE, lang="fr") != []

    # Neither English, nor xx, nor any persona language of this project.
    for lang in LANGS:
        assert build_category_path(["ca:cervesa-ipa"], TAXONOMY, EXCLUDE, lang=lang) == [], lang


def test_the_extractor_still_refuses_a_foreign_only_tag_end_to_end() -> None:
    """Through the CLI: refused as a flat value, and counted under its own reason."""
    runs = _extract_all()

    for lang, node, code in (
        ("en", "fr:pates-a-tartiner", SPREAD_PRODUCT),
        ("es", "fr:pates-a-tartiner", SPREAD_PRODUCT),
        ("en", "ca:cervesa-ipa", IPA_PRODUCT),
        ("fr", "ca:cervesa-ipa", IPA_PRODUCT),
    ):
        records, report = runs[lang]
        record = records[code.zfill(13)]
        label = display_label(TAXONOMY, node, lang)
        assert label not in record["taxonomy_tags"], (lang, node, record["taxonomy_tags"])
        refused = {
            entry["tag"]
            for entry in report["category_tag_curation"]["top_out_of_language_tags"]
        }
        assert node in refused, (lang, node, sorted(refused))

    # ... and the French catalog does file the French node, so the refusals above
    # are the filter working rather than the filter having stopped keeping things.
    records, _report = runs["fr"]
    record = records[SPREAD_PRODUCT.zfill(13)]
    assert display_label(TAXONOMY, "fr:pates-a-tartiner", "fr") in record["taxonomy_tags"]


# ---------------------------------------------------------------------------
# The forest still has one address per category and one chain per product
# ---------------------------------------------------------------------------

def test_admitting_the_neutral_nodes_leaves_one_address_per_category() -> None:
    """No node gains a second address and no label gains a second owner.

    The canonical parent map is language-blind and unchanged by this, so no
    node's ancestry can move; what changes is which nodes a chain may end at.
    That argument is the reason to check rather than a substitute for checking,
    since a label collision between a newly admitted node and an existing one
    would put one string at two addresses in the built index.
    """
    parents = build_canonical_parent_map(TAXONOMY, exclude=EXCLUDE)
    for lang in LANGS:
        eligible = eligible_nodes(
            TAXONOMY, keep_prefixes=default_keep_prefixes(lang), exclude=EXCLUDE
        )
        owners: Dict[str, List[str]] = {}
        for node in sorted(eligible):
            ancestry = canonical_ancestry(parents, node)
            assert ancestry and ancestry[-1] == node, (lang, node, ancestry)
            label = display_label(TAXONOMY, node, lang)
            owners.setdefault(label, []).append(node)
        shared = {label: ids for label, ids in owners.items() if len(ids) > 1}
        assert not shared, f"{lang}: label claimed by more than one category: {shared}"


def test_the_run_reports_no_addressing_or_labelling_conflict() -> None:
    """The extractor's own audit, over the records it actually wrote."""
    runs = _extract_all()
    for lang in LANGS:
        _records, report = runs[lang]
        counters = report["counters"]
        assert counters["categories_at_multiple_addresses"] == 0, lang
        assert counters["categories_under_multiple_labels"] == 0, lang
        assert counters["labels_shared_by_multiple_categories"] == 0, lang
        assert counters["written"] == len(PRODUCTS), (lang, counters["written"])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
