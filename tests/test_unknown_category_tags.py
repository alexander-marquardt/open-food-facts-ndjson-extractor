"""End-to-end tests: a non-taxonomy product tag must never reach ``categories``.

These run the real ``extract.main()`` CLI path over **real** Open Food Facts tag
data. ``fixtures/off_unknown_category_tags.json`` holds the ``categories_tags``
of eleven real products taken verbatim from the public JSONL export — including the
non-taxonomy tags that are the whole point — together with the ancestor closure
of those tags from the public category taxonomy. Only the surrounding product
envelope (title, description, image) is synthesised, so that every fixture record
clears the extractor's earlier filters and the *only* thing that can separate
them is category handling.

The eleven cases, and what each is for:

===========================  =======================================================
``groceries_with_lineage``   the 87% case — a complete valid lineage plus one junk
                             tag riding along. ``en:groceries`` was searchable on
                             6,299 documents of the built English catalog.
``salted_snacks_only``       a retired id as the product's *only* tag. Before the
                             alias map its path did not resolve and the product was
                             dropped outright by the category_path gate.
``salted_snacks_with_lineage`` a retired id and a junk tag on the same product.
``easter_food``              a retired id whose successor upstream still lists it
                             as a synonym.
``aoc_cheeses``              two label-attributes masquerading as categories, on a
                             product whose cheese lineage is intact.
``out_of_language``          a genuine French taxonomy node in an English catalog:
                             a real node that ``category_path`` already refuses.
``all_unknown``              nothing usable at all — the 2.5% that keep nothing
                             even when only values are dropped.
``clean_control``            no refusals; proves the curation is not eating
                             ordinary products.
``baking_decorations``       a retired id that strands 991 of its 1,098 carriers.
``long_tail_unknown``        ``en:salads``: a real category name with real volume
                             and no successor anywhere, now a recorded drop (#19).
``long_tail_typo``           ``en:potato-crips``: a contributor's misspelling. This
                             is the tail the report's top-N has to surface — a tag
                             nobody curated, as opposed to one we chose to refuse.
===========================  =======================================================

Run with ``pytest tests/`` or directly: ``python tests/test_unknown_category_tags.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from off_demo_extract.extract import main, pad_gtin13  # noqa: E402
from off_demo_extract.taxonomy import display_label  # noqa: E402

PRICING_CONFIG = REPO_ROOT / "config" / "pricing_buckets.json"

_FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "off_unknown_category_tags.json").read_text(
        encoding="utf-8"
    )
)
TAXONOMY: Dict[str, dict] = _FIXTURE["taxonomy"]
PRODUCTS: List[dict] = _FIXTURE["products"]
BY_CASE: Dict[str, dict] = {p["case"]: p for p in PRODUCTS}

# Tags in the fixture that must not survive into the indexed catalog, and why.
# Kept here rather than imported so this file exercises only the public CLI.
MUST_BE_REFUSED = {
    "en:groceries": "contentless catch-all",
    "en:aoc-cheeses": "label attribute, not a category",
    "en:labeled-cheeses": "label attribute, not a category",
    "en:empty": "placeholder",
    "en:proposed-for-deletion": "upstream maintenance marker",
    "en:salads": "real category name, no successor anywhere — a recorded drop",
    "en:potato-crips": "contributor typo — the anonymous tail nobody curated",
    "fr:pates-a-tartiner": "real node, wrong language for an English catalog",
}
MUST_BE_ALIASED = {
    "en:salted-snacks": "en:salty-snacks",
    "en:easter-food": "en:easter-foods-and-drinks",
    "en:baking-decorations": "en:food-decorations",
}


def _label(tag: str) -> str:
    """The label the extractor would render for ``tag`` in an English catalog."""
    return display_label(TAXONOMY, tag, "en")


# Every label an English catalog built on this taxonomy is allowed to emit.
# ``display_label`` is the extractor's own — and only — labeller, so this is an
# exact reverse-map of the indexed field onto the taxonomy, not an approximation.
ALLOWED_LABELS = {_label(n) for n in TAXONOMY if n.startswith("en:")}


def _envelope(product: dict) -> dict:
    """Wrap real tags in a product that clears every non-category filter."""
    return {
        "code": product["code"],
        "lang": "en",
        "product_name_en": f"Fixture item {product['case']}",
        "generic_name_en": "A product whose category tags come from the real export.",
        "categories_tags": product["categories_tags"],
        "images": {"front_en": {"rev": "3", "sizes": {"400": {"w": 400, "h": 400}}}},
    }


def _run_extract(tmp: Path, *extra_args: str) -> tuple[Dict[str, dict], dict]:
    """Run the extractor over the fixture; return ({id: record}, report)."""
    taxonomy_path = tmp / "categories.json"
    taxonomy_path.write_text(json.dumps(TAXONOMY), encoding="utf-8")

    input_path = tmp / "products.jsonl"
    input_path.write_text(
        "".join(json.dumps(_envelope(p)) + "\n" for p in PRODUCTS), encoding="utf-8"
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
            "--progress-every", "0",
            "--progress-seconds", "0",
            "--yes",
            *extra_args,
        ]
    )
    assert rc == 0, f"extractor exited {rc}"

    records = {
        json.loads(line)["id"]: json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return records, json.loads(report_path.read_text(encoding="utf-8"))


def _tmpdir() -> Any:
    return tempfile.TemporaryDirectory(prefix="off-unknown-tags-test-")


def _id(case: str) -> str:
    return pad_gtin13(BY_CASE[case]["code"])


def test_fixture_still_carries_the_offending_tags() -> None:
    """Guard on the guards below: they pass vacuously on a sanitised fixture.

    Every assertion in this file is about what happens to a tag that is not a
    usable taxonomy node. If a future fixture refresh dropped those tags, the
    tests would go green while testing nothing.
    """
    present = {t for p in PRODUCTS for t in p["categories_tags"]}
    for tag in [*MUST_BE_REFUSED, *MUST_BE_ALIASED]:
        assert tag in present, f"fixture no longer carries {tag}; these tests are now vacuous"
    for tag in MUST_BE_REFUSED:
        assert tag not in TAXONOMY or not tag.startswith("en:"), (
            f"{tag} is an eligible node of the fixture taxonomy, so refusing it "
            "would be wrong — the fixture and the expectation disagree"
        )


def test_no_indexed_category_falls_outside_the_taxonomy() -> None:
    """The defect, stated as a property: every emitted label maps back to a node.

    A label that does not reverse-map is a value that can be searched but can
    never be a facet path or a policy value — which is exactly what
    ``Groceries`` was on 6,299 documents.
    """
    with _tmpdir() as d:
        records, _ = _run_extract(Path(d))

    offenders = {
        record_id: sorted(set(rec["categories"]) - ALLOWED_LABELS)
        for record_id, rec in records.items()
        if set(rec["categories"]) - ALLOWED_LABELS
    }
    assert not offenders, (
        "categories that do not reverse-map onto an eligible taxonomy node: "
        f"{json.dumps(offenders, ensure_ascii=False, indent=2)}"
    )


def test_a_refused_tag_reaches_no_searchable_field() -> None:
    """``categories`` is not the only indexed field a tag can leak into.

    The primary tag also becomes ``attrs["Category"]``, which is folded into the
    generated ``description``. Validating only the flat list would leave the same
    junk searchable by another route.
    """
    with _tmpdir() as d:
        records, _ = _run_extract(Path(d))

    for tag in MUST_BE_REFUSED:
        label = _label(tag)
        for record_id, rec in records.items():
            assert label not in rec["categories"], f"{record_id}: {label!r} in categories"
            assert rec["attrs"].get("Category") != label, f"{record_id}: {label!r} in attrs"
            assert label not in rec["description"], f"{record_id}: {label!r} in description"


def test_the_record_survives_its_junk_tag() -> None:
    """Drop the value, never the record — the 87% case, end to end."""
    with _tmpdir() as d:
        records, _ = _run_extract(Path(d))

    rec = records.get(_id("groceries_with_lineage"))
    assert rec is not None, "a product was dropped for carrying one junk tag"
    assert _label("en:groceries") not in rec["categories"]
    assert _label("en:spices") in rec["categories"], rec["categories"]
    assert rec["category_path"], "the surviving record lost its hierarchy"


def test_a_renamed_tag_is_aliased_and_rescues_its_product() -> None:
    """The alias map is not cosmetic: it puts back products the gate was dropping.

    ``salted_snacks_only`` carries one retired id and nothing else. Without the
    alias its chain is empty and the default category_path gate drops it —
    2,387 of that tag's 2,998 carriers are in the same position.
    """
    with _tmpdir() as d:
        records, _ = _run_extract(Path(d))

    rec = records.get(_id("salted_snacks_only"))
    assert rec is not None, "the aliased product did not survive the category_path gate"
    assert rec["categories"] == [_label(MUST_BE_ALIASED["en:salted-snacks"])], rec["categories"]
    assert rec["category_path"], "aliased product has no hierarchy"

    for case, successor in (
        ("easter_food", MUST_BE_ALIASED["en:easter-food"]),
        ("baking_decorations", MUST_BE_ALIASED["en:baking-decorations"]),
    ):
        aliased = records.get(_id(case))
        assert aliased is not None, f"{case} was dropped"
        assert _label(successor) in aliased["categories"], (case, aliased["categories"])


def test_only_a_product_with_nothing_left_is_lost() -> None:
    """Exactly one fixture product has no usable tag at all; only it is dropped."""
    with _tmpdir() as d:
        records, report = _run_extract(Path(d))

    expected = {_id(p["case"]) for p in PRODUCTS} - {_id("all_unknown")}
    assert set(records) == expected, sorted(set(records) ^ expected)
    assert report["counters"]["missing_category_path"] == 1, report["counters"]


def test_the_report_surfaces_the_unknown_tag_rate() -> None:
    """Per-run visibility: the rate is a number in the report, not a hand audit.

    This defect was found by reverse-mapping a built index against the taxonomy.
    The point of these counters is that nobody has to do that again.
    """
    with _tmpdir() as d:
        _, report = _run_extract(Path(d))

    curation = report["category_tag_curation"]
    assert curation["products_with_tags"] == len(PRODUCTS)
    assert curation["tag_instances"] == sum(len(p["categories_tags"]) for p in PRODUCTS)
    assert curation["accepted_instances"] + curation["rejected_instances"] == curation["tag_instances"]
    assert curation["aliased_instances"] == 4, curation
    assert curation["products_with_no_accepted_tag"] == 1, curation
    assert curation["rejected_by_reason"] == {
        "curated_drop": 7,
        "not_in_taxonomy": 1,
        "out_of_language": 1,
    }, curation
    # ``en:salads`` is a *recorded* drop, so the tail this asserts on has to be a
    # tag nobody chose — ``en:potato-crips``, a contributor's typo, verbatim from
    # the export. Asserting the rate on a curated tag would make this test agree
    # with itself: every drop we ever record would leave it green while the
    # not_in_taxonomy path went unexercised.
    assert curation["distinct_unknown_tags"] == 1
    assert curation["top_unknown_tags"] == [{"tag": "en:potato-crips", "instances": 1}]
    assert curation["top_out_of_language_tags"] == [
        {"tag": "fr:pates-a-tartiner", "instances": 1}
    ]
    assert 0 < curation["unknown_tag_rate"] < 1
    assert report["counters"]["refused_category_tags"] == 9, report["counters"]
    assert report["counters"]["products_with_refused_category_tags"] == 7, report["counters"]


def test_curated_drops_are_reported_with_their_recorded_reason() -> None:
    """'Dropped explicitly, with the reason recorded' has to be visible per run."""
    with _tmpdir() as d:
        _, report = _run_extract(Path(d))

    drops = {row["tag"]: row for row in report["category_tag_curation"]["curated_drops"]}
    assert "en:groceries" in drops, drops
    assert drops["en:groceries"]["instances"] == 2, drops
    for tag, row in drops.items():
        assert row["reason"].strip(), f"{tag} was dropped with no reason recorded"


def test_without_a_taxonomy_the_flat_field_is_not_emptied() -> None:
    """``--no-taxonomy`` has no vocabulary, so it must not refuse everything.

    Refusing every tag with nothing to validate against would blank the flat
    ``categories`` field for the whole run — a worse outcome than the
    unvalidated field this change replaces. The curated drops still apply.
    """
    with _tmpdir() as d:
        records, report = _run_extract(Path(d), "--no-taxonomy")

    assert len(records) == len(PRODUCTS), "the gate ate records with no taxonomy loaded"
    # With no taxonomy there is no ``name`` to render, so labels fall back to the
    # slug — ask the labeller itself rather than assuming the string.
    rec = records[_id("groceries_with_lineage")]
    assert display_label({}, "en:spices", "en") in rec["categories"], rec["categories"]
    assert display_label({}, "en:groceries", "en") not in rec["categories"], rec["categories"]
    assert report["category_tag_curation"]["rejected_by_reason"] == {"curated_drop": 7}


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
