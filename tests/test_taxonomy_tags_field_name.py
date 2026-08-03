"""The flat tag field is emitted as ``taxonomy_tags`` — not ``categories``, not ``tags``.

This drives the real ``extract.main()`` CLI path and reads the key set of an
actual written record, because the field name is a property of the *emitted
document*, not of any function that helps build it. A test that asserted on
:func:`build_taxonomy_tags_list` would keep passing if the ``doc`` dict in
``main`` still wrote the old key.

Three names are in play and all three assertions are load-bearing:

``taxonomy_tags``
    What the field is: the product's own category tags, validated against the
    taxonomy and rendered with its display labels. Display-only — a chocolate
    tagged ``christmas`` is not filed in a "Christmas" category, it just carries
    that attribute.

``categories``
    The name it used to have, and the reason for the rename: ``categories`` is
    also the name of PRISM's **hierarchy facet**, which sources ``category_path``
    and renders the breadcrumb. One name for two different things, one of which
    is not a hierarchy at all.

``tags``
    The obvious short name, and a trap. PRISM's ingest
    (``sources.py``, ``normalized_json_resolve_row``) reads a row's ``tags`` key
    *in preference to* its ``dietary_restrictions`` key and assigns the result to
    the dietary field. A catalog emitting ``tags`` would silently overwrite real
    dietary data with the category tag set — no error, no empty field, just wrong
    values in a facet nobody would think to re-check. :func:`test_the_flat_field_is_not_named_tags`
    exists so that a later "simplification" back to ``tags`` fails here rather
    than in a rebuilt index.

``category_path`` is deliberately asserted *unchanged*: the rename moves the flat
field only, and a run that renamed the hierarchy too would still satisfy the
first two assertions on their own.

Run with ``pytest tests/`` or directly: ``python tests/test_taxonomy_tags_field_name.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from off_demo_extract.extract import main  # noqa: E402

PRICING_CONFIG = REPO_ROOT / "config" / "pricing_buckets.json"


def _n(name: str, parents: list[str]) -> dict:
    return {"name": {"en": name}, "parents": parents}


# Miniature taxonomy: a real root->leaf chain, so the emitted record carries a
# non-empty flat field AND a non-empty category_path and the two can be told
# apart by content, not just by presence.
TAXONOMY = {
    "en:beverages": _n("Beverages", []),
    "en:hot-beverages": _n("Hot beverages", ["en:beverages"]),
    "en:teas": _n("Teas", ["en:hot-beverages"]),
}

CODE = "3017620422003"

# ``labels_tags`` drives ``dietary_restrictions``, which is the field the ``tags``
# name would have collided with. Emitting it non-empty is what makes
# ``test_dietary_restrictions_keeps_its_own_name`` bite rather than pass on an
# absent field.
PRODUCT = {
    "code": CODE,
    "lang": "en",
    "product_name_en": "Earl Grey Tea Bags",
    "generic_name_en": "Black tea with bergamot, 20 bags",
    "categories_tags": ["en:beverages", "en:hot-beverages", "en:teas"],
    "labels_tags": ["en:vegan", "en:organic"],
    "images": {"front_en": {"rev": "7", "sizes": {"400": {"w": 400, "h": 400}}}},
}


def _run_extract(tmp: Path) -> list[dict]:
    """Run the real CLI over one product; return the written records."""
    taxonomy_path = tmp / "categories.json"
    taxonomy_path.write_text(json.dumps(TAXONOMY), encoding="utf-8")

    input_path = tmp / "products.jsonl"
    input_path.write_text(json.dumps(PRODUCT) + "\n", encoding="utf-8")

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
        ]
    )
    assert rc == 0, f"extractor exited {rc}"

    return [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tmpdir() -> Any:
    return tempfile.TemporaryDirectory(prefix="off-taxonomy-tags-name-test-")


def _record() -> dict:
    with _tmpdir() as tmp:
        records = _run_extract(Path(tmp))
    assert len(records) == 1, f"expected one record, got {len(records)}"
    return records[0]


def test_the_flat_field_is_emitted_as_taxonomy_tags() -> None:
    """The written document carries ``taxonomy_tags``, populated."""
    record = _record()
    assert "taxonomy_tags" in record, sorted(record)
    assert record["taxonomy_tags"] == ["Beverages", "Hot beverages", "Teas"], record["taxonomy_tags"]


def test_the_flat_field_is_not_named_categories() -> None:
    """``categories`` is the hierarchy facet's name and must not be emitted.

    Fails on the parent commit, where the same list is written under
    ``categories``.
    """
    record = _record()
    assert "categories" not in record, (
        "the flat tag field is still emitted as 'categories', which is also the "
        f"name of the hierarchy facet that sources category_path: {sorted(record)}"
    )


def test_the_flat_field_is_not_named_tags() -> None:
    """``tags`` would be read into the dietary field by PRISM's ingest.

    ``normalized_json_resolve_row`` prefers a row's ``tags`` key over its
    ``dietary_restrictions`` key. This is a standing guard, not a regression
    test for the rename: it must keep failing any future move to ``tags``.
    """
    record = _record()
    assert "tags" not in record, (
        "emitting 'tags' makes PRISM's ingest overwrite dietary_restrictions "
        f"with the category tag set: {sorted(record)}"
    )


def test_dietary_restrictions_keeps_its_own_name_and_value() -> None:
    """The field ``tags`` would have collided with is present and correct."""
    record = _record()
    assert record["dietary_restrictions"] == ["organic", "vegan"], record["dietary_restrictions"]


def test_the_hierarchy_field_is_unchanged() -> None:
    """``category_path`` is a hierarchy concern and keeps its name.

    Without this, renaming *both* fields would still satisfy the assertions
    above while breaking the breadcrumb facet.
    """
    record = _record()
    assert record["category_path"] == [
        "Beverages",
        "Beverages/Hot beverages",
        "Beverages/Hot beverages/Teas",
    ], record["category_path"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
