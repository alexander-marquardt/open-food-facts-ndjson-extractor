"""``description`` carries prose only, and the facts worth reaching get fields.

Two halves of one change, asserted together because either one alone is a
regression:

* ``build_description`` no longer appends the ``Key specifications:`` run of
  eighteen ``attrs`` entries. ``description`` is the title plus the product's
  own source text (``generic_name_<lang>`` or ``ingredients_text_<lang>``).
* The seven attributes worth reaching *exactly* are written as top-level fields,
  verbatim from the ``attrs`` entry they are taken from.

Removing the block without adding the fields would leave a window in which a
fact like ``no-gluten`` is reachable by neither path. Here they land in one
commit and one module holds both assertions, so neither can be reverted without
the other going red.

Why the block had to go, in one line each: it was padding for a thin tail that
is 1% of the catalog, it made up roughly three quarters of the median
description, BM25 normalises by field length so it down-weighted the real text
it was glued to, and its label words matched every document in the catalog —
``description ~ "Allergens"`` and ``~ "Nutri-Score"`` each returned all 108,379.

Why these seven and no more
---------------------------
``Labels``, ``Allergens``, ``Countries`` and ``Ingredients analysis`` are
shopper-facing facts that belong in recall. ``Nutri-Score``, ``Eco-Score`` and
``NOVA group`` are emitted as **data only**: present and targetable, and
deliberately not dressed up. The consuming search project's config records why —
this corpus carries per-serving figures in per-100g fields, so six SKUs record
``Sugars = 0 g/100g`` and compute Nutri-Score A for a ~94%-sugar confectionery.
Emitting the value is honest; presenting it as a graded health claim would not
be. That decision lives on the mapping and field-map side; what this module can
pin is the half it can see, which is that the value emitted is the source's own
and is not re-cased, re-labelled or "improved" on the way out.

The merchandising internals (margin, pricing bucket, the price and popularity
provenance stamps) and the nutrition numerics stay in ``attrs`` only, and
:func:`test_the_promote_list_is_exactly_these_seven` enumerates both directions
so an eighth field cannot arrive unannounced and one of the seven cannot quietly
leave.

``Category`` and ``Dietary restrictions`` are not promoted, and that is a
decision rather than an omission — see
:func:`test_category_is_not_promoted_because_the_document_already_carries_it`
and :func:`test_dietary_restrictions_is_not_given_a_second_spelling`.

``attrs`` keeps every promoted key
-----------------------------------
The blob is the inspection surface, and live readers downstream target it by
name: a business signal filters on ``attrs.NOVA group``, a merchandising script
derives the modelled margin from ``attrs.Labels``, and a retailer plugin
iterates ``attrs.items()`` generically. Dropping a key would break all three
silently — a ``terms`` clause that matches nothing raises nothing.
:func:`test_attrs_still_carries_every_promoted_key` pins that.

The data is real. ``fixtures/off_real_multivalue_attrs.json`` holds three
products from the public Open Food Facts JSONL export, reduced to the keys the
extractor reads and otherwise verbatim, plus the ancestor closure of their
category tags from the public taxonomy. They are reused here rather than
duplicated because they already cover what this change needs: a product with
seven labels and two allergens, one whose ``Countries`` prose carries a comma
with no following space, and one that carries no ``labels_tags`` at all and so
witnesses an omitted field. Two of the three carry no ``Eco-Score``, which is
the same witness for a scalar.

The tests drive the real ``extract.main()`` CLI and assert on written records,
because these fields are a property of the emitted document.

Run with ``pytest tests/`` or directly: ``python tests/test_promoted_attr_fields.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from off_demo_extract.extract import (  # noqa: E402
    PROMOTED_ATTR_FIELDS,
    build_description,
    get_description,
    main,
    promoted_attr_fields,
)

PRICING_CONFIG = REPO_ROOT / "config" / "pricing_buckets.json"

_FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "off_real_multivalue_attrs.json").read_text(
        encoding="utf-8"
    )
)
SOURCE_PRODUCTS: List[dict] = _FIXTURE["products"]
TAXONOMY: Dict[str, dict] = _FIXTURE["taxonomy"]

MULTI_LABEL = "0022506002357"
COMMA_IN_SCALARS = "0065633074712"
NO_LABELS = "0011152145969"

# The block that used to be appended to every description.
SPEC_MARKER = "Key specifications"

# The attribute keys that reach no top-level field at all. Enumerated rather than
# derived from the emitted document, so a key that stops being written is a
# failure here rather than a silently shorter list.
ATTRS_ONLY = [
    # Merchandising internals — retailer-private, never in shopper-facing recall.
    "Modelled margin",
    "Margin source",
    "Price source",
    "Pricing bucket",
    "Popularity source",
    "Estimated unit price",
    "Unique scans (Open Food Facts)",
    # Numerics — a number rendered as text matches no useful query.
    "Energy (kcal/100g)",
    "Fat (g/100g)",
    "Saturated fat (g/100g)",
    "Sugars (g/100g)",
    "Salt (g/100g)",
    "Protein (g/100g)",
    "Fiber (g/100g)",
    "Serving size",
    "Quantity",
]

# The two attributes a top-level field already carries, under a name of its own:
# ``Category`` is ``taxonomy_tags[0]``, and ``Dietary restrictions`` is the
# ``dietary_restrictions`` field. Neither is promoted, and neither is "attrs
# only" either — which is why they are listed apart from :data:`ATTRS_ONLY`.
ALREADY_A_FIELD = ["Category", "Dietary restrictions"]


def _run_extract(tmp: Path) -> List[dict]:
    """Run the real CLI over the fixture products; return the written records."""
    taxonomy_path = tmp / "categories.json"
    taxonomy_path.write_text(json.dumps(TAXONOMY), encoding="utf-8")

    input_path = tmp / "products.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in SOURCE_PRODUCTS) + "\n",
        encoding="utf-8",
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
        ]
    )
    assert rc == 0, f"extractor exited {rc}"

    return [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tmpdir() -> Any:
    return tempfile.TemporaryDirectory(prefix="off-promoted-attr-fields-test-")


def _records() -> Dict[str, dict]:
    with _tmpdir() as tmp:
        records = _run_extract(Path(tmp))
    assert len(records) == len(SOURCE_PRODUCTS), (
        f"expected {len(SOURCE_PRODUCTS)} records, got {len(records)}"
    )
    return {r["id"]: r for r in records}


def _source(code: str) -> dict:
    for product in SOURCE_PRODUCTS:
        if product["code"] == code:
            return product
    raise AssertionError(f"{code} not in fixture")


# ----------------------------
# §1 — description carries no attributes
# ----------------------------


def test_no_description_carries_the_attribute_block() -> None:
    """The ``Key specifications:`` run is gone from every emitted description.

    Red on the parent commit, where all three fixture products carry it.
    """
    offenders = {
        code: record["description"]
        for code, record in _records().items()
        if SPEC_MARKER in record["description"]
    }
    assert not offenders, offenders


def test_a_description_is_its_title_followed_by_the_products_own_text() -> None:
    """What is left is prose, and it *ends* where the source text ends.

    The block used to sit at the end, so an assertion on the description's prefix
    alone would have been green before the change too. This one pins the whole
    string against the source record, and the equality is the statement: there is
    nothing after the product's own text.
    """
    for code, record in _records().items():
        text = get_description(_source(code), "en")
        assert text, f"{code}: fixture product has no source text to compare against"
        assert record["description"] == build_description(title=record["title"], desc=text), (
            f"{code}: {record['description']!r}"
        )
        assert record["description"].endswith(text), code


def test_the_emitted_description_of_a_real_product_in_full() -> None:
    """One document's description, written out, so the change is readable.

    The literal is what the extractor emits for a real Open Food Facts record.
    Before this change the same product's description continued with ``Key
    specifications: Category: …; Nutri-Score: E; NOVA group: 3; Eco-Score: C;
    Dietary restrictions: kosher, organic; Allergens: eggs, mustard; …`` for
    another ~400 characters.
    """
    assert _records()[MULTI_LABEL]["description"] == (
        "Organic Mayonnaise. Organic expeller pressed soy and/or canola oil, organic "
        "whole eggs, organic egg yolks, organic distilled white vinegar, organic honey, "
        "filtered water, sea salt, organic mustard (organic distilled vinegar, water, "
        "organic mustard seed, salt, organic spices), organic lemon juice concentrate."
    )


def test_the_attribute_label_words_no_longer_match_every_document() -> None:
    """The words that used to be in all 108,379 descriptions are in none of them.

    ``Allergens`` and ``Nutri-Score`` were label words of the block, not product
    prose. A term present in every document cannot discriminate between any two
    of them; it can only dilute the text it shares a field with.
    """
    for code, record in _records().items():
        description = record["description"]
        for label in ("Allergens:", "Nutri-Score:", "NOVA group:", "Ingredients analysis:"):
            assert label not in description, f"{code}: {label!r} still in {description!r}"


def test_build_description_takes_no_attributes_at_all() -> None:
    """The signature is the guard: attrs cannot reach the field by any argument.

    A version that merely stopped *rendering* them would leave the door open for
    the next caller to pass a flag and put them back.
    """
    assert build_description(title="Olive oil", desc="Extra virgin olive oil.") == (
        "Olive oil. Extra virgin olive oil."
    )
    # The double-period guard is the one piece of the old renderer that stays.
    assert build_description(title="Olive oil.", desc="Extra virgin.") == (
        "Olive oil. Extra virgin."
    )


# ----------------------------
# §3 — the meaningful subset gets fields
# ----------------------------


def test_every_promoted_field_is_the_attrs_value_verbatim() -> None:
    """Each field equals the ``attrs`` entry it came from, on every record.

    This is the whole faithfulness rule in one assertion: no cleaning, no
    normalising, no re-casing. A field that "improved" its value would disagree
    with the blob the value was taken from, and the document would then say two
    different things about one fact.
    """
    for code, record in _records().items():
        for key, field in PROMOTED_ATTR_FIELDS.items():
            if key in record["attrs"]:
                assert record[field] == record["attrs"][key], f"{code} {field}"
            else:
                assert field not in record, f"{code}: {field} written without attrs[{key!r}]"


def test_the_promoted_values_are_the_ones_open_food_facts_supplied() -> None:
    """Named literals, so a silent change of source field is visible here.

    ``test_every_promoted_field_is_the_attrs_value_verbatim`` proves the field
    agrees with ``attrs``; it would stay green if *both* changed together. These
    are the values themselves.
    """
    record = _records()[MULTI_LABEL]

    assert record["labels"] == [
        "no-gluten",
        "organic",
        "kosher",
        "no-gmos",
        "usda-organic",
        "non-gmo-project",
        "orthodox-union-kosher",
    ], record["labels"]
    assert record["allergens"] == ["eggs", "mustard"], record["allergens"]
    assert record["ingredients_analysis"] == [
        "palm-oil-free",
        "non-vegan",
        "vegetarian-status-unknown",
    ], record["ingredients_analysis"]
    assert record["countries"] == "United States, World", record["countries"]
    assert record["nutri_score"] == "E", record["nutri_score"]
    assert record["eco_score"] == "C", record["eco_score"]
    assert record["nova_group"] == "3", record["nova_group"]


def test_each_label_is_its_own_value_so_an_exact_query_reaches_it() -> None:
    """The list shape is what makes the field worth having.

    ``no-gluten`` is the second most frequent label in the catalog and reached
    4,435 of its 12,229 carriers through ``attrs`` when the values were joined
    (#44, #45). One value per element is what a ``terms`` filter needs.
    """
    labels = _records()[MULTI_LABEL]["labels"]
    assert isinstance(labels, list), type(labels)
    assert "no-gluten" in labels, labels
    for value in labels:
        assert "," not in value, f"{value!r} is a combination, not a value"


def test_a_promoted_scalar_is_not_split_on_its_commas() -> None:
    """``countries`` is prose, and it comes out whole.

    ``0011152145969`` writes its commas with **no following space** and names
    France twice, in two languages, one of them as a raw tag id. Splitting it
    would file that product under three countries, one of which is not a country
    name. The free-text-versus-``countries_tags`` question is tracked in #50 and
    is not decided here; what is pinned here is that promotion changed neither
    the source field nor the value.
    """
    records = _records()
    assert records[NO_LABELS]["countries"] == "France,États-Unis,en:france"
    assert records[COMMA_IN_SCALARS]["countries"] == "Canada, France"
    for record in records.values():
        assert isinstance(record["countries"], str), type(record["countries"])


def test_an_absent_attribute_omits_its_field_rather_than_writing_an_empty_value() -> None:
    """Both shapes of "Open Food Facts does not carry this", on real products.

    ``0011152145969`` has no ``labels_tags``, so ``labels`` is absent — not
    ``[]``, which would claim the source was read and found empty. Two of the
    three products have no ``ecoscore_grade``, so ``eco_score`` is absent — not
    ``""``, which Elasticsearch would index as a real, empty keyword term.
    """
    records = _records()

    assert not _source(NO_LABELS).get("labels_tags"), "fixture product gained labels"
    assert "labels" not in records[NO_LABELS], records[NO_LABELS].get("labels")

    without_eco = [code for code, record in records.items() if "eco_score" not in record]
    assert sorted(without_eco) == sorted([COMMA_IN_SCALARS, NO_LABELS]), without_eco
    for code in without_eco:
        assert "Eco-Score" not in records[code]["attrs"], code


def test_attrs_still_carries_every_promoted_key() -> None:
    """Promotion adds a field; it does not move one.

    Three live readers target ``attrs`` by name — a business signal filtering on
    ``attrs.NOVA group``, the modelled-margin script reading ``attrs.Labels``,
    and a retailer plugin iterating ``attrs.items()``. Each fails silently on a
    missing key, so the loss would show up as a boost that stopped applying
    rather than as an error.
    """
    for code, record in _records().items():
        attrs = record["attrs"]
        for field, key in ((f, k) for k, f in PROMOTED_ATTR_FIELDS.items()):
            if field in record:
                assert key in attrs, f"{code}: {field} was promoted out of attrs, not alongside it"
        assert record["attr_keys"] == sorted(attrs), code


def test_the_promote_list_is_exactly_these_seven() -> None:
    """Enumerated in both directions, so neither list can drift alone.

    An eighth field arriving is a decision — the mapping side has to declare it,
    the field map has to display it — and this is where that decision has to be
    written down rather than discovered in a built index.
    """
    assert PROMOTED_ATTR_FIELDS == {
        "Labels": "labels",
        "Allergens": "allergens",
        "Countries": "countries",
        "Ingredients analysis": "ingredients_analysis",
        "Nutri-Score": "nutri_score",
        "Eco-Score": "eco_score",
        "NOVA group": "nova_group",
    }

    for code, record in _records().items():
        for key in ATTRS_ONLY + ALREADY_A_FIELD:
            assert key not in PROMOTED_ATTR_FIELDS, key
        for key in ATTRS_ONLY:
            if key not in record["attrs"]:
                continue
            # The merchandising internals and the numerics reach no top-level
            # field, under any spelling a reader would recognise.
            spelling = key.split(" (")[0].lower().replace(" ", "_").replace("-", "_")
            assert spelling not in record, (
                f"{code}: {key!r} reached a top-level field {spelling!r}"
            )


def test_category_is_not_promoted_because_the_document_already_carries_it() -> None:
    """``attrs['Category']`` is ``taxonomy_tags[0]``, and the path joins to it.

    Confirmed rather than assumed: a promoted ``category`` field would be a third
    spelling of a fact this document already carries twice, and the hierarchical
    facet reads ``category_path``, not an attribute.
    """
    assert "Category" not in PROMOTED_ATTR_FIELDS
    for code, record in _records().items():
        assert "category" not in record, code
        assert record["attrs"]["Category"] == record["taxonomy_tags"][0], code
        leaf = record["category_path"][-1].split("/")[-1]
        assert leaf in record["taxonomy_tags"], f"{code}: {leaf}"


def test_dietary_restrictions_is_not_given_a_second_spelling() -> None:
    """The existing field stays the only one, and its inputs are now fields too.

    ``dietary_restrictions`` is derived from ``labels_tags`` and
    ``ingredients_analysis_tags`` — the same two sources ``labels`` and
    ``ingredients_analysis`` are read from — so every value it carries is
    traceable to one of those two fields on the same document. That relationship
    is what makes a second promoted spelling redundant rather than merely
    duplicative.
    """
    assert "Dietary restrictions" not in PROMOTED_ATTR_FIELDS

    for code, record in _records().items():
        assert isinstance(record["dietary_restrictions"], list), code
        traceable = set(record.get("labels", [])) | set(record.get("ingredients_analysis", []))
        for value in record["dietary_restrictions"]:
            # ``gluten_free``/``lactose_free`` are underscored renames of a source
            # token; the rest are the token itself. Both spellings are checked so
            # this stays a statement about provenance rather than about casing.
            assert value in traceable or value.replace("_", "-") in traceable, (
                f"{code}: dietary_restrictions carries {value!r}, which neither "
                "labels nor ingredients_analysis on this document explains"
            )


def test_a_promoted_list_does_not_share_its_object_with_attrs() -> None:
    """Two keys of one document must never alias one mutable list.

    The same rule ``attrs['Dietary restrictions']`` already follows. Asserted on
    the function rather than on the written record, because JSON round-tripping
    would hide the aliasing.
    """
    attrs: Dict[str, Any] = {"Labels": ["organic"], "Countries": "United States"}
    fields = promoted_attr_fields(attrs)

    assert fields == {"labels": ["organic"], "countries": "United States"}
    fields["labels"].append("mutated")
    assert attrs["Labels"] == ["organic"], attrs["Labels"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
