"""``attrs`` writes list-sourced attributes as lists, and joins only at render.

``attrs`` is mapped ``flattened`` in Elasticsearch, and ``flattened`` indexes
**each element of an array** as its own keyword. Writing ``"no-gluten, organic,
kosher"`` therefore makes the *combination* the indexed key: a ``term`` query for
``no-gluten`` reaches only the products whose sole label is ``no-gluten``, and the
term dictionary carries one entry per distinct combination rather than one per
value. Measured on the live ``catalog_en_v14``, ``{"term": {"attrs.Labels":
"no-gluten"}}`` returned 4,435 of the 12,229 documents Open Food Facts labels
``en:no-gluten``.

Four attributes come from an Open Food Facts **list** and are written as lists:

    Labels · Allergens · Ingredients analysis · Dietary restrictions

``Countries`` is deliberately not among them, and
:func:`test_countries_is_still_a_scalar_read_from_the_free_text_field` pins that.
It looks like a fifth, and this call site never had a list to preserve: it reads
the free-text ``countries`` field. Open Food Facts publishes the canonical list
separately as ``countries_tags``, and reading that instead would change which
source field the attribute reads and the value it displays, which is a decision
about the catalog rather than a join to undo. Tracked on its own in #50.

**The trap this module exists to hold shut.** Four attributes carry a comma
*inside* one legitimate value:

    Modelled margin       42% (bucket=condiments_sauces base=30%, labels=…)
    Estimated unit price  7.59 USD/l (473ml, source=quantity, bucket=…)
    Serving size          per 2-pk, 46 g
    Quantity              10x 23g (5x 46 g), Net: 230 g

A fix that split ``attrs`` values on ``", "`` would shred all four while looking
like it worked, because a naive scan reports them as multi-valued. The
distinction between "a list of two" and "one string containing a comma" survives
in exactly one place -- the writer, where the source is still a list -- and that
is where the correction is made. :func:`test_comma_carrying_scalars_round_trip_whole`
and :func:`test_every_other_attribute_is_still_a_string` fail on any diff that
reintroduces a split, including one applied downstream.

The data is real. ``fixtures/off_real_multivalue_attrs.json`` holds three product
records from the public Open Food Facts JSONL export -- the same dump the
2026-08-03 build manifest pins by sha256 -- reduced to the keys the extractor
reads and otherwise verbatim, plus the ancestor closure of their category tags
from the public taxonomy. Nothing is synthesised, and each product is there for a
reason a made-up record would not have produced:

``0022506002357``
    Seven real labels including ``en:no-gluten``, and three of them
    (``no-gluten``, ``organic``, ``usda-organic``) drive margin uplift -- so this
    same product's ``Modelled margin`` carries an internal comma. The positive
    case (every label matchable) and the negative case (the comma-carrying scalar
    intact) are asserted on one product, which is what stops a change from
    passing one and quietly breaking the other.

``0065633074712``
    ``Quantity`` is ``"10x 23g (5x 46 g), Net: 230 g"`` and ``Serving size`` is
    ``"per 2-pk, 46 g"``. Both are single values written by a human with commas
    in them, and both must come out whole.

``0011152145969``
    Carries no ``labels_tags`` at all, so ``Labels`` is absent rather than an
    empty list -- the case that shows the writer omits a key instead of emitting
    ``[]``. It is also the ``Countries`` witness: its ``countries`` free-text
    field reads ``"France,États-Unis,en:france"`` while its ``countries_tags``
    reads ``["en:france", "en:united-states"]``. The free-text field names France
    twice, in two languages, one of them as a raw tag id, and uses no space after
    the comma. That is the evidence behind #50, and here it is what makes
    ``test_countries_is_still_a_scalar_read_from_the_free_text_field`` bite: the
    emitted value is that prose, unsplit.

The tests drive the real ``extract.main()`` CLI and assert on written records,
not on the helpers, because the shape is a property of the emitted document.

Run with ``pytest tests/`` or directly: ``python tests/test_attrs_list_shape.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from off_demo_extract.extract import main  # noqa: E402

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
COUNTRIES_FREE_TEXT_DISAGREES = "0011152145969"

# The attributes this call site reads from an Open Food Facts list. Written as
# lists. ``Countries`` is not here: see the module docstring and #50.
LIST_ATTRS = [
    "Labels",
    "Allergens",
    "Ingredients analysis",
    "Dietary restrictions",
]

# The attributes that carry a comma inside one legitimate value.
COMMA_CARRYING_SCALARS = [
    "Modelled margin",
    "Estimated unit price",
    "Serving size",
    "Quantity",
]


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
    return tempfile.TemporaryDirectory(prefix="off-attrs-list-shape-test-")


def _records() -> Dict[str, dict]:
    with _tmpdir() as tmp:
        records = _run_extract(Path(tmp))
    assert len(records) == len(SOURCE_PRODUCTS), (
        f"expected {len(SOURCE_PRODUCTS)} records, got {len(records)}"
    )
    return {r["id"]: r for r in records}


def _source(code: str) -> dict:
    for p in SOURCE_PRODUCTS:
        if p["code"] == code:
            return p
    raise AssertionError(f"{code} not in fixture")


def test_the_four_list_sourced_attributes_are_written_as_lists() -> None:
    """Each of the four arrives as a list of the source's own values.

    Fails on the parent commit, where every one of these is a joined string.
    """
    record = _records()[MULTI_LABEL]
    attrs = record["attrs"]

    assert attrs["Labels"] == [
        "no-gluten",
        "organic",
        "kosher",
        "no-gmos",
        "usda-organic",
        "non-gmo-project",
        "orthodox-union-kosher",
    ], attrs["Labels"]
    assert attrs["Allergens"] == ["eggs", "mustard"], attrs["Allergens"]
    assert attrs["Ingredients analysis"] == [
        "palm-oil-free",
        "non-vegan",
        "vegetarian-status-unknown",
    ], attrs["Ingredients analysis"]
    assert attrs["Dietary restrictions"] == ["kosher", "organic"], attrs["Dietary restrictions"]


def test_an_absent_source_list_omits_the_key_rather_than_writing_an_empty_list() -> None:
    """``0011152145969`` carries no ``labels_tags``, so ``Labels`` is not emitted.

    ``clean_tags`` returns ``None`` for an absent or all-empty source and the
    caller's truthiness check drops the key, exactly as the joined form did. An
    empty list would be a new shape for "no value" and would show up as an empty
    row on every display surface.
    """
    source = _source(COUNTRIES_FREE_TEXT_DISAGREES)
    assert not source.get("labels_tags"), source.get("labels_tags")

    attrs = _records()[COUNTRIES_FREE_TEXT_DISAGREES]["attrs"]
    assert "Labels" not in attrs, attrs.get("Labels")
    assert "Dietary restrictions" not in attrs, attrs.get("Dietary restrictions")


def test_each_label_of_a_multi_label_product_is_its_own_value() -> None:
    """The values are separable, which is the whole point of the change.

    ``flattened`` indexes each element as its own keyword, so this is the
    document-level form of "``term: no-gluten`` matches this product". Asserted
    as a set membership rather than by position, so it keeps biting if the source
    reorders its tags.
    """
    labels = _records()[MULTI_LABEL]["attrs"]["Labels"]
    assert isinstance(labels, list), type(labels)
    assert "no-gluten" in labels, labels
    assert "usda-organic" in labels, labels
    assert len(labels) == 7, labels
    for value in labels:
        assert "," not in value, f"{value!r} is a combination, not a value"


def test_comma_carrying_scalars_round_trip_whole() -> None:
    """The four attributes with an internal comma are untouched strings.

    This is the negative half of the change, asserted on the *same* corpus as the
    positive half: ``0022506002357`` is both the multi-label product and a product
    whose modelled margin names three uplift labels after a comma.
    """
    records = _records()

    margin = records[MULTI_LABEL]["attrs"]["Modelled margin"]
    assert margin == (
        "42% (bucket=condiments_sauces base=30%, "
        "labels=no-gluten+organic+usda-organic x1.40)"
    ), margin

    price = records[MULTI_LABEL]["attrs"]["Estimated unit price"]
    assert price == (
        "7.59 USD/l (473ml, source=quantity, bucket=condiments_sauces, scale=1.01, ratio=0.95)"
    ), price

    scalars = records[COMMA_IN_SCALARS]["attrs"]
    assert scalars["Quantity"] == "10x 23g (5x 46 g), Net: 230 g", scalars["Quantity"]
    assert scalars["Serving size"] == "per 2-pk, 46 g", scalars["Serving size"]

    for code in (MULTI_LABEL, COMMA_IN_SCALARS):
        for key in COMMA_CARRYING_SCALARS:
            value = records[code]["attrs"].get(key)
            if value is None:
                continue
            assert isinstance(value, str), f"{code} {key}: {type(value)} -- a split would do this"


def test_every_other_attribute_is_still_a_string() -> None:
    """Enumerated, so no attribute can quietly change shape in either direction.

    A test that only checked the four would pass a diff that also list-ified
    ``Quantity``; a test that only checked four scalars would miss the fifth
    attribute somebody adds next year. Both directions are covered by walking
    every key actually emitted -- which is also what keeps ``Countries`` pinned
    as a string for as long as #50 is undecided.
    """
    for code, record in _records().items():
        for key, value in record["attrs"].items():
            if key in LIST_ATTRS:
                assert isinstance(value, list), f"{code} {key}: expected a list, got {type(value)}"
                assert all(isinstance(v, str) for v in value), f"{code} {key}: {value}"
            else:
                assert isinstance(value, str), f"{code} {key}: expected a string, got {type(value)}"


def test_no_surface_rejoins_the_values_the_writer_kept_apart() -> None:
    """The one-value-per-element shape survives all the way to the document.

    This assertion used to be made on the generated ``description``, because that
    was where the join lived: ``attrs`` values were written joined, and the
    correction in #45 moved the join to render time so the human-readable surface
    stayed byte-identical while the indexed values came apart.

    That surface is gone -- ``description`` now carries the product's own prose
    and no attributes at all -- so the assertion moves to where those values are
    now read: the promoted top-level fields. The property is the same one and it
    is asserted on the same product; what changed is that a re-joined value would
    now be visible in a *field* rather than in a sentence.

    The comma-carrying scalars are the other half, and they stay in ``attrs``:
    ``test_comma_carrying_scalars_round_trip_whole`` above covers them, so a diff
    that reintroduced a split would still fail whichever half it touched.
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
    assert record["dietary_restrictions"] == ["kosher", "organic"], record["dietary_restrictions"]
    # ``Countries`` is the scalar that looks like a list: one string, unsplit.
    assert record["countries"] == "United States, World", record["countries"]

    for code, emitted in _records().items():
        assert "Key specifications" not in emitted["description"], (
            f"{code}: the description carries an attribute block again, and with it "
            "a second, joined spelling of every value this module keeps apart"
        )


def test_countries_is_still_a_scalar_read_from_the_free_text_field() -> None:
    """``Countries`` is out of scope here, and stays exactly as it was.

    Two different changes are available for this attribute and only one of them
    is ever acceptable:

    * Read ``countries_tags`` -- the canonical list Open Food Facts publishes
      alongside the prose. That is a source-field change with a visible
      consequence ("United States" becomes "united-states"), so it is a decision
      about the catalog rather than a join to undo, and it is tracked in #50.
    * Split ``countries`` on commas. That is never acceptable. It is the same
      heuristic that shreds ``Quantity``, and on this very product it would file
      France under two keys in two languages plus a raw tag id.

    ``0011152145969`` makes the difference visible: its prose and its tag list
    disagree about how many countries there are and what they are called. The
    assertions below fail on either change, so neither can arrive unannounced.

    Both fixture products are needed to cover the split, and the reason is a trap
    in the free-text field itself. ``0011152145969`` writes its commas with **no
    following space**, so a ``", "`` split passes straight over it; only
    ``0022506002357`` (``"United States, World"``) makes such a split visible.
    A guard written on the interesting product alone would have been inert.
    """
    records = _records()

    source = _source(COUNTRIES_FREE_TEXT_DISAGREES)
    assert source["countries"] == "France,États-Unis,en:france", source["countries"]
    assert source["countries_tags"] == ["en:france", "en:united-states"], source["countries_tags"]

    countries = records[COUNTRIES_FREE_TEXT_DISAGREES]["attrs"]["Countries"]
    assert isinstance(countries, str), f"{type(countries)} -- see #50 before changing this"
    assert countries == "France,États-Unis,en:france", countries

    naive_split = [part.strip() for part in source["countries"].split(",")]
    assert countries != naive_split, (
        "Countries was split on commas, which files France under two keys in two "
        f"languages plus a raw tag id: {naive_split}"
    )

    # The product whose prose does use ", ", so a `", "` split would show here.
    spaced = records[MULTI_LABEL]["attrs"]["Countries"]
    assert isinstance(spaced, str), f"{type(spaced)} -- see #50 before changing this"
    assert spaced == "United States, World", spaced


def test_attr_keys_still_names_every_attribute() -> None:
    """``attr_keys`` drives faceting and is keys-only, so the shape change is a no-op for it."""
    for code, record in _records().items():
        assert record["attr_keys"] == sorted(record["attrs"].keys()), code
        for key in LIST_ATTRS:
            if key in record["attrs"]:
                assert key in record["attr_keys"], f"{code}: {key}"


def test_dietary_restrictions_attr_agrees_with_the_dedicated_field() -> None:
    """The attr and the top-level field carry the same list, and the field is unchanged.

    ``dietary_restrictions`` has always been emitted as a list at the top level;
    only the ``attrs`` copy was joined. The derivation rules behind both are
    untouched by this change, and this pins that: a diff that altered which tokens
    are derived would fail here even though it would satisfy every shape assertion
    above.
    """
    record = _records()[MULTI_LABEL]
    assert record["dietary_restrictions"] == ["kosher", "organic"], record["dietary_restrictions"]
    assert record["attrs"]["Dietary restrictions"] == record["dietary_restrictions"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
