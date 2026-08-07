"""A category label starts with a capital, and is otherwise the name upstream wrote.

The defect
----------
A category's display label is the taxonomy's ``name``, which is the right source:
it carries hyphenation (``Plant-based foods``), disambiguating parentheticals
(``Crackers (Appetizers)``) and localisation that a de-slugged id destroys. But
upstream does not capitalise consistently. **92 of the taxonomy's 8,939
``en:``-prefixed nodes have an English name that begins with a lowercase
letter**, so those segments read

    Desserts/Frozen desserts/Ice creams and sorbets/ice creams/Ice cream tubs

and, since the flat ``taxonomy_tags`` field renders through the same labeller,
they read that way in a tag row too. 49 Spanish and 208 French names have the
same defect.

The rule, and why not the obvious ones
--------------------------------------
The first character is upper-cased when it is a lowercase letter. Nothing else
is touched.

* Not ``str.capitalize()``: it lower-cases the remainder, which would cost
  ``dried Toothed wrack`` its species capital and ``farmed Mediterranean bass``
  its proper noun — exactly the information the taxonomy ``name`` is preferred
  over a de-slug for.
* Not ``str.title()``: ``Crackers (Appetizers)`` and ``Plant-based foods`` would
  both move.
* Not "upper-case the first *letter*": ``10% red wine`` would become ``10% Red
  wine``.
* Not conditional on the language: the slug fallback already capitalised its
  first character, so before this the same node rendered ``Saint-émilion`` when
  the taxonomy had no name for it and ``saint-émilion`` when it did. One helper
  now decides both.

An unconditional upper-case would be the *wrong* fix if any of the 92 were
deliberately lowercase — a ``pH``-style term, a lowercase brand. All 92 were read
before the rule was chosen and none is: they are ordinary common nouns upstream
did not capitalise (``ice creams``, ``chorizo``, ``broths``, ``baker's yeast``).
No name in the snapshot, in any language, has the ``pH``/``iPhone`` shape of a
lowercase first character followed by an upper-case second one, and no name's
first character changes length under ``.upper()``. All 92 are recorded verbatim
in the fixture, so a taxonomy refresh that introduces a deliberately-lowercase
name is a question someone is made to answer rather than one this rule answers
silently.

**The source file is not modified.** This is presentation of a name the extractor
already takes; it is not a licence to correct the taxonomy.

What is asserted
----------------
Both directions, because each alone is satisfiable by a wrong rule:

* every one of the 92 real names renders with a capital first character and with
  its remaining characters **byte-identical**, which is what fails under
  ``capitalize`` and ``title``;
* the control names — a parenthetical, two hyphenations, a name starting with a
  digit, a name starting with ``%``, and an ordinary already-capitalised name —
  come back byte-identical, which is what fails under a rule that reaches past
  the first character.

Then end to end, through the CLI, on three real products: the label change
reaches both ``category_path`` and ``taxonomy_tags``, mid-breadcrumb as well as
at the leaf, and every segment that was already capitalised is untouched.

The data
--------
``fixtures/off_real_lowercase_names.json`` holds all 92 nodes, the controls, real
products carrying them verbatim from the public export, and the complete upward
closure of every id named. Nothing is synthesised and no name is edited.

Run with ``pytest tests/`` or directly:
``python tests/test_lowercase_taxonomy_names.py``.
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
from off_demo_extract.taxonomy import display_label  # noqa: E402

PRICING_CONFIG = REPO_ROOT / "config" / "pricing_buckets.json"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "off_real_lowercase_names.json"

LANGS = ("en", "es", "fr")

_FIXTURE = json.loads(FIXTURE.read_text(encoding="utf-8"))
TAXONOMY: Dict[str, Any] = _FIXTURE["taxonomy"]
PRODUCTS: List[Dict[str, Any]] = _FIXTURE["products"]
LOWERCASE_NAMES: Dict[str, str] = _FIXTURE["lowercase_first_en_names"]
CONTROLS: List[str] = _FIXTURE["byte_unchanged_controls"]

ICE_CREAM_PRODUCT = "00012444"       # `ice creams` sits mid-breadcrumb
CHORIZO_PRODUCT = "0012438000095"    # `chorizo` is the leaf
CORN_SALAD_PRODUCT = "2000491211006"  # `corn salad` is the leaf, under hyphenated ancestors


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


def _run_extract(tmp: Path, lang: str) -> Dict[str, dict]:
    """Run the real CLI over the fixture; return the written records by id."""
    taxonomy_path = tmp / "categories.json"
    taxonomy_path.write_text(json.dumps(TAXONOMY), encoding="utf-8")

    input_path = tmp / "products.jsonl"
    input_path.write_text(
        "".join(json.dumps(_input_record(p)) + "\n" for p in PRODUCTS), encoding="utf-8"
    )

    output_path = tmp / "out.ndjson"
    rc = main(
        [
            "--input", str(input_path),
            "--output", str(output_path),
            "--report", str(tmp / "report.json"),
            "--taxonomy", str(taxonomy_path),
            # A hand-built fixture is deliberately not the pinned snapshot.
            "--allow-unpinned-taxonomy",
            "--pricing-config", str(PRICING_CONFIG),
            "--lang", lang,
            "--progress-every", "0",
            "--progress-seconds", "0",
            "--yes",
        ]
    )
    assert rc == 0, f"extractor exited {rc}"
    return {
        json.loads(line)["id"]: json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _extract(lang: str) -> Dict[str, dict]:
    with tempfile.TemporaryDirectory(prefix="off-case-test-") as d:
        return _run_extract(Path(d), lang)


# ---------------------------------------------------------------------------
# The rule, over the real names it exists for
# ---------------------------------------------------------------------------

def test_the_fixture_still_holds_the_defect_it_was_built_from() -> None:
    """Every recorded name really does begin lowercase, and there are 92 of them.

    Without this the assertions below could go green on a fixture that had been
    tidied up rather than on a rule that works.
    """
    assert len(LOWERCASE_NAMES) == 92, len(LOWERCASE_NAMES)
    for node, name in LOWERCASE_NAMES.items():
        assert node.startswith("en:"), node
        assert name[:1].islower(), (node, name)
        assert TAXONOMY[node]["name"]["en"] == name, node


def test_every_lowercase_name_renders_capitalised_and_otherwise_unchanged() -> None:
    """All 92, and the tail of each string byte for byte.

    The second assertion is the one that fails under ``str.capitalize`` — on
    ``dried Toothed wrack``, ``farmed Mediterranean bass`` and ``sheep's-milk
    cheeses`` — and under ``str.title`` on all of them.
    """
    for node, name in LOWERCASE_NAMES.items():
        rendered = display_label(TAXONOMY, node, "en")
        assert rendered[:1] == name[:1].upper(), (node, name, rendered)
        assert rendered[1:] == name[1:], (node, name, rendered)


def test_the_named_examples_render_exactly_as_the_issue_asks() -> None:
    """Literals, so a failure reads as a label rather than as an index."""
    expected = {
        "en:ice-creams": "Ice creams",
        "en:chorizo": "Chorizo",
        "en:broths": "Broths",
        "en:books": "Books",
        "en:baker-s-yeast": "Baker's yeast",
        "en:chicken-skewers": "Chicken skewers",
        "en:corn-salad": "Corn salad",
        # Lowercase first character, deliberate capital further in: the case a
        # `.capitalize()` would destroy.
        "en:dried-toothed-wrack": "Dried Toothed wrack",
        "en:farmed-mediterranean-bass": "Farmed Mediterranean bass",
        # Hyphenation and an apostrophe inside the string.
        "en:sheep-s-milk-cheeses": "Sheep's-milk cheeses",
        "en:non-homogenized-milks": "Non-homogenized milks",
        "en:plant-based": "Plant-based",
    }
    for node, label in expected.items():
        assert display_label(TAXONOMY, node, "en") == label, node


def test_a_name_that_already_reads_correctly_is_byte_identical() -> None:
    """The control. A rule reaching past the first character fails here.

    ``10% red wine`` and ``% de matières grasses`` are the ones that catch a rule
    that capitalises the first *letter* wherever it sits rather than the first
    character.
    """
    expected = {
        ("en:crackers-appetizers", "en"): "Crackers (Appetizers)",
        ("en:plant-based-foods", "en"): "Plant-based foods",
        ("en:extra-virgin-olive-oils", "en"): "Extra-virgin olive oils",
        ("en:10-red-wine", "en"): "10% red wine",
        ("en:snacks", "en"): "Snacks",
        ("fr:de-matieres-grasses", "fr"): "% de matières grasses",
    }
    for (node, lang), label in expected.items():
        assert display_label(TAXONOMY, node, lang) == label, (node, lang)

    # And every control's rendered label is exactly the name in the file.
    for node in CONTROLS:
        for lang in LANGS:
            names = TAXONOMY[node].get("name", {})
            for key in (lang, "en", "xx"):
                if isinstance(names.get(key), str) and names[key].strip():
                    if not names[key].strip()[:1].islower():
                        assert display_label(TAXONOMY, node, lang) == names[key].strip()
                    break


def test_the_localised_label_is_capitalised_too() -> None:
    """The rule is about a label, not about English.

    A French or Spanish catalog renders the French or Spanish ``name``, and
    upstream's inconsistency is worse there than in English (208 French and 49
    Spanish names begin lowercase against 92 English ones).
    """
    assert display_label(TAXONOMY, "en:ice-creams", "fr") == "Glaces"
    assert display_label(TAXONOMY, "en:ice-creams", "es") == "Helado"
    assert display_label(TAXONOMY, "en:broths", "fr") == "Bouillons"
    assert display_label(TAXONOMY, "en:broths", "es") == "Caldos"


# ---------------------------------------------------------------------------
# End to end, on real products, in both emitted fields
# ---------------------------------------------------------------------------

def test_a_mid_breadcrumb_segment_is_capitalised_in_the_emitted_path() -> None:
    """The shape the issue reports: the label is not the leaf, it is a hop.

    The whole path is asserted so that a change to any other segment fails here
    too — the rule is supposed to move exactly one of them.
    """
    records = _extract("en")
    record = records[ICE_CREAM_PRODUCT.zfill(13)]
    assert record["category_path_primary"] == [
        "Desserts",
        "Desserts/Frozen desserts",
        "Desserts/Frozen desserts/Ice creams and sorbets",
        "Desserts/Frozen desserts/Ice creams and sorbets/Ice creams",
        "Desserts/Frozen desserts/Ice creams and sorbets/Ice creams/Ice cream tubs",
    ], record["category_path_primary"]
    assert "Ice creams" in record["taxonomy_tags"], record["taxonomy_tags"]


def test_a_leaf_segment_is_capitalised_and_its_ancestors_are_untouched() -> None:
    """``chorizo`` is the leaf; the four segments above it already read correctly."""
    records = _extract("en")
    record = records[CHORIZO_PRODUCT.zfill(13)]
    assert record["category_path_primary"] == [
        "Meats and their products",
        "Meats and their products/Prepared meats",
        "Meats and their products/Prepared meats/Cured sausages",
        "Meats and their products/Prepared meats/Cured sausages/Chorizo",
    ], record["category_path_primary"]
    assert "Chorizo" in record["taxonomy_tags"], record["taxonomy_tags"]


def test_a_hyphenated_ancestor_survives_the_leaf_being_capitalised() -> None:
    """``Plant-based foods and beverages`` must arrive with its hyphen intact."""
    records = _extract("en")
    record = records[CORN_SALAD_PRODUCT.zfill(13)]
    assert record["category_path_primary"][0] == "Plant-based foods and beverages"
    assert record["category_path_primary"][-1].endswith("/Corn salad"), record[
        "category_path_primary"
    ]
    assert "Corn salad" in record["taxonomy_tags"], record["taxonomy_tags"]
    assert "Plant-based foods and beverages" in record["taxonomy_tags"]


def test_the_two_emitted_fields_still_agree_on_every_self_tagged_node() -> None:
    """The invariant a second labelling rule would break.

    ``category_path`` and ``taxonomy_tags`` are joined on string downstream, so a
    casing rule that reached one field and not the other would be worse than the
    lowercase label. Scoped to the nodes the product itself tagged, because a
    chain also materialises ancestors the product never carried and those are
    legitimately absent from the flat field.
    """
    for lang in LANGS:
        records = _extract(lang)
        for product in PRODUCTS:
            record = records[product["code"].zfill(13)]
            flat = set(record["taxonomy_tags"])
            tagged = {
                display_label(TAXONOMY, tag, lang)
                for tag in product["categories_tags"]
                if tag in TAXONOMY
            }
            for segment in (p.rsplit("/", 1)[-1] for p in record["category_path"]):
                if segment in tagged:
                    assert segment in flat, (lang, product["code"], segment, sorted(flat))


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
