"""A category node must read the same in ``category_path`` and in ``taxonomy_tags``.

Both fields are derived from the product's ``categories_tags``. They used to
render a node's label with two different rules — the taxonomy's ``name`` for a
path segment, a mechanical de-slug of the tag id for a flat category value — so
the same node read ``Plant-based foods`` in one field and ``Plant based foods``
in the other, and the two fields could not be joined on string.

Measured over the first 200,000 lines of the public Open Food Facts export
(132,340 products with a resolved chain): under the de-slug rule only 75.10% of
products had every self-tagged chain node's label present verbatim in
``taxonomy_tags``; under the single-sourced rule, 100.00%. In a Spanish or French
run the two fields disagreed on *language* as well, because the de-slug only ever
produced English no matter the catalog.

What is asserted
----------------
For every node on a product's emitted chain **that the product actually tagged**,
the label in ``category_path`` must appear **verbatim** in ``taxonomy_tags``. The
comparison is deliberately exact. A normalised one (lowercase, hyphens to spaces)
passed before the fix too — it cannot tell a fixed field from a broken one — and
it is blind to the language half entirely.

The scope — nodes the product itself tagged — is what makes the assertion true
rather than merely mostly-true. A chain is anchored to a *global* taxonomy root,
so it materialises ancestors the product never tagged; those are legitimately
absent from ``taxonomy_tags``, which lists only the product's own tags. Asserting
plain containment of every chain node would fail on that structural difference
while saying nothing about labelling. The self-tagged nodes are identified from
:func:`category_chain`, which returns ids and never consults a label, so the
scoping cannot be biased by the rule under test.

The data
--------
``fixtures/off_real_label_divergence.json`` holds the ``categories_tags`` of 10
real products from the public Open Food Facts JSONL export, plus the ancestor
closure of those tags from the public category taxonomy with its English, Spanish
and French names. Nothing about the categories is synthesised; the closure is
closed under ``parents``, so depths inside the slice match the full 14,457-node
file and pruning cannot flatter the result. The products were chosen to carry
each shape the de-slug got wrong: hyphenation (``Plant-based foods``,
``Extra-virgin olive oils``), a disambiguating parenthetical (``Crackers
(Appetizers)``), and an upstream name whose casing the de-slug would "correct"
(``ice creams``). The surrounding product fields (title, description, image) are
scaffolding so the records clear the extractor's earlier filters and reach the
category code at all.

Run with ``pytest tests/`` or directly:
``python tests/test_category_label_agreement.py``.
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
from off_demo_extract.taxonomy import category_chain, default_keep_prefixes  # noqa: E402

PRICING_CONFIG = REPO_ROOT / "config" / "pricing_buckets.json"
PATH_SEPARATOR = "/"
LANGS = ("en", "es", "fr")

# The extractor's default --category-exclude.
EXCLUDE = {"en:null", "en:unknown", "en:undefined"}

_FIXTURE = json.loads(
    (REPO_ROOT / "tests" / "fixtures" / "off_real_label_divergence.json").read_text(
        encoding="utf-8"
    )
)
TAXONOMY: Dict[str, dict] = _FIXTURE["taxonomy"]
PRODUCTS: List[dict] = _FIXTURE["products"]


def _deslug(tag: str) -> str:
    """The rule ``taxonomy_tags`` used before the fix, kept only to prove the bite.

    No production code calls this any more — that is the whole point of the fix.
    It survives here so :func:`test_fixture_still_exercises_the_divergence` can
    assert that the fixture would still catch a relapse to it, rather than
    passing because the data got blander.
    """
    t = tag.split(":", 1)[1] if ":" in tag else tag
    t = t.replace("-", " ").replace("_", " ").strip()
    return tag if not t else t[0].upper() + t[1:]


def _input_record(product: dict) -> dict:
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


def _tmpdir() -> Any:
    return tempfile.TemporaryDirectory(prefix="off-label-test-")


def _chain_ids(tags: List[str], lang: str) -> List[str]:
    """Ids of the emitted chain, root→leaf. Label-free by construction."""
    return category_chain(
        tags, TAXONOMY, EXCLUDE, keep_prefixes=default_keep_prefixes(lang)
    )


def _self_tagged_labels(record: dict, tags: List[str], lang: str) -> List[str]:
    """The chain's labels for the nodes this product actually carried as tags."""
    ids = _chain_ids(tags, lang)
    paths = record["category_path"]
    assert len(ids) == len(paths), (
        f"chain length {len(ids)} != emitted category_path length {len(paths)} "
        f"for {record['id']}; the test's id walk and the extractor's disagree"
    )
    tagset = set(tags)
    return [
        path.rsplit(PATH_SEPARATOR, 1)[-1]
        for node, path in zip(ids, paths)
        if node in tagset
    ]


def _agreement_failures(lang: str) -> Tuple[List[tuple], int]:
    """(failures, nodes checked) for one language, over the extractor's output."""
    with _tmpdir() as d:
        records, _ = _run_extract(Path(d), lang)

    assert len(records) == len(PRODUCTS), (
        f"{len(PRODUCTS) - len(records)} fixture products did not survive the "
        f"extractor in {lang}; the assertion below would be weakened silently"
    )

    failures: List[tuple] = []
    checked = 0
    for product in PRODUCTS:
        record = records[product["code"].zfill(13)]
        taxonomy_tags = set(record["taxonomy_tags"])
        for label in _self_tagged_labels(record, product["categories_tags"], lang):
            checked += 1
            if label not in taxonomy_tags:
                failures.append((record["id"], label, sorted(taxonomy_tags)))
    return failures, checked


def _assert_agreement(lang: str) -> None:
    failures, checked = _agreement_failures(lang)
    assert checked >= len(PRODUCTS), (
        f"only {checked} self-tagged chain nodes checked in {lang}; the fixture "
        "is not exercising the join"
    )
    assert not failures, (
        f"[{lang}] category_path segments missing verbatim from taxonomy_tags: "
        + "; ".join(
            f"{code}: {label!r} not in {cats}" for code, label, cats in failures[:5]
        )
    )


def test_labels_agree_in_english() -> None:
    """Exact-string agreement in the English catalog — no normalisation allowed.

    This is the check the issue reports at 61% on the live English index. Before
    the fix it failed here on every hyphenated and parenthetical node in the
    fixture: the chain said ``Plant-based foods``, ``taxonomy_tags`` said
    ``Plant based foods``.
    """
    _assert_agreement("en")


def test_labels_agree_in_spanish() -> None:
    """The language half: a Spanish catalog must not disagree with itself.

    Before the fix ``category_path`` carried ``Alimentos de origen vegetal``
    while ``taxonomy_tags`` carried the English-derived ``Plant based foods``, in
    the *same document* — 0% agreement, and no amount of string normalisation
    could have closed it.
    """
    _assert_agreement("es")


def test_labels_agree_in_french() -> None:
    """Same, in French."""
    _assert_agreement("fr")


def test_taxonomy_tags_are_localised_not_english_derived() -> None:
    """``taxonomy_tags`` must speak the catalog's language where the taxonomy does.

    Agreement alone could be satisfied by dragging ``category_path`` down to the
    English de-slug. This pins the direction: the flat values carry the
    taxonomy's Spanish and French names, not a de-slugged English id.
    """
    for lang in ("es", "fr"):
        with _tmpdir() as d:
            records, _ = _run_extract(Path(d), lang)

        localised = 0
        for product in PRODUCTS:
            record = records[product["code"].zfill(13)]
            taxonomy_tags = set(record["taxonomy_tags"])
            for tag in product["categories_tags"]:
                name = (TAXONOMY.get(tag, {}).get("name") or {}).get(lang)
                if not name or name == _deslug(tag):
                    continue
                assert _deslug(tag) not in taxonomy_tags, (
                    f"[{lang}] {record['id']}: taxonomy_tags carries the "
                    f"English-derived {_deslug(tag)!r} instead of {name!r}"
                )
                if name in taxonomy_tags:
                    localised += 1
        assert localised >= len(PRODUCTS), (
            f"[{lang}] only {localised} localised labels observed; the fixture "
            "is not exercising the localisation half"
        )


def test_fixture_still_exercises_the_divergence() -> None:
    """Guard the guard: the fixture must still contain nodes the old rule broke.

    A passing agreement test proves nothing if the fixture happens to contain
    only nodes whose taxonomy name and de-slugged id coincide. This counts the
    nodes where they differ — every one of them is a case the old rule failed —
    and fails if the fixture ever drifts down to too few of them.
    """
    divergent = {
        tag
        for product in PRODUCTS
        for tag in product["categories_tags"]
        if (TAXONOMY.get(tag, {}).get("name") or {}).get("en", _deslug(tag))
        != _deslug(tag)
    }
    assert len(divergent) >= 8, (
        f"only {len(divergent)} nodes in the fixture differ between the taxonomy "
        f"name and the de-slugged id: {sorted(divergent)}"
    )
    # The three shapes the de-slug got wrong, each present by name.
    assert "en:plant-based-foods" in divergent, "hyphenation case missing"
    assert "en:crackers-appetizers" in divergent, "parenthetical case missing"
    assert "en:ice-creams" in divergent, "upstream-casing case missing"
    # The casing case is measured here between the taxonomy *name* and the
    # de-slug, which is why it still counts. It no longer separates the two rules
    # at render time: the labeller now upper-cases a lowercase first character,
    # so `en:ice-creams` renders `Ice creams` from either rule. The nodes that
    # keep the agreement tests biting are the other 12 — hyphenation,
    # parentheticals and localisation, which the de-slug cannot reach at all.
    # See tests/test_lowercase_taxonomy_names.py for the casing rule itself.


def test_run_report_audits_the_label_invariants() -> None:
    """The run must *report* label agreement, not just happen to have it.

    An invariant that holds by construction is the kind that quietly stops
    holding. These counters are what turns a relapse into a line in the
    extraction report instead of a hand audit of the built index.
    """
    with _tmpdir() as d:
        _, report = _run_extract(Path(d), "en")

    audit = report["category_path_addresses"]
    assert audit["categories_under_multiple_labels"] == 0, audit["label_examples"]
    assert audit["labels_seen"] >= audit["categories_seen"], audit

    counters = report["counters"]
    assert counters["categories_under_multiple_labels"] == 0, counters

    # The inverse count is *reported*, not asserted to be zero: on a full run it
    # is not, because upstream holds duplicate nodes under two language prefixes
    # that render to one string. This fixture has none, which is what makes the
    # equality below a real check on the counter rather than a placeholder.
    assert audit["labels_shared_by_multiple_categories"] == 0, audit[
        "shared_label_examples"
    ]
    assert counters["labels_shared_by_multiple_categories"] == 0, counters


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
