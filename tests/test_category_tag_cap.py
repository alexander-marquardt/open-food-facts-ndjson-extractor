"""The flat tag cap may drop a product's incidental tags, never its own chain.

The defect
----------
``taxonomy_tags`` is capped at :data:`MAX_NUM_TAXONOMY_TAGS` (20). The cap used
to be applied by walking the product's tags in order and stopping, which drops
the *tail*. Open Food Facts orders ``categories_tags`` roughly
general-to-specific, so the tail is where a product's most specific tags are —
and a dropped tag can be a node on the product's own emitted ``category_path``.
When it is, a path segment the product genuinely tagged has no counterpart in the
flat field, and a label-to-segment join misses it while looking exactly like the
labelling divergence #10 fixed.

Measured over the first 200,000 records of the public export, English catalog,
against the pinned 14,457-node taxonomy snapshot: 135,716 products carry at least
one eligible tag and all of them resolve an anchored chain; 6 of those have more
eligible tags than the cap; 3 lose a self-tagged chain node to it:

===============  ========  =========  ===================
code             eligible  emitted    segment lost
===============  ========  =========  ===================
0036800388352          22         20  ``Basmati rices``
0051933012707          21         20  ``Peas``
0078742086774          21         20  ``Peas``
===============  ========  =========  ===================

All three are in this fixture, and this file fails on all three without the fix.

What is asserted, and what stops it passing vacuously
-----------------------------------------------------
Two halves, because either alone is satisfiable by the wrong change:

* **Positive** — every self-tagged ``category_path`` node appears verbatim in
  ``taxonomy_tags``. Removing the cap satisfies this.
* **Negative** — the cap still bounds the list, and the tags it drops are the
  incidental ones: ``0078742022512`` still loses ``Frozen pineapples`` and
  ``Frozen papayas``, neither of which is on its chain. Keeping the old
  truncation satisfies this.

Only a fix that reserves the chain and caps the rest passes both. Each half also
asserts that the fixture *reached* the condition it is about — that products were
truncated at all, and that chain nodes were checked — so a fixture that stopped
exercising the cap fails here rather than going quiet.

The data
--------
``fixtures/off_real_category_cap.json`` holds the ``categories_tags`` of 8 real
products from the public export, unedited, plus the ancestor closure of those
tags from the public category taxonomy with its English, Spanish and French
names. The closure is closed under ``parents``, so depths inside the slice match
the full snapshot. The products are the 3 that lose a chain node, 2 more that the
cap truncates without losing one (the negative half), and 3 that fit under the
cap unchanged (19, 20 and 6 eligible tags — the last of which is an ordinary
product, so a change that only ever fires above the cap is visible here as
silence). The surrounding product fields (title, description, image) are
scaffolding so the records clear the extractor's earlier filters and reach the
category code at all — the same shape
``tests/test_category_label_agreement.py`` uses.

Run with ``pytest tests/`` or directly:
``python tests/test_category_tag_cap.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from off_demo_extract.category_tags import (  # noqa: E402
    CategoryVocabulary,
    curate_category_tags,
)
from off_demo_extract.extract import (  # noqa: E402
    MAX_NUM_TAXONOMY_TAGS,
    build_category_label_entries,
    main,
    pick_primary_category_tag,
)
from off_demo_extract.taxonomy import (  # noqa: E402
    category_chain,
    default_keep_prefixes,
)

PRICING_CONFIG = REPO_ROOT / "config" / "pricing_buckets.json"
PATH_SEPARATOR = "/"
LANGS = ("en", "es", "fr")

# The extractor's default --category-exclude.
EXCLUDE = {"en:null", "en:unknown", "en:undefined"}

# Larger than any list this fixture can produce, so the shared builder yields the
# full eligible list. Not ``None``: an int works against the capped and the
# uncapped signature alike, which keeps this file runnable either way.
NO_CAP = 1_000_000

_FIXTURE = json.loads(
    (REPO_ROOT / "tests" / "fixtures" / "off_real_category_cap.json").read_text(
        encoding="utf-8"
    )
)
TAXONOMY: Dict[str, dict] = _FIXTURE["taxonomy"]
PRODUCTS: List[dict] = _FIXTURE["products"]

# The products the issue names, with the segment each one lost. Spelled out so a
# fixture edit that quietly drops a case fails instead of shrinking the test.
LOST_SEGMENTS = {
    "0036800388352": "Basmati rices",
    "0051933012707": "Peas",
    "0078742086774": "Peas",
}

# Truncated, but everything the cap dropped is incidental — no node of this
# product's chain. These labels must still be absent after the fix: they are what
# says the cap is still doing its job.
STILL_DROPPED = {
    "0078742022512": ("Frozen pineapples", "Frozen papayas"),
}

# Fewer eligible tags than the cap, so the cap never applies and their output
# must be exactly the uncapped list, in order.
UNTRUNCATED = ("0041331029018", "0041449001104", "0000101209159")


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
    assert len(records) == len(PRODUCTS), (
        f"{len(PRODUCTS) - len(records)} fixture products did not survive the "
        f"extractor in {lang}; the assertions below would be weakened silently"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return records, report


def _tmpdir() -> Any:
    return tempfile.TemporaryDirectory(prefix="off-cap-test-")


def _eligible_labels(tags: List[str], lang: str) -> List[str]:
    """Every label the product is entitled to, uncapped, in the order they arrive.

    Built with the extractor's own curation and labelling, so this is what the
    flat field would carry if nothing were ever dropped.
    """
    vocabulary = CategoryVocabulary.for_catalog(
        TAXONOMY, default_keep_prefixes(lang), EXCLUDE
    )
    curated = curate_category_tags(tags, vocabulary, EXCLUDE).accepted
    primary = pick_primary_category_tag(curated)
    return [
        label
        for _tag, label in build_category_label_entries(
            primary, curated, TAXONOMY, lang, NO_CAP, vocabulary
        )
    ]


def _self_tagged_labels(record: dict, tags: List[str], lang: str) -> List[str]:
    """The chain's labels for the nodes this product actually carried as tags.

    Identified from :func:`category_chain`, which returns ids and never consults
    a label, so the scope cannot be biased by the field under test.
    """
    vocabulary = CategoryVocabulary.for_catalog(
        TAXONOMY, default_keep_prefixes(lang), EXCLUDE
    )
    curated = curate_category_tags(tags, vocabulary, EXCLUDE).accepted
    ids = category_chain(
        curated, TAXONOMY, EXCLUDE, keep_prefixes=default_keep_prefixes(lang)
    )
    paths = record["category_path"]
    assert len(ids) == len(paths), (
        f"chain length {len(ids)} != emitted category_path length {len(paths)} "
        f"for {record['id']}; the test's id walk and the extractor's disagree"
    )
    tagset = set(curated)
    return [
        path.rsplit(PATH_SEPARATOR, 1)[-1]
        for node, path in zip(ids, paths)
        if node in tagset
    ]


def _is_subsequence(short: List[str], long: List[str]) -> bool:
    it = iter(long)
    return all(value in it for value in short)


def _assert_chain_survives(lang: str) -> None:
    with _tmpdir() as d:
        records, _report = _run_extract(Path(d), lang)

    truncated = 0
    checked = 0
    failures: List[str] = []
    for product in PRODUCTS:
        record = records[product["code"].zfill(13)]
        emitted = record["taxonomy_tags"]
        if len(_eligible_labels(product["categories_tags"], lang)) > len(emitted):
            truncated += 1
        for label in _self_tagged_labels(record, product["categories_tags"], lang):
            checked += 1
            if label not in emitted:
                failures.append(f"{record['id']}: {label!r} not in {emitted}")

    # Without these the assertion below is satisfiable by a fixture that stopped
    # reaching the cap, or by an extractor that emitted no chain at all.
    assert truncated >= 3, (
        f"[{lang}] only {truncated} fixture products were truncated by the cap; "
        "the fixture no longer exercises it"
    )
    assert checked >= 2 * len(PRODUCTS), (
        f"[{lang}] only {checked} self-tagged chain nodes checked; the fixture is "
        "not exercising the join"
    )
    assert not failures, (
        f"[{lang}] category_path segments the product tagged, missing from "
        "taxonomy_tags: " + "; ".join(failures)
    )


def test_chain_survives_the_cap_in_english() -> None:
    """The three products the issue names keep their leaf.

    Fails without the fix on exactly those three: ``0036800388352`` emits 20 of
    its 22 eligible labels and ``Basmati rices`` — a segment of its own
    ``category_path`` — is one of the two dropped; ``0051933012707`` and
    ``0078742086774`` each lose ``Peas`` the same way.
    """
    _assert_chain_survives("en")


def test_chain_survives_the_cap_in_spanish() -> None:
    """The rule is about which tags survive, so it cannot be language-specific.

    A Spanish run labels the same nodes differently and, where the taxonomy has
    no Spanish name, falls back to English — neither changes which tags are
    *kept*, and this pins that.
    """
    _assert_chain_survives("es")


def test_chain_survives_the_cap_in_french() -> None:
    """As above, in the third catalog language the extractor builds."""
    _assert_chain_survives("fr")


def test_the_cap_still_drops_the_incidental_tags() -> None:
    """The negative half: reserving the chain must not amount to removing the cap.

    ``0078742022512`` carries 22 eligible tags and a 7-node chain; the two the
    cap discards, ``Frozen pineapples`` and ``Frozen papayas``, are on neither.
    They must still be gone. Without this, "keep everything" would pass the file.
    """
    with _tmpdir() as d:
        records, _report = _run_extract(Path(d), "en")

    for code, labels in STILL_DROPPED.items():
        emitted = records[code.zfill(13)]["taxonomy_tags"]
        eligible = _eligible_labels(
            next(p for p in PRODUCTS if p["code"] == code)["categories_tags"], "en"
        )
        for label in labels:
            assert label in eligible, (
                f"{code} is no longer entitled to {label!r}; the fixture stopped "
                "exercising the drop this test is about"
            )
            assert label not in emitted, (
                f"{code} emitted {label!r}, an incidental tag past the cap — the "
                f"cap is no longer bounding the list ({len(emitted)} values)"
            )

    for product in PRODUCTS:
        record = records[product["code"].zfill(13)]
        emitted = record["taxonomy_tags"]
        # The bound only means anything while no fixture product's own chain is
        # longer than the cap — reserving a longer one is defined to exceed it.
        # Checked from the chain, not from the run's own report, so this holds
        # the extractor to the rule rather than to what it reported.
        self_tagged = _self_tagged_labels(record, product["categories_tags"], "en")
        assert len(self_tagged) < MAX_NUM_TAXONOMY_TAGS, (
            f"{product['code']} has {len(self_tagged)} self-tagged chain nodes, "
            f"at or over the cap of {MAX_NUM_TAXONOMY_TAGS}; the bound below is "
            "then measuring the wrong thing"
        )
        assert len(emitted) <= MAX_NUM_TAXONOMY_TAGS, (
            f"{product['code']} emitted {len(emitted)} tags, over the cap of "
            f"{MAX_NUM_TAXONOMY_TAGS}"
        )


def test_products_under_the_cap_are_untouched() -> None:
    """A product the cap never applied to must emit exactly what it always did.

    19, 20 and 6 eligible tags: the boundary and an ordinary product. Equality is
    against the uncapped list including order, so a selection rule that reordered
    or re-picked below the cap is a failure here.
    """
    with _tmpdir() as d:
        records, _report = _run_extract(Path(d), "en")

    for code in UNTRUNCATED:
        product = next(p for p in PRODUCTS if p["code"] == code)
        eligible = _eligible_labels(product["categories_tags"], "en")
        assert len(eligible) <= MAX_NUM_TAXONOMY_TAGS, (
            f"{code} now has {len(eligible)} eligible tags; it is no longer the "
            "under-the-cap control this test needs"
        )
        assert records[code.zfill(13)]["taxonomy_tags"] == eligible


def test_selection_changes_but_order_does_not() -> None:
    """Which tags survive changes; the order the survivors sit in does not.

    ``taxonomy_tags[0]`` is read back as the product's primary category label —
    ``attrs["Category"]`` and the generated description both carry it — and the
    flat list feeds pricing-bucket matching. So the fix is a *selection* change:
    every emitted list must still be a subsequence of the uncapped list, and
    still lead with the primary tag's label.
    """
    with _tmpdir() as d:
        records, _report = _run_extract(Path(d), "en")

    for product in PRODUCTS:
        record = records[product["code"].zfill(13)]
        emitted = record["taxonomy_tags"]
        eligible = _eligible_labels(product["categories_tags"], "en")
        assert _is_subsequence(emitted, eligible), (
            f"{product['code']} emitted {emitted}, which is not a subsequence of "
            f"the uncapped {eligible}; the flat list has been reordered"
        )
        assert emitted[0] == eligible[0], (
            f"{product['code']} leads with {emitted[0]!r}, not the primary tag's "
            f"label {eligible[0]!r}"
        )
        assert record["attrs"]["Category"] == emitted[0], (
            f"{product['code']} carries attrs['Category']="
            f"{record['attrs']['Category']!r} against a flat list leading with "
            f"{emitted[0]!r}"
        )


def test_the_run_reports_what_the_cap_dropped() -> None:
    """A truncation nothing counts is invisible in the emitted catalog.

    The flat field has no marker for "there was more", so a shortened list reads
    exactly like a short one. The run report names the count, the dropped labels
    and the longest list seen — measured here against the fixture rather than
    asserted to be non-zero, so a report that stopped counting fails.
    """
    with _tmpdir() as d:
        _records, report = _run_extract(Path(d), "en")

    counters = report["counters"]
    cap = report["taxonomy_tags_cap"]

    assert counters["products_with_truncated_taxonomy_tags"] == 5, counters
    assert counters["truncated_taxonomy_tags"] == 9, counters
    assert cap["max_taxonomy_tags"] == MAX_NUM_TAXONOMY_TAGS
    assert cap["max_eligible_tags_seen"] == 23, cap
    assert cap["products_truncated"] == 5, cap
    assert cap["tags_dropped"] == 9, cap
    assert cap["products_with_chain_over_cap"] == 0, cap

    dropped = {row["label"] for row in cap["top_dropped_labels"]}
    for labels in STILL_DROPPED.values():
        for label in labels:
            assert label in dropped, (
                f"{label!r} was dropped by the cap but is not named in the "
                f"report: {sorted(dropped)}"
            )
    for label in LOST_SEGMENTS.values():
        assert label not in dropped, (
            f"{label!r} is a chain segment and must never be what the cap drops"
        )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
