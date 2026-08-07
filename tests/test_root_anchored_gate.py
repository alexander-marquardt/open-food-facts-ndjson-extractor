"""``--require-category-path`` must refuse a chain that stops short of a root.

Why this is a separate defect from an empty path
------------------------------------------------
Every chain is walked to a root of the parent map built for *this* run. A node
whose only parents were pruned out of that map is promoted to a root of it while
remaining a child of the taxonomy, and the chain that starts there is truncated:
it files its categories at an address that exists in no other catalog. The path
itself is non-empty and perfectly well-formed, so the emptiness test the gate
used to run cannot see it.

What can prune a parent, and what no longer can
-----------------------------------------------
The language filter used to, and that was the bulk of it — 90 of the 161 roots
an English run saw were manufactured that way, ``en:pate`` among them. Traversal
is language-blind now (``tests/test_language_blind_traversal.py``), so a foreign
parent is walked through rather than pruned and a default run has nothing left
to refuse.

``--category-exclude`` still prunes, and it is what these tests sever with. An
operator naming a mid-taxonomy node orphans everything beneath it, which is the
same defect from the chain's point of view and the reason the gate is still here.
Severing deliberately, rather than relying on a filter that no longer severs, is
also what keeps these tests honest: the rule is exercised by a mechanism that
really exists.

What is tested, and against what
--------------------------------
Both halves drive the real ``extract.main()`` CLI, because the gate's effective
value is resolved at runtime and only the full path proves it.

* :data:`TAXONOMY` is a hand-built miniature that isolates the mechanism — one
  branch severed by the exclusion, one intact — so a failure points at the rule
  rather than at the data.
* ``tests/fixtures/off_unanchored_chains.json`` is the same rule against **real
  records**: verbatim ``categories_tags`` from the public export and the complete
  upward closure of those tags from the public taxonomy. The closure is complete,
  so a node is a root in the fixture exactly when it is a root in the full
  taxonomy — which is the only property that makes the fixture able to reproduce
  this defect at all.

Run with ``pytest tests/`` or directly: ``python tests/test_root_anchored_gate.py``.
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
from off_demo_extract.taxonomy import global_roots, unanchored_head  # noqa: E402

PRICING_CONFIG = REPO_ROOT / "config" / "pricing_buckets.json"
REAL_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "off_unanchored_chains.json"


def _n(name: str, parents: List[str]) -> Dict[str, Any]:
    return {"name": {"en": name}, "parents": parents}


# Miniature taxonomy, shaped like the real defect. ``fr:charcuteries`` is a real
# node with a real English parent; excluding it (see :data:`SEVER`) takes
# ``en:pate``'s only ancestry with it and promotes ``en:pate`` to a root of the
# run's map. ``en:snacks`` is the control: a genuine taxonomy root, one segment,
# legitimate — and it is excluded from nothing, so it must survive.
TAXONOMY = {
    "en:prepared-meats": _n("Prepared meats", []),
    "fr:charcuteries": _n("Charcuteries", ["en:prepared-meats"]),
    "en:pate": _n("Pate", ["fr:charcuteries"]),
    "en:pork-pates": _n("Pork pates", ["en:pate"]),
    "en:snacks": _n("Snacks", []),
}

# Cutting the one node that stands between en:pate and the taxonomy's root. The
# default exclusion list is replaced wholesale, which is harmless here: none of
# its ids is in this taxonomy.
SEVER = ("--category-exclude", "fr:charcuteries")

TRUNCATED_CODE = "3017620422003"
ROOT_ONLY_CODE = "5449000000996"


def _product(code: str, categories_tags: List[str]) -> Dict[str, Any]:
    """A product that clears every filter *except* possibly the category_path gate."""
    return {
        "code": code,
        "lang": "en",
        "product_name_en": "Country Style Pork Pate",
        "generic_name_en": "Coarse pork pate, 180g terrine",
        "categories_tags": categories_tags,
        "images": {"front_en": {"rev": "7", "sizes": {"400": {"w": 400, "h": 400}}}},
    }


INPUT_PRODUCTS = [
    # Chain resolves to ["Pate", "Pate/Pork pates"] — non-empty, and truncated.
    _product(TRUNCATED_CODE, ["en:prepared-meats", "fr:charcuteries", "en:pate", "en:pork-pates"]),
    # Chain resolves to ["Snacks"] — one segment, and legitimately root-only.
    _product(ROOT_ONLY_CODE, ["en:snacks"]),
]


def _run_extract(
    tmp: Path, taxonomy: Dict[str, Any], products: List[Dict[str, Any]], *extra_args: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run the extractor over the given taxonomy/products; return (records, report)."""
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
            *extra_args,
        ]
    )
    assert rc == 0, f"extractor exited {rc}"

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return records, report


def _tmpdir() -> Any:
    return tempfile.TemporaryDirectory(prefix="off-anchor-test-")


# ---------------------------------------------------------------------------
# The rule, in isolation
# ---------------------------------------------------------------------------

def test_global_roots_ignores_what_the_run_pruned() -> None:
    """A node with a parent is not a root, however the run is configured.

    This is the fact the gate rests on, and it has to be read off the taxonomy
    file rather than off the run: ``build_canonical_parent_map`` calls
    ``en:pate`` a root once ``fr:charcuteries`` is excluded. The taxonomy does
    not, and that disagreement is precisely what the gate refuses.
    """
    roots = global_roots(TAXONOMY)
    assert roots == {"en:prepared-meats", "en:snacks"}, sorted(roots)
    assert "en:pate" not in roots


def test_unanchored_head_names_the_offending_node() -> None:
    """The refusal is specific: it returns *which* node the chain started at."""
    roots = global_roots(TAXONOMY)
    assert unanchored_head(["en:pate", "en:pork-pates"], roots) == "en:pate"
    assert unanchored_head(["en:snacks"], roots) is None
    assert unanchored_head(["en:prepared-meats", "fr:charcuteries"], roots) is None
    # An empty chain is the *other* defect, with its own counter — not this one.
    assert unanchored_head([], roots) is None


# ---------------------------------------------------------------------------
# The gate, end to end
# ---------------------------------------------------------------------------

def test_truncated_chain_is_dropped_and_root_only_is_kept() -> None:
    """The gate must separate the two one-time-indistinguishable cases.

    Rejecting every short path would throw away the legitimately root-only
    product; accepting every non-empty path is the defect. Only a rule that asks
    the taxonomy can keep one and drop the other, so both are asserted together.
    """
    with _tmpdir() as d:
        records, _report = _run_extract(Path(d), TAXONOMY, INPUT_PRODUCTS, *SEVER)

    ids = {r["id"] for r in records}
    assert ids == {ROOT_ONLY_CODE}, (
        f"expected only the root-anchored product to survive, got {sorted(ids)}"
    )
    assert records[0]["category_path"] == ["Snacks"], records[0]["category_path"]


def test_the_drop_is_counted_and_the_head_is_named() -> None:
    """A silent drop is indistinguishable from an input that had none.

    The counter moves *and* the report names the node the chain started at, so
    the refusal is a work item rather than a percentage.
    """
    with _tmpdir() as d:
        _records, report = _run_extract(Path(d), TAXONOMY, INPUT_PRODUCTS, *SEVER)

    counters = report["counters"]
    assert counters["unanchored_category_path"] == 1, counters
    assert counters["missing_category_path"] == 0, (
        "the truncated product has a non-empty path; it must not be counted as missing"
    )
    assert counters["written"] == 1, counters

    anchoring = report["category_path_anchoring"]
    assert anchoring["products_with_unanchored_path"] == 1, anchoring
    assert anchoring["distinct_unanchored_heads"] == 1, anchoring
    assert anchoring["top_unanchored_heads"] == [{"category": "en:pate", "products": 1}], (
        anchoring
    )
    assert "stops short of a taxonomy root" in report["filters"]["category_path"]


def test_override_keeps_the_truncated_record_but_still_reports_it() -> None:
    """``--no-require-category-path`` stops the drop, not the measurement.

    With the gate off this report block is the only place the number exists at
    all, so it must be populated even though nothing was refused.
    """
    with _tmpdir() as d:
        records, report = _run_extract(
            Path(d), TAXONOMY, INPUT_PRODUCTS, *SEVER, "--no-require-category-path"
        )

    by_id = {r["id"]: r for r in records}
    assert set(by_id) == {TRUNCATED_CODE, ROOT_ONLY_CODE}, sorted(by_id)
    assert by_id[TRUNCATED_CODE]["category_path_primary"] == ["Pate", "Pate/Pork pates"]

    assert report["counters"]["unanchored_category_path"] == 0, report["counters"]
    assert report["category_path_anchoring"]["products_with_unanchored_path"] == 1, (
        "the gate being off must not suppress the count"
    )


def test_a_fully_anchored_run_reports_nothing_unanchored() -> None:
    """The report block must be quiet when there is nothing to say.

    Without this, a block that is always populated — or an audit wired to record
    unconditionally — would satisfy the assertions above while meaning nothing.
    """
    with _tmpdir() as d:
        _records, report = _run_extract(
            Path(d), TAXONOMY, [_product(ROOT_ONLY_CODE, ["en:snacks"])], *SEVER
        )

    anchoring = report["category_path_anchoring"]
    assert anchoring["products_with_unanchored_path"] == 0, anchoring
    assert anchoring["top_unanchored_heads"] == [], anchoring
    assert report["counters"]["unanchored_category_path"] == 0, report["counters"]


# ---------------------------------------------------------------------------
# The same rule against real export records
# ---------------------------------------------------------------------------

def _real_fixture() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    data = json.loads(REAL_FIXTURE.read_text(encoding="utf-8"))
    products = [
        _product(p["code"], list(p["categories_tags"])) for p in data["products"]
    ]
    return data["taxonomy"], products


# The two fixture products whose lineage runs through a node with no other
# route to a root: excluding that node is the only way an operator can strand
# them. The other five reach a root by an English path regardless, which is why
# they stay anchored under the same exclusion and make the assertion two-sided.
REAL_SEVER = ("--category-exclude", "fr:cereales-preparees,xx:dumplings")
REAL_UNANCHORED = {
    "0017400140328": "en:prepared-rices",
    "0087703021877": "en:chinese-dumplings",
}
REAL_ANCHORED = {
    "0000168175589",
    "0000111301201",
    "0000111048403",
    "0055652001899",
    "0044700079195",
}


def test_real_records_with_a_severed_ancestry_are_refused() -> None:
    """Real tags, real taxonomy shape — an excluded ancestor strands its subtree.

    The synthetic taxonomy above proves the rule fires; this proves it fires on
    the shape the export actually produces, which is where the defect was found.
    """
    taxonomy, products = _real_fixture()
    with _tmpdir() as d:
        records, report = _run_extract(Path(d), taxonomy, products, *REAL_SEVER)

    kept = {r["id"] for r in records}
    assert kept == REAL_ANCHORED, f"unexpected survivors: {sorted(kept)}"
    assert report["counters"]["unanchored_category_path"] == len(REAL_UNANCHORED), (
        report["counters"]
    )

    named = {
        entry["category"]
        for entry in report["category_path_anchoring"]["top_unanchored_heads"]
    }
    assert named == set(REAL_UNANCHORED.values()), sorted(named)


def test_real_records_would_have_passed_the_old_emptiness_test() -> None:
    """Each refused record carries a non-empty path, so emptiness cannot catch it.

    This is what makes the defect a gate defect rather than a chain defect: the
    old rule is not merely weaker here, it is blind. Asserted on the paths the
    permissive run actually emits, including the one-segment ``Chinese
    dumplings`` — length is no signal either.
    """
    taxonomy, products = _real_fixture()
    with _tmpdir() as d:
        records, report = _run_extract(
            Path(d), taxonomy, products, *REAL_SEVER, "--no-require-category-path"
        )

    by_id = {r["id"]: r for r in records}
    assert set(by_id) == set(REAL_UNANCHORED) | REAL_ANCHORED, sorted(by_id)

    roots = global_roots(taxonomy)
    for code, head in REAL_UNANCHORED.items():
        path = by_id[code]["category_path"]
        assert path, f"{code} has no path at all; it is the wrong fixture for this test"
        assert head not in roots, f"{head} is a taxonomy root; fixture no longer bites"
    assert by_id["0087703021877"]["category_path_primary"] == ["Chinese dumplings"]
    assert by_id["0017400140328"]["category_path_primary"] == [
        "Prepared rices",
        "Prepared rices/Fried rice",
    ]

    assert report["counters"]["missing_category_path"] == 0, (
        "no fixture product should be failing the emptiness test"
    )
    assert report["category_path_anchoring"]["products_with_unanchored_path"] == len(
        REAL_UNANCHORED
    ), report["category_path_anchoring"]


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
