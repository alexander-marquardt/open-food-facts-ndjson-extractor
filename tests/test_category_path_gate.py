"""End-to-end tests for the ``--require-category-path`` clean-data gate.

These drive the real ``extract.main()`` CLI path — a synthetic taxonomy file, a
two-record input NDJSON, and the actual output/report files — rather than
re-implementing the gate against a mock. That matters because the gate is not a
pure function: its effective value is *resolved* at runtime (it auto-disables
when no taxonomy is loaded), so only the full path proves the default holds.

The fixture pairs one product whose category chain resolves against the taxonomy
with one whose ``categories_tags`` are absent from it. Both clear every earlier
filter (code, title, description, image), so the only thing that can separate
them is the category_path gate itself.

Run with ``pytest tests/`` or directly: ``python tests/test_category_path_gate.py``.
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


# Miniature taxonomy: enough to resolve a real root->leaf chain for one product.
TAXONOMY = {
    "en:beverages": _n("Beverages", []),
    "en:hot-beverages": _n("Hot beverages", ["en:beverages"]),
    "en:teas": _n("Teas", ["en:hot-beverages"]),
}


def _product(code: str, categories_tags: list[str]) -> dict:
    """A product that clears every filter *except* possibly the category_path gate."""
    return {
        "code": code,
        "lang": "en",
        "product_name_en": "Earl Grey Tea Bags",
        "generic_name_en": "Black tea with bergamot, 20 bags",
        "categories_tags": categories_tags,
        "images": {"front_en": {"rev": "7", "sizes": {"400": {"w": 400, "h": 400}}}},
    }


# One resolvable, one not. ``en:not-in-taxonomy`` has no node, so its chain is
# empty — the exact condition the gate drops.
RESOLVABLE_CODE = "3017620422003"
UNRESOLVABLE_CODE = "5449000000996"

INPUT_PRODUCTS = [
    _product(RESOLVABLE_CODE, ["en:beverages", "en:hot-beverages", "en:teas"]),
    _product(UNRESOLVABLE_CODE, ["en:not-in-taxonomy"]),
]


def _run_extract(tmp: Path, *extra_args: str) -> tuple[list[dict], dict]:
    """Run the extractor over the fixture; return (output records, report)."""
    taxonomy_path = tmp / "categories.json"
    taxonomy_path.write_text(json.dumps(TAXONOMY), encoding="utf-8")

    input_path = tmp / "products.jsonl"
    input_path.write_text(
        "".join(json.dumps(p) + "\n" for p in INPUT_PRODUCTS), encoding="utf-8"
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
    return tempfile.TemporaryDirectory(prefix="off-gate-test-")


def test_gate_is_on_by_default() -> None:
    """With no flags at all, the unresolvable product is dropped.

    This is the whole point of the default: the shipped NDJSON must not contain
    ``category_path: []`` records, which render as a hole in the hierarchical
    category facet.
    """
    with _tmpdir() as d:
        records, report = _run_extract(Path(d))

    ids = {r["id"] for r in records}
    assert ids == {RESOLVABLE_CODE}, f"expected only the resolvable product, got {ids}"
    assert all(r["category_path"] for r in records), (
        "a record with an empty category_path survived the default gate"
    )


def test_default_run_counts_the_dropped_record() -> None:
    """The drop is *reported*, not silent — ``missing_category_path`` must move.

    A gate that drops records without incrementing its counter is indistinguishable
    from an input that never had unresolvable products, which is how a silently
    inert gate would look.
    """
    with _tmpdir() as d:
        _, report = _run_extract(Path(d))

    counters = report["counters"]
    assert counters["missing_category_path"] == 1, counters
    assert counters["with_category_path"] == 1, counters
    assert counters["read"] == 2, counters
    assert counters["written"] == 1, counters
    assert "unresolved products dropped" in report["filters"]["category_path"]


def test_override_flag_keeps_unresolvable_records() -> None:
    """``--no-require-category-path`` restores the old permissive behaviour."""
    with _tmpdir() as d:
        records, report = _run_extract(Path(d), "--no-require-category-path")

    by_id = {r["id"]: r for r in records}
    assert set(by_id) == {RESOLVABLE_CODE, UNRESOLVABLE_CODE}, sorted(by_id)
    assert by_id[UNRESOLVABLE_CODE]["category_path"] == [], (
        "the kept record should carry the empty path that the default drops"
    )
    assert by_id[RESOLVABLE_CODE]["category_path"], "resolvable path went missing"

    counters = report["counters"]
    assert counters["missing_category_path"] == 0, counters
    assert counters["written"] == 2, counters
    assert "unresolved products dropped" not in report["filters"]["category_path"]


def test_explicit_flag_matches_the_default() -> None:
    """Passing ``--require-category-path`` explicitly is a no-op vs. the default.

    Callers (docs, scripts) may keep the flag for readability; it must not mean
    something different from omitting it.
    """
    with _tmpdir() as d:
        default_records, default_report = _run_extract(Path(d))
    with _tmpdir() as d:
        explicit_records, explicit_report = _run_extract(Path(d), "--require-category-path")

    assert default_records == explicit_records
    assert default_report["counters"] == explicit_report["counters"]


def test_gate_auto_disables_without_a_taxonomy() -> None:
    """With ``--no-taxonomy`` every path is empty, so the gate must not eat the run.

    Left naive, the default-on gate would drop 100% of records here. The resolver
    turns it off instead; this pins that safety valve.
    """
    with _tmpdir() as d:
        records, report = _run_extract(Path(d), "--no-taxonomy")

    assert len(records) == 2, f"the gate dropped records with no taxonomy loaded: {len(records)}"
    assert all(r["category_path"] == [] for r in records)
    assert report["counters"]["missing_category_path"] == 0
    assert report["filters"]["category_path"] == "disabled"


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
