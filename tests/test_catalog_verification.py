"""Tests for ``scripts/verify_catalog.py`` — does its exit status describe what
it just measured?

The script computed ``values_outside_pinned_snapshot``, printed it, and then
left it out of the sum that became the exit status. So it could report that
every value in a catalog was outside the snapshot the catalog was supposedly
built against and exit **0**, while ``scripts/verify_index.py`` treats the same
condition on the index side as a failure. These tests pin both ends of that rule
to the same answer, and pin the tolerance that is now the only way to get an
exception to it.

**How these run at the parent commit.** Every test drives ``main()`` through the
real argument parser, the real taxonomy load and the real NDJSON read, with
``sys.argv`` patched rather than an argv list passed — and the module is
imported as a module, with no ``from verify_catalog import ...`` of names the
fix introduced. Both together are what let this file be collected unchanged
against the commit before the fix, so the fail-before evidence is a *behavioural*
failure (exit 0 where 1 is asserted) rather than an ``ImportError``, which would
have proven only that a new symbol is new.

The taxonomy and the catalogs here are synthetic and tiny, which is deliberate:
the denominators have to be countable by hand for the fraction tolerance to be
testable at all (10 distinct values checked, 3 of them outside the snapshot, so
the difference between flooring and rounding 0.29 of 10 is the difference
between a pass and a failure). The same code was also run against the real
2026-08-03 en/es/fr catalogs; that evidence is in the pull request rather than
here, because those files are not in the repository.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import verify_catalog  # noqa: E402


# --------------------------------------------------------------------------- #
# a taxonomy small enough to count by hand
# --------------------------------------------------------------------------- #
TAXONOMY: Dict[str, Any] = {
    # The one node with no parents, so the one global root: label "Foods".
    "en:foods": {"name": {"en": "Foods"}},
    "en:snacks": {"name": {"en": "Snacks"}, "parents": ["en:foods"]},
    "en:biscuits": {"name": {"en": "Biscuits"}, "parents": ["en:snacks"]},
    "en:beverages": {"name": {"en": "Beverages"}, "parents": ["en:foods"]},
    "en:waters": {"name": {"en": "Waters"}, "parents": ["en:beverages"]},
    "en:dairy": {"name": {"en": "Dairy"}, "parents": ["en:foods"]},
    "en:cheeses": {"name": {"en": "Cheeses"}, "parents": ["en:dairy"]},
}

# Three records covering the seven labels above, each on a single anchored
# root->leaf address and each category at exactly one address. Every record
# carries ``category_path_primary`` as well: the verifier reads it for every gate
# scoped to the breadcrumb a product page leads with, and refuses a record that
# omits it rather than clearing those gates by having nothing to check.
CLEAN_RECORDS: List[Dict[str, Any]] = [
    {
        "id": "1",
        "taxonomy_tags": ["Foods", "Snacks", "Biscuits"],
        "category_path": ["Foods", "Foods/Snacks", "Foods/Snacks/Biscuits"],
        "category_path_primary": ["Foods", "Foods/Snacks", "Foods/Snacks/Biscuits"],
    },
    {
        "id": "2",
        "taxonomy_tags": ["Foods", "Beverages", "Waters"],
        "category_path": ["Foods", "Foods/Beverages", "Foods/Beverages/Waters"],
        "category_path_primary": ["Foods", "Foods/Beverages", "Foods/Beverages/Waters"],
    },
    {
        "id": "3",
        "taxonomy_tags": ["Foods", "Dairy", "Cheeses"],
        "category_path": ["Foods", "Foods/Dairy", "Foods/Dairy/Cheeses"],
        "category_path_primary": ["Foods", "Foods/Dairy", "Foods/Dairy/Cheeses"],
    },
]

# One more record whose flat values include three labels no node in the taxonomy
# above renders. Its path is a bare root, so it adds no new segment: the distinct
# values checked go 7 -> 10 and the values outside the snapshot go 0 -> 3.
OFF_SNAPSHOT_RECORD: Dict[str, Any] = {
    "id": "4",
    "taxonomy_tags": ["Foods", "Rocket fuel", "Gravel", "Neon"],
    "category_path": ["Foods"],
    "category_path_primary": ["Foods"],
}


def write_taxonomy(tmp_path: Path) -> Path:
    path = tmp_path / "categories.json"
    path.write_text(json.dumps(TAXONOMY), encoding="utf-8")
    return path


def write_catalog(tmp_path: Path, records: List[Dict[str, Any]], name: str = "cat.ndjson") -> Path:
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


def run(monkeypatch, capsys, *args: Any) -> Tuple[int, str, str]:
    """Invoke ``main()`` the way a shell does, and return (status, stdout, stderr)."""
    monkeypatch.setattr(sys, "argv", ["verify_catalog.py", *[str(a) for a in args]])
    status = verify_catalog.main()
    captured = capsys.readouterr()
    return status, captured.out, captured.err


# --------------------------------------------------------------------------- #
# the defect: a value outside the pinned snapshot has to reach the exit status
# --------------------------------------------------------------------------- #
def test_clean_catalog_exits_zero(tmp_path, monkeypatch, capsys):
    """The control. Without it, a verifier that always failed would pass the test
    below, and every assertion here would be about nothing."""
    taxonomy = write_taxonomy(tmp_path)
    catalog = write_catalog(tmp_path, CLEAN_RECORDS)

    status, out, _ = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    assert status == 0
    result = json.loads(out)
    assert result["values_outside_pinned_snapshot"] == 0
    assert result["values_checked_against_snapshot"] == 7


def test_value_outside_snapshot_exits_non_zero(tmp_path, monkeypatch, capsys):
    """**The fail-before test.** Reported and not gated, this exits 0.

    Note what is asserted about the report: the count was always printed. The bug
    was never that the number was missing, it was that the number was not fatal —
    so the assertion that bites is the one on the status, and the assertion on the
    count is there to prove the offending catalog really is offending.
    """
    taxonomy = write_taxonomy(tmp_path)
    catalog = write_catalog(tmp_path, CLEAN_RECORDS + [OFF_SNAPSHOT_RECORD])

    status, out, err = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    result = json.loads(out)
    assert result["values_outside_pinned_snapshot"] == 3
    assert status == 1, "a catalog using values the pinned snapshot does not explain must fail"
    assert "values_outside_pinned_snapshot" in result["failed"]
    assert "FAILED" in err


def test_off_snapshot_path_segment_also_fails(tmp_path, monkeypatch, capsys):
    """The other half of the same rule: a *path segment* outside the snapshot.

    The flat ``taxonomy_tags`` values and the ``category_path`` segments are checked
    against the same vocabulary and counted into the same number, so a catalog
    whose flat values are all fine can still be off-snapshot in its hierarchy.
    """
    taxonomy = write_taxonomy(tmp_path)
    record = {
        "id": "9",
        "taxonomy_tags": ["Foods"],
        "category_path": ["Foods", "Foods/Sawdust"],
        "category_path_primary": ["Foods", "Foods/Sawdust"],
    }
    catalog = write_catalog(tmp_path, CLEAN_RECORDS + [record])

    status, out, _ = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    result = json.loads(out)
    assert result["values_outside_pinned_snapshot"] == 1
    assert result["top_values_outside_pinned_snapshot"][0][0] == "Sawdust"
    assert status == 1


# --------------------------------------------------------------------------- #
# the tolerance: explicit, floored, and recorded
# --------------------------------------------------------------------------- #
def test_named_count_tolerance_permits_exactly_what_it_names(tmp_path, monkeypatch, capsys):
    taxonomy = write_taxonomy(tmp_path)
    catalog = write_catalog(tmp_path, CLEAN_RECORDS + [OFF_SNAPSHOT_RECORD])

    permitted, _, _ = run(
        monkeypatch,
        capsys,
        "--ndjson",
        catalog,
        "--taxonomy",
        taxonomy,
        "--allow-values-outside-snapshot",
        3,
    )
    refused, _, _ = run(
        monkeypatch,
        capsys,
        "--ndjson",
        catalog,
        "--taxonomy",
        taxonomy,
        "--allow-values-outside-snapshot",
        2,
    )

    assert permitted == 0
    assert refused == 1, "a tolerance of 2 must not absorb 3 off-snapshot values"


def test_fraction_tolerance_floors_rather_than_rounds(tmp_path, monkeypatch, capsys):
    """0.29 of the 10 distinct values checked is 2.9 values.

    Floored, that permits 2 and the catalog's 3 fail. Rounded, it would permit 3
    and the same catalog would pass — a fraction buying one more violation than
    anybody wrote down. This test is the difference between those two.
    """
    taxonomy = write_taxonomy(tmp_path)
    catalog = write_catalog(tmp_path, CLEAN_RECORDS + [OFF_SNAPSHOT_RECORD])

    status, out, _ = run(
        monkeypatch,
        capsys,
        "--ndjson",
        catalog,
        "--taxonomy",
        taxonomy,
        "--allow-values-outside-snapshot-fraction",
        0.29,
    )

    result = json.loads(out)
    assert result["values_checked_against_snapshot"] == 10
    assert result["values_outside_pinned_snapshot"] == 3
    assert result["values_outside_snapshot_tolerance"] == 2
    assert status == 1


def test_fraction_tolerance_that_covers_the_count_passes(tmp_path, monkeypatch, capsys):
    taxonomy = write_taxonomy(tmp_path)
    catalog = write_catalog(tmp_path, CLEAN_RECORDS + [OFF_SNAPSHOT_RECORD])

    status, out, _ = run(
        monkeypatch,
        capsys,
        "--ndjson",
        catalog,
        "--taxonomy",
        taxonomy,
        "--allow-values-outside-snapshot-fraction",
        0.3,
    )

    result = json.loads(out)
    assert result["values_outside_snapshot_tolerance"] == 3
    assert status == 0


def test_tolerance_in_force_is_recorded_in_the_json(tmp_path, monkeypatch, capsys):
    """The JSON is the artifact a build keeps; the terminal output is not.

    A tolerance that appears only on stdout leaves the recorded verdict saying
    "0 values outside the snapshot" with no trace of what was permitted, which is
    the same failure this whole issue is about one step removed.
    """
    taxonomy = write_taxonomy(tmp_path)
    catalog = write_catalog(tmp_path, CLEAN_RECORDS)
    out_path = tmp_path / "nested" / "verify.json"

    status, _, err = run(
        monkeypatch,
        capsys,
        "--ndjson",
        catalog,
        "--taxonomy",
        taxonomy,
        "--json",
        out_path,
        "--allow-values-outside-snapshot-fraction",
        0.5,
    )

    assert status == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["values_outside_snapshot_tolerance"] == 3  # floor(0.5 * 7)
    assert written["values_outside_snapshot_tolerance_source"] == (
        "--allow-values-outside-snapshot-fraction 0.5 (3 of 7)"
    )
    assert "0.5 (3 of 7)" in err


def test_default_tolerance_is_recorded_as_zero(tmp_path, monkeypatch, capsys):
    taxonomy = write_taxonomy(tmp_path)
    catalog = write_catalog(tmp_path, CLEAN_RECORDS)

    status, out, _ = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    result = json.loads(out)
    assert status == 0
    assert result["values_outside_snapshot_tolerance"] == 0
    assert result["values_outside_snapshot_tolerance_source"] == "0 (default: zero tolerance)"


@pytest.mark.parametrize(
    "args",
    [
        ["--allow-values-outside-snapshot", "-1"],
        ["--allow-values-outside-snapshot-fraction", "1.5"],
        ["--allow-values-outside-snapshot-fraction", "-0.1"],
        ["--allow-values-outside-snapshot", "1", "--allow-values-outside-snapshot-fraction", "0.1"],
    ],
)
def test_unusable_tolerances_are_refused_at_parse_time(tmp_path, monkeypatch, capsys, args):
    """A tolerance nobody can act on must not silently become a tolerance of
    everything — including the two forms being given at once, where the loser
    would be invisible."""
    taxonomy = write_taxonomy(tmp_path)
    catalog = write_catalog(tmp_path, CLEAN_RECORDS)

    with pytest.raises(SystemExit) as excinfo:
        run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy, *args)

    assert excinfo.value.code != 0


# --------------------------------------------------------------------------- #
# duplicate ids: reported by default, fatal only when asked for
# --------------------------------------------------------------------------- #
def test_duplicate_ids_are_reported_and_do_not_fail_by_default(tmp_path, monkeypatch, capsys):
    """The stated decision #35 asks for.

    Two records sharing an id is a property of the upstream dump, not of anything
    the extractor constructs, and the index is keyed by id — so the duplicate
    resolves to one document and ``distinct_ids``, which ``verify_index.py``
    compares ``_count`` against, already accounts for it. The 2026-08-03 build
    carries 1 / 3 / 81 of these in en / es / fr; failing on them by default would
    fail an accepted build against a uniqueness rule this project never adopted.
    """
    taxonomy = write_taxonomy(tmp_path)
    duplicate = dict(CLEAN_RECORDS[0])
    catalog = write_catalog(tmp_path, CLEAN_RECORDS + [duplicate])

    status, out, err = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    result = json.loads(out)
    assert result["records"] == 4
    assert result["distinct_ids"] == 3
    assert result["duplicate_id_instances"] == 1
    assert status == 0
    assert result["failed"] == []
    # Reported-only is a decision, so the output says so rather than leaving the
    # reader to infer it from a number that did nothing.
    assert "--require-unique-ids" in err


def test_require_unique_ids_makes_duplicates_fatal(tmp_path, monkeypatch, capsys):
    taxonomy = write_taxonomy(tmp_path)
    duplicate = dict(CLEAN_RECORDS[0])
    catalog = write_catalog(tmp_path, CLEAN_RECORDS + [duplicate])

    status, out, _ = run(
        monkeypatch,
        capsys,
        "--ndjson",
        catalog,
        "--taxonomy",
        taxonomy,
        "--require-unique-ids",
    )

    result = json.loads(out)
    assert status == 1
    assert result["failed"] == ["unique_ids"]
    assert result["require_unique_ids"] is True


def test_require_unique_ids_passes_a_catalog_without_duplicates(tmp_path, monkeypatch, capsys):
    taxonomy = write_taxonomy(tmp_path)
    catalog = write_catalog(tmp_path, CLEAN_RECORDS)

    status, _, _ = run(
        monkeypatch,
        capsys,
        "--ndjson",
        catalog,
        "--taxonomy",
        taxonomy,
        "--require-unique-ids",
    )

    assert status == 0


# --------------------------------------------------------------------------- #
# the gates that were already fatal must stay fatal
# --------------------------------------------------------------------------- #
def test_broken_chain_still_fails(tmp_path, monkeypatch, capsys):
    taxonomy = write_taxonomy(tmp_path)
    record = {
        "id": "5",
        "taxonomy_tags": ["Foods"],
        # Skips a level: the second element does not extend the first.
        "category_path": ["Foods", "Foods/Snacks/Biscuits"],
        "category_path_primary": ["Foods", "Foods/Snacks/Biscuits"],
    }
    catalog = write_catalog(tmp_path, [record])

    status, out, _ = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    result = json.loads(out)
    assert result["property_3_single_chain_violations"] == 1
    # The union check bites on the same record for its own reason: the ancestor
    # of "Foods/Snacks/Biscuits" is "Foods/Snacks", which is not a value.
    assert result["property_3_union_violations"] == 1
    assert status == 1
    assert set(result["failed"]) == {
        "property_3_single_primary_chain",
        "property_3_prefix_closed_union",
    }


def test_a_union_that_is_not_prefix_closed_fails(tmp_path, monkeypatch, capsys):
    """A second address whose ancestor levels were not emitted.

    The shape a naive "just append the alternates" change produces: the deeper
    value arrives, the breadcrumb level above it never does, and the drill-down
    facet has a level with no bucket to render.
    """
    taxonomy = write_taxonomy(tmp_path)
    record = {
        "id": "9",
        "taxonomy_tags": ["Foods", "Waters"],
        "category_path": [
            "Foods",
            "Foods/Beverages",
            "Foods/Beverages/Waters",
            "Foods/Snacks/Waters",
        ],
        "category_path_primary": ["Foods", "Foods/Beverages", "Foods/Beverages/Waters"],
    }
    catalog = write_catalog(tmp_path, [record])

    status, out, _ = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    result = json.loads(out)
    assert result["property_3_union_violations"] == 1
    assert result["property_3_single_chain_violations"] == 0
    assert status == 1
    assert result["failed"] == ["property_3_prefix_closed_union"]


def test_a_record_without_a_primary_address_fails(tmp_path, monkeypatch, capsys):
    """A catalog built before the primary existed must not pass vacuously.

    Every gate below the union check reads ``category_path_primary``. Without the
    field there is nothing to check, and "nothing was checked" would otherwise
    render as a clean verdict.
    """
    taxonomy = write_taxonomy(tmp_path)
    record = {
        "id": "10",
        "taxonomy_tags": ["Foods"],
        "category_path": ["Foods", "Foods/Beverages"],
    }
    catalog = write_catalog(tmp_path, [record])

    status, out, _ = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    result = json.loads(out)
    assert result["property_3_union_violations"] == 1
    assert status == 1
    assert "property_3_prefix_closed_union" in result["failed"]


def test_several_addresses_per_product_is_not_a_failure(tmp_path, monkeypatch, capsys):
    """The restored DAG itself: two addresses, both prefix-closed, primary first.

    The catalog this project now builds. It must pass — and the numbers a
    downstream aggregation has to be sized against must be reported from it.
    """
    taxonomy = write_taxonomy(tmp_path)
    record = {
        "id": "11",
        "taxonomy_tags": ["Foods", "Waters"],
        "category_path": [
            "Foods",
            "Foods/Beverages",
            "Foods/Beverages/Waters",
            "Foods/Snacks",
            "Foods/Snacks/Waters",
        ],
        "category_path_primary": ["Foods", "Foods/Beverages", "Foods/Beverages/Waters"],
    }
    catalog = write_catalog(tmp_path, [record])

    status, out, _ = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    result = json.loads(out)
    assert status == 0, result["failure_reasons"]
    assert result["records_at_multiple_addresses"] == 1
    assert result["categories_at_multiple_addresses"] == 1
    assert result["property_2_categories_at_multiple_primary_addresses"] == 0
    assert result["distinct_category_paths"] == 5
    assert result["max_category_path_values"] == 5


def test_category_at_two_addresses_still_fails(tmp_path, monkeypatch, capsys):
    taxonomy = write_taxonomy(tmp_path)
    records = [
        {
            "id": "6",
            "taxonomy_tags": ["Foods", "Waters"],
            "category_path": ["Foods", "Foods/Beverages", "Foods/Beverages/Waters"],
            "category_path_primary": [
                "Foods",
                "Foods/Beverages",
                "Foods/Beverages/Waters",
            ],
        },
        {
            "id": "7",
            "taxonomy_tags": ["Foods", "Waters"],
            "category_path": ["Foods", "Foods/Snacks", "Foods/Snacks/Waters"],
            "category_path_primary": ["Foods", "Foods/Snacks", "Foods/Snacks/Waters"],
        },
    ]
    catalog = write_catalog(tmp_path, records)

    status, out, _ = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    result = json.loads(out)
    assert result["property_2_categories_at_multiple_primary_addresses"] == 1
    assert status == 1
    assert result["failed"] == ["property_2_one_primary_address_per_category"]


def test_unanchored_chain_still_fails(tmp_path, monkeypatch, capsys):
    taxonomy = write_taxonomy(tmp_path)
    record = {
        "id": "8",
        "taxonomy_tags": ["Snacks"],
        # Starts mid-taxonomy: "Snacks" is not the label of a global root.
        "category_path": ["Snacks", "Snacks/Biscuits"],
        "category_path_primary": ["Snacks", "Snacks/Biscuits"],
    }
    catalog = write_catalog(tmp_path, [record])

    status, out, _ = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    result = json.loads(out)
    assert result["unanchored_chains"] == 1
    assert status == 1
    assert result["failed"] == ["anchoring"]


def test_every_failing_gate_is_named_at_once(tmp_path, monkeypatch, capsys):
    """One run, several broken properties: the report names all of them.

    The old sum collapsed the reasons into a single integer, so a failing run
    printed a blob of JSON and exited 1 without saying which rule it had broken.
    """
    taxonomy = write_taxonomy(tmp_path)
    records = [
        {
            "id": "a",
            "taxonomy_tags": ["Sawdust"],
            "category_path": ["Snacks", "Snacks/Biscuits"],
            "category_path_primary": ["Snacks", "Snacks/Biscuits"],
        },
        {
            "id": "b",
            "taxonomy_tags": ["Foods"],
            "category_path": ["Foods", "Foods/Waters"],
            "category_path_primary": ["Foods", "Foods/Waters"],
        },
        {
            "id": "c",
            "taxonomy_tags": ["Foods"],
            "category_path": ["Foods", "Foods/Beverages", "Foods/Beverages/Waters"],
            "category_path_primary": [
                "Foods",
                "Foods/Beverages",
                "Foods/Beverages/Waters",
            ],
        },
    ]
    catalog = write_catalog(tmp_path, records)

    status, out, _ = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    result = json.loads(out)
    assert status == 1
    assert set(result["failed"]) == {
        "anchoring",
        "property_2_one_primary_address_per_category",
        "values_outside_pinned_snapshot",
    }
    assert len(result["failure_reasons"]) == 3


# --------------------------------------------------------------------------- #
# a run that verified nothing is not a clean verdict
# --------------------------------------------------------------------------- #
def test_empty_catalog_does_not_report_clean(tmp_path, monkeypatch, capsys):
    """Every count is zero, so every gate passes vacuously — including the one
    this issue adds. A verifier that has looked at nothing has cleared nothing,
    and it is also the denominator a fraction tolerance would divide by."""
    taxonomy = write_taxonomy(tmp_path)
    catalog = write_catalog(tmp_path, [])

    status, out, _ = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    result = json.loads(out)
    assert result["records"] == 0
    assert status == 1
    assert result["failed"] == ["nothing_verified"]


def test_catalog_with_no_taxonomy_tags_at_all_does_not_report_clean(tmp_path, monkeypatch, capsys):
    """Records exist, but not one value was checked against the snapshot."""
    taxonomy = write_taxonomy(tmp_path)
    catalog = write_catalog(tmp_path, [{"id": "1"}, {"id": "2"}])

    status, out, _ = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    result = json.loads(out)
    assert result["records"] == 2
    assert result["values_checked_against_snapshot"] == 0
    assert status == 1
    assert result["failed"] == ["nothing_verified"]


# --------------------------------------------------------------------------- #
# "could not run" is exit 2, not exit 1
# --------------------------------------------------------------------------- #
def test_missing_catalog_exits_two(tmp_path, monkeypatch, capsys):
    taxonomy = write_taxonomy(tmp_path)

    status, out, err = run(
        monkeypatch, capsys, "--ndjson", tmp_path / "nope.ndjson", "--taxonomy", taxonomy
    )

    assert status == 2, "a missing file is not a verdict about a catalog"
    assert out == ""
    assert "cannot read the catalog" in err


def test_missing_taxonomy_exits_two(tmp_path, monkeypatch, capsys):
    catalog = write_catalog(tmp_path, CLEAN_RECORDS)

    status, _, err = run(
        monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", tmp_path / "nope.json"
    )

    assert status == 2
    assert "cannot read the taxonomy snapshot" in err


def test_unparseable_line_exits_two_and_names_the_line(tmp_path, monkeypatch, capsys):
    taxonomy = write_taxonomy(tmp_path)
    catalog = tmp_path / "broken.ndjson"
    catalog.write_text(
        json.dumps(CLEAN_RECORDS[0]) + "\n" + "{not json\n", encoding="utf-8"
    )

    status, _, err = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    assert status == 2
    assert "line 2" in err


def test_non_object_line_exits_two(tmp_path, monkeypatch, capsys):
    """A JSON array parses fine and then has no ``.get``. Without the type check
    this is an ``AttributeError`` traceback and an exit status of 1."""
    taxonomy = write_taxonomy(tmp_path)
    catalog = tmp_path / "array.ndjson"
    catalog.write_text("[1, 2, 3]\n", encoding="utf-8")

    status, _, err = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    assert status == 2
    assert "not a JSON object" in err


# --------------------------------------------------------------------------- #
# stdout stays machine-readable
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("records", [CLEAN_RECORDS, CLEAN_RECORDS + [OFF_SNAPSHOT_RECORD]])
def test_stdout_is_json_and_nothing_else(tmp_path, monkeypatch, capsys, records):
    """The build workflow redirects stdout to ``verify_<lang>_v14.json``.

    A summary line printed to stdout on the failing path would corrupt exactly
    the runs whose record matters most, and would do it only on failure — the
    condition least likely to be exercised before it ships.
    """
    taxonomy = write_taxonomy(tmp_path)
    catalog = write_catalog(tmp_path, records)

    _, out, err = run(monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy)

    assert json.loads(out)  # parses whole, so nothing else was written there
    assert "records" in err  # ... and the human summary did go somewhere


def test_json_file_matches_stdout(tmp_path, monkeypatch, capsys):
    taxonomy = write_taxonomy(tmp_path)
    catalog = write_catalog(tmp_path, CLEAN_RECORDS + [OFF_SNAPSHOT_RECORD])
    out_path = tmp_path / "verify.json"

    status, out, _ = run(
        monkeypatch, capsys, "--ndjson", catalog, "--taxonomy", taxonomy, "--json", out_path
    )

    assert status == 1
    assert json.loads(out_path.read_text(encoding="utf-8")) == json.loads(out)
