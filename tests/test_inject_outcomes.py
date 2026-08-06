"""Tests for ``scripts/inject_category_path.py`` — does the loader's exit status
describe what the run actually did?

**Where the proof sits, and where it does not.** Unlike
``test_index_verification.py``, whose envelopes are live captures, the bulk
responses here are *constructed*, not recorded. They could not be recorded: a
``_bulk`` response can only be obtained by writing to a cluster, and the cluster
these tests were written against is shared and read-only for this work. So the
seam under test is the loader's own boundary — the ``bulk`` callable it is
handed — and what these tests prove is that *given* a response of the documented
shape, the loader classifies it and gates on it correctly. What they cannot
prove is that Elasticsearch emits exactly that shape; that half rests on the
documented bulk contract and on the status codes the script was already reading
before this change (``200``/``201``/``404``), which is the same evidence the
original code was written from.

The shapes replayed below are therefore kept minimal and literal: an ``items``
array of one-key objects whose key is the operation (``update``), carrying
``status``, ``_id``, ``result`` on success and an ``error`` object on failure.
``document_missing_exception`` is the error Elasticsearch attaches to a partial
update against an id the index does not hold — the whole subject of the bug.

Every test drives ``main()`` through the real argument parser, the real file
read and the real flush path, patching only the HTTP call. ``main`` is invoked
with no arguments and ``sys.argv`` patched, rather than by passing an argv list,
so this file runs unchanged against the commit before the fix — which is how the
fail-before evidence for the central test was produced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import inject_category_path  # noqa: E402
from inject_category_path import (  # noqa: E402
    Outcome,
    Tolerance,
    gate,
    inject,
    main,
)


# --------------------------------------------------------------------------- #
# response construction: the shape a bulk of partial updates comes back as
# --------------------------------------------------------------------------- #
def updated_item(doc_id: str, result: str = "updated") -> Dict[str, Any]:
    return {
        "update": {
            "_index": "catalog_en_v8",
            "_id": doc_id,
            "_version": 2,
            "result": result,
            "status": 200,
        }
    }


def missing_item(doc_id: str) -> Dict[str, Any]:
    return {
        "update": {
            "_index": "catalog_en_v8",
            "_id": doc_id,
            "status": 404,
            "error": {
                "type": "document_missing_exception",
                "reason": f"[{doc_id}]: document missing",
                "index": "catalog_en_v8",
            },
        }
    }


def conflict_item(doc_id: str) -> Dict[str, Any]:
    return {
        "update": {
            "_index": "catalog_en_v8",
            "_id": doc_id,
            "status": 409,
            "error": {"type": "version_conflict_engine_exception", "reason": "current version"},
        }
    }


def rejected_item(doc_id: str) -> Dict[str, Any]:
    return {
        "update": {
            "_index": "catalog_en_v8",
            "_id": doc_id,
            "status": 400,
            "error": {"type": "mapper_parsing_exception", "reason": "failed to parse field"},
        }
    }


def envelope(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """A whole bulk response. ``errors`` reflects only *bulk-level* trouble.

    It is ``false`` for a page of nothing but 404s, which is the fact the bug
    rests on: the request succeeded, so a loader reading this flag sees success.
    """
    return {"took": 12, "errors": False, "items": list(items)}


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
def write_ndjson(tmp_path: Path, records: Sequence[Dict[str, Any]]) -> Path:
    path = tmp_path / "extract.ndjson"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


def catalog(count: int, empty_paths: int = 0) -> List[Dict[str, Any]]:
    records = [
        {"id": f"gtin-{i:03d}", "category_path": ["Foods", "Foods/Snacks"]}
        for i in range(count)
    ]
    records += [{"id": f"empty-{i:03d}", "category_path": []} for i in range(empty_paths)]
    return records


def run_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    records: Sequence[Dict[str, Any]],
    responder: Callable[[List[str]], Dict[str, Any]],
    extra_argv: Sequence[str] = (),
) -> int:
    ndjson = write_ndjson(tmp_path, records)
    monkeypatch.setattr(inject_category_path, "_bulk", responder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inject_category_path.py",
            "--index",
            "catalog_en_v8",
            "--ndjson",
            str(ndjson),
            *extra_argv,
        ],
    )
    return main()


def responder_for(
    classify: Callable[[str], Dict[str, Any]], calls: List[List[str]] | None = None
) -> Callable[[List[str]], Dict[str, Any]]:
    """Answer each batch by classifying every id the loader actually sent."""

    def respond(lines: List[str]) -> Dict[str, Any]:
        if calls is not None:
            calls.append(list(lines))
        ids = [json.loads(line)["update"]["_id"] for line in lines[::2]]
        return envelope([classify(doc_id) for doc_id in ids])

    return respond


# --------------------------------------------------------------------------- #
# the bug: a run that touched nothing
# --------------------------------------------------------------------------- #
def test_a_run_whose_every_target_is_absent_does_not_report_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole issue in one test.

    Wrong index, empty index, index built from another locale's extract — all
    three arrive here: every update 404s, the bulk request itself succeeded, and
    the process must not tell its caller that the load worked.
    """
    status = run_loader(monkeypatch, tmp_path, catalog(12), responder_for(missing_item))

    assert status != 0
    captured = capsys.readouterr()
    assert "sent=12" in captured.out
    assert "applied=0" in captured.out
    assert "not_found=12" in captured.out
    assert "FAILED" in captured.err


def test_the_failure_names_ids_so_wrong_index_is_distinguishable_from_stale_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_loader(monkeypatch, tmp_path, catalog(12), responder_for(missing_item))

    out = capsys.readouterr().out
    assert "first ids not in the index:" in out
    assert "gtin-000" in out


def test_a_run_that_genuinely_updates_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = run_loader(monkeypatch, tmp_path, catalog(12), responder_for(updated_item))

    assert status == 0
    out = capsys.readouterr().out
    assert "applied=12" in out
    assert "applied rate: 100.00%" in out


# --------------------------------------------------------------------------- #
# the threshold policy
# --------------------------------------------------------------------------- #
def _partial(missing_ids: Sequence[str]) -> Callable[[str], Dict[str, Any]]:
    def classify(doc_id: str) -> Dict[str, Any]:
        return missing_item(doc_id) if doc_id in missing_ids else updated_item(doc_id)

    return classify


def test_a_partial_miss_fails_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """60% applied is not a success unless somebody said so in the command line."""
    missing = [f"gtin-{i:03d}" for i in range(4)]
    status = run_loader(
        monkeypatch, tmp_path, catalog(10), responder_for(_partial(missing))
    )

    assert status == 1
    captured = capsys.readouterr()
    assert "applied rate: 60.00%" in captured.out
    assert "4 of 10 ids are not in the index (tolerance 0)" in captured.err


def test_an_explicit_absolute_tolerance_is_honoured_and_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = [f"gtin-{i:03d}" for i in range(4)]
    status = run_loader(
        monkeypatch,
        tmp_path,
        catalog(10),
        responder_for(_partial(missing)),
        extra_argv=["--allow-missing", "4"],
    )

    assert status == 0
    assert "missing tolerance: --allow-missing 4" in capsys.readouterr().out


def test_a_tolerance_one_short_still_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = [f"gtin-{i:03d}" for i in range(4)]
    status = run_loader(
        monkeypatch,
        tmp_path,
        catalog(10),
        responder_for(_partial(missing)),
        extra_argv=["--allow-missing", "3"],
    )

    assert status == 1


def test_a_fractional_tolerance_is_floored_and_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """0.15 of 20 is 3 documents, not 3.0 — and 4 misses is still a failure."""
    status = run_loader(
        monkeypatch,
        tmp_path,
        catalog(20),
        responder_for(_partial([f"gtin-{i:03d}" for i in range(3)])),
        extra_argv=["--allow-missing-fraction", "0.15"],
    )
    assert status == 0
    assert "(3 of 20)" in capsys.readouterr().out

    status = run_loader(
        monkeypatch,
        tmp_path,
        catalog(20),
        responder_for(_partial([f"gtin-{i:03d}" for i in range(4)])),
        extra_argv=["--allow-missing-fraction", "0.15"],
    )
    assert status == 1


def test_no_tolerance_can_authorise_a_run_that_applied_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one rule the operator cannot switch off.

    ``--allow-missing 1000000`` is a statement about a few stale ids. It is not
    a statement that the index may be the wrong one, which is what a zero-match
    run means — so the zero-match check runs ahead of the tolerance.
    """
    for extra in (["--allow-missing", "1000000"], ["--allow-missing-fraction", "1.0"]):
        status = run_loader(
            monkeypatch, tmp_path, catalog(9), responder_for(missing_item), extra_argv=extra
        )
        assert status == 1, extra
        assert "nothing was applied" in capsys.readouterr().err


def test_an_input_that_addresses_no_document_is_not_a_successful_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Zero updates issued: an empty extract, or one whose every path is empty.

    Nothing 404s because nothing was sent, so a gate written only around the
    response would pass this. It is the same failure — a loader reporting a
    completed load having loaded nothing.
    """
    status = run_loader(
        monkeypatch,
        tmp_path,
        catalog(0, empty_paths=5),
        responder_for(updated_item),
    )

    assert status == 1
    assert "nothing was sent" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the other per-item outcomes
# --------------------------------------------------------------------------- #
def test_noop_counts_as_applied_but_is_reported_separately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Re-running the loader is legitimate: the documents already carry the value.

    A ``noop`` is not a miss — the index holds the id and holds the field — so
    it must not fail the run. It is still worth its own number, because "every
    document was a noop" and "every document was updated" describe different
    situations and only one of them means this run did anything.
    """
    status = run_loader(
        monkeypatch,
        tmp_path,
        catalog(6),
        responder_for(lambda doc_id: updated_item(doc_id, result="noop")),
    )

    assert status == 0
    out = capsys.readouterr().out
    assert "applied=6 (updated=0 noop=6)" in out


def test_a_version_conflict_is_counted_apart_from_a_rejection_and_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """409 and 400 are different problems and are not summed into one number."""

    def classify(doc_id: str) -> Dict[str, Any]:
        if doc_id == "gtin-000":
            return conflict_item(doc_id)
        if doc_id == "gtin-001":
            return rejected_item(doc_id)
        return updated_item(doc_id)

    status = run_loader(monkeypatch, tmp_path, catalog(8), responder_for(classify))

    assert status == 1
    captured = capsys.readouterr()
    assert "conflict=1" in captured.out
    assert "failed=1" in captured.out
    assert "version conflict" in captured.err
    assert "1 updates failed" in captured.err


def test_documents_the_response_never_mentions_are_not_assumed_applied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A short ``items`` array is silence, not consent.

    Counting only what the response talks about would let a truncated response
    turn 100 sent documents into "everything reported succeeded".
    """

    def respond(lines: List[str]) -> Dict[str, Any]:
        ids = [json.loads(line)["update"]["_id"] for line in lines[::2]]
        return envelope([updated_item(doc_id) for doc_id in ids[:-2]])

    status = run_loader(monkeypatch, tmp_path, catalog(7), respond)

    assert status == 1
    captured = capsys.readouterr()
    assert "unaccounted=2" in captured.out
    assert "accounted for neither success nor failure" in captured.err


def test_an_item_for_another_operation_is_not_read_as_an_update(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loader issues ``update`` and reads ``update``.

    An ``index`` item here would mean the response does not describe the request
    that was sent; treating its 200 as an applied update would make the totals
    self-consistent while describing something else.
    """
    status = run_loader(
        monkeypatch,
        tmp_path,
        catalog(3),
        responder_for(lambda doc_id: {"index": {"_id": doc_id, "status": 200}}),
    )

    assert status == 1


# --------------------------------------------------------------------------- #
# the request the loader builds, and the operational failure mode
# --------------------------------------------------------------------------- #
def test_every_document_is_sent_once_across_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: List[List[str]] = []
    status = run_loader(
        monkeypatch,
        tmp_path,
        catalog(7),
        responder_for(updated_item, calls),
        extra_argv=["--batch", "2"],
    )

    assert status == 0
    assert [len(lines) // 2 for lines in calls] == [2, 2, 2, 1]
    actions = [json.loads(line) for call in calls for line in call[::2]]
    payloads = [json.loads(line) for call in calls for line in call[1::2]]
    assert [action["update"]["_id"] for action in actions] == [
        f"gtin-{i:03d}" for i in range(7)
    ]
    assert {action["update"]["_index"] for action in actions} == {"catalog_en_v8"}
    assert payloads[0] == {"doc": {"category_path": ["Foods", "Foods/Snacks"]}}


# --------------------------------------------------------------------------- #
# the empty-path policy: which one ran, and can the operator choose it
# --------------------------------------------------------------------------- #
def _sent_ids(calls: Sequence[List[str]]) -> List[str]:
    return [json.loads(line)["update"]["_id"] for call in calls for line in call[::2]]


def _payloads(calls: Sequence[List[str]]) -> List[Dict[str, Any]]:
    return [json.loads(line) for call in calls for line in call[1::2]]


def test_an_empty_path_is_not_sent_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A partial update writes what it is handed, so ``[]`` would erase the field.

    The default is a backfill: a record the extract resolves no path for is
    counted and reported, not turned into an instruction to empty the document.
    """
    calls: List[List[str]] = []
    status = run_loader(
        monkeypatch, tmp_path, catalog(6, empty_paths=4), responder_for(updated_item, calls)
    )

    assert status == 0
    assert _sent_ids(calls) == [f"gtin-{i:03d}" for i in range(6)]
    assert {"doc": {"category_path": []}} not in _payloads(calls)
    out = capsys.readouterr().out
    assert "sent=6" in out
    assert "empty(skipped)=4" in out
    assert "empty(overwritten)=0" in out


def test_naming_the_default_explicitly_sends_exactly_what_omitting_it_sends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--skip-empty`` stays accepted, and stays the default.

    ``builds/2026-08-03/VERIFICATION.md`` records the load that produced the
    current indices as ``inject_category_path.py --skip-empty``; that command
    has to keep meaning what it meant, and keep parsing.
    """
    records = catalog(6, empty_paths=4)
    implicit: List[List[str]] = []
    explicit: List[List[str]] = []

    assert run_loader(monkeypatch, tmp_path, records, responder_for(updated_item, implicit)) == 0
    assert (
        run_loader(
            monkeypatch,
            tmp_path,
            records,
            responder_for(updated_item, explicit),
            extra_argv=["--skip-empty"],
        )
        == 0
    )

    assert implicit == explicit


def test_no_skip_empty_sends_the_empty_path_as_an_explicit_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other reading: make the index agree with the extract exactly.

    A document holding a path an earlier extract generation resolved and the
    current one does not is only repairable by writing the empty list over it.
    """
    calls: List[List[str]] = []
    status = run_loader(
        monkeypatch,
        tmp_path,
        catalog(6, empty_paths=4),
        responder_for(updated_item, calls),
        extra_argv=["--no-skip-empty"],
    )

    assert status == 0
    assert _sent_ids(calls) == [f"gtin-{i:03d}" for i in range(6)] + [
        f"empty-{i:03d}" for i in range(4)
    ]
    assert _payloads(calls)[6:] == [{"doc": {"category_path": []}}] * 4
    out = capsys.readouterr().out
    assert "sent=10" in out
    assert "empty(skipped)=0" in out
    assert "empty(overwritten)=4" in out


def test_the_two_settings_differ_on_the_same_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The lever moves something. Same records, two settings, two requests."""
    records = catalog(6, empty_paths=4)
    skipping: List[List[str]] = []
    overwriting: List[List[str]] = []

    run_loader(monkeypatch, tmp_path, records, responder_for(updated_item, skipping))
    run_loader(
        monkeypatch,
        tmp_path,
        records,
        responder_for(updated_item, overwriting),
        extra_argv=["--no-skip-empty"],
    )

    assert len(_sent_ids(skipping)) == 6
    assert len(_sent_ids(overwriting)) == 10
    assert set(_sent_ids(overwriting)) - set(_sent_ids(skipping)) == {
        f"empty-{i:03d}" for i in range(4)
    }


def test_the_report_states_which_empty_path_policy_ran(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stated on every run, including one whose corpus has no empty path at all.

    A record of a load says what was permitted, not only what happened to occur
    — the same reason the missing tolerance is printed when nothing is missing.
    """
    run_loader(monkeypatch, tmp_path, catalog(3), responder_for(updated_item))
    assert "empty paths: skipped" in capsys.readouterr().out

    run_loader(
        monkeypatch,
        tmp_path,
        catalog(3),
        responder_for(updated_item),
        extra_argv=["--no-skip-empty"],
    )
    assert "empty paths: overwritten with [] (--no-skip-empty)" in capsys.readouterr().out


def test_inject_overwrites_an_empty_path_only_when_asked(tmp_path: Path) -> None:
    """The function under the CLI, both ways, without the parser in between."""
    records = catalog(2, empty_paths=3)

    skipping = inject(iter(records), "catalog_en_v8", bulk=responder_for(updated_item))
    overwriting = inject(
        iter(records), "catalog_en_v8", bulk=responder_for(updated_item), skip_empty=False
    )

    assert (skipping.sent, skipping.empty, skipping.empty_overwritten) == (2, 3, 0)
    assert (overwriting.sent, overwriting.empty, overwriting.empty_overwritten) == (5, 0, 3)


def test_a_failed_bulk_request_is_an_operational_exit_not_a_gate_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """"The cluster refused the request" and "the run applied nothing" are
    different answers and get different exit codes, so a caller can tell a
    broken connection from a broken load."""

    def explode(lines: List[str]) -> Dict[str, Any]:
        raise inject_category_path.BulkRequestError("bulk failed: HTTP 503\nunavailable")

    status = run_loader(monkeypatch, tmp_path, catalog(4), explode)

    assert status == 2
    assert "HTTP 503" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the accounting itself, without the CLI around it
# --------------------------------------------------------------------------- #
def test_outcome_accounts_for_each_class_once() -> None:
    outcome = Outcome(sent=4)
    outcome.record_response(
        envelope(
            [
                updated_item("a"),
                updated_item("b", result="noop"),
                missing_item("c"),
                rejected_item("d"),
            ]
        )
    )

    assert (outcome.updated, outcome.noop, outcome.not_found, outcome.failed) == (1, 1, 1, 1)
    assert outcome.applied == 2
    assert outcome.unaccounted == 0
    assert outcome.missing_examples == ["c"]


def test_missing_examples_are_capped_but_the_count_is_not() -> None:
    outcome = Outcome(sent=50)
    outcome.record_response(envelope([missing_item(f"gtin-{i}") for i in range(50)]))

    assert outcome.not_found == 50
    assert len(outcome.missing_examples) == inject_category_path.MAX_EXAMPLES


def test_gate_passes_only_a_run_that_applied_everything_it_sent() -> None:
    clean = Outcome(sent=10, updated=10)
    assert gate(clean, Tolerance()) == []


def test_inject_skips_records_without_an_id_without_counting_them_sent() -> None:
    records = [
        {"id": "", "category_path": ["Foods"]},
        {"category_path": ["Foods"]},
        {"id": "gtin-1", "category_path": ["Foods"]},
    ]
    outcome = inject(
        iter(records), "catalog_en_v8", bulk=responder_for(updated_item), batch_size=10
    )

    assert outcome.sent == 1
    assert outcome.applied == 1
