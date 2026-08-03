"""Tests for ``scripts/reindex_v7_to_v8.py`` — does the exit status describe a
copy that was shown to have happened?

**Where the proof sits, and where it does not.** The seam under test is the
script's own boundary: the two functions it reaches the cluster through
(``_req`` and ``_exists``) are replaced with a stub, and everything above them —
the argument parser, the ordering of the calls, the polling loop, the gate and
the exit status — is the real code. The task envelopes replayed below are
therefore **constructed, not recorded**, and that is a limitation stated here
rather than left for a reader to discover: a reindex response can only be
obtained by writing to a cluster, and the cluster available for this work is
shared and read-only. So what these tests prove is that *given* a task of the
documented shape, this script polls it, reads it and gates on it correctly. That
Elasticsearch emits that shape rests on the documented ``_reindex`` / ``_tasks``
contract and on the fields the script was already reading before this change
(``completed``, ``task.status.created``, ``response.failures``) — the same
evidence the original code was written from. It is weaker evidence than
``test_index_verification.py``'s live captures, and it is worth saying so.

Two shapes matter and both appear below. A task that finished carries
``{"completed": true, "task": {...}, "response": {...}}`` with the reindex
counters in ``response``. A task that ended in an error carries ``error``
instead and **no** ``response`` at all — which is why a check written only
around ``response["failures"]`` sees an empty list and calls it clean.

Every test drives ``main()`` through the real argument parser, and invokes it
with no arguments and ``sys.argv`` patched rather than by passing an argv list,
so this file runs unchanged against the commit before the fix — which is how the
fail-before evidence for the two central tests was produced. For the same reason
the module is imported *as a module*: no ``from reindex_v7_to_v8 import ...`` of
a name the fix introduces, which would make the whole file fail to collect at the
parent for a reason that proves nothing.

Tests that pass a flag the fix adds cannot run at the parent at all — argparse
refuses the unknown option there. They are marked in their docstrings and are
not part of the fail-before proof.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import reindex_v7_to_v8  # noqa: E402

SOURCE = "catalog_en_v7"
DEST = "catalog_en_v8"
TASK_ID = "oTUltX4IQMOUUVeiohTt8A:124"


def reindex_response(
    total: int,
    created: Optional[int] = None,
    version_conflicts: int = 0,
    noops: int = 0,
    failures: Sequence[Any] = (),
    canceled: Optional[str] = None,
) -> Dict[str, Any]:
    """The ``response`` block a finished ``_reindex`` task carries."""
    body: Dict[str, Any] = {
        "took": 8_411,
        "timed_out": False,
        "total": total,
        "updated": 0,
        "created": total if created is None else created,
        "deleted": 0,
        "batches": max(1, total // 1000),
        "version_conflicts": version_conflicts,
        "noops": noops,
        "retries": {"bulk": 0, "search": 0},
        "throttled_millis": 0,
        "requests_per_second": -1.0,
        "throttled_until_millis": 0,
        "failures": list(failures),
    }
    if canceled is not None:
        body["canceled"] = canceled
    return body


class Cluster:
    """A stub of exactly the endpoints this script calls, and nothing else."""

    def __init__(
        self,
        src_count: int,
        dst_count: int = 0,
        response: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        dest_exists: bool = False,
    ) -> None:
        self.src_count = src_count
        self.dst_count = dst_count
        self.response = reindex_response(src_count) if response is None else response
        self.error = error
        self.dest_exists = dest_exists
        self.calls: List[Tuple[str, str]] = []

    # -- the two seams the script reaches a cluster through ----------------- #
    def exists(self, index: str) -> bool:
        self.calls.append(("HEAD", f"/{index}"))
        return self.dest_exists

    def request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        self.calls.append((method, path))
        if path == f"/{SOURCE}/_count":
            return {"count": self.src_count}
        if path == f"/{DEST}/_count":
            return {"count": self.dst_count}
        if path == f"/{SOURCE}/_mapping":
            return {SOURCE: {"mappings": {"properties": {"title": {"type": "text"}}}}}
        if path == f"/{SOURCE}/_settings":
            return {
                SOURCE: {
                    "settings": {
                        "index": {
                            "number_of_shards": "1",
                            "default_pipeline": "off-embed",
                            "uuid": "9Qh1",
                        }
                    }
                }
            }
        if path.startswith("/_reindex"):
            return {"task": TASK_ID}
        if path == f"/_tasks/{TASK_ID}":
            return self.task_status()
        if method in ("PUT", "DELETE") and path == f"/{DEST}":
            return {"acknowledged": True}
        if path == f"/{DEST}/_refresh":
            return {"_shards": {"total": 2, "successful": 1, "failed": 0}}
        raise AssertionError(
            f"the script called an endpoint the stub does not know: {method} {path}"
        )

    def task_status(self) -> Dict[str, Any]:
        status = {
            "created": self.response.get("created", 0),
            "total": self.response.get("total", 0),
        }
        if self.error is not None:
            # A task that ended in an error carries no ``response`` at all.
            return {"completed": True, "task": {"status": status}, "error": self.error}
        return {"completed": True, "task": {"status": status}, "response": self.response}

    def paths(self, method: str) -> List[str]:
        return [path for verb, path in self.calls if verb == method]


def run_reindex(
    monkeypatch: pytest.MonkeyPatch, cluster: Cluster, extra_argv: Sequence[str] = ()
) -> int:
    monkeypatch.setattr(reindex_v7_to_v8, "_req", cluster.request)
    monkeypatch.setattr(reindex_v7_to_v8, "_exists", cluster.exists)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["reindex_v7_to_v8.py", "--source", SOURCE, "--dest", DEST, *extra_argv],
    )
    return reindex_v7_to_v8.main()


# --------------------------------------------------------------------------- #
# hole 1: --no-wait returned the status of a verified copy
# --------------------------------------------------------------------------- #
def test_no_wait_does_not_return_the_status_a_verified_copy_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first half of the issue, as one comparison.

    Both runs are of the same 500-document source. One polls the task to
    completion, refreshes the destination and counts it; the other submits and
    returns immediately, knowing only that the cluster accepted the request. If
    those two end in the same status, no caller can tell them apart — an ``&&``
    chain reads "submitted" as "copied and verified".
    """
    verified = run_reindex(monkeypatch, Cluster(src_count=500, dst_count=500))
    started = run_reindex(monkeypatch, Cluster(src_count=500, dst_count=0), ["--no-wait"])

    assert verified == 0
    assert started != verified
    assert started == 3


def test_no_wait_says_in_words_that_nothing_has_been_verified(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The status is for the shell; the sentence is for whoever reads the log."""
    run_reindex(monkeypatch, Cluster(src_count=500), ["--no-wait"])

    captured = capsys.readouterr()
    assert "NOT VERIFIED" in captured.err
    assert TASK_ID in captured.out


def test_no_wait_returns_without_polling_the_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag still does what it is for: submit and hand back the task id.

    Refusing to claim success is not the same as removing the flag, and a caller
    who wants to poll separately must still get out of here without waiting.
    """
    cluster = Cluster(src_count=500)
    run_reindex(monkeypatch, cluster, ["--no-wait"])

    assert f"/_tasks/{TASK_ID}" not in cluster.paths("GET")
    assert f"/{DEST}/_count" not in cluster.paths("GET")


# --------------------------------------------------------------------------- #
# hole 2: a source of 0 documents satisfied dst_count == src_count
# --------------------------------------------------------------------------- #
def test_a_zero_document_source_is_not_a_successful_copy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The second half of the issue.

    An empty or misnamed source produces an empty destination, and ``0 == 0``
    satisfies every count-based check there is. This is the same rule
    ``inject_category_path.py`` settled for a run that sent nothing.
    """
    status = run_reindex(monkeypatch, Cluster(src_count=0, dst_count=0))

    assert status == 1
    assert "FAILED [empty_source]" in capsys.readouterr().err


def test_an_empty_source_is_refused_before_the_destination_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refused at the top, not after the fact.

    Nothing about an empty source becomes copyable by creating an index for it,
    and the cheapest place to say so is before the cluster is asked to do any
    work at all.
    """
    cluster = Cluster(src_count=0, dst_count=0)
    run_reindex(monkeypatch, cluster)

    assert f"/{DEST}" not in cluster.paths("PUT")
    assert not [path for _, path in cluster.calls if path.startswith("/_reindex")]


def test_an_empty_source_can_be_permitted_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cannot run at the parent commit: --allow-empty-source does not exist there.

    Zero tolerance by default, and the exception named on the command line
    rather than assumed — the idiom the three sibling scripts use.
    """
    status = run_reindex(
        monkeypatch,
        Cluster(src_count=0, dst_count=0, response=reindex_response(0)),
        ["--allow-empty-source"],
    )

    assert status == 0


# --------------------------------------------------------------------------- #
# the copy that did happen
# --------------------------------------------------------------------------- #
def test_a_short_copy_fails_the_gate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A count mismatch is a completed run whose outcome is bad: exit 1, not 2.

    It used to exit 2, which under the convention the sibling scripts settled
    means "the run could not be carried out" — the status a broken connection
    deserves. Both cases exiting the same value is the thing that convention
    exists to prevent.
    """
    cluster = Cluster(src_count=100, dst_count=97, response=reindex_response(100, created=97))
    status = run_reindex(monkeypatch, cluster)

    assert status == 1
    captured = capsys.readouterr()
    assert "3 missing" in captured.out
    assert "FAILED [destination_count]" in captured.err


def test_a_verified_copy_prints_the_tolerance_in_force(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    status = run_reindex(monkeypatch, Cluster(src_count=100, dst_count=100))

    assert status == 0
    out = capsys.readouterr().out
    assert "missing tolerance: 0 (default: zero tolerance)" in out
    assert "=> verified" in out


def test_an_explicit_tolerance_is_honoured_and_recorded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cannot run at the parent commit: --allow-missing does not exist there."""
    cluster = Cluster(src_count=100, dst_count=97, response=reindex_response(100, created=97))
    status = run_reindex(monkeypatch, cluster, ["--allow-missing", "3"])

    assert status == 0
    assert "missing tolerance: --allow-missing 3" in capsys.readouterr().out


def test_a_fractional_tolerance_is_floored_never_rounded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cannot run at the parent commit: the flag does not exist there.

    0.15 of 20 documents is 3, not 3.0 and not 4. Rounding up would let a
    fraction quietly buy one more missing document than anybody wrote down.
    """
    permitted = run_reindex(
        monkeypatch,
        Cluster(src_count=20, dst_count=17, response=reindex_response(20, created=17)),
        ["--allow-missing-fraction", "0.15"],
    )
    assert permitted == 0
    assert "(3 of 20)" in capsys.readouterr().out

    one_too_many = run_reindex(
        monkeypatch,
        Cluster(src_count=20, dst_count=16, response=reindex_response(20, created=16)),
        ["--allow-missing-fraction", "0.15"],
    )
    assert one_too_many == 1


def test_no_tolerance_can_authorise_a_destination_that_holds_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cannot run at the parent commit: the flags do not exist there.

    ``--allow-missing 1000000`` is a statement about some documents that did not
    make it. It is not a statement that the copy may be absent, which is what an
    empty destination for a non-empty source means.
    """
    for extra in (["--allow-missing", "1000000"], ["--allow-missing-fraction", "1.0"]):
        status = run_reindex(
            monkeypatch,
            Cluster(src_count=100, dst_count=0, response=reindex_response(100, created=0)),
            extra,
        )
        assert status == 1, extra
        assert "FAILED [nothing_copied]" in capsys.readouterr().err


def test_a_destination_holding_more_than_the_source_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cannot run at the parent commit: --allow-missing does not exist there.

    Extra documents are not covered by a tolerance for missing ones: the two are
    different findings, and a destination holding documents the source does not
    is not a copy of the source.
    """
    cluster = Cluster(src_count=100, dst_count=140)
    status = run_reindex(monkeypatch, cluster, ["--allow-missing", "1000"])

    assert status == 1
    assert "more than the source" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# what the finished task reports, including the fields nobody was reading
# --------------------------------------------------------------------------- #
def test_a_task_that_ended_in_an_error_is_not_read_as_a_clean_copy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``error`` and ``response`` are alternatives, and only one was being read.

    The counts are deliberately made to agree here so the test isolates whether
    the task's own verdict is read at all: a failure that is only ever caught by
    the count check is a failure caught by luck, and says nothing about the
    fields the report is built from.
    """
    cluster = Cluster(
        src_count=100,
        dst_count=100,
        error={"type": "search_phase_execution_exception", "reason": "all shards failed"},
    )
    status = run_reindex(monkeypatch, cluster)

    assert status == 1
    assert "FAILED [task_error]" in capsys.readouterr().err


def test_a_cancelled_task_is_not_a_successful_copy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cluster = Cluster(
        src_count=100,
        dst_count=100,
        response=reindex_response(100, canceled="by user request"),
    )
    status = run_reindex(monkeypatch, cluster)

    assert status == 1
    assert "FAILED [task_canceled]" in capsys.readouterr().err


def test_version_conflicts_are_named_rather_than_left_to_the_count_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The decision the issue asked for, stated as a gate rather than implied.

    ``op_type: create`` into an index this script just created means a conflict
    is an id the destination already held. Each one is a document that was not
    copied. The shortfall would show up in the count anyway — but as an
    unexplained one, which is a worse report than the same number with its cause
    attached.
    """
    cluster = Cluster(
        src_count=100,
        dst_count=96,
        response=reindex_response(100, created=96, version_conflicts=4),
    )
    status = run_reindex(monkeypatch, cluster)

    assert status == 1
    captured = capsys.readouterr()
    assert "version_conflicts=4" in captured.out
    assert "FAILED [version_conflicts]" in captured.err


def test_reindex_failures_still_fail_the_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A verdict that was already right and must stay right.

    The exit status of 1 is not new — the pre-fix script already read
    ``response["failures"]``. What is new is where the cause is said: with every
    other reason, on stderr and under the name of the check it broke, rather
    than as a bare ``FAILURES:`` dump on stdout among the progress lines.
    """
    cluster = Cluster(
        src_count=100,
        dst_count=99,
        response=reindex_response(
            100,
            created=99,
            failures=[
                {"index": DEST, "id": "gtin-1", "cause": {"type": "mapper_parsing_exception"}}
            ],
        ),
    )
    status = run_reindex(monkeypatch, cluster)

    assert status == 1
    assert "mapper_parsing_exception" in capsys.readouterr().err


def test_a_counter_the_task_did_not_report_is_not_gated_as_a_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """"Not reported" and "0" are different facts and the report says which.

    A response missing ``created`` must not be read as "created 0 documents" —
    that would invent a failure — nor printed as ``0``, which would assert a
    measurement nobody made.
    """
    response = reindex_response(100)
    del response["created"]
    del response["version_conflicts"]
    status = run_reindex(monkeypatch, Cluster(src_count=100, dst_count=100, response=response))

    assert status == 0
    out = capsys.readouterr().out
    assert "created=not reported" in out
    assert "version_conflicts=not reported" in out


# --------------------------------------------------------------------------- #
# the run that could not happen at all
# --------------------------------------------------------------------------- #
def test_a_failed_request_is_an_operational_exit_not_a_gate_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cannot run at the parent commit: ReindexError does not exist there.

    "The cluster refused the request" and "the copy is short" are different
    answers and get different statuses, so a caller can tell a broken connection
    from a broken copy. Both used to be 1.
    """
    cluster = Cluster(src_count=100, dst_count=100)

    def explode(method: str, path: str, body: Optional[dict] = None) -> dict:
        if path == f"/{SOURCE}/_count":
            raise reindex_v7_to_v8.ReindexError(f"GET {path} -> HTTP 503\nunavailable")
        return cluster.request(method, path, body)

    monkeypatch.setattr(reindex_v7_to_v8, "_req", explode)
    monkeypatch.setattr(reindex_v7_to_v8, "_exists", cluster.exists)
    monkeypatch.setattr(sys, "argv", ["reindex_v7_to_v8.py", "--source", SOURCE, "--dest", DEST])

    assert reindex_v7_to_v8.main() == 2
    assert "HTTP 503" in capsys.readouterr().err


def test_an_existing_destination_without_recreate_is_an_operational_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cannot run at the parent commit: it exits through ``sys.exit`` there.

    No verdict was reached about any copy — the copy was never attempted.
    """
    status = run_reindex(monkeypatch, Cluster(src_count=100, dest_exists=True))

    assert status == 2
    assert "already exists" in capsys.readouterr().err


def test_recreate_deletes_the_destination_before_copying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = Cluster(src_count=100, dst_count=100, dest_exists=True)
    status = run_reindex(monkeypatch, cluster, ["--recreate"])

    assert status == 0
    assert cluster.paths("DELETE") == [f"/{DEST}"]


# --------------------------------------------------------------------------- #
# the record the run leaves behind
# --------------------------------------------------------------------------- #
def test_the_record_says_a_no_wait_run_verified_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cannot run at the parent commit: --json does not exist there.

    A tolerance and a verdict visible only in a terminal is the same defect one
    step removed. The record is the artifact a build keeps, so it carries the
    task id a caller needs to discharge the verification it is owed.
    """
    out = tmp_path / "reindex.json"
    run_reindex(monkeypatch, Cluster(src_count=500), ["--no-wait", "--json", str(out)])
    record = json.loads(out.read_text(encoding="utf-8"))

    assert record["verified"] is False
    assert record["task"] == TASK_ID
    assert record["dest_count"] is None
    assert record["exit_status"] == 3


def test_the_record_carries_the_counts_and_the_tolerance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cannot run at the parent commit: the flags do not exist there."""
    out = tmp_path / "reindex.json"
    cluster = Cluster(src_count=100, dst_count=97, response=reindex_response(100, created=97))
    run_reindex(monkeypatch, cluster, ["--allow-missing", "3", "--json", str(out)])
    record = json.loads(out.read_text(encoding="utf-8"))

    assert record["verified"] is True
    assert (record["source_count"], record["dest_count"]) == (100, 97)
    assert record["documents_missing"] == 3
    assert record["missing_tolerance"] == 3
    assert record["missing_tolerance_source"] == "--allow-missing 3"
    assert record["failed"] == []


def test_the_record_names_the_checks_that_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cannot run at the parent commit: --json does not exist there."""
    out = tmp_path / "reindex.json"
    run_reindex(monkeypatch, Cluster(src_count=0, dst_count=0), ["--json", str(out)])
    record = json.loads(out.read_text(encoding="utf-8"))

    assert record["failed"] == ["empty_source"]
    assert record["exit_status"] == 1


# --------------------------------------------------------------------------- #
# the gate, without the CLI around it
# --------------------------------------------------------------------------- #
def test_gate_passes_a_copy_that_matches_its_source() -> None:
    """Cannot run at the parent commit: neither Copy nor gate exists there."""
    copy = reindex_v7_to_v8.Copy(source=SOURCE, dest=DEST, src_count=10, dst_count=10)
    copy.read_response(reindex_response(10))

    assert reindex_v7_to_v8.gate(copy, reindex_v7_to_v8.Tolerance(), False) == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--allow-missing", "-1"],
        ["--allow-missing-fraction", "1.5"],
        ["--allow-missing-fraction", "-0.1"],
        ["--allow-missing", "1", "--allow-missing-fraction", "0.1"],
    ],
)
def test_unusable_tolerances_are_refused_at_parse_time(
    monkeypatch: pytest.MonkeyPatch, extra: List[str]
) -> None:
    """Passes vacuously at the parent commit: argparse refuses unknown flags too."""
    with pytest.raises(SystemExit) as excinfo:
        run_reindex(monkeypatch, Cluster(src_count=100, dst_count=100), extra)

    assert excinfo.value.code != 0
