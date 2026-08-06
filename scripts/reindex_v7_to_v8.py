#!/usr/bin/env python3
"""Copy an existing PRISM catalog index to a new ``_v8`` *without re-embedding*.

The embedding vectors (``content_embedding_elser`` / ``_e5`` / ``_jina``) are
physically stored in ``_source`` on the v7 indexes, so a server-side ``_reindex``
with **no ingest pipeline** copies them verbatim — no inference is run. The new
index is created from the source's own mapping/settings, minus the per-index
``default_pipeline`` (which is what would otherwise re-embed), plus a new
``category_path`` keyword field for the hierarchical category facet.

Why the exit status is the interesting part
-------------------------------------------
A reindex is submitted with ``wait_for_completion=false``: the cluster answers
with a task id and the copy happens afterwards. Two ways of reporting success
for a copy that has not been shown to have happened were closed here.

* ``--no-wait`` used to ``return 0`` having only *started* the task. That is the
  same value a run returns when it has polled to completion, refreshed the
  destination and found the counts equal, so an ``&&`` chain or a CI step could
  not tell "copied and verified" from "submitted". The flag is still useful — a
  caller may legitimately fire a long reindex and poll separately — so it is
  kept, and it now exits **3**: started, nothing verified, verification owed.
* A source of 0 documents used to satisfy ``dst_count == src_count``, because
  ``0 == 0``. Reindexing an empty or misnamed source therefore reported a clean
  copy. An empty source now fails before the destination is created, which is
  the rule ``inject_category_path.py`` settled for a run that sent nothing.

Everything the completed task reports is now read and named rather than left
implied. ``version_conflicts`` in particular: with ``op_type: create`` into a
freshly created destination it should stay at 0, and a non-zero value means an
id was already present, so each conflict is a document that was *not* copied. It
would surface through the count check anyway — but as an unexplained shortfall,
which is a worse report than the same number with its cause attached.

Tolerances are zero by default and named on the command line when an exception
is wanted (``--allow-missing`` / ``--allow-missing-fraction``, floored, the same
flags and the same meaning as ``inject_category_path.py``), and whichever is in
force is printed with the result and recorded in ``--json``. One rule no
tolerance can switch off: a non-empty source whose destination ends up holding
**nothing** always fails, because no value of ``--allow-missing`` expresses "the
whole copy may be absent".

Exit status:
    0  the copy completed and the destination matches the source (within any
       tolerance named on the command line)
    1  the copy completed and its outcome fails a gate
    2  the run could not complete (a request failed, or the destination exists
       and --recreate was not given). Previously these exited 1, which was
       indistinguishable from a verdict of "the copy is bad".
    3  --no-wait only: the reindex task was started and **nothing about it has
       been verified**. Not an error and not a success; the caller owes the
       verification and gets the task id to do it with.

Connection comes from the environment:
    PRISM_ELASTICSEARCH_URL, PRISM_ELASTICSEARCH_API_KEY

Usage:
    python scripts/reindex_v7_to_v8.py --source catalog_en_v7 --dest catalog_en_v8
    python scripts/reindex_v7_to_v8.py --source catalog_en_v7 --dest catalog_en_v8 --recreate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tolerance import Tolerance

MAX_EXAMPLES = 3

EXIT_OK = 0
EXIT_GATE = 1
EXIT_OPERATIONAL = 2
# Deliberately a fourth value rather than a reuse of one of the three. 0/1/2 keep
# exactly the meanings verify_index.py, inject_category_path.py and
# verify_catalog.py give them; this is the one state in this pipeline none of
# them describes. 2 ("could not complete") would be wrong: nothing failed, the
# run did precisely what it was asked to do. 1 ("fails a gate") would be wrong
# too: no gate was evaluated, because there is nothing yet to evaluate one on.
EXIT_STARTED_NOT_VERIFIED = 3


class ReindexError(RuntimeError):
    """The run could not complete. Distinct from a copy that failed a gate."""


# Settings keys that are index-instance-specific and must not be carried to a new index.
_DROP_SETTINGS = {
    "uuid",
    "creation_date",
    "version",
    "provided_name",
    "routing",
    "default_pipeline",  # <- the whole point: dropping this prevents re-embedding
    "history",
}


def _es() -> tuple[str, str]:
    url = os.environ.get("PRISM_ELASTICSEARCH_URL", "").rstrip("/")
    key = os.environ.get("PRISM_ELASTICSEARCH_API_KEY", "")
    if not url or not key:
        raise ReindexError("PRISM_ELASTICSEARCH_URL and PRISM_ELASTICSEARCH_API_KEY must be set")
    return url, key


def _req(method: str, path: str, body: dict | None = None) -> dict:
    url, key = _es()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + path, data=data, method=method)
    req.add_header("Authorization", f"ApiKey {key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise ReindexError(f"{method} {path} -> HTTP {exc.code}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise ReindexError(f"{method} {path} -> {exc.reason}") from exc


def _exists(index: str) -> bool:
    url, key = _es()
    req = urllib.request.Request(f"{url}/{index}", method="HEAD")
    req.add_header("Authorization", f"ApiKey {key}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise ReindexError(f"HEAD /{index} -> HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ReindexError(f"HEAD /{index} -> {exc.reason}") from exc


def missing_tolerance(
    allow_missing: Optional[int] = None,
    allow_missing_fraction: Optional[float] = None,
) -> Tolerance:
    """How many uncopied documents the operator has explicitly agreed to.

    Same two flags and the same names as ``inject_category_path.py``, and now
    literally the same rule: both call ``scripts/tolerance.py``, so a fraction
    cannot round up into permitting one more missing document than anybody
    wrote down, and cannot start doing so here without starting there.
    """
    return Tolerance(
        "--allow-missing",
        "--allow-missing-fraction",
        allow_missing,
        allow_missing_fraction,
    )


@dataclass
class Copy:
    """What the reindex actually did, measured rather than assumed.

    Every field the completed task reports has a home here, including the ones
    the previous version of this script never read. ``None`` means "the task did
    not report it", which is not the same as 0 and must not be gated as if it
    were.
    """

    source: str
    dest: str
    src_count: int = 0
    dst_count: int = 0
    task_id: Optional[str] = None
    total: Optional[int] = None
    created: Optional[int] = None
    version_conflicts: Optional[int] = None
    noops: Optional[int] = None
    canceled: Optional[str] = None
    task_error: Optional[Dict[str, Any]] = None
    failures: List[Any] = field(default_factory=list)

    @property
    def missing(self) -> int:
        """Documents the source holds that the destination does not."""
        return max(self.src_count - self.dst_count, 0)

    @property
    def copied_fraction(self) -> float:
        return self.dst_count / self.src_count if self.src_count else 0.0

    def read_response(self, response: Dict[str, Any]) -> None:
        self.total = response.get("total")
        self.created = response.get("created")
        self.version_conflicts = response.get("version_conflicts")
        self.noops = response.get("noops")
        self.canceled = response.get("canceled")
        self.failures = list(response.get("failures") or [])


def check_source(copy: Copy, allow_empty_source: bool) -> List[Tuple[str, str]]:
    """The one gate that can be decided before any copying happens.

    0 == 0 satisfies every count-based check there is, so an empty or misnamed
    source used to pass all of them vacuously. This is the rule
    ``inject_category_path.py`` settled for a run that sent nothing — a loader
    reporting a completed load having loaded nothing — and deciding it here lets
    the run refuse before a destination index is created for a copy of nothing.
    """
    if copy.src_count == 0 and not allow_empty_source:
        return [
            (
                "empty_source",
                f"the source {copy.source} holds 0 documents: there is nothing to copy, and a "
                "destination that matches it matches it vacuously. Check --source, or pass "
                "--allow-empty-source if an empty index really is the intended input",
            )
        ]
    return []


def gate(copy: Copy, tolerance: Tolerance, allow_empty_source: bool) -> List[Tuple[str, str]]:
    """The reasons this copy must not report success, as ``(check, reason)``.

    The single place the exit status of a completed copy is derived from, so a
    reader of the output can see which numbers are fatal instead of inferring it.
    """
    reasons = check_source(copy, allow_empty_source)
    if reasons:
        return reasons

    if copy.task_error is not None:
        reasons.append(
            (
                "task_error",
                f"the reindex task ended in an error: {json.dumps(copy.task_error)[:300]}",
            )
        )
    if copy.canceled:
        reasons.append(("task_canceled", f"the reindex task was cancelled: {copy.canceled}"))
    if copy.failures:
        reasons.append(
            (
                "reindex_failures",
                f"the reindex task reported {len(copy.failures):,} failures: "
                f"{json.dumps(copy.failures[:MAX_EXAMPLES])[:300]}",
            )
        )
    if copy.version_conflicts:
        # op_type is `create` into an index this script just created, so a
        # conflict means the destination already held that id. Each one is a
        # document that was not copied; the count check would show the shortfall
        # but not its cause.
        reasons.append(
            (
                "version_conflicts",
                f"{copy.version_conflicts:,} documents hit a version conflict and were not "
                f"written: {copy.dest} already held those ids, or something else is writing to it",
            )
        )

    if copy.src_count and copy.dst_count == 0:
        # Not overridable by any tolerance, for the reason inject_category_path.py
        # gives for its own zero-applied rule: --allow-missing is a statement
        # about a few documents, not a statement that the copy may be absent.
        reasons.append(
            (
                "nothing_copied",
                f"{copy.dest} holds 0 documents after a reindex of {copy.src_count:,}: nothing "
                "was copied",
            )
        )
        return reasons

    permitted = tolerance.permitted(copy.src_count)
    if copy.missing > permitted:
        reasons.append(
            (
                "destination_count",
                f"{copy.dest} holds {copy.dst_count:,} of the source's {copy.src_count:,} "
                f"documents — {copy.missing:,} missing (tolerance {permitted:,})",
            )
        )
    if copy.dst_count > copy.src_count:
        # Never tolerable: --allow-missing permits documents that did not arrive,
        # it says nothing about documents that arrived from somewhere else.
        reasons.append(
            (
                "destination_count",
                f"{copy.dest} holds {copy.dst_count:,} documents, more than the source's "
                f"{copy.src_count:,}: it is not a copy of {copy.source} alone",
            )
        )
    if copy.total is not None and copy.created is not None:
        not_created = copy.total - copy.created
        if not_created > permitted:
            reasons.append(
                (
                    "documents_not_created",
                    f"the task created {copy.created:,} of the {copy.total:,} documents it "
                    f"processed (tolerance {permitted:,})",
                )
            )
    return reasons


def build_dest_body(source: str) -> dict:
    """Construct the create-index body from the source's mapping + settings."""
    mapping = _req("GET", f"/{source}/_mapping")[source]["mappings"]
    settings_idx = _req("GET", f"/{source}/_settings")[source]["settings"]["index"]

    clean_settings = {
        k: v for k, v in settings_idx.items() if k not in _DROP_SETTINGS
    }

    # Add the hierarchical category field. Keyword for exact faceting / drill-down;
    # a `.text` sub-field mirrors how retail catalogs typically expose the
    # category for optional full-text boosting later.
    props = dict(mapping.get("properties", {}))
    if "category_path" not in props:
        props["category_path"] = {
            "type": "keyword",
            "fields": {"text": {"type": "text", "analyzer": "base"}},
        }
    new_mapping = dict(mapping)
    new_mapping["properties"] = props

    return {"settings": {"index": clean_settings}, "mappings": new_mapping}


def await_task(task_id: str, copy: Copy, poll_seconds: float) -> None:
    """Poll until the task completes, recording everything it reports."""
    while True:
        time.sleep(poll_seconds)
        status = _req("GET", f"/_tasks/{task_id}")
        st = status.get("task", {}).get("status", {})
        created = st.get("created", 0)
        total = st.get("total", copy.src_count)
        print(f"  reindexed {created:,}/{total:,}", flush=True)
        if status.get("completed"):
            # A task that ended in an error carries `error` and no `response` at
            # all. Reading only `response.failures` saw an empty list there and
            # treated the copy as clean.
            copy.task_error = status.get("error")
            copy.read_response(status.get("response") or {})
            return


def report(copy: Copy, tolerance: Tolerance, failures: List[Tuple[str, str]]) -> str:
    lines = [
        f"Done. {copy.dest}: {copy.dst_count:,} docs (source {copy.source} had "
        f"{copy.src_count:,}; {copy.missing:,} missing)",
        f"  copied: {copy.copied_fraction * 100:.2f}% "
        f"| missing tolerance: {tolerance.describe(copy.src_count)}",
        f"  task {copy.task_id}: total={_shown(copy.total)} created={_shown(copy.created)} "
        f"version_conflicts={_shown(copy.version_conflicts)} noops={_shown(copy.noops)} "
        f"failures={len(copy.failures):,}",
        "  => "
        + ("FAILED: " + "; ".join(reason for _, reason in failures) if failures else "verified"),
    ]
    return "\n".join(lines)


def _shown(value: Optional[int]) -> str:
    """``None`` is "the task did not report it", which is not ``0``."""
    return "not reported" if value is None else f"{value:,}"


def _record(
    copy: Copy,
    tolerance: Tolerance,
    allow_empty_source: bool,
    verified: bool,
    failures: List[Tuple[str, str]],
    status: int,
) -> Dict[str, Any]:
    return {
        "source": copy.source,
        "dest": copy.dest,
        "task": copy.task_id,
        # The one field a --no-wait caller has to read: it says in the record,
        # not only in a terminal, that nothing about this copy has been checked.
        "verified": verified,
        "source_count": copy.src_count,
        "dest_count": copy.dst_count if verified else None,
        "documents_missing": copy.missing if verified else None,
        "task_total": copy.total,
        "task_created": copy.created,
        "task_version_conflicts": copy.version_conflicts,
        "task_noops": copy.noops,
        "task_failures": len(copy.failures),
        "task_failure_examples": copy.failures[:MAX_EXAMPLES],
        "missing_tolerance": tolerance.permitted(copy.src_count),
        "missing_tolerance_source": tolerance.describe(copy.src_count),
        "allow_empty_source": allow_empty_source,
        "failed": [check for check, _ in failures],
        "failure_reasons": [reason for _, reason in failures],
        "exit_status": status,
    }


def _write_record(path: Optional[Path], record: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Copy a v7 catalog index to a new v8 without re-embedding, and exit non-zero "
            "unless the destination has been shown to hold the source's documents."
        ),
        epilog=(
            "Exit status: 0 copied and verified; 1 the copy completed and fails a gate; "
            "2 the run could not complete; 3 --no-wait, the task was started and nothing "
            "about it was verified."
        ),
    )
    ap.add_argument("--source", required=True, help="e.g. catalog_en_v7")
    ap.add_argument("--dest", required=True, help="e.g. catalog_en_v8")
    ap.add_argument("--recreate", action="store_true", help="delete dest if it exists")
    ap.add_argument(
        "--no-wait",
        action="store_true",
        help=(
            "submit the reindex and return without polling. Exits 3, not 0: the task has been "
            "started and nothing about it has been verified"
        ),
    )
    ap.add_argument(
        "--poll-seconds", type=float, default=5.0, help="seconds between task polls"
    )
    ap.add_argument("--json", type=Path, default=None, help="write the run's record here")
    ap.add_argument(
        "--allow-empty-source",
        action="store_true",
        help=(
            "permit a source holding 0 documents. Off by default: a destination that matches an "
            "empty source matches it vacuously"
        ),
    )
    allowance = ap.add_mutually_exclusive_group()
    allowance.add_argument(
        "--allow-missing",
        type=int,
        default=None,
        metavar="N",
        help="permit up to N of the source's documents to be absent from dest (default: 0)",
    )
    allowance.add_argument(
        "--allow-missing-fraction",
        type=float,
        default=None,
        metavar="F",
        help=(
            "permit a fraction of the source's documents to be absent, e.g. 0.01; floored to a "
            "whole number of documents (default: 0)"
        ),
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    tolerance = missing_tolerance(args.allow_missing, args.allow_missing_fraction)
    problem = tolerance.problem()
    if problem is not None:
        ap.error(problem)

    copy = Copy(source=args.source, dest=args.dest)

    try:
        return _run(args, copy, tolerance)
    except ReindexError as exc:
        print(f"reindex_v7_to_v8: {exc}", file=sys.stderr)
        failures = [("could_not_run", str(exc))]
        _write_record(
            args.json,
            _record(
                copy, tolerance, args.allow_empty_source, False, failures, EXIT_OPERATIONAL
            ),
        )
        return EXIT_OPERATIONAL


def _finish(
    args: argparse.Namespace,
    copy: Copy,
    tolerance: Tolerance,
    failures: List[Tuple[str, str]],
    verified: bool,
    status: int,
) -> int:
    _write_record(
        args.json,
        _record(copy, tolerance, args.allow_empty_source, verified, failures, status),
    )
    for check, reason in failures:
        print(f"FAILED [{check}]: {reason}", file=sys.stderr)
    return status


def _run(args: argparse.Namespace, copy: Copy, tolerance: Tolerance) -> int:
    if _exists(args.dest):
        if not args.recreate:
            raise ReindexError(f"{args.dest} already exists (use --recreate to replace)")
        print(f"Deleting existing {args.dest} ...", flush=True)
        _req("DELETE", f"/{args.dest}")

    copy.src_count = _req("GET", f"/{args.source}/_count")["count"]
    print(f"Source {args.source}: {copy.src_count:,} docs", flush=True)

    # Before the destination is created and before anything is submitted: an
    # empty source cannot produce a copy worth verifying, and every count-based
    # check downstream would pass it on 0 == 0.
    empty = check_source(copy, args.allow_empty_source)
    if empty:
        return _finish(args, copy, tolerance, empty, False, EXIT_GATE)

    body = build_dest_body(args.source)
    has_dp = "default_pipeline" in body["settings"]["index"]
    print(
        f"Creating {args.dest} (default_pipeline carried over: {has_dp}; "
        f"category_path added: {'category_path' in body['mappings']['properties']})",
        flush=True,
    )
    _req("PUT", f"/{args.dest}", body)

    print("Starting server-side reindex (no pipeline -> embeddings copied as-is) ...", flush=True)
    task = _req(
        "POST",
        "/_reindex?wait_for_completion=false&slices=auto&refresh=false",
        {"source": {"index": args.source}, "dest": {"index": args.dest, "op_type": "create"}},
    )
    copy.task_id = task.get("task")
    if not copy.task_id:
        raise ReindexError(f"the reindex request named no task: {json.dumps(task)[:300]}")
    print(f"Reindex task: {copy.task_id}", flush=True)

    if args.no_wait:
        print(f"Poll with: GET /_tasks/{copy.task_id}", flush=True)
        print(
            f"NOT VERIFIED: the reindex of {args.source} -> {args.dest} was started and nothing "
            f"about it has been checked — not the document count, not the task's failures. "
            f"Exit status {EXIT_STARTED_NOT_VERIFIED} says so; 0 is reserved for a copy this "
            "script has polled to completion and counted.",
            file=sys.stderr,
        )
        return _finish(args, copy, tolerance, [], False, EXIT_STARTED_NOT_VERIFIED)

    await_task(copy.task_id, copy, args.poll_seconds)

    _req("POST", f"/{args.dest}/_refresh")
    copy.dst_count = _req("GET", f"/{args.dest}/_count")["count"]

    failures = gate(copy, tolerance, args.allow_empty_source)
    print(report(copy, tolerance, failures), flush=True)
    return _finish(args, copy, tolerance, failures, True, EXIT_GATE if failures else EXIT_OK)


if __name__ == "__main__":
    raise SystemExit(main())
