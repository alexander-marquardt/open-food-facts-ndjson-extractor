#!/usr/bin/env python3
"""Bulk-add the hierarchical category fields onto an existing index.

Writes both ``category_path`` — every address the product sits at, as cumulative
``/``-joined strings — and ``category_path_primary``, the one address a product
page leads with. They go in one partial update because a document holding one
without the other is a document that cannot render its own breadcrumb.

Reads an extractor NDJSON (which contains ``id`` = GTIN and both fields)
and issues partial ``_update`` operations keyed by ``_id`` (= GTIN on PRISM
catalog indexes). Partial updates merge the one field and **do not** run any
ingest pipeline, so existing fields — including the copied embedding vectors —
are left untouched.

Why the exit status is the interesting part
-------------------------------------------
A partial update cannot create a document. Elasticsearch reports an update
against an id the index does not hold as a **per-item 404**, not as a
bulk-level error — the request itself succeeded, so ``response["errors"]`` can
be ``false`` while every single document was dropped. A loader that reads only
the bulk-level flag prints a large number under a heading nobody has to read
and then exits 0, and a shell ``&&`` chain or a CI step reads that as success.

So every per-item outcome is counted and classified, and the exit status is a
function of those counts rather than of whether the HTTP call worked:

* ``updated`` / ``noop`` — the id was in the index and now carries the value.
  ``noop`` means it already did; both are *applied*.
* ``not_found`` — a 404. The index does not hold that id. This is a **result**,
  not a non-event: it is the signature of the wrong index, a stale id set, or an
  index built from a different locale's extract.
* ``conflict`` — a 409. Somebody else wrote the document mid-run; the update
  did not apply.
* ``failed`` — anything else (a mapping rejection, a malformed item).
* ``unaccounted`` — documents sent for which the response carried no item at
  all. A run that cannot account for every document it sent has not verified
  anything about them.

The default is zero tolerance for anything but ``updated``/``noop``. There is no
steady-state reason for the extract that built an index to address a document
that index does not hold, so "some ids are missing" is a finding, not noise. A
tolerance can be named explicitly with ``--allow-missing`` /
``--allow-missing-fraction``, and whatever tolerance is in force is printed in
the report so the record says what was permitted. One rule is not overridable:
a run that applied **nothing** always fails, because no tolerance expresses
"pointing at the wrong index is fine".

Exit status:
    0  every document sent was applied, or misses stayed inside a named tolerance
    1  the run completed and its outcomes fail the gate
    2  the run could not complete (the bulk request itself failed)

Records whose ``category_path`` is empty
----------------------------------------
A partial update writes the value it is handed, so sending an empty path sets
the field to ``[]`` — over whatever the document already holds. An extract can
legitimately contain path-less records (``off-extract
--no-require-category-path``, or any run without a taxonomy), so by default they
are **skipped**: this is a backfill, and "this extract resolves no path for the
product" is not a reason to erase the one the index has. They are still counted,
as ``empty(skipped)``, rather than passed over silently.

``--no-skip-empty`` asks for the other reading — make the index agree with the
extract exactly, including for the products the extract resolves no path for.
That is the repair for a document carrying a path an earlier extract generation
produced and the current one does not; it destroys a value by construction, so
it has to be named on the command line. Whichever policy is in force is printed
in the report, and the overwrites it caused are counted separately as
``empty(overwritten)``.

Connection comes from the environment:
    PRISM_ELASTICSEARCH_URL, PRISM_ELASTICSEARCH_API_KEY

Usage:
    python scripts/inject_category_path.py \
        --index catalog_en_v8 \
        --ndjson data/products/off_en_hierarchy_full.ndjson
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, TextIO

from tolerance import Tolerance

MAX_EXAMPLES = 5

EXIT_OK = 0
EXIT_OUTCOME = 1
EXIT_OPERATIONAL = 2


class BulkRequestError(RuntimeError):
    """The bulk request itself failed — the run could not complete."""


def _es() -> tuple[str, str]:
    url = os.environ.get("PRISM_ELASTICSEARCH_URL", "").rstrip("/")
    key = os.environ.get("PRISM_ELASTICSEARCH_API_KEY", "")
    if not url or not key:
        sys.exit("PRISM_ELASTICSEARCH_URL and PRISM_ELASTICSEARCH_API_KEY must be set")
    return url, key


def _bulk(lines: List[str]) -> Dict[str, Any]:
    url, key = _es()
    payload = ("\n".join(lines) + "\n").encode()
    req = urllib.request.Request(f"{url}/_bulk", data=payload, method="POST")
    req.add_header("Authorization", f"ApiKey {key}")
    req.add_header("Content-Type", "application/x-ndjson")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise BulkRequestError(f"bulk failed: HTTP {exc.code}\n{detail}") from exc


@dataclass
class Outcome:
    """What a run actually did, per document rather than per request."""

    sent: int = 0
    updated: int = 0
    noop: int = 0
    not_found: int = 0
    conflict: int = 0
    failed: int = 0
    empty: int = 0
    empty_overwritten: int = 0
    missing_examples: List[str] = field(default_factory=list)
    failure_examples: List[str] = field(default_factory=list)

    @property
    def applied(self) -> int:
        """Documents the index now holds the value for."""
        return self.updated + self.noop

    @property
    def accounted(self) -> int:
        return self.applied + self.not_found + self.conflict + self.failed

    @property
    def unaccounted(self) -> int:
        """Documents sent that the response said nothing about."""
        return self.sent - self.accounted

    @property
    def applied_fraction(self) -> float:
        return self.applied / self.sent if self.sent else 0.0

    def record_item(self, item: Dict[str, Any]) -> None:
        """Classify one entry of a bulk response's ``items`` array."""
        result = item.get("update")
        if not isinstance(result, dict):
            # Not the operation we issued. Counting it as a success would make
            # the totals agree with themselves while describing another request.
            self.failed += 1
            self._remember_failure(json.dumps(item)[:200])
            return
        status = result.get("status", 0)
        doc_id = str(result.get("_id", ""))
        if status in (200, 201):
            if result.get("result") == "noop":
                self.noop += 1
            else:
                self.updated += 1
        elif status == 404:
            self.not_found += 1
            if len(self.missing_examples) < MAX_EXAMPLES and doc_id:
                self.missing_examples.append(doc_id)
        elif status == 409:
            self.conflict += 1
            self._remember_failure(json.dumps(result)[:200])
        else:
            self.failed += 1
            self._remember_failure(json.dumps(result)[:200])

    def record_response(self, response: Dict[str, Any]) -> None:
        for item in response.get("items", []):
            self.record_item(item)

    def _remember_failure(self, detail: str) -> None:
        if len(self.failure_examples) < MAX_EXAMPLES:
            self.failure_examples.append(detail)


def missing_tolerance(
    allow_missing: Optional[int] = None,
    allow_missing_fraction: Optional[float] = None,
) -> Tolerance:
    """How many misses the operator has explicitly agreed to.

    The rule itself — a whole number beats a fraction, a fraction is floored
    and never rounded, nothing named means zero — is ``scripts/tolerance.py``'s
    and is shared with ``verify_catalog.py`` and ``reindex_v7_to_v8.py``. All
    this script contributes is the names of its own two flags, which is what
    the report and any usage error quote back.
    """
    return Tolerance(
        "--allow-missing",
        "--allow-missing-fraction",
        allow_missing,
        allow_missing_fraction,
    )


def read_records(handle: Iterable[str]) -> Iterable[Dict[str, Any]]:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def inject(
    records: Iterable[Dict[str, Any]],
    index: str,
    bulk: Callable[[List[str]], Dict[str, Any]],
    batch_size: int = 1000,
    skip_empty: bool = True,
    progress: Optional[TextIO] = None,
) -> Outcome:
    """Send partial updates for ``records`` and account for every one of them."""
    outcome = Outcome()
    batch: List[str] = []

    def flush() -> None:
        if not batch:
            return
        outcome.record_response(bulk(list(batch)))
        batch.clear()

    for doc in records:
        gtin = str(doc.get("id") or "").strip()
        path = doc.get("category_path") or []
        # Written in the same partial update, never in a second pass: an index
        # holding the addresses but not the primary cannot render "one address
        # plus also categorized as …", and the two arriving separately would
        # leave a window where the document disagrees with itself.
        primary = doc.get("category_path_primary") or []
        if not gtin:
            continue
        if not path:
            if skip_empty:
                outcome.empty += 1
                continue
            # Asked for: write [] over whatever the document holds, so that the
            # index says exactly what this extract says.
            outcome.empty_overwritten += 1
        batch.append(json.dumps({"update": {"_id": gtin, "_index": index}}))
        batch.append(
            json.dumps(
                {"doc": {"category_path": path, "category_path_primary": primary}}
            )
        )
        outcome.sent += 1
        if len(batch) >= batch_size * 2:
            flush()
            if progress is not None and outcome.sent % (batch_size * 20) == 0:
                print(
                    f"  sent {outcome.sent:,} (applied {outcome.applied:,}, "
                    f"not found {outcome.not_found:,})",
                    file=progress,
                    flush=True,
                )
    flush()
    return outcome


def gate(outcome: Outcome, tolerance: Tolerance) -> List[str]:
    """The reasons this run must not report success. Empty means it may."""
    reasons: List[str] = []

    if outcome.sent == 0:
        reasons.append(
            f"nothing was sent: 0 updates issued ({outcome.empty:,} records skipped for an "
            "empty category_path). An input that addresses no document cannot have loaded one"
        )
        return reasons

    # Checked before the tolerance, and deliberately not overridable by it: no
    # value of --allow-missing means "the whole index being absent is fine".
    if outcome.applied == 0:
        reasons.append(
            f"nothing was applied: 0 of {outcome.sent:,} updates reached a document "
            f"({outcome.not_found:,} not found, {outcome.conflict:,} conflicted, "
            f"{outcome.failed:,} failed). Check --index and the extract this id set came from"
        )
        return reasons

    permitted = tolerance.permitted(outcome.sent)
    if outcome.not_found > permitted:
        reasons.append(
            f"{outcome.not_found:,} of {outcome.sent:,} ids are not in the index "
            f"(tolerance {permitted:,})"
        )
    if outcome.conflict:
        reasons.append(f"{outcome.conflict:,} updates hit a version conflict and did not apply")
    if outcome.failed:
        reasons.append(f"{outcome.failed:,} updates failed")
    if outcome.unaccounted:
        reasons.append(
            f"{outcome.unaccounted:,} of {outcome.sent:,} documents were sent but the "
            "response accounted for neither success nor failure"
        )
    return reasons


def report(outcome: Outcome, index: str, tolerance: Tolerance, skip_empty: bool = True) -> str:
    # Stated on every run, whether or not the input held an empty path, for the
    # reason the tolerance is: the record of a load says what was permitted.
    empty_policy = (
        "skipped" if skip_empty else "overwritten with [] (--no-skip-empty)"
    )
    lines = [
        f"Done. index={index} sent={outcome.sent:,} applied={outcome.applied:,} "
        f"(updated={outcome.updated:,} noop={outcome.noop:,}) "
        f"not_found={outcome.not_found:,} conflict={outcome.conflict:,} "
        f"failed={outcome.failed:,} unaccounted={outcome.unaccounted:,} "
        f"empty(skipped)={outcome.empty:,} empty(overwritten)={outcome.empty_overwritten:,}",
        f"  applied rate: {outcome.applied_fraction * 100:.2f}% "
        f"| missing tolerance: {tolerance.describe(outcome.sent)} "
        f"| empty paths: {empty_policy}",
    ]
    if outcome.missing_examples:
        lines.append(
            f"  first ids not in the index: {', '.join(outcome.missing_examples)}"
            + (" ..." if outcome.not_found > len(outcome.missing_examples) else "")
        )
    for detail in outcome.failure_examples:
        lines.append(f"  failure: {detail}")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge category_path onto the documents of an existing index, and exit non-zero "
            "unless the updates actually reached them."
        ),
        epilog=(
            "Exit status: 0 applied (within any named tolerance); 1 the run completed and its "
            "outcomes fail the gate; 2 the run could not complete."
        ),
    )
    ap.add_argument("--index", required=True, help="e.g. catalog_en_v8")
    ap.add_argument("--ndjson", required=True, help="extractor output with category_path")
    ap.add_argument("--batch", type=int, default=1000, help="docs per bulk request")
    ap.add_argument(
        "--skip-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="do not send updates for records with an empty category_path "
        "(default: on). --no-skip-empty sends them, which writes an empty list "
        "over whatever category_path the document already holds",
    )
    allowance = ap.add_mutually_exclusive_group()
    allowance.add_argument(
        "--allow-missing",
        type=int,
        default=None,
        metavar="N",
        help="permit up to N ids that the index does not hold (default: 0)",
    )
    allowance.add_argument(
        "--allow-missing-fraction",
        type=float,
        default=None,
        metavar="F",
        help="permit a fraction of the sent ids to be missing, e.g. 0.01 (default: 0)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    tolerance = missing_tolerance(args.allow_missing, args.allow_missing_fraction)
    problem = tolerance.problem()
    if problem is not None:
        ap.error(problem)

    try:
        with open(args.ndjson, encoding="utf-8") as handle:
            outcome = inject(
                read_records(handle),
                args.index,
                bulk=_bulk,
                batch_size=args.batch,
                skip_empty=args.skip_empty,
                progress=sys.stdout,
            )
    except BulkRequestError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_OPERATIONAL

    print(report(outcome, args.index, tolerance, skip_empty=args.skip_empty))

    reasons = gate(outcome, tolerance)
    for reason in reasons:
        print(f"FAILED: {reason}", file=sys.stderr)
    return EXIT_OUTCOME if reasons else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
