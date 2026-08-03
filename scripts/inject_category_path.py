#!/usr/bin/env python3
"""Bulk-add the hierarchical ``category_path`` field onto an existing index.

Reads an extractor NDJSON (which contains ``id`` = GTIN and ``category_path``)
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
import math
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, TextIO

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


@dataclass
class Tolerance:
    """How many misses the operator has explicitly agreed to."""

    allow_missing: Optional[int] = None
    allow_missing_fraction: Optional[float] = None

    def permitted(self, sent: int) -> int:
        if self.allow_missing is not None:
            return self.allow_missing
        if self.allow_missing_fraction is not None:
            return math.floor(self.allow_missing_fraction * sent)
        return 0

    def describe(self, sent: int) -> str:
        if self.allow_missing is not None:
            return f"--allow-missing {self.allow_missing}"
        if self.allow_missing_fraction is not None:
            return (
                f"--allow-missing-fraction {self.allow_missing_fraction:g} "
                f"({self.permitted(sent):,} of {sent:,})"
            )
        return "0 (default: zero tolerance)"


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
        if not gtin:
            continue
        if skip_empty and not path:
            outcome.empty += 1
            continue
        batch.append(json.dumps({"update": {"_id": gtin, "_index": index}}))
        batch.append(json.dumps({"doc": {"category_path": path}}))
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


def report(outcome: Outcome, index: str, tolerance: Tolerance) -> str:
    lines = [
        f"Done. index={index} sent={outcome.sent:,} applied={outcome.applied:,} "
        f"(updated={outcome.updated:,} noop={outcome.noop:,}) "
        f"not_found={outcome.not_found:,} conflict={outcome.conflict:,} "
        f"failed={outcome.failed:,} unaccounted={outcome.unaccounted:,} "
        f"empty(skipped)={outcome.empty:,}",
        f"  applied rate: {outcome.applied_fraction * 100:.2f}% "
        f"| missing tolerance: {tolerance.describe(outcome.sent)}",
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
        action="store_true",
        default=True,
        help="do not send updates for records with an empty category_path",
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

    if args.allow_missing is not None and args.allow_missing < 0:
        ap.error("--allow-missing must not be negative")
    if args.allow_missing_fraction is not None and not 0.0 <= args.allow_missing_fraction <= 1.0:
        ap.error("--allow-missing-fraction must be between 0 and 1")

    tolerance = Tolerance(args.allow_missing, args.allow_missing_fraction)

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

    print(report(outcome, args.index, tolerance))

    reasons = gate(outcome, tolerance)
    for reason in reasons:
        print(f"FAILED: {reason}", file=sys.stderr)
    return EXIT_OUTCOME if reasons else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
