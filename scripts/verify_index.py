#!/usr/bin/env python3
"""Verify a **built index** against the build manifest that describes it.

``scripts/verify_catalog.py`` reads the NDJSON before it is loaded.
``scripts/build_manifest.py`` pins what a build was made from. Nothing had ever
read the other end — the index — and put its numbers next to the manifest's.
That gap is not hypothetical: ``catalog_fr_v13`` held 195,209 documents while the
extract it was loaded from holds 222,955 distinct ids, a 12.5% shortfall that
survived months and three index generations because no check ever compared the
two numbers.

This script is that comparison. It is read-only, it is cheap, and it is a CLI, so
an operator can run it at load time against any index/manifest pair rather than
discovering the shortfall by archaeology.

What it checks
--------------

``manifest_identity``
    Whether the index says which build produced it. An index built by this
    project currently records **nothing** about its dump, taxonomy or extractor
    commit, so this check reports ``unverifiable`` rather than pass or fail, and
    prints the exact ``_meta`` block that would make it answerable (see
    "Making an index self-describing" below). It also confirms that the taxonomy
    file passed to ``--taxonomy`` is byte-for-byte the snapshot the manifest
    pins, so the vocabulary checks below are known to run against the right
    snapshot rather than a look-alike.

``document_count``
    ``_count`` against the manifest's ``distinct_ids`` — **not** its record
    count. An index keyed by id holds one document per distinct id, and the two
    differ (by 1 / 3 / 81 for en / es / fr in the 2026-08-03 build). Comparing
    against records would have made a correct index look 81 documents short and
    invited someone to explain the difference away.

``category_path_coverage``
    How many documents carry ``category_path`` at all. This separates two
    failures that produce the same count shortfall: documents that were never
    written, and documents that were written but never received the field
    (``scripts/inject_category_path.py`` issues partial ``_update`` operations,
    which cannot create a document and count the rest as ``missing``).

``category_vocabulary`` (needs ``--taxonomy``)
    Every distinct ``taxonomy_tags`` value and every ``category_path`` segment in
    the index, checked against the display labels of the pinned snapshot — the
    index-side mirror of the rule ``verify_catalog.py`` applies to the NDJSON, so
    a catalog and its index are judged by one rule rather than two. Reported in
    both directions: values the index uses that the snapshot does not explain
    (a failure — the index's vocabulary is not the snapshot's), and labels the
    snapshot has that the index never uses (informational — no catalog is
    obliged to use every label).

``document_identity`` (opt-in, ``--catalog``)
    The exact id-set difference between the index and an extract NDJSON, plus
    the run-length profile of the missing ids in extract order. This is what
    turns "27,746 short" into "13 contiguous stretches of the extract were never
    written", which is the difference between a load that dropped batches and an
    index built from an older extract. Opt-in because it costs one pass over the
    NDJSON and ~20 paginated requests; the checks above cost three requests.

Read-only, on a cluster other people are using
----------------------------------------------

Every request goes through one helper that refuses any endpoint outside a
four-entry allowlist (``_search``, ``_count``, ``_mapping``, ``_settings``) and
any method other than GET and POST. ``POST`` appears only because ``_search``
needs a body; nothing here creates, updates, deletes or reindexes anything. The
API key is read from the environment and never accepted as an argument, so it
cannot end up in shell history or in another user's ``ps`` output.

Cheap, by construction
----------------------

The whole default run is **one** ``_search`` with ``size: 0`` — exact total via
``track_total_hits``, the ``category_path`` coverage filter, and both vocabulary
terms aggregations in a single round trip — plus one ``_mapping`` read. No
scroll, no per-document fetch, nothing that scales with catalog size on the
client.

Truncation is read, not assumed
-------------------------------

A ``terms`` aggregation that hits its ``size`` silently returns a *prefix* of the
vocabulary, and a vocabulary check that reads a prefix reports "no unknown
values" for exactly the wrong reason — the same class of bug as the one this
script exists to catch. So all three truncation signals are read and reported:

* ``len(buckets) == size`` — the reliable tell, and the escalation trigger.
* ``sum_other_doc_count`` — reported, but **not** trusted alone: on a
  multi-valued field (``taxonomy_tags`` and ``category_path`` both are) a document
  is counted in every bucket it lands in, so the sum of bucket doc counts can
  exceed the document total and drive this to zero while terms are still
  missing.
* ``doc_count_error_upper_bound`` — reported, but note it is structurally zero on
  a single-shard index, which every catalog index here is. A zero is therefore
  not evidence of completeness.

On any of those, the field is re-enumerated with a ``composite`` aggregation,
which paginates deterministically over the whole term space and is exhaustive by
construction. The result records that it escalated, so a reader can tell an
exhaustive answer from a lucky one. ``cardinality`` is deliberately not used to
cross-check the bucket count: it is a HyperLogLog estimate even below its
precision threshold, and it already disagrees with the exhaustive enumeration on
one of these indices (6,500 vs 6,499 for ``catalog_fr_v13``).

Making an index self-describing
-------------------------------

Nothing in an index today names the build it came from. Elasticsearch has a place
for exactly this — ``_meta`` on the mapping, which is free-form, survives
``dynamic: strict``, and can be set at creation *or* added afterwards with
``PUT /<index>/_mapping`` without a reindex. The loader should write::

    "_meta": {"off_catalog_build": {
        "manifest_schema": "off-catalog-build-manifest/1",
        "manifest_sha256": "<digest of the manifest file>",
        "generated_utc": "<manifest generated_utc>",
        "lang": "fr",
        "extractor_commit": "<manifest extractor.commit>",
        "dump_sha256": "<manifest dump.sha256>",
        "taxonomy_sha256": "<manifest taxonomy.sha256>",
        "catalog_sha256": "<manifest locale catalog.sha256>",
        "expected_distinct_ids": 222955
    }}

With that block present this script stops needing to be *told* which manifest to
trust: it reads the claim off the index and refuses a manifest that does not
match it. Until then ``--manifest`` is an assertion by the operator, and the
``manifest_identity`` check says so instead of pretending otherwise.

Usage
-----

::

    export PRISM_ELASTICSEARCH_URL=...  PRISM_ELASTICSEARCH_API_KEY=...

    python scripts/verify_index.py \\
        --index catalog_fr_v13 \\
        --manifest builds/2026-08-03/build_manifest.json \\
        --taxonomy data/taxonomy/categories.json

    # exact id-set diff, when a count shortfall needs a mechanism
    python scripts/verify_index.py --index catalog_fr_v13 \\
        --manifest builds/2026-08-03/build_manifest.json \\
        --catalog data/products/off_fr_v14.ndjson

Exit status is 1 if any check failed, 0 otherwise. ``unverifiable`` checks do not
fail the run unless ``--require-self-describing`` is passed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from off_demo_extract.taxonomy import (  # noqa: E402
    PATH_SEPARATOR,
    display_label,
    global_roots,
    load_taxonomy,
)

SCHEMA = "off-index-verification/1"
MAX_EXAMPLES = 10
CHUNK = 1 << 22

# Every endpoint this script is permitted to touch. Membership is checked on the
# last path segment of the request, so a typo or a future edit cannot reach a
# write endpoint by accident. Read-only-ness is a property of the code path, not
# of the author's intentions on the day.
READ_ONLY_ENDPOINTS = frozenset({"_search", "_count", "_mapping", "_settings"})

# Enough for every catalog built here (the largest uses 16,743 distinct
# ``taxonomy_tags`` values) with room to spare, and saturation escalates rather than
# truncates, so this is a performance knob and not a correctness one.
DEFAULT_TERMS_SIZE = 30000
COMPOSITE_PAGE = 10000


class VerificationError(RuntimeError):
    """A problem with the inputs or the cluster, as opposed to a failed check."""


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #


class ReadOnlyClient:
    """Minimal Elasticsearch client that can only read.

    Written against ``urllib`` rather than the official client because this
    package declares no runtime dependencies (see ``pyproject.toml``) and the
    other scripts here do the same.
    """

    def __init__(self, url: str, api_key: str, timeout: float = 180.0) -> None:
        self.url = url.rstrip("/")
        self._api_key = api_key
        self.timeout = timeout
        self.requests: List[str] = []

    @classmethod
    def from_env(cls, url: Optional[str] = None, timeout: float = 180.0) -> "ReadOnlyClient":
        resolved = url or os.environ.get("PRISM_ELASTICSEARCH_URL", "")
        key = os.environ.get("PRISM_ELASTICSEARCH_API_KEY", "")
        if not resolved or not key:
            raise VerificationError(
                "PRISM_ELASTICSEARCH_URL and PRISM_ELASTICSEARCH_API_KEY must be set "
                "(the key is read from the environment on purpose: passing it as an "
                "argument would put it in shell history and in ps output)"
            )
        return cls(resolved, key, timeout=timeout)

    def request(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        endpoint = urllib.parse.urlsplit(path).path.rstrip("/").rsplit("/", 1)[-1]
        if endpoint not in READ_ONLY_ENDPOINTS:
            raise VerificationError(
                f"refusing {path!r}: {endpoint!r} is not one of the read-only endpoints "
                f"{sorted(READ_ONLY_ENDPOINTS)}"
            )
        self.requests.append(path)
        data = json.dumps(body).encode() if body is not None else None
        # POST only because ``_search`` needs a body; GET and POST are the only
        # methods this class can issue, and neither one mutates a read endpoint.
        req = urllib.request.Request(
            f"{self.url}{path}", data=data, method="POST" if data is not None else "GET"
        )
        req.add_header("Authorization", f"ApiKey {self._api_key}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode(errors="replace")[:2000]
            raise VerificationError(f"{path} failed: HTTP {exc.code}\n{detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise VerificationError(f"{path} failed: {exc.reason}") from exc


# --------------------------------------------------------------------------- #
# aggregation reading
# --------------------------------------------------------------------------- #


def terms_truncation(agg: Dict[str, Any], size: int) -> Dict[str, Any]:
    """Read every truncation signal a ``terms`` aggregation offers.

    Returns the raw numbers alongside the verdict, because the numbers are what
    a reader needs to judge the verdict — and because two of the three are
    weaker than they look (see the module docstring).
    """
    buckets = agg.get("buckets", [])
    saturated = len(buckets) >= size
    other = agg.get("sum_other_doc_count", 0)
    error = agg.get("doc_count_error_upper_bound", 0)
    return {
        "requested_size": size,
        "buckets_returned": len(buckets),
        "size_saturated": saturated,
        "sum_other_doc_count": other,
        "doc_count_error_upper_bound": error,
        "complete": not saturated and not other and not error,
    }


def composite_terms(
    request: Callable[[str, Optional[Dict[str, Any]]], Dict[str, Any]],
    index: str,
    field: str,
    page: int = COMPOSITE_PAGE,
) -> Dict[str, int]:
    """Enumerate every distinct value of ``field`` exactly.

    A ``composite`` aggregation paginates over the whole term space in a defined
    order and stops when it runs out, so unlike ``terms`` it cannot return a
    prefix and call it the answer. It is the escalation path, not the default,
    because it costs one request per ``page`` values.
    """
    values: Dict[str, int] = {}
    after: Optional[Dict[str, Any]] = None
    while True:
        source: Dict[str, Any] = {
            "composite": {"size": page, "sources": [{"v": {"terms": {"field": field}}}]}
        }
        if after is not None:
            source["composite"]["after"] = after
        agg = request(f"/{index}/_search", {"size": 0, "aggs": {"values": source}})[
            "aggregations"
        ]["values"]
        buckets = agg.get("buckets", [])
        if not buckets:
            return values
        for bucket in buckets:
            values[bucket["key"]["v"]] = bucket["doc_count"]
        after = agg.get("after_key")
        if after is None:
            return values


def resolve_vocabulary(
    request: Callable[[str, Optional[Dict[str, Any]]], Dict[str, Any]],
    index: str,
    field: str,
    agg: Dict[str, Any],
    size: int,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """The field's distinct values, escalating past a truncated ``terms`` read."""
    truncation = terms_truncation(agg, size)
    if truncation["complete"]:
        truncation["escalated_to_composite"] = False
        return {b["key"]: b["doc_count"] for b in agg.get("buckets", [])}, truncation
    values = composite_terms(request, index, field)
    truncation["escalated_to_composite"] = True
    truncation["exhaustive_distinct_values"] = len(values)
    return values, truncation


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_lang(index: str) -> Optional[str]:
    """``catalog_fr_v13`` -> ``fr``. Returns None when the name says nothing."""
    parts = index.split("_")
    for part in parts[1:]:
        if len(part) == 2 and part.isalpha():
            return part.lower()
    return None


def manifest_locale(manifest: Dict[str, Any], lang: str) -> Dict[str, Any]:
    for entry in manifest.get("locales", []):
        if entry.get("lang") == lang:
            return entry
    known = [entry.get("lang") for entry in manifest.get("locales", [])]
    raise VerificationError(f"manifest has no locale {lang!r}; it has {known}")


def expected_distinct_ids(entry: Dict[str, Any], lang: str) -> int:
    verification = entry.get("artifact_verification") or {}
    value = verification.get("distinct_ids")
    if value is None:
        raise VerificationError(
            f"manifest locale {lang!r} carries no artifact_verification.distinct_ids, so "
            "there is no number to check the index against. Run scripts/verify_catalog.py "
            "on the catalog and rebuild the manifest with its output."
        )
    return int(value)


def _check(name: str, status: str, **detail: Any) -> Dict[str, Any]:
    return {"check": name, "status": status, **detail}


def top(counts: Dict[str, int], limit: int = MAX_EXAMPLES) -> List[Dict[str, Any]]:
    return [
        {"value": value, "doc_count": count}
        for value, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    ]


def contiguous_runs(positions: Sequence[int]) -> List[Tuple[int, int]]:
    """Collapse sorted positions into ``(start, end)`` inclusive runs.

    The run-length profile is the whole point of the id-set check: 928 missing
    ids in 700 runs is a per-record difference between two extracts, and 27,746
    missing ids in 48 runs — 13 of which cover 99.9% of them — is a load that
    dropped batches. The counts alone cannot tell those apart.
    """
    runs: List[Tuple[int, int]] = []
    start = previous = None
    for position in positions:
        if start is None:
            start = previous = position
        elif position == previous + 1:
            previous = position
        else:
            runs.append((start, previous))
            start = previous = position
    if start is not None and previous is not None:
        runs.append((start, previous))
    return runs


def read_catalog_ids(path: Path) -> List[str]:
    """Distinct ids in the order the catalog emits them, duplicates dropped."""
    seen: Set[str] = set()
    order: List[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            identifier = str(json.loads(line).get("id") or "")
            if identifier and identifier not in seen:
                seen.add(identifier)
                order.append(identifier)
    return order


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #


def check_manifest_identity(
    mapping_meta: Dict[str, Any],
    manifest: Dict[str, Any],
    manifest_path: Path,
    lang: str,
    taxonomy_path: Optional[Path],
) -> Dict[str, Any]:
    claimed = (mapping_meta or {}).get("off_catalog_build")
    taxonomy_pinned = (manifest.get("taxonomy") or {}).get("sha256")

    snapshot: Dict[str, Any] = {"pinned_taxonomy_sha256": taxonomy_pinned}
    if taxonomy_path is not None:
        actual = sha256(taxonomy_path)
        snapshot.update({"taxonomy_file": str(taxonomy_path), "taxonomy_file_sha256": actual})
        if taxonomy_pinned and actual != taxonomy_pinned:
            return _check(
                "manifest_identity",
                "fail",
                reason=(
                    "the --taxonomy file is not the snapshot this manifest pins, so the "
                    "vocabulary checks would judge the index against the wrong taxonomy"
                ),
                **snapshot,
            )

    if not claimed:
        return _check(
            "manifest_identity",
            "unverifiable",
            reason=(
                "the index records nothing about the build that produced it: its mapping "
                "has no _meta.off_catalog_build block. The manifest named on the command "
                "line is therefore an assertion by the operator, not something the index "
                "confirms."
            ),
            remedy=(
                "have the loader write _meta.off_catalog_build (manifest_sha256, "
                "generated_utc, lang, extractor_commit, dump_sha256, taxonomy_sha256, "
                "catalog_sha256, expected_distinct_ids). _meta is free-form, survives "
                "dynamic:strict, and can be added to a live index with "
                "PUT /<index>/_mapping without a reindex."
            ),
            manifest=str(manifest_path),
            manifest_sha256=sha256(manifest_path),
            asserted={
                "extractor_commit": (manifest.get("extractor") or {}).get("commit"),
                "dump_sha256": (manifest.get("dump") or {}).get("sha256"),
                "taxonomy_sha256": taxonomy_pinned,
            },
            **snapshot,
        )

    expected = {
        "extractor_commit": (manifest.get("extractor") or {}).get("commit"),
        "dump_sha256": (manifest.get("dump") or {}).get("sha256"),
        "taxonomy_sha256": taxonomy_pinned,
        "lang": lang,
    }
    disagreements = {
        key: {"index_claims": claimed.get(key), "manifest_says": value}
        for key, value in expected.items()
        if claimed.get(key) is not None and claimed.get(key) != value
    }
    status = "fail" if disagreements else "pass"
    return _check(
        "manifest_identity",
        status,
        index_claims=claimed,
        manifest=str(manifest_path),
        manifest_sha256=sha256(manifest_path),
        disagreements=disagreements,
        **snapshot,
    )


def check_document_count(
    indexed: int, entry: Dict[str, Any], lang: str, expected: int
) -> Dict[str, Any]:
    records = (entry.get("counters") or {}).get("written")
    delta = indexed - expected
    return _check(
        "document_count",
        "pass" if delta == 0 else "fail",
        indexed_documents=indexed,
        manifest_distinct_ids=expected,
        delta=delta,
        manifest_records=records,
        # Named explicitly so nobody reconciles against the wrong number and
        # declares a correct index broken, or a broken one correct.
        note=(
            "compared against distinct_ids, not records: an index keyed by id holds one "
            f"document per distinct id, and this locale's catalog has "
            f"{'' if records is None else records - expected} duplicate id instances"
        ),
    )


def check_category_path_coverage(indexed: int, with_path: int) -> Dict[str, Any]:
    missing = indexed - with_path
    return _check(
        "category_path_coverage",
        "pass" if missing == 0 else "fail",
        indexed_documents=indexed,
        with_category_path=with_path,
        without_category_path=missing,
        note=(
            "documents present but without the field are the signature of a partial "
            "_update pass over an index that already held other documents; documents "
            "absent entirely are the signature of an incomplete load"
        ),
    )


def check_category_vocabulary(
    taxonomy_tags: Dict[str, int],
    addresses: Dict[str, int],
    labels: Set[str],
    root_labels: Set[str],
) -> Dict[str, Any]:
    outside = {value: count for value, count in taxonomy_tags.items() if value not in labels}

    segments: Set[str] = set()
    for address in addresses:
        segments.update(address.split(PATH_SEPARATOR))
    segments_outside = sorted(segment for segment in segments if segment not in labels)

    heads = {address.split(PATH_SEPARATOR)[0] for address in addresses}
    heads_outside = sorted(head for head in heads if head not in root_labels)

    # Every document emits its whole cumulative chain, so each address deeper
    # than a root must also appear one segment shorter. One that does not means
    # a chain reached the index without its own prefix.
    orphans = sorted(
        address
        for address in addresses
        if PATH_SEPARATOR in address
        and address.rsplit(PATH_SEPARATOR, 1)[0] not in addresses
    )

    unused = sorted(labels - set(taxonomy_tags))
    failed = bool(outside or segments_outside or heads_outside or orphans)
    return _check(
        "category_vocabulary",
        "fail" if failed else "pass",
        snapshot_labels=len(labels),
        snapshot_root_labels=len(root_labels),
        distinct_categories_in_index=len(taxonomy_tags),
        distinct_category_path_addresses=len(addresses),
        distinct_category_path_segments=len(segments),
        values_outside_snapshot=len(outside),
        value_instances_outside_snapshot=sum(outside.values()),
        top_values_outside_snapshot=top(outside),
        path_segments_outside_snapshot=len(segments_outside),
        top_path_segments_outside_snapshot=segments_outside[:MAX_EXAMPLES],
        path_heads_outside_taxonomy_roots=len(heads_outside),
        top_path_heads_outside_taxonomy_roots=heads_outside[:MAX_EXAMPLES],
        orphan_addresses=len(orphans),
        top_orphan_addresses=orphans[:MAX_EXAMPLES],
        # The reverse direction the issue asks for. Not a failure on its own: a
        # catalog covering part of the taxonomy is normal, and a catalog using a
        # label the snapshot does not have is not.
        snapshot_labels_unused_by_index=len(unused),
        sample_snapshot_labels_unused_by_index=unused[:MAX_EXAMPLES],
    )


def check_document_identity(
    request: Callable[[str, Optional[Dict[str, Any]]], Dict[str, Any]],
    index: str,
    catalog: Path,
) -> Dict[str, Any]:
    indexed_ids = set(composite_terms(request, index, "id"))
    catalog_ids = read_catalog_ids(catalog)
    position = {identifier: order for order, identifier in enumerate(catalog_ids)}

    missing = [identifier for identifier in catalog_ids if identifier not in indexed_ids]
    extra = sorted(indexed_ids - set(catalog_ids))
    runs = contiguous_runs([position[identifier] for identifier in missing])
    long_runs = [run for run in runs if run[1] - run[0] + 1 >= 100]

    return _check(
        "document_identity",
        "pass" if not missing and not extra else "fail",
        catalog=str(catalog),
        catalog_distinct_ids=len(catalog_ids),
        indexed_ids=len(indexed_ids),
        catalog_ids_absent_from_index=len(missing),
        index_ids_absent_from_catalog=len(extra),
        sample_index_ids_absent_from_catalog=extra[:MAX_EXAMPLES],
        missing_contiguous_runs=len(runs),
        missing_runs_of_100_or_more=len(long_runs),
        ids_in_runs_of_100_or_more=sum(end - start + 1 for start, end in long_runs),
        largest_missing_runs=[
            {"catalog_position": start, "length": end - start + 1}
            for start, end in sorted(runs, key=lambda run: -(run[1] - run[0]))[:MAX_EXAMPLES]
        ],
        note=(
            "many short runs is a per-record difference between two extracts; a few long "
            "runs is a load that dropped batches"
        ),
    )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def verify(
    client: ReadOnlyClient,
    index: str,
    manifest_path: Path,
    *,
    lang: Optional[str] = None,
    taxonomy_path: Optional[Path] = None,
    catalog_path: Optional[Path] = None,
    terms_size: int = DEFAULT_TERMS_SIZE,
) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_lang = lang or infer_lang(index)
    if not resolved_lang:
        raise VerificationError(f"cannot infer a locale from index name {index!r}; pass --lang")
    entry = manifest_locale(manifest, resolved_lang)
    expected = expected_distinct_ids(entry, resolved_lang)

    def request(path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return client.request(path, body)

    mappings = request(f"/{index}/_mapping")[index]["mappings"]

    # One round trip: exact total, coverage, and both vocabularies.
    response = request(
        f"/{index}/_search",
        {
            "size": 0,
            "track_total_hits": True,
            "aggs": {
                "with_category_path": {"filter": {"exists": {"field": "category_path"}}},
                "taxonomy_tags": {"terms": {"field": "taxonomy_tags", "size": terms_size}},
                "category_path": {"terms": {"field": "category_path", "size": terms_size}},
            },
        },
    )
    indexed = response["hits"]["total"]["value"]
    aggregations = response["aggregations"]
    with_path = aggregations["with_category_path"]["doc_count"]

    checks: List[Dict[str, Any]] = [
        check_manifest_identity(
            mappings.get("_meta") or {}, manifest, manifest_path, resolved_lang, taxonomy_path
        ),
        check_document_count(indexed, entry, resolved_lang, expected),
        check_category_path_coverage(indexed, with_path),
    ]

    truncation: Dict[str, Any] = {}
    if taxonomy_path is not None:
        taxonomy_tags, truncation["taxonomy_tags"] = resolve_vocabulary(
            request, index, "taxonomy_tags", aggregations["taxonomy_tags"], terms_size
        )
        addresses, truncation["category_path"] = resolve_vocabulary(
            request, index, "category_path", aggregations["category_path"], terms_size
        )
        taxonomy = load_taxonomy(taxonomy_path)
        labels = {display_label(taxonomy, node, resolved_lang) for node in taxonomy}
        root_labels = {
            display_label(taxonomy, node, resolved_lang) for node in global_roots(taxonomy)
        }
        checks.append(check_category_vocabulary(taxonomy_tags, addresses, labels, root_labels))
    else:
        checks.append(
            _check(
                "category_vocabulary",
                "skipped",
                reason="pass --taxonomy to check the index vocabulary against the snapshot",
            )
        )

    if catalog_path is not None:
        checks.append(check_document_identity(request, index, catalog_path))
    else:
        checks.append(
            _check(
                "document_identity",
                "skipped",
                reason=(
                    "pass --catalog to diff the index id set against the extract; it costs "
                    "one pass over the NDJSON and one paginated enumeration of the index"
                ),
            )
        )

    return {
        "schema": SCHEMA,
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index": index,
        "lang": resolved_lang,
        "manifest": {
            "path": str(manifest_path),
            "schema": manifest.get("schema"),
            "generated_utc": manifest.get("generated_utc"),
        },
        "checks": checks,
        "aggregation_truncation": truncation,
        "failed": [c["check"] for c in checks if c["status"] == "fail"],
        "unverifiable": [c["check"] for c in checks if c["status"] == "unverifiable"],
        "requests": client.requests,
    }


def summarise(result: Dict[str, Any]) -> str:
    lines = [f"{result['index']} ({result['lang']}) against {result['manifest']['path']}"]
    for check in result["checks"]:
        marker = {
            "pass": "PASS",
            "fail": "FAIL",
            "skipped": "SKIP",
            "unverifiable": "????",
        }[check["status"]]
        lines.append(f"  [{marker}] {check['check']}")
        if check["check"] == "document_count":
            lines.append(
                f"          indexed {check['indexed_documents']:,} vs manifest "
                f"distinct_ids {check['manifest_distinct_ids']:,} "
                f"(delta {check['delta']:+,})"
            )
        elif check["check"] == "category_path_coverage" and check["status"] != "skipped":
            lines.append(
                f"          {check['with_category_path']:,} of "
                f"{check['indexed_documents']:,} documents carry category_path"
            )
        elif check["check"] == "category_vocabulary" and check["status"] not in ("skipped",):
            lines.append(
                f"          {check['values_outside_snapshot']:,} index values outside the "
                f"snapshot ({check['value_instances_outside_snapshot']:,} instances); "
                f"{check['snapshot_labels_unused_by_index']:,} snapshot labels unused"
            )
        elif check["check"] == "document_identity" and check["status"] != "skipped":
            lines.append(
                f"          {check['catalog_ids_absent_from_index']:,} catalog ids absent "
                f"from the index in {check['missing_contiguous_runs']:,} runs; "
                f"{check['index_ids_absent_from_catalog']:,} index ids absent from the catalog"
            )
    lines.append(
        "  => "
        + (
            "FAILED: " + ", ".join(result["failed"])
            if result["failed"]
            else "all checks passed"
        )
        + (
            "; unverifiable: " + ", ".join(result["unverifiable"])
            if result["unverifiable"]
            else ""
        )
    )
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a built index against the build manifest that describes it."
    )
    parser.add_argument("--index", required=True, help="e.g. catalog_fr_v13")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lang", default=None, help="defaults to the locale in the index name")
    parser.add_argument(
        "--taxonomy", type=Path, default=None, help="pinned snapshot; enables the vocabulary check"
    )
    parser.add_argument(
        "--catalog", type=Path, default=None, help="extract NDJSON; enables the id-set check"
    )
    parser.add_argument("--url", default=None, help="overrides PRISM_ELASTICSEARCH_URL")
    parser.add_argument("--terms-size", type=int, default=DEFAULT_TERMS_SIZE)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--json", type=Path, default=None, help="also write the result here")
    parser.add_argument(
        "--require-self-describing",
        action="store_true",
        help="treat an index that records no build identity as a failure",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        client = ReadOnlyClient.from_env(args.url, timeout=args.timeout)
        result = verify(
            client,
            args.index,
            args.manifest,
            lang=args.lang,
            taxonomy_path=args.taxonomy,
            catalog_path=args.catalog,
            terms_size=args.terms_size,
        )
    except VerificationError as exc:
        print(f"verify_index: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    print("\n" + summarise(result), file=sys.stderr)

    if result["failed"]:
        return 1
    if args.require_self_describing and result["unverifiable"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
