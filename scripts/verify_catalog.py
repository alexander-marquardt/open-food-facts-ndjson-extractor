#!/usr/bin/env python3
"""Verify a built catalog NDJSON against the taxonomy snapshot it was built from.

The extractor's own report records what the *run* believed it emitted. This
script re-derives the same properties from the **artifact on disk**, so a claim
about a catalog does not rest on the code that wrote it. It reads only the
NDJSON and the pinned taxonomy file; it never contacts a cluster.

What it checks, per catalog:

* **Property 3 — a prefix-closed union of chains per product.** ``category_path``
  holds the cumulative ``/``-joined addresses of every position the product sits
  at, so every value's ancestor prefix must be a value of its own, and no value
  may repeat. ``category_path_primary`` — the one address the product page leads
  with — must additionally be a *single* chain (element *i* is element *i-1* plus
  one segment), and must be the head of ``category_path``.

  This is the restatement of the older "exactly one chain per product". The
  source taxonomy is a DAG, 2,545 of whose 14,457 nodes have several parents, and
  a catalog that files each of them at one address is asserting something the
  data does not say. What was load-bearing about the old reading survives on the
  primary; what it cost was every second breadcrumb to a product.
* **Property 2 — every category at exactly one PRIMARY address.** A category (a
  path segment) may occur at several cumulative addresses — that is the DAG — but
  its address in ``category_path_primary`` must be the same on every product in
  the catalog. A primary that differs between two products means the breadcrumb a
  product page leads with is not a property of the category, which is the defect
  the original check was written for. The number of segments at several
  *non-primary* addresses is measured and reported, and is not a failure.
* **Anchoring.** Every chain's first segment must be the display label of one of
  the taxonomy's global roots. This is the artifact-side reading of the
  extractor's ``category_path_anchoring`` block.
* **Vocabulary against the pinned snapshot.** Every ``taxonomy_tags`` value and
  every path segment must be a display label of some node in the snapshot. Values
  present in the catalog but not in the snapshot are reported — the offline half
  of "compare the indexed vocabulary against the pinned snapshot", checked
  against the file that will be indexed rather than after the fact.

What fails the run
------------------

Every one of the checks above is fatal, at zero tolerance, and every one of them
names its reason in ``failure_reasons`` and on stderr. The vocabulary check used
to be the exception: it was measured, printed, and then left out of the sum that
became the exit status, so a catalog none of whose values the pinned snapshot
explains exited **0**. ``scripts/verify_index.py`` gates the same rule on the
index side, which left the two ends of one rule disagreeing about whether it was
fatal.

An exception is named on the command line rather than budgeted for silently:
``--allow-values-outside-snapshot N``, or ``--allow-values-outside-snapshot-
fraction F`` of the distinct values actually checked (floored, so a fraction
never rounds up into permitting one more). The tolerance in force is printed with
the result *and* recorded in the JSON, so the record says what was permitted
rather than what was requested.

``duplicate_id_instances`` is deliberately **not** fatal by default; pass
``--require-unique-ids`` to make it so. Two records sharing an id is a property
of the upstream dump, not of anything this extractor constructs, and the index
resolves it deterministically — it is keyed by id, so the last record wins and
the index holds one document. That is why ``distinct_ids``, not ``records``, is
the number ``verify_index.py`` compares ``_count`` against: the duplicates are
already accounted for downstream rather than unexplained. Failing on them by
default would assert a uniqueness rule this project has never adopted and cannot
enforce without changing what the extractor emits.

Exit status:
    0   the catalog passes every gate (within any tolerance named on the command
        line)
    1   the catalog fails a gate
    2   the verification could not be carried out — a file is missing, or a line
        of the NDJSON is not a JSON object. No verdict was reached, which is a
        different thing from a verdict of "bad", and previously both exited 1.

Usage:
    python scripts/verify_catalog.py \\
        --ndjson data/products/off_en_v14.ndjson \\
        --taxonomy data/json_source/categories.json \\
        --lang en [--json out.json]

stdout is the JSON result and nothing else — the build workflow captures it to a
file. The human-readable summary, including the tolerance in force and any
failure reason, goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tolerance import Tolerance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from off_demo_extract.taxonomy import (  # noqa: E402
    PATH_SEPARATOR,
    display_label,
    global_roots,
    load_taxonomy,
)

MAX_EXAMPLES = 5

EXIT_OK = 0
EXIT_GATE = 1
EXIT_OPERATIONAL = 2


class CatalogError(Exception):
    """The verification could not be carried out.

    Distinct from a failing catalog: this says no verdict was reached. Raised for
    a missing or unreadable file and for a line that is not a JSON object, all of
    which used to surface as an uncaught traceback and an exit status of 1 —
    indistinguishable from "the artifact fails a gate".
    """


def values_tolerance(
    allow_values: Optional[int] = None,
    allow_values_fraction: Optional[float] = None,
) -> Tolerance:
    """How many off-snapshot values the operator has explicitly agreed to.

    Counted in **distinct values**, which is the number ``verify_index.py`` gates
    on for the same rule, and the number that says how much of the vocabulary is
    unexplained rather than how often the unexplained part occurs. That choice of
    denominator is this script's; what it does with it — floor a fraction, never
    round it — is ``scripts/tolerance.py``'s, shared with the two loaders.
    """
    return Tolerance(
        "--allow-values-outside-snapshot",
        "--allow-values-outside-snapshot-fraction",
        allow_values,
        allow_values_fraction,
    )


def _union_segments(path: List[str]) -> List[Tuple[str, str]]:
    """``(segment, address)`` for a prefix-closed union, or raise on a broken one.

    Every element must be either a single root segment or an extension of another
    element by exactly one segment. Derived by looking the parent prefix up in the
    value set rather than by trusting adjacency: a union is not ordered as a
    chain, so "extends its predecessor" is no longer the question — "its ancestor
    is present" is. Repeats are a violation of their own; ``category_path`` is a
    set of addresses and a duplicate means something emitted the same address
    twice.
    """
    values = set(path)
    if len(values) != len(path):
        raise ValueError("a cumulative path value is emitted more than once")
    out: List[Tuple[str, str]] = []
    for element in path:
        if not element:
            raise ValueError("empty path value")
        if PATH_SEPARATOR not in element:
            out.append((element, element))
            continue
        parent, _, tail = element.rpartition(PATH_SEPARATOR)
        if not tail:
            raise ValueError(f"{element!r} ends in the path separator")
        if parent not in values:
            raise ValueError(
                f"{element!r} is emitted but its ancestor {parent!r} is not: "
                "the union is not prefix-closed"
            )
        out.append((tail, element))
    return out


def _chain_segments(path: List[str]) -> List[str]:
    """The segment each cumulative element adds, or raise on a broken chain.

    Derived by subtracting the previous element rather than by splitting on the
    separator: labels have the separator neutralised at render time, but reading
    the delta is what actually proves the elements nest. Applied to the PRIMARY
    address only — the union is checked by :func:`_union_segments`.
    """
    segments: List[str] = []
    previous = ""
    for index, element in enumerate(path):
        if index == 0:
            if not element:
                raise ValueError("empty first element")
            segments.append(element)
        else:
            prefix = previous + "/"
            if not element.startswith(prefix):
                raise ValueError(f"element {index} does not extend its predecessor: {element!r}")
            tail = element[len(prefix) :]
            if not tail or "/" in tail:
                raise ValueError(f"element {index} adds {tail!r}, not exactly one segment")
            segments.append(tail)
        previous = element
    return segments


def verify(ndjson: Path, taxonomy_path: Path, lang: str) -> Dict[str, Any]:
    try:
        taxonomy = load_taxonomy(taxonomy_path)
    except OSError as exc:
        raise CatalogError(f"cannot read the taxonomy snapshot {taxonomy_path}: {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise CatalogError(f"the taxonomy snapshot {taxonomy_path} is not usable: {exc}") from exc
    root_labels: Set[str] = {display_label(taxonomy, node, lang) for node in global_roots(taxonomy)}
    vocabulary: Set[str] = {display_label(taxonomy, node, lang) for node in taxonomy}

    records = 0
    with_path = 0
    empty_path = 0
    ids: Dict[str, int] = defaultdict(int)
    chain_violations: List[Dict[str, Any]] = []
    union_violations: List[Dict[str, Any]] = []
    unanchored: List[Dict[str, Any]] = []
    primary_addresses: Dict[str, Set[str]] = defaultdict(set)
    addresses: Dict[str, Set[str]] = defaultdict(set)
    multi_address_records = 0
    path_values = 0
    max_path_values = 0
    distinct_paths: Set[str] = set()
    off_vocabulary: Dict[str, int] = defaultdict(int)
    # The denominator the off-snapshot count is a numerator of. Without it, "0
    # values outside the snapshot" cannot be told apart from "no value was ever
    # looked at", and a fraction tolerance has nothing to be a fraction of.
    checked_values: Set[str] = set()
    depths: Dict[int, int] = defaultdict(int)

    try:
        handle = ndjson.open(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"cannot read the catalog {ndjson}: {exc}") from exc

    with handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            records += 1
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CatalogError(f"{ndjson} line {number} is not valid JSON: {exc}") from exc
            if not isinstance(doc, dict):
                raise CatalogError(
                    f"{ndjson} line {number} is a {type(doc).__name__}, not a JSON object"
                )
            ids[str(doc.get("id") or "")] += 1
            path = doc.get("category_path") or []
            for value in doc.get("taxonomy_tags") or []:
                checked_values.add(value)
                if value not in vocabulary:
                    off_vocabulary[value] += 1
            primary = doc.get("category_path_primary") or []
            if not path:
                empty_path += 1
                continue
            with_path += 1
            depths[len(path)] += 1
            path_values += len(path)
            max_path_values = max(max_path_values, len(path))
            distinct_paths.update(path)
            if len(path) > len(primary):
                multi_address_records += 1

            # The union: prefix-closed, no repeats, and headed by the primary.
            try:
                union = _union_segments(path)
                if not primary:
                    # Refused rather than skipped. Every gate below is scoped to
                    # the primary, so a record without one would clear property 2
                    # and the anchoring check by having nothing to check — a
                    # vacuous pass indistinguishable from a clean catalog. A
                    # pre-restoration catalog, which carries no such field at all,
                    # therefore fails here by name instead of silently.
                    raise ValueError(
                        "the record carries a category_path but no "
                        "category_path_primary, so the address it leads with is "
                        "unstated and nothing below can be checked"
                    )
                if path[: len(primary)] != primary:
                    raise ValueError(
                        "category_path does not lead with category_path_primary"
                    )
            except ValueError as exc:
                if len(union_violations) < MAX_EXAMPLES:
                    union_violations.append(
                        {"id": doc.get("id"), "reason": str(exc), "path": path}
                    )
                else:
                    union_violations.append({"id": doc.get("id"), "reason": str(exc)})
                union = []

            # The primary: one chain, anchored at a taxonomy root, one address per
            # segment across the whole catalog.
            try:
                segments = _chain_segments(primary)
            except ValueError as exc:
                if len(chain_violations) < MAX_EXAMPLES:
                    chain_violations.append(
                        {"id": doc.get("id"), "reason": str(exc), "path": primary}
                    )
                else:
                    chain_violations.append({"id": doc.get("id"), "reason": str(exc)})
                segments = []
            if segments and segments[0] not in root_labels:
                if len(unanchored) < MAX_EXAMPLES:
                    unanchored.append({"id": doc.get("id"), "head": segments[0]})
                else:
                    unanchored.append({"id": doc.get("id")})
            for segment, address in zip(segments, primary):
                primary_addresses[segment].add(address)
            for segment, address in union:
                addresses[segment].add(address)
                checked_values.add(segment)
                if segment not in vocabulary:
                    off_vocabulary[segment] += 1

    multi_primary = {
        seg: sorted(addr) for seg, addr in primary_addresses.items() if len(addr) > 1
    }
    multi_address = {seg: sorted(addr) for seg, addr in addresses.items() if len(addr) > 1}
    return {
        "ndjson": str(ndjson),
        "lang": lang,
        "records": records,
        "with_category_path": with_path,
        "empty_category_path": empty_path,
        # An index keyed by ``id`` holds one document per distinct id, so this is
        # the number a post-index ``_count`` has to match — not the record count.
        # A silent gap between the two is how a catalog and its index start
        # disagreeing without either side looking wrong.
        "distinct_ids": len(ids),
        "duplicate_id_instances": records - len(ids),
        "distinct_categories_in_paths": len(addresses),
        "property_3_single_chain_violations": len(chain_violations),
        "property_3_examples": chain_violations[:MAX_EXAMPLES],
        "property_3_union_violations": len(union_violations),
        "property_3_union_examples": union_violations[:MAX_EXAMPLES],
        "property_2_categories_at_multiple_primary_addresses": len(multi_primary),
        "property_2_examples": [
            {"category": seg, "addresses": addr}
            for seg, addr in list(multi_primary.items())[:MAX_EXAMPLES]
        ],
        # Measured, not gated: this is the restored DAG, and the numbers below are
        # what a downstream ``category_path`` aggregation has to be sized against.
        # An aggregation still sized for the collapsed cardinality truncates
        # silently, and a truncated facet panel lies.
        "categories_at_multiple_addresses": len(multi_address),
        "multi_address_examples": [
            {"category": seg, "addresses": addr}
            for seg, addr in list(multi_address.items())[:MAX_EXAMPLES]
        ],
        "records_at_multiple_addresses": multi_address_records,
        "distinct_category_paths": len(distinct_paths),
        "mean_category_path_values": round(path_values / with_path, 3) if with_path else 0.0,
        "max_category_path_values": max_path_values,
        "unanchored_chains": len(unanchored),
        "unanchored_examples": unanchored[:MAX_EXAMPLES],
        "taxonomy_root_labels": len(root_labels),
        "values_checked_against_snapshot": len(checked_values),
        "values_outside_pinned_snapshot": len(off_vocabulary),
        "value_instances_outside_pinned_snapshot": sum(off_vocabulary.values()),
        "top_values_outside_pinned_snapshot": sorted(
            off_vocabulary.items(), key=lambda kv: -kv[1]
        )[:MAX_EXAMPLES],
        "path_depth_histogram": {str(k): depths[k] for k in sorted(depths)},
    }


def gate(
    result: Dict[str, Any], tolerance: Tolerance, require_unique_ids: bool = False
) -> List[Tuple[str, str]]:
    """The reasons this catalog must not report clean, as ``(check, reason)``.

    The single place the exit status is derived from. Previously the sum lived
    inline in ``main`` and three of the measured properties were in it while
    ``values_outside_pinned_snapshot`` was not — an omission no reader of the
    JSON could see, because nothing in the output said which numbers were fatal.
    """
    reasons: List[Tuple[str, str]] = []

    # Checked first and returned early: a run that examined nothing has not
    # cleared the catalog of anything, and every count below would be a
    # vacuous zero. It is also the denominator a fraction tolerance divides by.
    if result["records"] == 0:
        reasons.append(
            ("nothing_verified", "the catalog holds no records: nothing was verified")
        )
        return reasons
    if result["values_checked_against_snapshot"] == 0:
        reasons.append(
            (
                "nothing_verified",
                f"none of the {result['records']:,} records carried a taxonomy_tags value or a "
                "category_path segment, so no value was checked against the snapshot",
            )
        )
        return reasons

    if result["property_3_single_chain_violations"]:
        reasons.append(
            (
                "property_3_single_primary_chain",
                f"{result['property_3_single_chain_violations']:,} products whose "
                "category_path_primary is not a single root->leaf chain",
            )
        )
    if result["property_3_union_violations"]:
        reasons.append(
            (
                "property_3_prefix_closed_union",
                f"{result['property_3_union_violations']:,} products whose category_path is not "
                "a prefix-closed union headed by category_path_primary",
            )
        )
    if result["property_2_categories_at_multiple_primary_addresses"]:
        reasons.append(
            (
                "property_2_one_primary_address_per_category",
                f"{result['property_2_categories_at_multiple_primary_addresses']:,} categories "
                "have more than one PRIMARY address, so the breadcrumb a product page leads "
                "with is not a property of the category",
            )
        )
    if result["unanchored_chains"]:
        reasons.append(
            (
                "anchoring",
                f"{result['unanchored_chains']:,} chains do not start at a global taxonomy root",
            )
        )

    permitted = tolerance.permitted(result["values_checked_against_snapshot"])
    if result["values_outside_pinned_snapshot"] > permitted:
        reasons.append(
            (
                "values_outside_pinned_snapshot",
                f"{result['values_outside_pinned_snapshot']:,} of "
                f"{result['values_checked_against_snapshot']:,} distinct values are outside the "
                f"pinned snapshot ({result['value_instances_outside_pinned_snapshot']:,} "
                f"instances, tolerance {permitted:,}). The catalog's vocabulary is not the "
                "snapshot's",
            )
        )

    if require_unique_ids and result["duplicate_id_instances"]:
        reasons.append(
            (
                "unique_ids",
                f"{result['duplicate_id_instances']:,} records share an id already used by "
                f"another record ({result['records']:,} records, "
                f"{result['distinct_ids']:,} distinct ids)",
            )
        )
    return reasons


def summarise(result: Dict[str, Any]) -> str:
    duplicates = (
        f"  duplicate ids: {result['duplicate_id_instances']:,} records share an id"
        + (
            " (fatal: --require-unique-ids)"
            if result["require_unique_ids"]
            else " (reported, not gated — pass --require-unique-ids to fail on them)"
        )
    )
    return "\n".join(
        [
            f"{result['ndjson']} ({result['lang']}): {result['records']:,} records, "
            f"{result['distinct_ids']:,} distinct ids",
            f"  property 3 (one primary chain per product): "
            f"{result['property_3_single_chain_violations']:,} violations; "
            f"(prefix-closed union): {result['property_3_union_violations']:,} violations",
            f"  property 2 (one primary address per category): "
            f"{result['property_2_categories_at_multiple_primary_addresses']:,} categories at 2+ "
            "primary addresses",
            f"  addressing (measured, not gated): "
            f"{result['records_at_multiple_addresses']:,} records at 2+ addresses, "
            f"{result['categories_at_multiple_addresses']:,} categories at 2+ addresses, "
            f"{result['distinct_category_paths']:,} distinct category_path values "
            f"(mean {result['mean_category_path_values']}, max "
            f"{result['max_category_path_values']:,} per record)",
            f"  anchoring: {result['unanchored_chains']:,} chains not headed by a taxonomy root",
            f"  vocabulary: {result['values_outside_pinned_snapshot']:,} of "
            f"{result['values_checked_against_snapshot']:,} distinct values outside the pinned "
            f"snapshot ({result['value_instances_outside_pinned_snapshot']:,} instances) "
            f"| tolerance: {result['values_outside_snapshot_tolerance_source']}",
            duplicates,
            "  => "
            + (
                "FAILED: " + "; ".join(result["failure_reasons"])
                if result["failed"]
                else "all checks passed"
            ),
        ]
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a built catalog NDJSON against the taxonomy snapshot it was built from, "
            "and exit non-zero on every property it measures that does not hold."
        ),
        epilog=(
            "Exit status: 0 the catalog passes every gate (within any named tolerance); "
            "1 the catalog fails a gate; 2 the verification could not be carried out."
        ),
    )
    parser.add_argument("--ndjson", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--json", type=Path, default=None, help="also write the result here")
    allowance = parser.add_mutually_exclusive_group()
    allowance.add_argument(
        "--allow-values-outside-snapshot",
        type=int,
        default=None,
        metavar="N",
        help="permit up to N distinct values the pinned snapshot does not explain (default: 0)",
    )
    allowance.add_argument(
        "--allow-values-outside-snapshot-fraction",
        type=float,
        default=None,
        metavar="F",
        help=(
            "permit a fraction of the distinct values checked to be outside the snapshot, "
            "e.g. 0.01; floored to a whole number of values (default: 0)"
        ),
    )
    parser.add_argument(
        "--require-unique-ids",
        action="store_true",
        help=(
            "treat records sharing an id as a failure. Off by default: duplicate ids come from "
            "the dump, and the index is keyed by id, so verify_index.py compares its _count "
            "against distinct_ids rather than records"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    tolerance = values_tolerance(
        args.allow_values_outside_snapshot,
        args.allow_values_outside_snapshot_fraction,
    )
    problem = tolerance.problem()
    if problem is not None:
        parser.error(problem)

    try:
        result = verify(args.ndjson, args.taxonomy, args.lang)
    except CatalogError as exc:
        print(f"verify_catalog: {exc}", file=sys.stderr)
        return EXIT_OPERATIONAL

    checked = result["values_checked_against_snapshot"]
    result["values_outside_snapshot_tolerance"] = tolerance.permitted(checked)
    result["values_outside_snapshot_tolerance_source"] = tolerance.describe(checked)
    result["require_unique_ids"] = bool(args.require_unique_ids)
    failures = gate(result, tolerance, args.require_unique_ids)
    result["failed"] = [check for check, _ in failures]
    result["failure_reasons"] = [reason for _, reason in failures]

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json:
        try:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"verify_catalog: cannot write {args.json}: {exc}", file=sys.stderr)
            return EXIT_OPERATIONAL
    # stdout stays pure JSON — the build workflow captures it to a file. The
    # summary, the tolerance in force and every failure reason go to stderr.
    print(text)
    print("\n" + summarise(result), file=sys.stderr)
    for check, reason in failures:
        print(f"FAILED [{check}]: {reason}", file=sys.stderr)
    return EXIT_GATE if failures else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
